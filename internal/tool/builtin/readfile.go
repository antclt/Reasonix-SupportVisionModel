// Package builtin provides Reasonix's compile-time built-in tools. Each tool
// self-registers via init(); main blank-imports this package to wire them in.
package builtin

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/text/transform"

	fileenc "reasonix/internal/fileutil/encoding"
	"reasonix/internal/imagedata"
	"reasonix/internal/tool"
)

const (
	readFileBinaryPeek   = 8 * 1024   // bytes scanned for NUL before reading further
	readFileDetectSample = 256 * 1024 // bytes sampled for encoding detection before streaming
)

func init() { tool.RegisterBuiltin(readFile{}) }

// readFile reads a text file. workDir, when non-empty, is the directory a
// relative path is resolved against (see resolveIn). paths maps session-scoped
// external read aliases to local roots without changing the model-visible tool
// schema. forbidRoots lists directories the tool may not read from (resolved,
// absolute paths).
type readFile struct {
	workDir     string
	paths       *PathResolver
	forbidRoots []string
	// overlay, when non-nil, serves content from the host transport (unsaved
	// editor buffers) before falling back to disk. Consulted only after path
	// resolution and read confinement, and never for external alias paths.
	overlay FileOverlay
}

const (
	readFileDefaultLimit = 2000 // lines returned when limit is unset
)

func (readFile) Name() string { return "read_file" }

func (readFile) Description() string {
	return "Read a text file or a supported image file (PNG, JPEG, GIF, or WebP). Image files are returned through structured image content so the current multimodal model, or the configured vision fallback, can inspect the actual pixels. When a shell command, archive extraction, folder listing, or document extraction produces image paths that must be visually inspected, call read_file on the actual image files; printing paths, sizes, or dimensions does not expose image content. Text output prefixes each line with its 1-based number (e.g. `   42→...`) so subsequent edit_file calls can target exact lines. Use `offset` and `limit` to page through large text files; the tool reports total length and pagination hints in a trailer."
}

func (readFile) Schema() json.RawMessage {
	return json.RawMessage(`{
"type":"object",
"properties":{
  "path":{"type":"string","description":"Path to a text file or a supported PNG/JPEG/GIF/WebP image file"},
  "offset":{"type":"integer","description":"0-based line offset to start reading from (default 0)","minimum":0},
  "limit":{"type":"integer","description":"Maximum lines to return (default 2000)","minimum":1}
},
"required":["path"]
}`)
}

func (readFile) ReadOnly() bool { return true }

// SnipHint front-loads file content: the most relevant lines are near the top,
// so keep a generous head and a short tail when an old read is shortened.
func (readFile) SnipHint() tool.SnipHint {
	return tool.SnipHint{Head: 120, Tail: 12, HeadChars: 12000, TailChars: 2000}
}

func (r readFile) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	text, _, err := r.ExecuteWithImages(ctx, args)
	return text, err
}

// ExecuteWithImages implements tool.ImageTool: text files behave exactly like
// Execute (images nil), while PNG/JPEG/GIF/WebP files return a short text
// summary plus the image as a data URL. The base64 payload never enters the
// tool text — the output-truncation budget would corrupt it and it would bloat
// the context — it rides the structured Images channel instead. Any other
// binary keeps the historical error.
func (r readFile) ExecuteWithImages(ctx context.Context, args json.RawMessage) (string, []string, error) {
	var p struct {
		Path   string `json:"path"`
		Offset int    `json:"offset,omitempty"`
		Limit  int    `json:"limit,omitempty"`
	}
	if err := json.Unmarshal(args, &p); err != nil {
		return "", nil, fmt.Errorf("invalid args: %w", err)
	}
	if p.Path == "" {
		return "", nil, fmt.Errorf("path is required")
	}
	rp := resolveReadablePath(r.workDir, p.Path, r.paths)
	p.Path = rp.Path
	displayPath := rp.DisplayPath
	if confineRead(r.forbidRoots, p.Path) {
		err := &os.PathError{Op: "open", Path: p.Path, Err: os.ErrNotExist}
		if rp.External {
			return "", nil, fmt.Errorf("read %s: %s", displayPath, rp.ErrorText(err))
		}
		return "", nil, err
	}
	if p.Offset < 0 {
		p.Offset = 0
	}
	if p.Limit <= 0 {
		p.Limit = readFileDefaultLimit
	}

	// The host overlay (unsaved editor buffers) wins over the disk when it can
	// serve the path. Content arrives already decoded as text, so the encoding
	// and binary-detection pipeline below applies to the disk fallback only.
	if r.overlay != nil && !rp.External && filepath.IsAbs(p.Path) {
		if content, ok := r.overlay.ReadTextFile(ctx, p.Path); ok {
			text, serr := r.scan(strings.NewReader(content), p.Offset, p.Limit)
			return text, nil, serr
		}
	}

	// A directory can be os.Open'd but not read as text — catch it up front with
	// an actionable message (and avoid the doubled "read X: read X:" the scanner's
	// error would otherwise produce) so the model switches to the ls tool.
	if info, err := os.Stat(p.Path); err == nil && info.IsDir() {
		return "", nil, fmt.Errorf("%s is a directory, not a file — use the ls tool to list it, or read a specific file inside it", displayPath)
	}

	f, err := os.Open(p.Path)
	if err != nil {
		if rp.External {
			return "", nil, fmt.Errorf("read %s: %s", displayPath, rp.ErrorText(err))
		}
		return "", nil, fmt.Errorf("read %s: %w", displayPath, err)
	}
	defer f.Close()

	// Peek the first 8 KiB to reject binary files cheaply (a NUL byte) before
	// reading further — keeps a multi-GB archive from being slurped just to be
	// discarded.
	peek := make([]byte, readFileBinaryPeek)
	pn, perr := io.ReadFull(f, peek)
	peek = peek[:pn]
	peekEOF := perr != nil // whole file fit in the peek (EOF / ErrUnexpectedEOF)

	// A NUL-carrying payload that sniffs as a supported image is returned as
	// structured image output instead of a binary error — this is the seam
	// where read_file learns to read workspace screenshots and diagrams. The
	// check runs before the text pipeline (and independent of the NUL probe)
	// so even an image whose first bytes are NUL-free is never mis-scanned as
	// text.
	if mime := imagedata.DetectMIME(peek); mime != "" {
		return r.readImage(f, peek, displayPath)
	}

	// BOM check first: UTF-16 files contain 0x00 for every ASCII character, so a
	// naive NUL check would misidentify them as binary.
	switch fileenc.DetectQuick(peek) {
	case fileenc.UTF16LE, fileenc.UTF16BE:
		// UTF-16 is not self-synchronising and can't be streamed line-by-line, so
		// buffer it fully (these files are rare and usually small).
		rest, rerr := io.ReadAll(f)
		if rerr != nil {
			if rp.External {
				return "", nil, fmt.Errorf("read %s: %s", displayPath, rp.ErrorText(rerr))
			}
			return "", nil, fmt.Errorf("read %s: %w", displayPath, rerr)
		}
		all := append(peek, rest...)
		bom := fileenc.DetectQuick(all)
		text, serr := r.scan(bytes.NewReader(fileenc.Decode(all, bom)), p.Offset, p.Limit)
		return text, nil, serr
	case fileenc.UTF8BOM:
		// Strip the 3-byte BOM; the content is valid UTF-8 and streams directly.
		body := peek
		if len(body) >= 3 {
			body = body[3:]
		}
		text, serr := r.scan(io.MultiReader(bytes.NewReader(body), f), p.Offset, p.Limit)
		return text, nil, serr
	}

	// BOM-less UTF-16 (Windows source files) has a NUL for every ASCII char but
	// no BOM, so it reaches here; recognise it by its NUL pattern and decode it
	// rather than rejecting it as binary.
	if k, ok := fileenc.DetectUTF16NoBOM(peek); ok {
		rest, rerr := io.ReadAll(f)
		if rerr != nil {
			if rp.External {
				return "", nil, fmt.Errorf("read %s: %s", displayPath, rp.ErrorText(rerr))
			}
			return "", nil, fmt.Errorf("read %s: %w", displayPath, rerr)
		}
		all := append(peek, rest...)
		text, serr := r.scan(bytes.NewReader(fileenc.Decode(all, k)), p.Offset, p.Limit)
		return text, nil, serr
	}

	if bytes.IndexByte(peek, 0) >= 0 {
		if rp.External {
			return "", nil, fmt.Errorf("binary file %s (NUL byte detected); not shown by read_file", displayPath)
		}
		return "", nil, fmt.Errorf("binary file %s (NUL byte detected); use `bash hexdump` or another tool", displayPath)
	}

	// Read up to a bounded sample for encoding detection, then stream the rest —
	// so a large text file isn't slurped whole just to return a few lines.
	head := peek
	if !peekEOF {
		more := make([]byte, readFileDetectSample-len(peek))
		mn, merr := io.ReadFull(f, more)
		head = append(peek, more[:mn]...)
		peekEOF = merr != nil
	}

	// Detect from a char-safe slice: when more file follows, trim to the last
	// newline so the sample never ends mid multi-byte sequence (UTF-8 and GB18030
	// are ASCII-transparent, so '\n' is always a clean boundary).
	sample := head
	if !peekEOF {
		if i := bytes.LastIndexByte(head, '\n'); i >= 0 {
			sample = head[:i+1]
		}
	}
	enc, _ := fileenc.Detect(sample)

	src := io.MultiReader(bytes.NewReader(head), f)
	if dec := fileenc.Decoder(enc); dec != nil {
		text, serr := r.scan(transform.NewReader(src, dec), p.Offset, p.Limit)
		return text, nil, serr
	}
	text, serr := r.scan(src, p.Offset, p.Limit)
	return text, nil, serr
}

// readImage turns an already-opened image file into a short text summary plus
// one data URL. f is positioned after the peek bytes; raw content is assembled
// from peek + the remainder. The text stays small and path-relative; the image
// bytes travel only in the structured Images result so the tool-text truncation
// budget can never corrupt a base64 payload. The stat/read re-stat sequence
// detects a file swapped or modified mid-read.
func (r readFile) readImage(f *os.File, peek []byte, displayPath string) (string, []string, error) {
	before, err := f.Stat()
	if err != nil {
		return "", nil, fmt.Errorf("read %s: %w", displayPath, err)
	}
	if before.Size() > imagedata.MaxBytes {
		return "", nil, fmt.Errorf("image %s exceeds 10 MB limit", displayPath)
	}
	rest, rerr := io.ReadAll(io.LimitReader(f, imagedata.MaxBytes+1))
	if rerr != nil {
		return "", nil, fmt.Errorf("read %s: %w", displayPath, rerr)
	}
	raw := make([]byte, 0, len(peek)+len(rest))
	raw = append(raw, peek...)
	raw = append(raw, rest...)
	if len(raw) > imagedata.MaxBytes {
		return "", nil, fmt.Errorf("image %s exceeds 10 MB limit", displayPath)
	}
	after, aerr := f.Stat()
	if aerr != nil {
		return "", nil, fmt.Errorf("read %s: %w", displayPath, aerr)
	}
	if !os.SameFile(before, after) || before.Size() != after.Size() {
		return "", nil, fmt.Errorf("image %s changed while reading; re-read it", displayPath)
	}
	img := imagedata.Decode(raw)
	if img.MIME == "" {
		// Content no longer matches a supported image (spoofed extension).
		return "", nil, fmt.Errorf("binary file %s (not a supported image); use `bash hexdump` or another tool", displayPath)
	}
	text := fmt.Sprintf("已读取图片文件：%s\n格式：%s\n大小：%d bytes", displayPath, img.MIME, len(raw))
	return text, []string{imagedata.DataURL(img.Raw, img.MIME)}, nil
}

// scan reads lines from src and returns the formatted output with line numbers.
func (r readFile) scan(src io.Reader, offset, limit int) (string, error) {
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var collected []string
	lineNo := 0
	hasMore := false
	for scanner.Scan() {
		lineNo++
		if lineNo <= offset {
			continue
		}
		if len(collected) < limit {
			collected = append(collected, scanner.Text())
			continue
		}
		// A line past the requested window exists — stop here rather than reading
		// the rest of the file just to count the remainder.
		hasMore = true
		break
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scan: %w", err)
	}

	if lineNo == 0 {
		return "(empty file)", nil
	}
	if len(collected) == 0 {
		return fmt.Sprintf("(offset %d is past EOF — file has %d lines)", offset, lineNo), nil
	}

	maxShown := offset + len(collected)
	w := len(fmt.Sprint(maxShown))

	var b strings.Builder
	for i, line := range collected {
		fmt.Fprintf(&b, "%*d→%s\n", w, offset+i+1, line)
	}
	if hasMore {
		fmt.Fprintf(&b, "\n[more lines below; pass offset=%d to continue]\n", offset+len(collected))
	}
	return b.String(), nil
}
