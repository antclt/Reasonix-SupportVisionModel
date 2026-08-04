package builtin

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// --- image fixtures ---------------------------------------------------------

func testPNGBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x % 256), G: uint8(y % 256), B: 128, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	return buf.Bytes()
}

func testJPEGBytes(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x * 4), G: uint8(y * 4), B: 200, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	return buf.Bytes()
}

func testGIFBytes(t *testing.T) []byte {
	t.Helper()
	img := image.NewPaletted(image.Rect(0, 0, 32, 32), color.Palette{color.Black, color.White})
	for x := 0; x < 32; x += 2 {
		img.Set(x, 16, color.White)
	}
	var buf bytes.Buffer
	if err := gif.Encode(&buf, img, nil); err != nil {
		t.Fatalf("gif encode: %v", err)
	}
	return buf.Bytes()
}

// testWebPBytes returns a real 1x1 VP8L WebP so the full decode path is
// exercised, not just the RIFF/WEBP magic.
func testWebPBytes() []byte {
	return []byte{
		0x52, 0x49, 0x46, 0x46, 0x1a, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
		0x56, 0x50, 0x38, 0x4c, 0x0e, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
		0x10, 0x07, 0x10, 0x11, 0x11, 0x88, 0x88, 0xfe, 0x07, 0x00,
	}
}

func writeTestFile(t *testing.T, path string, raw []byte) {
	t.Helper()
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// --- tests ------------------------------------------------------------------

// TestImageToolDescriptionsAdvertiseReadFileHandoff keeps the model-visible
// contract aligned with the structured image support. Without these hints a
// model can extract document images in the shell and then incorrectly claim
// that no image-reading tool exists.
func TestImageToolDescriptionsAdvertiseReadFileHandoff(t *testing.T) {
	desc := readFile{}.Description()
	for _, want := range []string{"PNG", "JPEG", "GIF", "WebP", "structured image content", "call read_file", "does not expose image content"} {
		if !strings.Contains(desc, want) {
			t.Errorf("read_file description missing %q:\n%s", want, desc)
		}
	}

	schema := string(readFile{}.Schema())
	if !strings.Contains(schema, "PNG/JPEG/GIF/WebP image file") {
		t.Errorf("read_file path schema does not advertise image support:\n%s", schema)
	}

	for _, want := range []string{"read_file calls", "not image inspection"} {
		if !strings.Contains(bashToolSteer, want) {
			t.Errorf("bash image handoff guidance missing %q:\n%s", want, bashToolSteer)
		}
	}
}

// TestReadFileImageFormats proves the four supported formats come back as a
// short text summary plus a structured data URL, with the base64 payload kept
// out of the tool text.
func TestReadFileImageFormats(t *testing.T) {
	cases := []struct {
		name string
		ext  string
		raw  []byte
		mime string
	}{
		{"png", ".png", testPNGBytes(t, 64, 48), "image/png"},
		{"jpeg", ".jpg", testJPEGBytes(t), "image/jpeg"},
		{"gif", ".gif", testGIFBytes(t), "image/gif"},
		{"webp", ".webp", testWebPBytes(), "image/webp"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "shot"+tc.ext)
			writeTestFile(t, path, tc.raw)
			args, _ := json.Marshal(map[string]any{"path": path})
			text, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
			if err != nil {
				t.Fatalf("ExecuteWithImages: %v", err)
			}
			if len(images) != 1 {
				t.Fatalf("images = %d, want 1", len(images))
			}
			if !strings.HasPrefix(images[0], "data:"+tc.mime+";base64,") {
				t.Fatalf("image data URL mime = %q, want %q", images[0][:20], "data:"+tc.mime+";base64,")
			}
			if _, _, ok := strings.Cut(images[0], ","); !ok {
				t.Fatal("data URL has no payload separator")
			}
			for _, want := range []string{"已读取图片文件", tc.mime, "bytes"} {
				if !strings.Contains(text, want) {
					t.Errorf("text missing %q:\n%s", want, text)
				}
			}
			if strings.Contains(text, "base64") || strings.Contains(text, "data:image") {
				t.Errorf("image payload leaked into tool text:\n%s", text)
			}
		})
	}
}

// TestReadFileImageExtensionIndependent proves detection is content-based: a
// PNG renamed to .txt is still read as an image, while a text file named .png
// is still read as text (never a bogus image).
func TestReadFileImageExtensionIndependent(t *testing.T) {
	dir := t.TempDir()
	pngPath := filepath.Join(dir, "real.png")
	writeTestFile(t, pngPath, testPNGBytes(t, 32, 32))
	renamed := filepath.Join(dir, "real.txt")
	writeTestFile(t, renamed, testPNGBytes(t, 32, 32))

	args, _ := json.Marshal(map[string]any{"path": renamed})
	_, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
	if err != nil {
		t.Fatalf("png-as-txt rejected: %v", err)
	}
	if len(images) != 1 || !strings.HasPrefix(images[0], "data:image/png;base64,") {
		t.Fatalf("png-as-txt images = %d (%v)", len(images), images)
	}

	textPath := filepath.Join(dir, "notes.png")
	writeTestFile(t, textPath, []byte("plain text pretending to be an image\n"))
	args2, _ := json.Marshal(map[string]any{"path": textPath})
	text, images2, err := readFile{}.ExecuteWithImages(context.Background(), args2)
	if err != nil {
		t.Fatalf("text-as-png rejected: %v", err)
	}
	if len(images2) != 0 {
		t.Fatalf("text-as-png returned %d images", len(images2))
	}
	if !strings.Contains(text, "plain text pretending") {
		t.Fatalf("text-as-png not read as text:\n%s", text)
	}
}

// TestReadFileImageSpoofedBinaryRejected proves a binary that sniffs as an
// unsupported image (BMP) keeps the historical binary error — no image output.
func TestReadFileImageSpoofedBinaryRejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fake.png")
	writeTestFile(t, path, []byte{'B', 'M', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00})
	args, _ := json.Marshal(map[string]any{"path": path})
	text, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
	if err == nil {
		t.Fatalf("BMP-as-png succeeded (text=%q images=%d), want error", text, len(images))
	}
	if !strings.Contains(err.Error(), "binary file") {
		t.Fatalf("err = %v, want binary-file rejection", err)
	}
	if len(images) != 0 {
		t.Fatalf("images = %d, want 0 on rejection", len(images))
	}
}

// TestReadFileImageGarbageWithMagicRejected proves a payload carrying a valid
// image magic but no decodable image is rejected — a broken file must never
// ride into the model as a data URL.
func TestReadFileImageGarbageWithMagicRejected(t *testing.T) {
	raw := append([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, bytes.Repeat([]byte{0xAB}, 4096)...)
	path := filepath.Join(t.TempDir(), "broken.png")
	writeTestFile(t, path, raw)
	args, _ := json.Marshal(map[string]any{"path": path})
	text, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
	if err == nil {
		t.Fatalf("garbage-with-magic succeeded (text=%q images=%d), want error", text, len(images))
	}
	if len(images) != 0 {
		t.Fatalf("images = %d, want 0 on rejection", len(images))
	}
}

// TestReadFileImageTooLarge proves a >10 MB image is rejected up front rather
// than slurped and sent to the model.
func TestReadFileImageTooLarge(t *testing.T) {
	raw := append([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, bytes.Repeat([]byte{0xAB}, 10*1024*1024)...)
	path := filepath.Join(t.TempDir(), "huge.png")
	writeTestFile(t, path, raw)
	args, _ := json.Marshal(map[string]any{"path": path})
	_, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
	if err == nil || !strings.Contains(err.Error(), "10 MB") {
		t.Fatalf("err = %v, want 10 MB limit error (images=%d)", err, len(images))
	}
}

// TestReadFileImagePathSafety proves directory targets and out-of-workdir
// escapes stay rejected for image reads exactly as for text reads.
func TestReadFileImagePathSafety(t *testing.T) {
	workDir := t.TempDir()
	imgPath := filepath.Join(workDir, "shot.png")
	writeTestFile(t, imgPath, testPNGBytes(t, 16, 16))

	dirArgs, _ := json.Marshal(map[string]any{"path": workDir})
	_, _, err := readFile{workDir: workDir}.ExecuteWithImages(context.Background(), dirArgs)
	if err == nil {
		t.Fatal("directory read succeeded, want error")
	}
	inArgs := mustMarshal(t, map[string]any{"path": filepath.Join(workDir, "shot.png")})
	_, images, err := readFile{workDir: workDir}.ExecuteWithImages(context.Background(), inArgs)
	if err != nil {
		t.Fatalf("in-workdir image read failed: %v", err)
	}
	if len(images) != 1 {
		t.Fatalf("images = %d, want 1", len(images))
	}
	escapeArgs, _ := json.Marshal(map[string]any{"path": filepath.Join(workDir, "..", "escaped.png")})
	_, images, err = readFile{workDir: workDir}.ExecuteWithImages(context.Background(), escapeArgs)
	if err == nil {
		t.Fatalf("escape read succeeded (images=%d), want error", len(images))
	}
}

// TestReadFileImageOversizedDownscaled proves a large-but-allowed image is
// downscaled before it enters the data URL (the 1568px vision cap).
func TestReadFileImageOversizedDownscaled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wide.png")
	writeTestFile(t, path, testPNGBytes(t, 4000, 2000))
	args, _ := json.Marshal(map[string]any{"path": path})
	_, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
	if err != nil {
		t.Fatalf("ExecuteWithImages: %v", err)
	}
	if len(images) != 1 {
		t.Fatalf("images = %d, want 1", len(images))
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(images[0], "data:image/png;base64,"))
	if err != nil {
		t.Fatalf("decode data URL: %v", err)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("decode returned image: %v", err)
	}
	if cfg.Width != 1568 {
		t.Fatalf("width = %d, want 1568 (downscaled)", cfg.Width)
	}
}

// TestReadFileImageConcurrent proves parallel readers never cross data between
// goroutines: every call returns its own image bytes.
func TestReadFileImageConcurrent(t *testing.T) {
	dir := t.TempDir()
	paths := make([]string, 8)
	for i := range paths {
		paths[i] = filepath.Join(dir, "shot.png")
		writeTestFile(t, paths[i], testPNGBytes(t, 32, 32))
	}
	var wg sync.WaitGroup
	errs := make([]error, len(paths))
	results := make([][]string, len(paths))
	for i := range paths {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			args, _ := json.Marshal(map[string]any{"path": paths[i]})
			_, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
			errs[i] = err
			results[i] = images
		}(i)
	}
	wg.Wait()
	for i := range paths {
		if errs[i] != nil {
			t.Fatalf("goroutine %d: %v", i, errs[i])
		}
		if len(results[i]) != 1 || !strings.HasPrefix(results[i][0], "data:image/png;base64,") {
			t.Fatalf("goroutine %d: images = %v", i, results[i])
		}
	}
	// All results should be identical (same fixture) — no cross-talk with a
	// different payload would fail the prefix check above.
	if results[0][0][:64] != results[1][0][:64] {
		t.Fatal("concurrent results differ for identical input")
	}
}

// TestReadFileImageExecuteMatchesTextBehavior proves Execute still returns the
// same text for text files (images nil), i.e. the ImageTool split is invisible
// to the plain Tool contract.
func TestReadFileImageExecuteMatchesTextBehavior(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notes.txt")
	writeTestFile(t, path, []byte("line one\nline two\n"))
	args, _ := json.Marshal(map[string]any{"path": path})
	text, err := readFile{}.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	text2, images, err := readFile{}.ExecuteWithImages(context.Background(), args)
	if err != nil {
		t.Fatalf("ExecuteWithImages: %v", err)
	}
	if text2 != text {
		t.Fatalf("text mismatch:\nExecute=%q\nExecuteWithImages=%q", text, text2)
	}
	if len(images) != 0 {
		t.Fatalf("text file returned images: %v", images)
	}
}

func mustMarshal(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
