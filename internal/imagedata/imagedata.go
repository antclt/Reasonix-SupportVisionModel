// Package imagedata provides format detection, downscaling, and data-URL
// encoding for images that tools return to the model. It is the neutral
// building block shared by file-reading tools (read_file) and any future
// binary-capable tool: detection is content-based (never extension-based),
// the supported set is exactly PNG/JPEG/GIF/WebP, and every payload is
// bounded so a large or hostile file can never inflate a prompt.
package imagedata

import (
	"bytes"
	"encoding/base64"
	"image"
	_ "image/gif" // register gif decoder
	"image/jpeg"
	"image/png"

	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/webp" // register webp decoder
)

// MaxBytes is the largest single image payload accepted. It matches the
// attachment limit: anything larger is rejected rather than truncated, so the
// model never silently receives a partial image.
const MaxBytes = 10 * 1024 * 1024

// MaxVisionDimension caps the longest image side sent to a model. It is
// exported so every image entry point (user attachments and tool results) uses
// the same multimodal request budget.
const MaxVisionDimension = 1568

const maxVisionDim = MaxVisionDimension

// maxDecodePixels guards against decompression-bomb images: a tiny file can
// declare enormous dimensions. Beyond this we skip decoding and send as-is
// (still bounded by MaxBytes).
const maxDecodePixels = 50_000_000

var (
	pngMagic  = []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}
	jpegMagic = []byte{0xFF, 0xD8, 0xFF}
	gif87a    = []byte("GIF87a")
	gif89a    = []byte("GIF89a")
	webpRIFF  = []byte("RIFF")
	webpTag   = []byte("WEBP")
)

// DetectMIME sniffs the image format from raw bytes by magic number. It
// returns only the four supported media types ("image/png", "image/jpeg",
// "image/gif", "image/webp") and "" for anything else, so callers can reject
// BMP/TIFF/SVG and arbitrary binary without trusting file extensions.
func DetectMIME(raw []byte) string {
	switch {
	case bytes.HasPrefix(raw, pngMagic):
		return "image/png"
	case bytes.HasPrefix(raw, jpegMagic):
		return "image/jpeg"
	case bytes.HasPrefix(raw, gif87a), bytes.HasPrefix(raw, gif89a):
		return "image/gif"
	case isWebP(raw):
		return "image/webp"
	}
	return ""
}

// isWebP matches the RIFF container layout: "RIFF" + 4-byte little-endian
// size + "WEBP" at offset 8.
func isWebP(raw []byte) bool {
	if len(raw) < 12 || !bytes.HasPrefix(raw, webpRIFF) || !bytes.Equal(raw[8:12], webpTag) {
		return false
	}
	return true
}

// CompressForVision downscales an oversized image to maxVisionDim and
// re-encodes it — PNG/GIF stay lossless (screenshots, text, transparency),
// JPEG/WebP go to JPEG. Best-effort: an undecodable format, a decode/encode
// failure, or an image already within budget returns the original bytes and
// mime unchanged.
func CompressForVision(raw []byte, mime string) ([]byte, string) {
	switch mime {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
	default:
		return raw, mime // unsupported: no decoder wired, send original
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil || cfg.Width*cfg.Height > maxDecodePixels {
		return raw, mime
	}
	if cfg.Width <= maxVisionDim && cfg.Height <= maxVisionDim {
		return raw, mime // within budget — no point re-encoding
	}
	src, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return raw, mime
	}
	w, h := scaledDims(cfg.Width, cfg.Height, maxVisionDim)
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), xdraw.Over, nil)

	var buf bytes.Buffer
	if mime == "image/png" || mime == "image/gif" {
		if err := png.Encode(&buf, dst); err != nil {
			return raw, mime
		}
		return buf.Bytes(), "image/png"
	}
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 85}); err != nil {
		return raw, mime
	}
	return buf.Bytes(), "image/jpeg"
}

// scaledDims returns dimensions with the longest side clamped to m, preserving
// aspect ratio (each side at least 1px).
func scaledDims(w, h, m int) (int, int) {
	if w >= h {
		nh := h * m / w
		if nh < 1 {
			nh = 1
		}
		return m, nh
	}
	nw := w * m / h
	if nw < 1 {
		nw = 1
	}
	return nw, m
}

// DataURL encodes raw image bytes as a data URL of the given media type
// (data:<mime>;base64,<payload>), the transport shape provider request
// builders and the vision describer expect.
func DataURL(raw []byte, mime string) string {
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw)
}

// Image is one decoded image: its detected media type and the raw (possibly
// downscaled) bytes. An Image with MIME "" signals that the payload is not a
// supported image.
type Image struct {
	MIME string
	Raw  []byte
}

// Decode validates, and optionally downscales, raw image bytes. It returns
// MIME "" when raw is not one of the four supported formats or when the magic
// header matches but the payload is truncated/corrupt (a broken image must not
// ride into a provider request); an oversized image is still returned
// uncompressed (callers keep the MaxBytes cap). Compressing up front keeps one
// giant screenshot from ever entering a provider request.
func Decode(raw []byte) Image {
	mime := DetectMIME(raw)
	if mime == "" {
		return Image{}
	}
	if _, _, err := image.DecodeConfig(bytes.NewReader(raw)); err != nil {
		return Image{}
	}
	raw, mime = CompressForVision(raw, mime)
	return Image{MIME: mime, Raw: raw}
}
