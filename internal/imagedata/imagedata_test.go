package imagedata

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"
)

func TestDetectMIMEByMagic(t *testing.T) {
	cases := []struct {
		name string
		raw  []byte
		want string
	}{
		{"png", []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R'}, "image/png"},
		{"jpeg", []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J', 'F', 'I', 'F'}, "image/jpeg"},
		{"gif87a", []byte("GIF87a\x01\x00\x01\x00\x80\x00\x00"), "image/gif"},
		{"gif89a", []byte("GIF89a\x01\x00\x01\x00\x80\x00\x00"), "image/gif"},
		{"webp", []byte("RIFF\x24\x00\x00\x00WEBPVP8 \x0a\x00\x00\x00"), "image/webp"},
		{"tiff", []byte{0x49, 0x49, 0x2A, 0x00}, ""},
		{"bmp", []byte{0x42, 0x4D}, ""},
		{"svg", []byte("<svg xmlns='http://www.w3.org/2000/svg'>"), ""},
		{"empty", nil, ""},
		{"text", []byte("hello world\n"), ""},
		{"webp-short", []byte("RIFFWEBP"), ""},
	}
	for _, tc := range cases {
		if got := DetectMIME(tc.raw); got != tc.want {
			t.Errorf("%s: DetectMIME = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestDecodeKeepsSmallImage(t *testing.T) {
	raw := encodePNG(t, 100, 80)
	img := Decode(raw)
	if img.MIME != "image/png" {
		t.Fatalf("MIME = %q, want image/png", img.MIME)
	}
	if !bytes.Equal(img.Raw, raw) {
		t.Fatal("small image must be returned unchanged (no re-encode)")
	}
}

func TestDecodeDownscalesOversizedImage(t *testing.T) {
	raw := encodePNG(t, 3000, 1500)
	img := Decode(raw)
	if img.MIME != "image/png" {
		t.Fatalf("MIME = %q, want image/png", img.MIME)
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(img.Raw))
	if err != nil {
		t.Fatalf("decode compressed: %v", err)
	}
	if cfg.Width != maxVisionDim || cfg.Height != 1500*maxVisionDim/3000 {
		t.Errorf("dims = %dx%d, want %dx%d", cfg.Width, cfg.Height, maxVisionDim, 1500*maxVisionDim/3000)
	}
}

// TestDecodeRejectsGarbageWithMagic proves a payload that only carries a valid
// magic header but no decodable image is rejected — a broken file must not
// ride into a provider request.
func TestDecodeRejectsGarbageWithMagic(t *testing.T) {
	cases := [][]byte{
		{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0xAB, 0xAB, 0xAB},
		{0xFF, 0xD8, 0xFF, 0xE0},
		[]byte("GIF89a\x01\x00\x01\x00"),
		[]byte("RIFF\x24\x00\x00\x00WEBPVP8 \x0a\x00\x00\x00"),
	}
	for _, raw := range cases {
		if img := Decode(raw); img.MIME != "" {
			t.Errorf("Decode(% x) = %q, want rejected (empty MIME)", raw[:8], img.MIME)
		}
	}
}

// TestDecodeAcceptsMinimalWebP proves a real (minimal) WebP payload decodes
// and round-trips through the pipeline.
func TestDecodeAcceptsMinimalWebP(t *testing.T) {
	raw := []byte{
		0x52, 0x49, 0x46, 0x46, 0x1a, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
		0x56, 0x50, 0x38, 0x4c, 0x0e, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
		0x10, 0x07, 0x10, 0x11, 0x11, 0x88, 0x88, 0xfe, 0x07, 0x00,
	}
	img := Decode(raw)
	if img.MIME != "image/webp" {
		t.Fatalf("MIME = %q, want image/webp", img.MIME)
	}
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(img.Raw)); err != nil || cfg.Width != 1 || cfg.Height != 1 {
		t.Fatalf("decoded minimal webp = %dx%d (err=%v), want 1x1", cfg.Width, cfg.Height, err)
	}
}

func TestCompressForVisionConvertsGIFToPNG(t *testing.T) {
	// A wide GIF is losslessly re-encoded as PNG so transparency survives.
	img := image.NewPaletted(image.Rect(0, 0, 2000, 200), color.Palette{color.Black, color.White})
	for x := 0; x < 2000; x += 2 {
		img.Set(x, 100, color.White)
	}
	var buf bytes.Buffer
	if err := gif.Encode(&buf, img, nil); err != nil {
		t.Fatalf("gif encode: %v", err)
	}
	out, mime := CompressForVision(buf.Bytes(), "image/gif")
	if mime != "image/png" {
		t.Fatalf("mime = %q, want image/png", mime)
	}
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(out)); err != nil || cfg.Width != maxVisionDim {
		t.Fatalf("compressed gif decodes to %dx%d (err=%v), want width %d", cfg.Width, cfg.Height, err, maxVisionDim)
	}
}

func TestCompressForVisionJPEGStaysJPEG(t *testing.T) {
	raw := encodeJPEG(t, 2000, 200)
	out, mime := CompressForVision(raw, "image/jpeg")
	if mime != "image/jpeg" {
		t.Fatalf("mime = %q, want image/jpeg", mime)
	}
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(out)); err != nil || cfg.Width != maxVisionDim {
		t.Fatalf("compressed jpeg decodes to %dx%d (err=%v), want width %d", cfg.Width, cfg.Height, err, maxVisionDim)
	}
}

func TestDataURLRoundTrip(t *testing.T) {
	raw := []byte{1, 2, 3, 4}
	url := DataURL(raw, "image/png")
	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)
	if url != want {
		t.Fatalf("DataURL = %q, want %q", url, want)
	}
	if !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Fatalf("DataURL missing prefix: %q", url)
	}
}

func encodePNG(t *testing.T, w, h int) []byte {
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

func encodeJPEG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x % 256), G: uint8(y % 256), B: 128, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	return buf.Bytes()
}
