package control

import "reasonix/internal/imagedata"

// maxVisionDim caps the longest image side sent to a model. OpenAI and Anthropic
// downscale to roughly this server-side anyway, so a larger upload only wastes
// request bytes and image tokens without adding fidelity.
const maxVisionDim = imagedata.MaxVisionDimension

// compressForVision downscales an oversized image to maxVisionDim and re-encodes
// it — PNG/GIF stay lossless (screenshots, text, transparency), JPEG/WebP go to
// JPEG. Best-effort: an undecodable format, a decode/encode failure, or an image
// already within budget returns the original bytes and mime unchanged.
func compressForVision(raw []byte, mime string) ([]byte, string) {
	return imagedata.CompressForVision(raw, mime)
}
