package control

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"reasonix/internal/config"
)

func writeVisionTestConfig(t *testing.T, root string) {
	t.Helper()
	cfg := config.Default()
	cfg.DefaultModel = "custom/vision-pro"
	cfg.Providers = []config.ProviderEntry{{
		Name:         "custom",
		Kind:         "openai",
		BaseURL:      "https://example.invalid/v1",
		Models:       []string{"text-only", "vision-pro"},
		VisionModels: []string{"vision-pro"},
	}}
	if err := cfg.SaveTo(filepath.Join(root, "reasonix.toml")); err != nil {
		t.Fatalf("save config: %v", err)
	}
}

func TestControllerInputImagesResolvesAttachment(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeVisionTestConfig(t, dir)
	ref, err := SaveImageDataURL("data:image/png;base64," + tinyPNG)
	if err != nil {
		t.Fatalf("SaveImageDataURL: %v", err)
	}
	urls := (&Controller{modelRef: "custom/vision-pro"}).inputImages("look at @" + ref)
	if len(urls) != 1 {
		t.Fatalf("inputImages = %v, want one resolved data URL", urls)
	}
	if !strings.HasPrefix(urls[0], "data:image/png;base64,") {
		t.Errorf("resolved url = %q, want a png data URL", urls[0])
	}
}

func TestControllerInputImagesResolvesWorkspaceAttachmentOutsideProcessCWD(t *testing.T) {
	workspace := t.TempDir()
	writeVisionTestConfig(t, workspace)
	ref, err := SaveImageBytesInRoot(workspace, "image/png", mustBase64(t, tinyPNG))
	if err != nil {
		t.Fatalf("SaveImageBytesInRoot: %v", err)
	}
	// Desktop image saving is scoped to the active workspace, but the process
	// working directory is restored before the controller resolves the turn.
	t.Chdir(t.TempDir())

	urls := (&Controller{workspaceRoot: workspace, modelRef: "custom/vision-pro"}).inputImages("look at @" + ref)
	if len(urls) != 1 {
		t.Fatalf("inputImages outside workspace cwd = %v, want one resolved data URL", urls)
	}
	if !strings.HasPrefix(urls[0], "data:image/png;base64,") {
		t.Errorf("resolved url = %q, want a png data URL", urls[0])
	}
}

func TestControllerInputImagesIgnoresNonAttachmentRefs(t *testing.T) {
	t.Chdir(t.TempDir())
	if urls := New(Options{}).inputImages("plain text with @missing.png"); len(urls) != 0 {
		t.Errorf("inputImages = %v, want none for a non-existent / non-attachment ref", urls)
	}
}

func TestControllerInputImagesResolvesWorkspaceImage(t *testing.T) {
	workspace := t.TempDir()
	writeVisionTestConfig(t, workspace)
	path := filepath.Join(workspace, "docs", "diagram.png")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, mustBase64(t, tinyPNG), 0o644); err != nil {
		t.Fatal(err)
	}

	urls := (&Controller{workspaceRoot: workspace, modelRef: "custom/vision-pro"}).inputImages("look at @docs/diagram.png")
	if len(urls) != 1 {
		t.Fatalf("inputImages = %v, want one resolved data URL", urls)
	}
	if !strings.HasPrefix(urls[0], "data:image/png;base64,") {
		t.Errorf("resolved url = %q, want a png data URL", urls[0])
	}
}

func TestControllerInputImagesResolvesAbsoluteWorkspaceImage(t *testing.T) {
	workspace := t.TempDir()
	writeVisionTestConfig(t, workspace)
	path := filepath.Join(workspace, "diagram.png")
	if err := os.WriteFile(path, mustBase64(t, tinyPNG), 0o644); err != nil {
		t.Fatal(err)
	}

	urls := (&Controller{workspaceRoot: workspace, modelRef: "custom/vision-pro"}).inputImages("look at @" + path)
	if len(urls) != 1 {
		t.Fatalf("inputImages = %v, want one resolved data URL", urls)
	}
	if !strings.HasPrefix(urls[0], "data:image/png;base64,") {
		t.Errorf("resolved url = %q, want a png data URL", urls[0])
	}
}

func TestControllerInputImagesRequiresWorkspaceForFileImageRefs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "diagram.png")
	if err := os.WriteFile(path, mustBase64(t, tinyPNG), 0o644); err != nil {
		t.Fatal(err)
	}

	urls := New(Options{}).inputImages("look at @" + path)
	if len(urls) != 0 {
		t.Fatalf("inputImages without a workspace = %v, want no file image refs", urls)
	}
}

func TestControllerInputImagesSkipsModelImagesWhenSelectedModelIsTextOnly(t *testing.T) {
	workspace := t.TempDir()
	cfg := config.Default()
	cfg.DefaultModel = "custom/text-only"
	cfg.Providers = []config.ProviderEntry{{
		Name:         "custom",
		Kind:         "openai",
		BaseURL:      "https://example.invalid/v1",
		Models:       []string{"text-only", "vision-pro"},
		VisionModels: []string{"vision-pro"},
	}}
	if err := cfg.SaveTo(filepath.Join(workspace, "reasonix.toml")); err != nil {
		t.Fatalf("save workspace config: %v", err)
	}
	path := filepath.Join(workspace, "diagram.png")
	if err := os.WriteFile(path, mustBase64(t, tinyPNG), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &Controller{workspaceRoot: workspace, modelRef: "custom/text-only"}
	if urls := c.inputImages("look at @diagram.png"); len(urls) != 0 {
		t.Fatalf("text-only model should suppress image payloads, got %v", urls)
	}

	c.modelRef = "custom/vision-pro"
	if urls := c.inputImages("look at @diagram.png"); len(urls) != 1 {
		t.Fatalf("vision model should keep image payloads, got %v", urls)
	}
}

func TestControllerImageInputEnabledDoesNotFallbackFromUnknownRef(t *testing.T) {
	workspace := t.TempDir()
	writeVisionTestConfig(t, workspace)

	c := &Controller{workspaceRoot: workspace, modelRef: "deleted/model"}
	if c.imageInputEnabled() {
		t.Fatal("unknown ref should not inherit image input from the default fallback model")
	}
}

func TestControllerResolveInputImagesIgnoresModelCapability(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeVisionTestConfig(t, dir)
	ref, err := SaveImageDataURL("data:image/png;base64," + tinyPNG)
	if err != nil {
		t.Fatalf("SaveImageDataURL: %v", err)
	}
	// The same config: vision-pro is the vision model, text-only is not.
	// inputImages gates on the main model; resolveInputImages must not.
	textOnly := (&Controller{modelRef: "custom/text-only"})
	if got := textOnly.inputImages("look at @" + ref); len(got) != 0 {
		t.Fatalf("inputImages for text-only model = %v, want nil (gated)", got)
	}
	imgs := textOnly.resolveInputImages("look at @" + ref)
	if len(imgs) != 1 {
		t.Fatalf("resolveInputImages = %+v, want one resolved image regardless of model", imgs)
	}
	if !strings.HasPrefix(imgs[0].DataURL, "data:image/png;base64,") {
		t.Errorf("DataURL = %q, want png data url", imgs[0].DataURL)
	}
	if imgs[0].Ref == "" || imgs[0].Path == "" {
		t.Errorf("ResolvedImage keeps ref/path for notices: %+v", imgs[0])
	}
}

func TestControllerResolveInputImagesPreservesOrder(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeVisionTestConfig(t, dir)
	r1, err := SaveImageDataURL("data:image/png;base64," + tinyPNG)
	if err != nil {
		t.Fatalf("SaveImageDataURL: %v", err)
	}
	r2, err := SaveImageDataURL("data:image/png;base64," + tinyPNG)
	if err != nil {
		t.Fatalf("SaveImageDataURL: %v", err)
	}
	imgs := (&Controller{modelRef: "custom/vision-pro"}).resolveInputImages("first @" + r1 + " second @" + r2)
	if len(imgs) != 2 {
		t.Fatalf("resolveInputImages = %+v, want two images", imgs)
	}
	if imgs[0].Ref != r1 || imgs[1].Ref != r2 {
		t.Errorf("order not preserved: %+v", imgs)
	}
}

func TestControllerResolveInputImagesKeepsUnreadableImageCandidate(t *testing.T) {
	workspace := t.TempDir()
	badPath := filepath.Join(workspace, "bad.png")
	if err := os.WriteFile(badPath, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workspace, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c := &Controller{workspaceRoot: workspace, modelRef: "custom/text-only"}
	images := c.resolveInputImages("look at @bad.png")
	if len(images) != 1 {
		t.Fatalf("resolveInputImages = %+v, want one failed image candidate", images)
	}
	if images[0].DataURL != "" || images[0].Error == "" {
		t.Fatalf("failed image = %+v, want empty DataURL and non-empty Error", images[0])
	}
	if got := c.resolveInputImages("read @main.go"); len(got) != 0 {
		t.Fatalf("non-image ref = %+v, want ignored", got)
	}
}

func TestImageRefNoteReflectsRouterState(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writeVisionTestConfig(t, dir) // custom/vision-pro supports vision, custom/text-only does not

	vision := (&Controller{modelRef: "custom/vision-pro"})
	textOnly := (&Controller{modelRef: "custom/text-only"})
	textOnlyWithFallback := (&Controller{modelRef: "custom/text-only", visionModelRef: "custom/vision-pro"})

	// Vision-capable main model: note says direct image input.
	if got := vision.imageRefNote("shot.png", "image/png", 42, true); !strings.Contains(got, "vision-capable main model") {
		t.Errorf("vision-capable note = %q, want direct-input wording", got)
	}

	// Text-only main model + configured vision fallback: note points at the router.
	if got := textOnlyWithFallback.imageRefNote("shot.png", "image/png", 42, true); !strings.Contains(got, "configured vision model will describe") {
		t.Errorf("fallback note = %q, want vision-model wording", got)
	}

	// Text-only main model + vision_model that itself cannot read images:
	// the note must NOT promise a description (the router will go path-only).
	badFallback := (&Controller{modelRef: "custom/text-only", visionModelRef: "custom/text-only"})
	if got := badFallback.imageRefNote("shot.png", "image/png", 42, true); strings.Contains(got, "configured vision model will describe") {
		t.Errorf("unsupported vision_model note = %q, must not promise a description", got)
	}

	// Text-only without fallback: keep the default OCR/tool guidance.
	if got := textOnly.imageRefNote("shot.png", "image/png", 42, true); !strings.Contains(got, "OCR/image/vision tool") {
		t.Errorf("text-only note = %q, want default OCR wording", got)
	}

	// Attachment form (empty mime) also reflects the fallback.
	if got := textOnlyWithFallback.imageRefNote("shot.png", "", 0, true); !strings.Contains(got, "configured vision model will describe") {
		t.Errorf("attachment fallback note = %q, want vision-model wording", got)
	}
}
