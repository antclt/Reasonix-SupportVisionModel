package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"reasonix/internal/event"
	"reasonix/internal/provider"
	"reasonix/internal/tool"
	"reasonix/internal/vision"
)

// fakeToolImageProcessor records every routed input and returns a scripted
// output, letting agent-level tests drive the text/vison split precisely.
type fakeToolImageProcessor struct {
	mu    sync.Mutex
	calls []vision.ToolImageInput
	out   vision.ToolImageOutput
}

func (f *fakeToolImageProcessor) ProcessToolImages(ctx context.Context, in vision.ToolImageInput) vision.ToolImageOutput {
	f.mu.Lock()
	f.calls = append(f.calls, in)
	f.mu.Unlock()
	return f.out
}

func (f *fakeToolImageProcessor) inputs() []vision.ToolImageInput {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]vision.ToolImageInput(nil), f.calls...)
}

const shotDataURL = "data:image/png;base64,QUFB"

func runShotTurn(t *testing.T, opts Options) (*Agent, *fakeToolImageProcessor, *Session) {
	t.Helper()
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "已读取图片文件：shot.png", images: []string{shotDataURL}})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{toolCallChunk("c1", "shot", `{}`), {Type: provider.ChunkDone}},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, opts, event.Discard)
	if err := a.Run(context.Background(), "look at the screenshot"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return a, nil, sess
}

func toolMessageImages(s *Session, name string) []string {
	for i := range s.Messages {
		if s.Messages[i].Role == provider.RoleTool && s.Messages[i].Name == name {
			return s.Messages[i].Images
		}
	}
	return nil
}

// TestAgentToolImagesTextModelDescribesAndClearsImages proves the routed
// description replaces the raw images on the tool message for a text model.
func TestAgentToolImagesTextModelDescribesAndClearsImages(t *testing.T) {
	fp := &fakeToolImageProcessor{out: vision.ToolImageOutput{
		Text:     "已读取图片文件：shot.png\n\n<tool-image-description source=\"shot\">\n图里有报错\n</tool-image-description>\n",
		Images:   nil,
		Success:  true,
		Attempts: 1,
	}}
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "已读取图片文件：shot.png", images: []string{shotDataURL}})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{toolCallChunk("c1", "shot", `{}`), {Type: provider.ChunkDone}},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, Options{ToolImages: fp, ModelRef: "qwen/qwen3.7-plus"}, event.Discard)
	if err := a.Run(context.Background(), "look at the screenshot"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if img := toolMessageImages(sess, "shot"); img != nil {
		t.Fatalf("tool message images = %v, want nil (text model sees description)", img)
	}
	if content := lastToolResult(sess, "shot"); !strings.Contains(content, "<tool-image-description") {
		t.Fatalf("tool message missing description:\n%s", content)
	}
	in := fp.inputs()
	if len(in) != 1 {
		t.Fatalf("processor calls = %d, want 1", len(in))
	}
	if in[0].ToolName != "shot" || in[0].ToolCallID != "c1" || in[0].ModelRef != "qwen/qwen3.7-plus" {
		t.Fatalf("input = %+v, want tool identity + model ref", in[0])
	}
	if in[0].ModelSupportsImages {
		t.Fatal("text model flagged as vision-capable")
	}
	if len(in[0].Images) != 1 || in[0].Images[0] != shotDataURL {
		t.Fatalf("processor images = %v", in[0].Images)
	}
	if !strings.Contains(in[0].TaskContext, "look at the screenshot") {
		t.Fatalf("task context missing:\n%s", in[0].TaskContext)
	}
}

// TestAgentToolImagesVisionCapableKeepsOriginals proves a multimodal agent
// model passes the raw images through to the provider (the processor is told
// ModelSupportsImages and keeps them).
func TestAgentToolImagesVisionCapableKeepsOriginals(t *testing.T) {
	fp := &fakeToolImageProcessor{out: vision.ToolImageOutput{Text: "x", Images: []string{shotDataURL}}}
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "已读取图片文件：shot.png", images: []string{shotDataURL}})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{toolCallChunk("c1", "shot", `{}`), {Type: provider.ChunkDone}},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, Options{ToolImages: fp, ModelSupportsImages: true}, event.Discard)
	if err := a.Run(context.Background(), "look"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if img := toolMessageImages(sess, "shot"); len(img) != 1 || img[0] != shotDataURL {
		t.Fatalf("tool message images = %v, want original kept", img)
	}
	if in := fp.inputs(); len(in) != 1 || !in[0].ModelSupportsImages {
		t.Fatalf("processor input = %+v, want ModelSupportsImages=true", in)
	}
}

// TestAgentToolImagesNoProcessorFallback proves the honest fallback for
// direct construction: no processor and a text model → images dropped, status
// appended.
func TestAgentToolImagesNoProcessorFallback(t *testing.T) {
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "已读取图片文件：shot.png", images: []string{shotDataURL}})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{toolCallChunk("c1", "shot", `{}`), {Type: provider.ChunkDone}},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, Options{}, event.Discard)
	if err := a.Run(context.Background(), "look"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if img := toolMessageImages(sess, "shot"); img != nil {
		t.Fatalf("text-model fallback kept images: %v", img)
	}
	if content := lastToolResult(sess, "shot"); !strings.Contains(content, "<tool-image-status>") || !strings.Contains(content, "不得声称已经看到了图片") {
		t.Fatalf("fallback status missing:\n%s", content)
	}
}

// TestAgentToolImagesNoProcessorVisionKeepsImages proves direct construction
// with a vision-capable model and no processor keeps the originals.
func TestAgentToolImagesNoProcessorVisionKeepsImages(t *testing.T) {
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "shot", images: []string{shotDataURL}})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{toolCallChunk("c1", "shot", `{}`), {Type: provider.ChunkDone}},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, Options{ModelSupportsImages: true}, event.Discard)
	if err := a.Run(context.Background(), "look"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if img := toolMessageImages(sess, "shot"); len(img) != 1 || img[0] != shotDataURL {
		t.Fatalf("vision-capable fallback dropped images: %v", img)
	}
}

// TestAgentToolImagesNoImagesSkipsProcessor proves a tool result without
// images never touches the processor.
func TestAgentToolImagesNoImagesSkipsProcessor(t *testing.T) {
	fp := &fakeToolImageProcessor{}
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "plain text"})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{toolCallChunk("c1", "shot", `{}`), {Type: provider.ChunkDone}},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, Options{ToolImages: fp}, event.Discard)
	if err := a.Run(context.Background(), "look"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if in := fp.inputs(); len(in) != 0 {
		t.Fatalf("processor called %d times for image-less result", len(in))
	}
	if content := lastToolResult(sess, "shot"); content != "plain text" {
		t.Fatalf("tool result changed: %q", content)
	}
}

// TestAgentToolImagesParallelBatchOrder proves the processor receives parallel
// tool results in call order, matching the session message order.
func TestAgentToolImagesParallelBatchOrder(t *testing.T) {
	fp := &fakeToolImageProcessor{out: vision.ToolImageOutput{Text: "seen", Images: nil}}
	reg := tool.NewRegistry()
	reg.Add(&fakeImageTool{text: "已读取图片文件：shot.png", images: []string{shotDataURL}})
	prov := &scriptedProvider{name: "p", turns: [][]provider.Chunk{
		{
			toolCallChunk("c1", "shot", `{}`),
			toolCallChunk("c2", "shot", `{}`),
			toolCallChunk("c3", "shot", `{}`),
			{Type: provider.ChunkDone},
		},
		{{Type: provider.ChunkText, Text: "done"}, {Type: provider.ChunkDone}},
	}}
	sess := NewSession("sys")
	a := New(prov, reg, sess, Options{ToolImages: fp}, event.Discard)
	if err := a.Run(context.Background(), "look"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	in := fp.inputs()
	if len(in) != 3 {
		t.Fatalf("processor calls = %d, want 3", len(in))
	}
	for i, want := range []string{"c1", "c2", "c3"} {
		if in[i].ToolCallID != want {
			t.Fatalf("call %d = %q, want %q (order must match call order)", i, in[i].ToolCallID, want)
		}
	}
	// Session messages stay in call order too.
	var toolIDs []string
	for i := range sess.Messages {
		if sess.Messages[i].Role == provider.RoleTool {
			toolIDs = append(toolIDs, sess.Messages[i].ToolCallID)
		}
	}
	if strings.Join(toolIDs, ",") != "c1,c2,c3" {
		t.Fatalf("session tool order = %v, want c1,c2,c3", toolIDs)
	}
}

// TestCurrentTaskContextPrefersRawUserContent pins the privacy boundary for
// the secondary vision provider: host-composed controller context may be sent
// to the main model, but only the user-authored text may orient tool-image
// recognition.
func TestCurrentTaskContextPrefersRawUserContent(t *testing.T) {
	sess := NewSession("sys")
	sess.Add(provider.Message{
		Role:       provider.RoleUser,
		Content:    "<host-context>internal routing data</host-context>\n\n用户原问题",
		RawContent: "用户原问题",
	})
	a := &Agent{session: sess}
	if got := a.currentTaskContext(); got != "用户原问题" {
		t.Fatalf("task context = %q, want raw user content only", got)
	}
}

// TestCurrentTaskContextPrefersPristineSubagentTask covers task, parallel and
// skill children. Their context inherits the root turn's RawUserInput, so the
// child-specific trusted task must win over both that root text and host
// framing stored in Content.
func TestCurrentTaskContextPrefersPristineSubagentTask(t *testing.T) {
	sess := NewSession("sys")
	sess.Add(provider.Message{
		Role:       provider.RoleUser,
		Content:    "<subagent-context>host framing</subagent-context>\n\n检查 child.png",
		RawContent: "根任务中的其他内容",
	})
	a := &Agent{session: sess, classifierTaskText: "检查 child.png"}
	if got := a.currentTaskContext(); got != "检查 child.png" {
		t.Fatalf("task context = %q, want pristine child task", got)
	}
}
