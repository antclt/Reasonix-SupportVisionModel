package vision

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"reasonix/internal/event"
	"reasonix/internal/provider"
)

// fakeProvider streams scripted chunks and records the last request.
type fakeProvider struct {
	name  string
	chunk []provider.Chunk
	mu    sync.Mutex
	req   provider.Request
	ctx   context.Context
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Stream(ctx context.Context, req provider.Request) (<-chan provider.Chunk, error) {
	f.mu.Lock()
	f.req = req
	f.ctx = ctx
	f.mu.Unlock()
	ch := make(chan provider.Chunk)
	go func() {
		defer close(ch)
		for _, c := range f.chunk {
			select {
			case ch <- c:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch, nil
}

func (f *fakeProvider) lastReq() provider.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.req
}

func (f *fakeProvider) lastCtx() context.Context {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.ctx
}

func testImages() []Image {
	return []Image{
		{Ref: "a.png", Path: "a.png", DataURL: "data:image/png;base64,AAAA"},
		{Ref: "b.jpg", Path: "b.jpg", DataURL: "data:image/jpeg;base64,BBBB"},
	}
}

func TestDescribeOnceBuildsToolLessRequest(t *testing.T) {
	fake := &fakeProvider{name: "qwen", chunk: []provider.Chunk{{Type: provider.ChunkText, Text: "图1: 一只猫; 图2: 一只狗"}}}
	d := NewProviderDescriber(fake, nil, nil)

	desc, usage, err := d.DescribeOnce(context.Background(), "qwen/qwen-vl", testImages(), "图里有什么？")
	if err != nil {
		t.Fatalf("DescribeOnce: %v", err)
	}
	if !strings.Contains(desc, "猫") || !strings.Contains(desc, "狗") {
		t.Fatalf("description = %q, want collected text", desc)
	}
	if usage != nil {
		t.Fatalf("usage = %v, want nil (no usage chunk)", usage)
	}

	req := fake.lastReq()
	if req.Tools != nil {
		t.Fatalf("vision request must have nil Tools, got %+v", req.Tools)
	}
	if req.Temperature == nil || *req.Temperature != 0 {
		t.Fatalf("vision request temperature = %v, want 0", req.Temperature)
	}
	if req.MaxTokens != 4096 {
		t.Fatalf("vision request MaxTokens = %d, want 4096", req.MaxTokens)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("messages = %d, want system + user", len(req.Messages))
	}
	if req.Messages[0].Role != provider.RoleSystem || !strings.Contains(req.Messages[0].Content, "图片内容识别") {
		t.Fatalf("system message wrong: %+v", req.Messages[0])
	}
	user := req.Messages[1]
	if user.Role != provider.RoleUser {
		t.Fatalf("user role = %q", user.Role)
	}
	if len(user.Images) != 2 || user.Images[0] != "data:image/png;base64,AAAA" || user.Images[1] != "data:image/jpeg;base64,BBBB" {
		t.Fatalf("user images = %v, want both data URLs in order", user.Images)
	}
	if !strings.Contains(user.Content, "图里有什么？") || !strings.Contains(user.Content, "图片1") || !strings.Contains(user.Content, "图片2") {
		t.Fatalf("user prompt missing question/order markers: %q", user.Content)
	}
}

func TestDescribeOnceReportsUsageAndEmitsEvent(t *testing.T) {
	got := &usageSink{}
	fake := &fakeProvider{name: "qwen", chunk: []provider.Chunk{
		{Type: provider.ChunkText, Text: "desc"},
		{Type: provider.ChunkUsage, Usage: &provider.Usage{PromptTokens: 10, CompletionTokens: 5}},
	}}
	d := NewProviderDescriber(fake, nil, got)

	_, usage, err := d.DescribeOnce(context.Background(), "qwen/qwen-vl", testImages(), "")
	if err != nil {
		t.Fatalf("DescribeOnce: %v", err)
	}
	if usage == nil || usage.PromptTokens != 10 || usage.CompletionTokens != 5 {
		t.Fatalf("usage = %+v, want 10/5", usage)
	}
	if len(got.events) != 1 {
		t.Fatalf("events = %d, want one usage event", len(got.events))
	}
	if got.events[0].UsageSource != event.UsageSourceVision {
		t.Fatalf("usage source = %q, want %q", got.events[0].UsageSource, event.UsageSourceVision)
	}
	if got.events[0].ModelRef != "qwen/qwen-vl" {
		t.Fatalf("usage model ref = %q, want configured vision model", got.events[0].ModelRef)
	}
}

func TestDescribeOnceRejectsToolCall(t *testing.T) {
	for _, chunkType := range []provider.ChunkType{
		provider.ChunkToolCallStart,
		provider.ChunkToolCallArgsDelta,
		provider.ChunkToolCall,
	} {
		t.Run(fmt.Sprint(chunkType), func(t *testing.T) {
			fake := &fakeProvider{name: "qwen", chunk: []provider.Chunk{{Type: chunkType, ToolCall: &provider.ToolCall{ID: "c", Name: "bash"}}}}
			d := NewProviderDescriber(fake, nil, nil)

			_, _, err := d.DescribeOnce(context.Background(), "qwen/qwen-vl", testImages(), "")
			if !errors.Is(err, ErrUnexpectedVisionToolCall) {
				t.Fatalf("err = %v, want ErrUnexpectedVisionToolCall", err)
			}
		})
	}
}

func TestDescribeOnceEmptyOutputIsAllowedAndStreamErrorSurfaces(t *testing.T) {
	empty := &fakeProvider{name: "qwen", chunk: nil}
	d := NewProviderDescriber(empty, nil, nil)
	desc, _, err := d.DescribeOnce(context.Background(), "qwen/qwen-vl", testImages(), "")
	if err != nil {
		t.Fatalf("empty output should not error from the describer (router judges emptiness): %v", err)
	}
	if desc != "" {
		t.Fatalf("desc = %q, want empty", desc)
	}

	errFake := &fakeProvider{name: "qwen", chunk: []provider.Chunk{{Type: provider.ChunkError, Err: errors.New("boom")}}}
	d2 := NewProviderDescriber(errFake, nil, nil)
	if _, _, err := d2.DescribeOnce(context.Background(), "qwen/qwen-vl", testImages(), ""); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err = %v, want stream error surfaced", err)
	}
}

func TestDescribeOnceRequiresImages(t *testing.T) {
	fake := &fakeProvider{name: "qwen"}
	d := NewProviderDescriber(fake, nil, nil)
	if _, _, err := d.DescribeOnce(context.Background(), "qwen/qwen-vl", nil, "hi"); err == nil {
		t.Fatal("expected error for empty images")
	}
}

// TestDescribeOncePinsSingleHTTPAttempt proves the transport retry budget is
// zeroed per describe call — the outer router, never SendWithRetry or the
// reconnect loop, owns retries.
func TestDescribeOncePinsSingleHTTPAttempt(t *testing.T) {
	fake := &fakeProvider{name: "qwen", chunk: []provider.Chunk{{Type: provider.ChunkText, Text: "d"}}}
	d := NewProviderDescriber(fake, nil, nil)
	if _, _, err := d.DescribeOnce(context.Background(), "qwen/qwen-vl", testImages(), ""); err != nil {
		t.Fatalf("DescribeOnce: %v", err)
	}
	if max, ok := provider.MaxRetriesFromContext(fake.lastCtx()); !ok || max != 0 {
		t.Fatalf("MaxRetriesFromContext = (%d, %v), want (0, true)", max, ok)
	}
}

// TestDescribeToolImagesOnceBuildsScopedRequest proves tool-result images get
// a dedicated system prompt and a prompt scoped to the tool result — the
// session history never reaches the vision model.
func TestDescribeToolImagesOnceBuildsScopedRequest(t *testing.T) {
	fake := &fakeProvider{name: "qwen", chunk: []provider.Chunk{{Type: provider.ChunkText, Text: "图1: 报错信息"}}}
	d := NewProviderDescriber(fake, nil, nil)
	desc, _, err := d.DescribeToolImagesOnce(context.Background(), "qwen/qwen-vl", ToolImageDescribeInput{
		ToolName:    "read_file",
		ToolText:    "已读取图片文件：screenshots/error.png",
		TaskContext: "排查构建失败",
		Images:      testImages(),
	})
	if err != nil {
		t.Fatalf("DescribeToolImagesOnce: %v", err)
	}
	if desc != "图1: 报错信息" {
		t.Fatalf("desc = %q", desc)
	}
	req := fake.lastReq()
	if len(req.Messages) != 2 {
		t.Fatalf("messages = %d, want system + user", len(req.Messages))
	}
	sys := req.Messages[0]
	if sys.Role != provider.RoleSystem || !strings.Contains(sys.Content, "工具执行返回的图片") {
		t.Fatalf("tool-image system prompt wrong: %+v", sys)
	}
	user := req.Messages[1]
	for _, want := range []string{"read_file", "已读取图片文件", "排查构建失败", "图片1", "图片2"} {
		if !strings.Contains(user.Content, want) {
			t.Errorf("user prompt missing %q: %s", want, user.Content)
		}
	}
	if len(user.Images) != 2 || user.Images[0] != "data:image/png;base64,AAAA" || user.Images[1] != "data:image/jpeg;base64,BBBB" {
		t.Fatalf("user images = %v, want both in order", user.Images)
	}
	if req.Tools != nil {
		t.Fatalf("tool-image request must not carry tools")
	}
}

type usageSink struct {
	events []event.Event
}

func (s *usageSink) Emit(e event.Event) { s.events = append(s.events, e) }
