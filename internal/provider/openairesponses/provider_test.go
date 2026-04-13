package openairesponses

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/frediansh/rcopenai/internal/provider"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

// ---------------------------------------------------------------------------
// Mock
// ---------------------------------------------------------------------------

type mockResponsesAPI struct {
	NewFunc func(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error)
}

func (m *mockResponsesAPI) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	return m.NewFunc(ctx, body, opts...)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func mustUnmarshalResponse(raw string) *responses.Response {
	var r responses.Response
	if err := json.Unmarshal([]byte(raw), &r); err != nil {
		panic(fmt.Sprintf("mustUnmarshalResponse: %v\nJSON: %s", err, raw))
	}
	return &r
}

func makeTextResponse(id, text string, input, output, total int64) *responses.Response {
	textJSON, _ := json.Marshal(text)
	idJSON, _ := json.Marshal(id)
	raw := fmt.Sprintf(`{
		"id":%s,"created_at":0,"error":{},"incomplete_details":{},"metadata":{},
		"model":"gpt-4o","object":"response","parallel_tool_calls":false,
		"output":[{"type":"message","id":"msg-1","role":"assistant","status":"completed",
		           "content":[{"type":"output_text","text":%s,"annotations":[]}]}],
		"usage":{"input_tokens":%d,"output_tokens":%d,"total_tokens":%d,
		         "input_tokens_details":{},"output_tokens_details":{}}
	}`, idJSON, textJSON, input, output, total)
	return mustUnmarshalResponse(raw)
}

func makeEmptyResponse(id string) *responses.Response {
	idJSON, _ := json.Marshal(id)
	raw := fmt.Sprintf(`{
		"id":%s,"created_at":0,"error":{},"incomplete_details":{},"metadata":{},
		"model":"gpt-4o","object":"response","parallel_tool_calls":false,
		"output":[],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,
		         "input_tokens_details":{},"output_tokens_details":{}}
	}`, idJSON)
	return mustUnmarshalResponse(raw)
}

func makeFunctionCallResponse(id, callID, name, arguments string) *responses.Response {
	idJSON, _ := json.Marshal(id)
	callIDJSON, _ := json.Marshal(callID)
	nameJSON, _ := json.Marshal(name)
	argsJSON, _ := json.Marshal(arguments)
	raw := fmt.Sprintf(`{
		"id":%s,"created_at":0,"error":{},"incomplete_details":{},"metadata":{},
		"model":"gpt-4o","object":"response","parallel_tool_calls":false,
		"output":[{"type":"function_call","id":"fc-1","call_id":%s,"name":%s,"arguments":%s,"status":"completed"}],
		"usage":{"input_tokens":5,"output_tokens":10,"total_tokens":15,
		         "input_tokens_details":{},"output_tokens_details":{}}
	}`, idJSON, callIDJSON, nameJSON, argsJSON)
	return mustUnmarshalResponse(raw)
}

// pngMagic is the first 8 bytes of a PNG file (PNG signature).
var pngMagic = []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}

func writeTempFile(t *testing.T, ext string, data []byte) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "testimg*"+ext)
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatalf("Write: %v", err)
	}
	f.Close()
	return f.Name()
}

func newTestProvider(api responsesAPI) *Provider {
	return &Provider{api: api}
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

func TestNew_MissingAPIKey(t *testing.T) {
	_, err := New("", "")
	if err == nil {
		t.Fatal("expected error for empty api key")
	}
}

func TestNew_ValidAPIKey(t *testing.T) {
	p, err := New("test-key", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p == nil {
		t.Fatal("expected non-nil Provider")
	}
	if p.api == nil {
		t.Fatal("expected api to be set")
	}
}

// ---------------------------------------------------------------------------
// resolveImageSource
// ---------------------------------------------------------------------------

func TestResolveImageSource_Empty(t *testing.T) {
	_, err := resolveImageSource("")
	if err == nil {
		t.Fatal("expected error for empty image source")
	}
}

func TestResolveImageSource_Whitespace(t *testing.T) {
	_, err := resolveImageSource("   ")
	if err == nil {
		t.Fatal("expected error for whitespace image source")
	}
}

func TestResolveImageSource_HTTPS(t *testing.T) {
	src := "https://example.com/image.png"
	got, err := resolveImageSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != src {
		t.Errorf("got %q, want %q", got, src)
	}
}

func TestResolveImageSource_HTTP(t *testing.T) {
	src := "http://example.com/image.jpg"
	got, err := resolveImageSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != src {
		t.Fatal("expected HTTP URL returned as-is")
	}
}

func TestResolveImageSource_DataURL(t *testing.T) {
	src := "data:image/png;base64,abc123"
	got, err := resolveImageSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != src {
		t.Fatal("expected data URL returned as-is")
	}
}

func TestResolveImageSource_FileScheme_ValidPNG(t *testing.T) {
	path := writeTempFile(t, ".png", pngMagic)
	src := "file://" + path
	got, err := resolveImageSource(src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("expected data URL, got %q", got[:min(len(got), 40)])
	}
}

func TestResolveImageSource_LocalPath_ValidPNG(t *testing.T) {
	path := writeTempFile(t, ".png", pngMagic)
	got, err := resolveImageSource(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("expected data URL, got %q", got[:min(len(got), 40)])
	}
}

func TestResolveImageSource_LocalPath_NotFound(t *testing.T) {
	_, err := resolveImageSource("/nonexistent/path/image.png")
	if err == nil {
		t.Fatal("expected error for non-existent file")
	}
}

// ---------------------------------------------------------------------------
// localPathToDataURL
// ---------------------------------------------------------------------------

func TestLocalPathToDataURL_Empty(t *testing.T) {
	_, err := localPathToDataURL("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestLocalPathToDataURL_Whitespace(t *testing.T) {
	_, err := localPathToDataURL("   ")
	if err == nil {
		t.Fatal("expected error for whitespace path")
	}
}

func TestLocalPathToDataURL_NotFound(t *testing.T) {
	_, err := localPathToDataURL("/definitely/does/not/exist.png")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLocalPathToDataURL_EmptyFile(t *testing.T) {
	path := writeTempFile(t, ".png", []byte{})
	_, err := localPathToDataURL(path)
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestLocalPathToDataURL_NonImageFile(t *testing.T) {
	path := writeTempFile(t, ".txt", []byte("just text content here, not an image at all"))
	_, err := localPathToDataURL(path)
	if err == nil {
		t.Fatal("expected error for non-image file")
	}
}

func TestLocalPathToDataURL_ValidPNG(t *testing.T) {
	path := writeTempFile(t, ".png", pngMagic)
	got, err := localPathToDataURL(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("expected data URL with image/png, got %q", got[:min(len(got), 50)])
	}
}

// ---------------------------------------------------------------------------
// toTurnOutput (tested directly — same package)
// ---------------------------------------------------------------------------

func TestToTurnOutput_EmptyOutput(t *testing.T) {
	resp := makeEmptyResponse("resp-empty")
	out := toTurnOutput(resp)
	if out.ResponseID != "resp-empty" {
		t.Errorf("ResponseID: got %q, want %q", out.ResponseID, "resp-empty")
	}
	if out.Text != "" {
		t.Errorf("Text: got %q, want empty", out.Text)
	}
	if len(out.ToolCalls) != 0 {
		t.Errorf("ToolCalls: got %d, want 0", len(out.ToolCalls))
	}
}

func TestToTurnOutput_TextResponse(t *testing.T) {
	resp := makeTextResponse("resp-txt", "hello world", 10, 20, 30)
	out := toTurnOutput(resp)
	if out.Text != "hello world" {
		t.Errorf("Text: got %q, want %q", out.Text, "hello world")
	}
	if out.Usage.InputTokens != 10 || out.Usage.OutputTokens != 20 || out.Usage.TotalTokens != 30 {
		t.Errorf("Usage: got %+v", out.Usage)
	}
	if len(out.ToolCalls) != 0 {
		t.Errorf("ToolCalls: unexpected %d items", len(out.ToolCalls))
	}
}

func TestToTurnOutput_FunctionCall(t *testing.T) {
	resp := makeFunctionCallResponse("resp-fn", "call-99", "my-tool", `{"key":"val"}`)
	out := toTurnOutput(resp)
	if len(out.ToolCalls) != 1 {
		t.Fatalf("ToolCalls: got %d, want 1", len(out.ToolCalls))
	}
	tc := out.ToolCalls[0]
	if tc.CallID != "call-99" {
		t.Errorf("CallID: got %q, want %q", tc.CallID, "call-99")
	}
	if tc.Name != "my-tool" {
		t.Errorf("Name: got %q, want %q", tc.Name, "my-tool")
	}
	if tc.Arguments != `{"key":"val"}` {
		t.Errorf("Arguments: got %q", tc.Arguments)
	}
}

// ---------------------------------------------------------------------------
// RunTurn
// ---------------------------------------------------------------------------

func TestRunTurn_TextResponse(t *testing.T) {
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, _ responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			return makeTextResponse("resp-1", "hello world", 5, 10, 15), nil
		},
	}
	p := newTestProvider(mock)
	out, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Text != "hello world" {
		t.Errorf("Text: got %q, want %q", out.Text, "hello world")
	}
	if out.ResponseID != "resp-1" {
		t.Errorf("ResponseID: got %q", out.ResponseID)
	}
}

func TestRunTurn_FunctionCall(t *testing.T) {
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, _ responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			return makeFunctionCallResponse("resp-fn", "call-1", "search", `{"q":"go"}`), nil
		},
	}
	p := newTestProvider(mock)
	out, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user", Content: "search for go"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("ToolCalls: got %d, want 1", len(out.ToolCalls))
	}
	tc := out.ToolCalls[0]
	if tc.Name != "search" || tc.CallID != "call-1" {
		t.Errorf("ToolCall: %+v", tc)
	}
}

func TestRunTurn_WithInstructions(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:        "gpt-4o",
		Instructions: "Be helpful.",
		Messages:     []provider.Message{{Role: "user", Content: "hi"}},
	})
	if !capturedBody.Instructions.Valid() || capturedBody.Instructions.Value != "Be helpful." {
		t.Errorf("Instructions not set correctly: %v", capturedBody.Instructions)
	}
}

func TestRunTurn_NoInstructions(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if capturedBody.Instructions.Valid() {
		t.Errorf("Instructions should not be set, got %v", capturedBody.Instructions.Value)
	}
}

func TestRunTurn_WithPreviousResponseID(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:              "gpt-4o",
		PreviousResponseID: "prev-resp-id",
		FunctionOutputs: []provider.FunctionCallOutput{
			{CallID: "call-1", Output: "result text"},
		},
	})
	if !capturedBody.PreviousResponseID.Valid() || capturedBody.PreviousResponseID.Value != "prev-resp-id" {
		t.Errorf("PreviousResponseID not set correctly: %v", capturedBody.PreviousResponseID)
	}
}

func TestRunTurn_WithTools(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model: "gpt-4o",
		Tools: []provider.ToolDefinition{
			{Name: "search", Description: "Search the web", Parameters: map[string]any{"type": "object"}},
		},
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if len(capturedBody.Tools) != 1 {
		t.Errorf("Tools: got %d, want 1", len(capturedBody.Tools))
	}
}

func TestRunTurn_MessageRoles(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model: "gpt-4o",
		Messages: []provider.Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "world"},
			{Role: "system", Content: "sys"},
			{Role: "developer", Content: "dev"},
		},
	})
	if capturedBody.Input.OfInputItemList == nil {
		t.Fatal("expected input item list to be set")
	}
	if len(capturedBody.Input.OfInputItemList) != 4 {
		t.Errorf("messages: got %d, want 4", len(capturedBody.Input.OfInputItemList))
	}
}

func TestRunTurn_APIError(t *testing.T) {
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, _ responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			return nil, errors.New("api unavailable")
		},
	}
	p := newTestProvider(mock)
	_, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "gpt-4o",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "api unavailable") {
		t.Errorf("error should mention api unavailable, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// DescribeImage
// ---------------------------------------------------------------------------

func TestDescribeImage_EmptySource(t *testing.T) {
	called := false
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, _ responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			called = true
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	_, err := p.DescribeImage(context.Background(), "gpt-4o", "", "describe this", "")
	if err == nil {
		t.Fatal("expected error for empty image source")
	}
	if called {
		t.Error("API should not be called when imageSource is invalid")
	}
}

func TestDescribeImage_HTTPSSource(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeTextResponse("resp-img", "a cat", 10, 20, 30), nil
		},
	}
	p := newTestProvider(mock)
	out, err := p.DescribeImage(context.Background(), "gpt-4o", "", "what is this?", "https://example.com/cat.png")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Text != "a cat" {
		t.Errorf("Text: got %q, want %q", out.Text, "a cat")
	}
	if capturedBody.Input.OfInputItemList == nil {
		t.Fatal("input not set")
	}
}

func TestDescribeImage_WithInstructions(t *testing.T) {
	var capturedBody responses.ResponseNewParams
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, body responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			capturedBody = body
			return makeEmptyResponse("r"), nil
		},
	}
	p := newTestProvider(mock)
	p.DescribeImage(context.Background(), "gpt-4o", "You are an image analyzer.", "describe", "https://example.com/img.png")
	if !capturedBody.Instructions.Valid() || capturedBody.Instructions.Value != "You are an image analyzer." {
		t.Errorf("Instructions not set correctly: %v", capturedBody.Instructions)
	}
}

func TestDescribeImage_APIError(t *testing.T) {
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, _ responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			return nil, errors.New("vision api down")
		},
	}
	p := newTestProvider(mock)
	_, err := p.DescribeImage(context.Background(), "gpt-4o", "", "describe", "https://example.com/img.png")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDescribeImage_LocalPNGFile(t *testing.T) {
	path := writeTempFile(t, ".png", pngMagic)
	mock := &mockResponsesAPI{
		NewFunc: func(_ context.Context, _ responses.ResponseNewParams, _ ...option.RequestOption) (*responses.Response, error) {
			return makeTextResponse("resp-local", "local image", 5, 10, 15), nil
		},
	}
	p := newTestProvider(mock)
	out, err := p.DescribeImage(context.Background(), "gpt-4o", "", "describe", path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Text != "local image" {
		t.Errorf("Text: got %q", out.Text)
	}
}

// ---------------------------------------------------------------------------
// Close
// ---------------------------------------------------------------------------

func TestClose_ReturnsNil(t *testing.T) {
	p := newTestProvider(nil)
	if err := p.Close(); err != nil {
		t.Errorf("Close() = %v, want nil", err)
	}
}

// ---------------------------------------------------------------------------
// localPathToDataURL — octet-stream branch
// ---------------------------------------------------------------------------

// TestLocalPathToDataURL_OctetStreamWithImageExt covers the branch where
// http.DetectContentType returns "application/octet-stream" but the file
// extension is a known image type, so mime.TypeByExtension succeeds.
func TestLocalPathToDataURL_OctetStreamWithImageExt(t *testing.T) {
	// Arbitrary bytes that don't match any magic number → "application/octet-stream"
	arbitraryBytes := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13}
	path := writeTempFile(t, ".jpg", arbitraryBytes)
	got, err := localPathToDataURL(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "data:image/jpeg;base64,") {
		t.Errorf("expected data URL with image/jpeg, got %q", got[:min(len(got), 50)])
	}
}

// TestLocalPathToDataURL_OctetStreamUnknownExt covers the branch where
// http.DetectContentType returns "application/octet-stream" and the
// extension is not a known image type → error.
func TestLocalPathToDataURL_OctetStreamUnknownExt(t *testing.T) {
	arbitraryBytes := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13}
	// ".xyz" is not a known MIME type — mime.TypeByExtension returns ""
	path := writeTempFile(t, ".xyz", arbitraryBytes)
	_, err := localPathToDataURL(path)
	if err == nil {
		t.Fatal("expected error for non-image octet-stream file")
	}
}

// min is needed for Go < 1.21 compatibility.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
