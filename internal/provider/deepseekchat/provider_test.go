package deepseekchat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/frediansh/rcopenai/internal/provider"
)

// ---------------------------------------------------------------------------
// Mock
// ---------------------------------------------------------------------------

type mockChatCompletionsAPI struct {
	NewFunc func(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
}

func (m *mockChatCompletionsAPI) New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	return m.NewFunc(ctx, body, opts...)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeTextCompletion(id, content string, promptTokens, completionTokens, totalTokens int64) *openai.ChatCompletion {
	idJSON, _ := json.Marshal(id)
	contentJSON, _ := json.Marshal(content)
	raw := fmt.Sprintf(`{
		"id":%s,"object":"chat.completion","created":0,"model":"deepseek-chat",
		"choices":[{
			"finish_reason":"stop","index":0,
			"logprobs":{"content":[],"refusal":[]},
			"message":{"content":%s,"refusal":"","role":"assistant"}
		}],
		"usage":{"completion_tokens":%d,"prompt_tokens":%d,"total_tokens":%d}
	}`, idJSON, contentJSON, completionTokens, promptTokens, totalTokens)
	var c openai.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		panic(fmt.Sprintf("makeTextCompletion: %v", err))
	}
	return &c
}

func makeToolCallCompletion(id, tcID, name, arguments string) *openai.ChatCompletion {
	idJSON, _ := json.Marshal(id)
	tcIDJSON, _ := json.Marshal(tcID)
	nameJSON, _ := json.Marshal(name)
	argsJSON, _ := json.Marshal(arguments) // arguments is already a JSON string — this wraps it
	raw := fmt.Sprintf(`{
		"id":%s,"object":"chat.completion","created":0,"model":"deepseek-chat",
		"choices":[{
			"finish_reason":"tool_calls","index":0,
			"logprobs":{"content":[],"refusal":[]},
			"message":{
				"content":"","refusal":"","role":"assistant",
				"tool_calls":[{"id":%s,"type":"function","function":{"name":%s,"arguments":%s}}]
			}
		}],
		"usage":{"completion_tokens":10,"prompt_tokens":5,"total_tokens":15}
	}`, idJSON, tcIDJSON, nameJSON, argsJSON)
	var c openai.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		panic(fmt.Sprintf("makeToolCallCompletion: %v", err))
	}
	return &c
}

func makeEmptyChoicesCompletion() *openai.ChatCompletion {
	raw := `{"id":"empty","object":"chat.completion","created":0,"model":"deepseek-chat","choices":[],"usage":{"completion_tokens":0,"prompt_tokens":0,"total_tokens":0}}`
	var c openai.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		panic(fmt.Sprintf("makeEmptyChoicesCompletion: %v", err))
	}
	return &c
}

func newTestProvider(api chatCompletionsAPI) *Provider {
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
// RunTurn
// ---------------------------------------------------------------------------

func TestRunTurn_TextResponse(t *testing.T) {
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			return makeTextCompletion("chat-1", "hello world", 5, 10, 15), nil
		},
	}
	p := newTestProvider(mock)
	out, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Text != "hello world" {
		t.Errorf("Text: got %q, want %q", out.Text, "hello world")
	}
	if out.ResponseID != "chat-1" {
		t.Errorf("ResponseID: got %q", out.ResponseID)
	}
	if out.Usage.InputTokens != 5 || out.Usage.OutputTokens != 10 || out.Usage.TotalTokens != 15 {
		t.Errorf("Usage: got %+v", out.Usage)
	}
}

func TestRunTurn_FunctionCall(t *testing.T) {
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			return makeToolCallCompletion("chat-fn", "tc-1", "search", `{"q":"go"}`), nil
		},
	}
	p := newTestProvider(mock)
	out, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "search go"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("ToolCalls: got %d, want 1", len(out.ToolCalls))
	}
	tc := out.ToolCalls[0]
	if tc.CallID != "tc-1" {
		t.Errorf("CallID: got %q, want %q", tc.CallID, "tc-1")
	}
	if tc.Name != "search" {
		t.Errorf("Name: got %q, want %q", tc.Name, "search")
	}
	if tc.Arguments != `{"q":"go"}` {
		t.Errorf("Arguments: got %q", tc.Arguments)
	}
}

func TestRunTurn_WithInstructions(t *testing.T) {
	var capturedBody openai.ChatCompletionNewParams
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			capturedBody = body
			return makeTextCompletion("r", "ok", 1, 1, 2), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:        "deepseek-chat",
		Instructions: "Be concise.",
		Messages:     []provider.Message{{Role: "user", Content: "hi"}},
	})
	// Instructions become a system message at index 0
	if len(capturedBody.Messages) == 0 {
		t.Fatal("expected messages to be set")
	}
	// Verify first message is system
	firstMsg := capturedBody.Messages[0]
	if firstMsg.OfSystem == nil {
		t.Errorf("expected first message to be system, got: %+v", firstMsg)
	}
}

func TestRunTurn_MessageRoles(t *testing.T) {
	var capturedBody openai.ChatCompletionNewParams
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			capturedBody = body
			return makeTextCompletion("r", "ok", 1, 1, 2), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model: "deepseek-chat",
		Messages: []provider.Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "world"},
			{Role: "system", Content: "sys"},
			{Role: "developer", Content: "dev"},
		},
	})
	// 4 messages (no instructions)
	if len(capturedBody.Messages) != 4 {
		t.Errorf("messages: got %d, want 4", len(capturedBody.Messages))
	}
}

func TestRunTurn_APIError(t *testing.T) {
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			return nil, errors.New("service unavailable")
		},
	}
	p := newTestProvider(mock)
	_, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "service unavailable") {
		t.Errorf("error: got %v", err)
	}
}

func TestRunTurn_EmptyChoices(t *testing.T) {
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			return makeEmptyChoicesCompletion(), nil
		},
	}
	p := newTestProvider(mock)
	_, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for empty choices")
	}
	if !strings.Contains(err.Error(), "empty choices") {
		t.Errorf("error: got %v", err)
	}
}

func TestRunTurn_WithTools(t *testing.T) {
	var capturedBody openai.ChatCompletionNewParams
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			capturedBody = body
			return makeTextCompletion("r", "ok", 1, 1, 2), nil
		},
	}
	p := newTestProvider(mock)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model: "deepseek-chat",
		Tools: []provider.ToolDefinition{
			{Name: "search", Description: "Search the web", Parameters: map[string]any{"type": "object"}},
			{Name: "calc", Parameters: map[string]any{"type": "object"}},
		},
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if len(capturedBody.Tools) != 2 {
		t.Errorf("Tools: got %d, want 2", len(capturedBody.Tools))
	}
}

func TestRunTurn_PreviousResponseID_AppendsFunctionOutputs(t *testing.T) {
	var capturedBody openai.ChatCompletionNewParams
	call1 := 0
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			call1++
			capturedBody = body
			if call1 == 1 {
				return makeToolCallCompletion("chat-1", "tc-1", "search", `{"q":"go"}`), nil
			}
			return makeTextCompletion("chat-2", "result", 5, 10, 15), nil
		},
	}
	p := newTestProvider(mock)

	// First turn: gets a tool call
	out1, err := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "search go"}},
	})
	if err != nil {
		t.Fatalf("turn 1 error: %v", err)
	}
	if len(out1.ToolCalls) != 1 {
		t.Fatalf("turn 1: expected 1 tool call, got %d", len(out1.ToolCalls))
	}

	// Second turn: provides function output
	_, err = p.RunTurn(context.Background(), provider.TurnInput{
		Model:              "deepseek-chat",
		PreviousResponseID: out1.ResponseID,
		FunctionOutputs: []provider.FunctionCallOutput{
			{CallID: "tc-1", Output: "Go is great"},
		},
	})
	if err != nil {
		t.Fatalf("turn 2 error: %v", err)
	}
	// capturedBody should contain the tool message from the second turn
	// The messages slice should include the function output
	hasToolMsg := false
	for _, msg := range capturedBody.Messages {
		if msg.OfTool != nil {
			hasToolMsg = true
			break
		}
	}
	if !hasToolMsg {
		t.Error("expected tool message in second turn, none found")
	}
}

func TestRunTurn_ResetHistory_OnNewTurn(t *testing.T) {
	callCount := 0
	var capturedBodies []openai.ChatCompletionNewParams
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			callCount++
			capturedBodies = append(capturedBodies, body)
			return makeTextCompletion(fmt.Sprintf("chat-%d", callCount), "ok", 1, 1, 2), nil
		},
	}
	p := newTestProvider(mock)

	// Turn 1
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "msg1"}},
	})

	// Turn 2 (no PreviousResponseID → reset)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "msg2"}},
	})

	// Turn 2 should only have its own messages (history reset)
	turn2Msgs := capturedBodies[1].Messages
	if len(turn2Msgs) != 1 {
		t.Errorf("turn 2 messages: got %d, want 1 (history reset)", len(turn2Msgs))
	}
}

func TestRunTurn_MessageAccumulation(t *testing.T) {
	call := 0
	var capturedBody2 openai.ChatCompletionNewParams
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			call++
			if call == 2 {
				capturedBody2 = body
			}
			return makeTextCompletion(fmt.Sprintf("r%d", call), "reply", 1, 1, 2), nil
		},
	}
	p := newTestProvider(mock)

	// Turn 1
	out1, _ := p.RunTurn(context.Background(), provider.TurnInput{
		Model:    "deepseek-chat",
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})

	// Turn 2 via PreviousResponseID path (appends function outputs + the prior assistant msg)
	p.RunTurn(context.Background(), provider.TurnInput{
		Model:              "deepseek-chat",
		PreviousResponseID: out1.ResponseID,
		FunctionOutputs:    []provider.FunctionCallOutput{},
	})

	// After turn 1, the assistant reply is appended to messages.
	// Turn 2 continues from that accumulation.
	if len(capturedBody2.Messages) == 0 {
		t.Error("expected non-empty messages in turn 2 (history accumulation)")
	}
}

func TestRunTurn_Concurrent(t *testing.T) {
	mock := &mockChatCompletionsAPI{
		NewFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			return makeTextCompletion("r", "ok", 1, 1, 2), nil
		},
	}
	p := newTestProvider(mock)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.RunTurn(context.Background(), provider.TurnInput{
				Model:    "deepseek-chat",
				Messages: []provider.Message{{Role: "user", Content: "hi"}},
			})
		}()
	}
	wg.Wait()
}

func TestClose(t *testing.T) {
	p := newTestProvider(nil)
	if err := p.Close(); err != nil {
		t.Errorf("Close: unexpected error: %v", err)
	}
}
