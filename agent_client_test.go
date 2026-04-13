package rcopenai

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/frediansh/rcopenai/internal/provider"
)

// ---- helper ----

func newTestClient(t *testing.T, mp *MockProvider, opts ...func(*AgentClientConfig)) *AgentClient {
	t.Helper()
	cfg := AgentClientConfig{ProviderOverride: mp}
	for _, o := range opts {
		o(&cfg)
	}
	c, err := NewAgentClient(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewAgentClient: %v", err)
	}
	return c
}

// ---- NewAgentClient ----

func TestNewAgentClient_MissingToken(t *testing.T) {
	_, err := NewAgentClient(context.Background(), AgentClientConfig{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestNewAgentClient_InvalidProvider(t *testing.T) {
	_, err := NewAgentClient(context.Background(), AgentClientConfig{
		OpenAIToken: "tok",
		Provider:    "unknown_xyz",
	})
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
}

func TestNewAgentClient_Defaults(t *testing.T) {
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: &MockProvider{},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c == nil {
		t.Fatal("expected non-nil client")
	}
	if c.maxIterations != defaultMaxIterations {
		t.Errorf("maxIterations=%d, want %d", c.maxIterations, defaultMaxIterations)
	}
	if c.model != defaultOpenAIModel {
		t.Errorf("model=%q, want %q", c.model, defaultOpenAIModel)
	}
}

func TestNewAgentClient_WithTools(t *testing.T) {
	tools := []Tool{
		&MockTool{NameVal: "tool1"},
		&MockTool{NameVal: "tool2"},
	}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: &MockProvider{},
		Tools:            tools,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(c.toolsByName) != 2 {
		t.Errorf("toolsByName len=%d, want 2", len(c.toolsByName))
	}
	if _, ok := c.toolsByName["tool1"]; !ok {
		t.Error("tool1 not indexed")
	}
}

func TestNewAgentClient_WithHistory(t *testing.T) {
	history := []ChatMessage{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi"},
	}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: &MockProvider{},
		InitialHistory:   history,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(c.history) != 2 {
		t.Errorf("history len=%d, want 2", len(c.history))
	}
}

// ---- Chat ----

func TestChat_SimpleTextResponse(t *testing.T) {
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			return provider.TurnOutput{Text: "hello world"}, nil
		},
	}
	c := newTestClient(t, mp)
	out, err := c.Chat(context.Background(), "apa kabar?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "hello world" {
		t.Errorf("output=%q, want %q", out, "hello world")
	}
	if mp.CallCount != 1 {
		t.Errorf("CallCount=%d, want 1", mp.CallCount)
	}
}

func TestChat_WithToolCall(t *testing.T) {
	mockTool := &MockTool{
		NameVal: "search",
		CallFunc: func(ctx context.Context, args string) (string, error) {
			return "result 1", nil
		},
	}
	spy := &SpyHandler{}
	callNum := 0
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			callNum++
			if callNum == 1 {
				return provider.TurnOutput{
					ToolCalls: []provider.FunctionCall{
						{CallID: "c1", Name: "search", Arguments: `{"q":"go"}`},
					},
				}, nil
			}
			return provider.TurnOutput{Text: "berdasarkan hasil: result 1"}, nil
		},
	}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: mp,
		Tools:            []Tool{mockTool},
		Callback:         spy,
	})
	if err != nil {
		t.Fatalf("NewAgentClient: %v", err)
	}
	out, err := c.Chat(context.Background(), "search go")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "berdasarkan hasil: result 1" {
		t.Errorf("output=%q", out)
	}
	if mockTool.CallCount != 1 {
		t.Errorf("tool CallCount=%d, want 1", mockTool.CallCount)
	}
	if len(spy.ToolStartCalls) != 1 {
		t.Errorf("ToolStartCalls=%d, want 1", len(spy.ToolStartCalls))
	}
	if spy.ToolStartCalls[0].Name != "search" {
		t.Errorf("ToolStart name=%q, want search", spy.ToolStartCalls[0].Name)
	}
}

func TestChat_ToolCallError(t *testing.T) {
	toolErr := errors.New("tool failure")
	mockTool := &MockTool{
		NameVal: "fail_tool",
		CallFunc: func(ctx context.Context, args string) (string, error) {
			return "", toolErr
		},
	}
	spy := &SpyHandler{}
	callNum := 0
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			callNum++
			if callNum == 1 {
				return provider.TurnOutput{
					ToolCalls: []provider.FunctionCall{
						{CallID: "c1", Name: "fail_tool", Arguments: "{}"},
					},
				}, nil
			}
			return provider.TurnOutput{Text: "handled error"}, nil
		},
	}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: mp,
		Tools:            []Tool{mockTool},
		Callback:         spy,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), "run fail tool")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(spy.ToolErrorCalls) != 1 {
		t.Errorf("ToolErrorCalls=%d, want 1", len(spy.ToolErrorCalls))
	}
	if !errors.Is(spy.ToolErrorCalls[0].Err, toolErr) {
		t.Errorf("ToolError Err=%v, want %v", spy.ToolErrorCalls[0].Err, toolErr)
	}
}

func TestChat_MaxIterationsReached(t *testing.T) {
	maxReachedCalled := false
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			return provider.TurnOutput{
				ToolCalls: []provider.FunctionCall{
					{CallID: "c1", Name: "loop_tool", Arguments: "{}"},
				},
			}, nil
		},
	}
	loopTool := &MockTool{NameVal: "loop_tool"}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: mp,
		MaxIterations:    3,
		Tools:            []Tool{loopTool},
		OnMaxIterationsReached: func(current, max int) {
			maxReachedCalled = true
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), "loop")
	if err == nil {
		t.Fatal("expected error for max iterations")
	}
	if !strings.Contains(err.Error(), "max iterations") {
		t.Errorf("error=%q, want 'max iterations'", err)
	}
	if !maxReachedCalled {
		t.Error("OnMaxIterationsReached not called")
	}
	if mp.CallCount != 3 {
		t.Errorf("CallCount=%d, want 3", mp.CallCount)
	}
}

func TestChat_IterationWarning(t *testing.T) {
	warningCalled := false
	callNum := 0
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			callNum++
			if callNum < 4 {
				return provider.TurnOutput{
					ToolCalls: []provider.FunctionCall{
						{CallID: "c1", Name: "t", Arguments: "{}"},
					},
				}, nil
			}
			return provider.TurnOutput{Text: "done"}, nil
		},
	}
	tool := &MockTool{NameVal: "t"}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride:          mp,
		MaxIterations:             4,
		Tools:                     []Tool{tool},
		IterationWarningThreshold: 0.5,
		OnIterationWarning: func(current, max int) {
			warningCalled = true
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !warningCalled {
		t.Error("OnIterationWarning not called")
	}
}

func TestChat_UnknownTool(t *testing.T) {
	callNum := 0
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			callNum++
			if callNum == 1 {
				return provider.TurnOutput{
					ToolCalls: []provider.FunctionCall{
						{CallID: "c1", Name: "unknown_tool", Arguments: "{}"},
					},
				}, nil
			}
			return provider.TurnOutput{Text: "ok"}, nil
		},
	}
	c := newTestClient(t, mp)
	out, err := c.Chat(context.Background(), "use unknown tool")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "ok" {
		t.Errorf("output=%q, want ok", out)
	}
}

func TestChat_WithTemplate(t *testing.T) {
	var gotInput string
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			if len(in.Messages) > 0 {
				gotInput = in.Messages[len(in.Messages)-1].Content
			}
			return provider.TurnOutput{Text: "ok"}, nil
		},
	}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride:    mp,
		HumanPromptTemplate: "Process: {{.Input}}",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), "my data")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotInput != "Process: my data" {
		t.Errorf("gotInput=%q, want %q", gotInput, "Process: my data")
	}
}

func TestChat_HistoryAccumulation(t *testing.T) {
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			return provider.TurnOutput{Text: "reply"}, nil
		},
	}
	c := newTestClient(t, mp)

	if _, err := c.Chat(context.Background(), "first"); err != nil {
		t.Fatal(err)
	}
	if len(c.history) != 2 {
		t.Errorf("history after first chat=%d, want 2", len(c.history))
	}

	if _, err := c.Chat(context.Background(), "second"); err != nil {
		t.Fatal(err)
	}
	if len(c.history) != 4 {
		t.Errorf("history after second chat=%d, want 4", len(c.history))
	}
}

func TestChat_ContextCancellation(t *testing.T) {
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			return provider.TurnOutput{}, ctx.Err()
		},
	}
	c := newTestClient(t, mp)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := c.Chat(ctx, "test")
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestChat_Concurrent(t *testing.T) {
	mp := &MockProvider{
		RunTurnFunc: func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
			return provider.TurnOutput{Text: "ok"}, nil
		},
	}
	c := newTestClient(t, mp)
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Chat(context.Background(), "concurrent") //nolint:errcheck
		}()
	}
	wg.Wait()
}

// ---- DescribeImage ----

func TestDescribeImage_ProviderDoesNotSupportImage(t *testing.T) {
	c := newTestClient(t, &MockProvider{})
	_, err := c.DescribeImage(context.Background(), "file.jpg", "describe it")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "does not support") {
		t.Errorf("error=%q, want 'does not support'", err)
	}
}

func TestDescribeImage_EmptyPrompt(t *testing.T) {
	mp := &MockImageProvider{}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{ProviderOverride: mp})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.DescribeImage(context.Background(), "cat.jpg", "")
	if err == nil {
		t.Fatal("expected error for empty prompt")
	}
}

func TestDescribeImage_Success(t *testing.T) {
	mp := &MockImageProvider{
		DescribeImageFunc: func(ctx context.Context, model, instructions, prompt, imageSource string) (provider.TurnOutput, error) {
			return provider.TurnOutput{Text: "a cat"}, nil
		},
	}
	c, err := NewAgentClient(context.Background(), AgentClientConfig{ProviderOverride: mp})
	if err != nil {
		t.Fatal(err)
	}
	out, err := c.DescribeImage(context.Background(), "cat.jpg", "what is this?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "a cat" {
		t.Errorf("output=%q, want %q", out, "a cat")
	}
}

// ---- normalizeProviderName ----

func TestNormalizeProviderName(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"", ProviderOpenAIResponses},
		{"openai", ProviderOpenAIResponses},
		{"openai_responses", ProviderOpenAIResponses},
		{"OPENAI-RESPONSES", ProviderOpenAIResponses},
		{"responses", ProviderOpenAIResponses},
		{"deepseek", ProviderDeepSeekChat},
		{"deepseek_chat", ProviderDeepSeekChat},
		{"DEEPSEEK-CHAT", ProviderDeepSeekChat},
		{"chat_completions", ProviderDeepSeekChat},
		{"unknown_xyz", "unknown_xyz"},
	}
	for _, tc := range tests {
		got := normalizeProviderName(tc.in)
		if got != tc.want {
			t.Errorf("normalizeProviderName(%q)=%q, want %q", tc.in, got, tc.want)
		}
	}
}

// ---- toProviderMessages ----

func TestToProviderMessages_Nil(t *testing.T) {
	out := toProviderMessages(nil)
	if out != nil {
		t.Errorf("expected nil, got %v", out)
	}
}

func TestToProviderMessages_NonEmpty(t *testing.T) {
	msgs := []ChatMessage{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi"},
	}
	out := toProviderMessages(msgs)
	if len(out) != 2 {
		t.Errorf("len=%d, want 2", len(out))
	}
	if out[0].Role != "user" || out[0].Content != "hello" {
		t.Errorf("out[0]=%+v", out[0])
	}
	if out[1].Role != "assistant" || out[1].Content != "hi" {
		t.Errorf("out[1]=%+v", out[1])
	}
}

// ---- toProviderTools ----

func TestToProviderTools_Empty(t *testing.T) {
	c, _ := NewAgentClient(context.Background(), AgentClientConfig{ProviderOverride: &MockProvider{}})
	out := c.toProviderTools()
	if out != nil {
		t.Errorf("expected nil, got %v", out)
	}
}

func TestToProviderTools_WithTools(t *testing.T) {
	tool := &MockTool{NameVal: "search"}
	c, _ := NewAgentClient(context.Background(), AgentClientConfig{
		ProviderOverride: &MockProvider{},
		Tools:            []Tool{tool},
	})
	out := c.toProviderTools()
	if len(out) != 1 {
		t.Errorf("len=%d, want 1", len(out))
	}
	if out[0].Name != "search" {
		t.Errorf("Name=%q, want search", out[0].Name)
	}
	if out[0].Description != "mock tool" {
		t.Errorf("Description=%q, want 'mock tool'", out[0].Description)
	}
}
