package rcopenai

import (
	"context"
	"strings"
	"sync"
	"testing"
)

func TestTokenUsageHandler_SingleCall(t *testing.T) {
	h := &TokenUsageHandler{}
	h.HandleLLMCallEnd(context.Background(), TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30})

	if h.CallCount != 1 {
		t.Errorf("CallCount = %d, want 1", h.CallCount)
	}
	if h.PromptTokens != 10 {
		t.Errorf("PromptTokens = %d, want 10", h.PromptTokens)
	}
	if h.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", h.CompletionTokens)
	}
	if h.TotalTokens != 30 {
		t.Errorf("TotalTokens = %d, want 30", h.TotalTokens)
	}
}

func TestTokenUsageHandler_MultipleCalls_Accumulates(t *testing.T) {
	h := &TokenUsageHandler{}
	ctx := context.Background()
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 5, OutputTokens: 10, TotalTokens: 15})
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 3, OutputTokens: 7, TotalTokens: 10})
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5})

	if h.CallCount != 3 {
		t.Errorf("CallCount = %d, want 3", h.CallCount)
	}
	if h.PromptTokens != 10 {
		t.Errorf("PromptTokens = %d, want 10", h.PromptTokens)
	}
	if h.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", h.CompletionTokens)
	}
	if h.TotalTokens != 30 {
		t.Errorf("TotalTokens = %d, want 30", h.TotalTokens)
	}
}

func TestTokenUsageHandler_GetStats_ZeroCalls(t *testing.T) {
	h := &TokenUsageHandler{}
	s := h.GetStats()
	if s != "No LLM calls made yet" {
		t.Errorf("GetStats() = %q, want 'No LLM calls made yet'", s)
	}
}

func TestTokenUsageHandler_GetStats_Format(t *testing.T) {
	h := &TokenUsageHandler{}
	h.HandleLLMCallEnd(context.Background(), TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30})
	h.HandleLLMCallEnd(context.Background(), TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30})

	s := h.GetStats()
	checks := []string{"Total calls: 2", "Total tokens: 60", "prompt: 20", "completion: 40", "Average tokens/call: 30.00"}
	for _, want := range checks {
		if !strings.Contains(s, want) {
			t.Errorf("GetStats() = %q, want to contain %q", s, want)
		}
	}
}

func TestTokenUsageHandler_GetStats_AverageCalculation(t *testing.T) {
	h := &TokenUsageHandler{}
	// 1 call, 100 total tokens → avg = 100.00
	h.HandleLLMCallEnd(context.Background(), TokenUsage{InputTokens: 40, OutputTokens: 60, TotalTokens: 100})
	s := h.GetStats()
	if !strings.Contains(s, "100.00") {
		t.Errorf("GetStats() = %q, want avg 100.00", s)
	}
}

func TestTokenUsageHandler_Reset(t *testing.T) {
	h := &TokenUsageHandler{}
	ctx := context.Background()
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30})
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30})

	h.Reset()

	if h.CallCount != 0 || h.TotalTokens != 0 || h.PromptTokens != 0 || h.CompletionTokens != 0 {
		t.Errorf("after Reset: CallCount=%d TotalTokens=%d PromptTokens=%d CompletionTokens=%d, want all 0",
			h.CallCount, h.TotalTokens, h.PromptTokens, h.CompletionTokens)
	}
	s := h.GetStats()
	if s != "No LLM calls made yet" {
		t.Errorf("GetStats() after Reset = %q, want 'No LLM calls made yet'", s)
	}
}

func TestTokenUsageHandler_NoOpMethods_NoPanic(t *testing.T) {
	h := &TokenUsageHandler{}
	ctx := context.Background()
	// These should all be no-ops and not panic
	h.HandleLLMCallStart(ctx, 1)
	h.HandleToolStart(ctx, "mytool", `{"x":1}`)
	h.HandleToolEnd(ctx, "mytool", "output")
	h.HandleToolError(ctx, "mytool", nil)
}

func TestTokenUsageHandler_ImplementsHandler(t *testing.T) {
	var _ Handler = (*TokenUsageHandler)(nil)
}

func TestTokenUsageHandler_Concurrent(t *testing.T) {
	h := &TokenUsageHandler{}
	ctx := context.Background()

	var wg sync.WaitGroup
	n := 100
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3})
		}()
	}
	wg.Wait()

	if h.CallCount != n {
		t.Errorf("CallCount = %d, want %d", h.CallCount, n)
	}
	if h.TotalTokens != n*3 {
		t.Errorf("TotalTokens = %d, want %d", h.TotalTokens, n*3)
	}
}

func TestTokenUsageHandler_Reset_ThenAccumulate(t *testing.T) {
	h := &TokenUsageHandler{}
	ctx := context.Background()
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 10, OutputTokens: 10, TotalTokens: 20})
	h.Reset()
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 5, OutputTokens: 5, TotalTokens: 10})

	if h.CallCount != 1 {
		t.Errorf("CallCount = %d after reset+call, want 1", h.CallCount)
	}
	if h.TotalTokens != 10 {
		t.Errorf("TotalTokens = %d after reset+call, want 10", h.TotalTokens)
	}
}
