package glopenai

import (
	"context"
	"fmt"
	"sync"
)

// TokenUsageHandler accumulates token usage across model calls.
type TokenUsageHandler struct {
	mu sync.Mutex

	TotalTokens      int
	PromptTokens     int
	CompletionTokens int
	CallCount        int
}

func (h *TokenUsageHandler) HandleLLMCallStart(ctx context.Context, iteration int) {}

func (h *TokenUsageHandler) HandleLLMCallEnd(ctx context.Context, usage TokenUsage) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.CallCount++
	h.PromptTokens += usage.InputTokens
	h.CompletionTokens += usage.OutputTokens
	h.TotalTokens += usage.TotalTokens
}

func (h *TokenUsageHandler) HandleToolStart(ctx context.Context, name, arguments string) {}
func (h *TokenUsageHandler) HandleToolEnd(ctx context.Context, name, output string)      {}
func (h *TokenUsageHandler) HandleToolError(ctx context.Context, name string, err error) {}

func (h *TokenUsageHandler) GetStats() string {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.CallCount == 0 {
		return "No LLM calls made yet"
	}
	avg := float64(h.TotalTokens) / float64(h.CallCount)
	return fmt.Sprintf(
		"Total calls: %d, Total tokens: %d (prompt: %d, completion: %d), Average tokens/call: %.2f",
		h.CallCount,
		h.TotalTokens,
		h.PromptTokens,
		h.CompletionTokens,
		avg,
	)
}

func (h *TokenUsageHandler) Reset() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.TotalTokens = 0
	h.PromptTokens = 0
	h.CompletionTokens = 0
	h.CallCount = 0
}
