package rcopenai

import (
	"context"
	"errors"
	"testing"
)

// compile-time check: NopHandler satisfies Handler interface
var _ Handler = NopHandler{}

func TestNopHandler_AllMethods_NoPanic(t *testing.T) {
	h := NopHandler{}
	ctx := context.Background()

	h.HandleLLMCallStart(ctx, 1)
	h.HandleLLMCallEnd(ctx, TokenUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15})
	h.HandleToolStart(ctx, "mytool", `{"key":"val"}`)
	h.HandleToolEnd(ctx, "mytool", "some output")
	h.HandleToolError(ctx, "mytool", errors.New("something failed"))
}

func TestNopHandler_HandleLLMCallStart_VariousIterations(t *testing.T) {
	h := NopHandler{}
	ctx := context.Background()
	for _, iter := range []int{0, 1, 5, 100, -1} {
		h.HandleLLMCallStart(ctx, iter) // must not panic
	}
}

func TestNopHandler_HandleLLMCallEnd_ZeroUsage(t *testing.T) {
	h := NopHandler{}
	h.HandleLLMCallEnd(context.Background(), TokenUsage{})
}

func TestNopHandler_HandleToolError_NilError(t *testing.T) {
	h := NopHandler{}
	h.HandleToolError(context.Background(), "tool", nil)
}
