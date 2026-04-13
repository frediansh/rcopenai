package glopenai

import "context"

// Handler receives execution events from AgentClient.
type Handler interface {
	HandleLLMCallStart(ctx context.Context, iteration int)
	HandleLLMCallEnd(ctx context.Context, usage TokenUsage)
	HandleToolStart(ctx context.Context, name, arguments string)
	HandleToolEnd(ctx context.Context, name, output string)
	HandleToolError(ctx context.Context, name string, err error)
}

// NopHandler ignores all events.
type NopHandler struct{}

func (NopHandler) HandleLLMCallStart(ctx context.Context, iteration int)       {}
func (NopHandler) HandleLLMCallEnd(ctx context.Context, usage TokenUsage)      {}
func (NopHandler) HandleToolStart(ctx context.Context, name, arguments string) {}
func (NopHandler) HandleToolEnd(ctx context.Context, name, output string)      {}
func (NopHandler) HandleToolError(ctx context.Context, name string, err error) {}
