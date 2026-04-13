package rcopenai

import (
	"context"
	"sync"

	"github.com/frediansh/rcopenai/internal/provider"
)

// MockProvider implements provider.Provider for testing.
type MockProvider struct {
	mu          sync.Mutex
	RunTurnFunc func(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error)
	CloseFunc   func() error
	CallCount   int
}

func (m *MockProvider) RunTurn(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
	m.mu.Lock()
	m.CallCount++
	m.mu.Unlock()
	if m.RunTurnFunc != nil {
		return m.RunTurnFunc(ctx, in)
	}
	return provider.TurnOutput{}, nil
}

func (m *MockProvider) Close() error {
	if m.CloseFunc != nil {
		return m.CloseFunc()
	}
	return nil
}

// MockImageProvider implements provider.Provider + imageDescriber.
type MockImageProvider struct {
	MockProvider
	DescribeImageFunc func(ctx context.Context, model, instructions, prompt, imageSource string) (provider.TurnOutput, error)
}

func (m *MockImageProvider) DescribeImage(ctx context.Context, model, instructions, prompt, imageSource string) (provider.TurnOutput, error) {
	if m.DescribeImageFunc != nil {
		return m.DescribeImageFunc(ctx, model, instructions, prompt, imageSource)
	}
	return provider.TurnOutput{Text: "image description"}, nil
}

// MockTool implements Tool for testing.
type MockTool struct {
	NameVal   string
	CallFunc  func(ctx context.Context, args string) (string, error)
	CallCount int
	LastArgs  string
}

func (m *MockTool) Name() string               { return m.NameVal }
func (m *MockTool) Description() string        { return "mock tool" }
func (m *MockTool) JSONSchema() map[string]any { return map[string]any{"type": "object"} }
func (m *MockTool) Call(ctx context.Context, args string) (string, error) {
	m.CallCount++
	m.LastArgs = args
	if m.CallFunc != nil {
		return m.CallFunc(ctx, args)
	}
	return "mock output", nil
}

// SpyHandler implements Handler, recording all calls.
type SpyHandler struct {
	mu             sync.Mutex
	LLMStartCalls  []int
	LLMEndCalls    []TokenUsage
	ToolStartCalls []struct{ Name, Args string }
	ToolEndCalls   []struct{ Name, Output string }
	ToolErrorCalls []struct {
		Name string
		Err  error
	}
}

func (s *SpyHandler) HandleLLMCallStart(ctx context.Context, i int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LLMStartCalls = append(s.LLMStartCalls, i)
}

func (s *SpyHandler) HandleLLMCallEnd(ctx context.Context, u TokenUsage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LLMEndCalls = append(s.LLMEndCalls, u)
}

func (s *SpyHandler) HandleToolStart(ctx context.Context, name, args string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ToolStartCalls = append(s.ToolStartCalls, struct{ Name, Args string }{name, args})
}

func (s *SpyHandler) HandleToolEnd(ctx context.Context, name, output string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ToolEndCalls = append(s.ToolEndCalls, struct{ Name, Output string }{name, output})
}

func (s *SpyHandler) HandleToolError(ctx context.Context, name string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ToolErrorCalls = append(s.ToolErrorCalls, struct {
		Name string
		Err  error
	}{name, err})
}
