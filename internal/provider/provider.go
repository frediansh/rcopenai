package provider

import "context"

type Message struct {
	Role    string
	Content string
}

type ToolDefinition struct {
	Name        string
	Description string
	Parameters  map[string]any
}

type FunctionCall struct {
	CallID    string
	Name      string
	Arguments string
}

type FunctionCallOutput struct {
	CallID string
	Output string
}

type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

type TurnInput struct {
	Model              string
	Instructions       string
	Messages           []Message
	PreviousResponseID string
	FunctionOutputs    []FunctionCallOutput
	Tools              []ToolDefinition
}

type TurnOutput struct {
	ResponseID string
	Text       string
	ToolCalls  []FunctionCall
	Usage      Usage
}

type Provider interface {
	RunTurn(ctx context.Context, in TurnInput) (TurnOutput, error)
	Close() error
}
