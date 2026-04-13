package rcopenai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/frediansh/rcopenai/internal/provider"
	"github.com/frediansh/rcopenai/internal/provider/deepseekchat"
	"github.com/frediansh/rcopenai/internal/provider/openairesponses"
)

const (
	defaultOpenAIModel   = "gpt-4.1-mini"
	defaultMaxIterations = 8

	ProviderOpenAIResponses = "openai_responses"
	ProviderDeepSeekChat    = "deepseek_chat"
)

type AgentClientConfig struct {
	OpenAIToken   string
	OpenAIModel   string
	OpenAIBaseURL string
	Provider      string

	// Optional system prompt passed as request instructions.
	SystemPrompt string

	// Optional template to wrap latest user prompt before sending.
	// Go-template variables: {{.Input}}, {{.HumanPrompt}}.
	HumanPromptTemplate string

	// Optional seed history loaded into in-memory conversation state.
	InitialHistory []ChatMessage

	MaxIterations int

	// Optional list of tools.
	Tools []Tool

	// Optional callback handler for observability.
	Callback Handler

	// Optional: warning threshold for iteration usage (0.0-1.0).
	IterationWarningThreshold float64

	// Optional callbacks for iteration status.
	OnIterationWarning     func(current, max int)
	OnMaxIterationsReached func(current, max int)

	// ProviderOverride is used only for testing.
	// If set, OpenAIToken and Provider fields are ignored.
	ProviderOverride provider.Provider
}

type AgentClient struct {
	provider provider.Provider

	model               string
	systemPrompt        string
	humanPromptTemplate string
	maxIterations       int
	tools               []Tool
	toolsByName         map[string]Tool
	callback            Handler

	iterationWarningThreshold float64
	onIterationWarning        func(current, max int)
	onMaxIterationsReached    func(current, max int)

	mu      sync.Mutex
	history []ChatMessage
}

type imageDescriber interface {
	DescribeImage(ctx context.Context, model, instructions, prompt, imageSource string) (provider.TurnOutput, error)
}

func NewAgentClient(ctx context.Context, cfg AgentClientConfig) (*AgentClient, error) {
	_ = ctx
	if strings.TrimSpace(cfg.OpenAIModel) == "" {
		cfg.OpenAIModel = defaultOpenAIModel
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = defaultMaxIterations
	}

	var (
		p   provider.Provider
		err error
	)

	if cfg.ProviderOverride != nil {
		p = cfg.ProviderOverride
	} else {
		if strings.TrimSpace(cfg.OpenAIToken) == "" {
			return nil, errors.New("OpenAIToken is required")
		}
		selectedProvider := normalizeProviderName(cfg.Provider)
		switch selectedProvider {
		case ProviderOpenAIResponses:
			p, err = openairesponses.New(cfg.OpenAIToken, cfg.OpenAIBaseURL)
		case ProviderDeepSeekChat:
			p, err = deepseekchat.New(cfg.OpenAIToken, cfg.OpenAIBaseURL)
		default:
			return nil, fmt.Errorf("unknown provider %q", cfg.Provider)
		}
		if err != nil {
			return nil, fmt.Errorf("create provider %q: %w", selectedProvider, err)
		}
	}

	cb := cfg.Callback
	if cb == nil {
		cb = NopHandler{}
	}

	history := make([]ChatMessage, 0, len(cfg.InitialHistory))
	history = append(history, cfg.InitialHistory...)

	return &AgentClient{
		provider:                  p,
		model:                     cfg.OpenAIModel,
		systemPrompt:              cfg.SystemPrompt,
		humanPromptTemplate:       cfg.HumanPromptTemplate,
		maxIterations:             cfg.MaxIterations,
		tools:                     cfg.Tools,
		toolsByName:               indexTools(cfg.Tools),
		callback:                  cb,
		iterationWarningThreshold: cfg.IterationWarningThreshold,
		onIterationWarning:        cfg.OnIterationWarning,
		onMaxIterationsReached:    cfg.OnMaxIterationsReached,
		history:                   history,
	}, nil
}

func normalizeProviderName(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "openai", "openai_responses", "openai-responses", "responses":
		return ProviderOpenAIResponses
	case "deepseek", "deepseek_chat", "deepseek-chat", "chat_completions", "chat-completions":
		return ProviderDeepSeekChat
	default:
		return strings.ToLower(strings.TrimSpace(name))
	}
}

func (c *AgentClient) Close() {
	if c == nil || c.provider == nil {
		return
	}
	_ = c.provider.Close()
}

func (c *AgentClient) Chat(ctx context.Context, humanPrompt string) (string, error) {
	if c == nil || c.provider == nil {
		return "", errors.New("AgentClient not initialized")
	}

	input := humanPrompt
	if strings.TrimSpace(c.humanPromptTemplate) != "" {
		var err error
		input, err = renderHumanPromptTemplate(c.humanPromptTemplate, humanPrompt)
		if err != nil {
			return "", err
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	turnMessages := make([]ChatMessage, 0, len(c.history)+1)
	turnMessages = append(turnMessages, c.history...)
	turnMessages = append(turnMessages, ChatMessage{Role: RoleUser, Content: input})

	monitorEnabled := c.iterationWarningThreshold > 0
	var monitor *IterationMonitor
	if monitorEnabled {
		monitor = NewIterationMonitor(c.maxIterations, c.iterationWarningThreshold)
	}

	prevResponseID := ""
	var pendingOutputs []provider.FunctionCallOutput

	for i := 1; i <= c.maxIterations; i++ {
		if monitor != nil {
			current, warn, _ := monitor.Increment()
			if warn && c.onIterationWarning != nil {
				c.onIterationWarning(current, c.maxIterations)
			}
		}

		c.callback.HandleLLMCallStart(ctx, i)

		in := provider.TurnInput{
			Model:        c.model,
			Instructions: c.systemPrompt,
			Tools:        c.toProviderTools(),
		}
		if prevResponseID == "" {
			in.Messages = toProviderMessages(turnMessages)
		} else {
			in.PreviousResponseID = prevResponseID
			in.FunctionOutputs = pendingOutputs
		}

		out, err := c.provider.RunTurn(ctx, in)
		if err != nil {
			return "", err
		}

		c.callback.HandleLLMCallEnd(ctx, TokenUsage{
			InputTokens:  out.Usage.InputTokens,
			OutputTokens: out.Usage.OutputTokens,
			TotalTokens:  out.Usage.TotalTokens,
		})

		if out.ResponseID != "" {
			prevResponseID = out.ResponseID
		}

		if len(out.ToolCalls) == 0 {
			finalText := strings.TrimSpace(out.Text)
			c.history = append(c.history, ChatMessage{Role: RoleUser, Content: input})
			c.history = append(c.history, ChatMessage{Role: RoleAssistant, Content: finalText})
			return finalText, nil
		}

		pendingOutputs = pendingOutputs[:0]
		for _, call := range out.ToolCalls {
			tool, ok := c.toolsByName[call.Name]
			if !ok {
				pendingOutputs = append(pendingOutputs, provider.FunctionCallOutput{
					CallID: call.CallID,
					Output: fmt.Sprintf("tool %q is not available", call.Name),
				})
				continue
			}

			c.callback.HandleToolStart(ctx, call.Name, call.Arguments)
			toolOutput, err := tool.Call(ctx, call.Arguments)
			if err != nil {
				c.callback.HandleToolError(ctx, call.Name, err)
				pendingOutputs = append(pendingOutputs, provider.FunctionCallOutput{
					CallID: call.CallID,
					Output: fmt.Sprintf("tool %q error: %v", call.Name, err),
				})
				continue
			}
			c.callback.HandleToolEnd(ctx, call.Name, toolOutput)
			pendingOutputs = append(pendingOutputs, provider.FunctionCallOutput{
				CallID: call.CallID,
				Output: toolOutput,
			})
		}
	}

	if c.onMaxIterationsReached != nil {
		c.onMaxIterationsReached(c.maxIterations, c.maxIterations)
	}
	return "", fmt.Errorf("max iterations reached before final response")
}

// DescribeImage sends a prompt plus image input (http/https URL, file:// URL, or local path)
// and returns the model text output.
func (c *AgentClient) DescribeImage(ctx context.Context, imageSource, prompt string) (string, error) {
	if c == nil || c.provider == nil {
		return "", errors.New("AgentClient not initialized")
	}
	if strings.TrimSpace(prompt) == "" {
		return "", errors.New("prompt is required")
	}

	describer, ok := c.provider.(imageDescriber)
	if !ok {
		return "", errors.New("provider does not support image input")
	}

	c.callback.HandleLLMCallStart(ctx, 1)
	out, err := describer.DescribeImage(ctx, c.model, c.systemPrompt, prompt, imageSource)
	if err != nil {
		return "", err
	}
	c.callback.HandleLLMCallEnd(ctx, TokenUsage{
		InputTokens:  out.Usage.InputTokens,
		OutputTokens: out.Usage.OutputTokens,
		TotalTokens:  out.Usage.TotalTokens,
	})

	return strings.TrimSpace(out.Text), nil
}

func (c *AgentClient) toProviderTools() []provider.ToolDefinition {
	if len(c.tools) == 0 {
		return nil
	}
	out := make([]provider.ToolDefinition, 0, len(c.tools))
	for _, t := range c.tools {
		if t == nil {
			continue
		}
		schema := t.JSONSchema()
		if schema == nil {
			schema = map[string]any{
				"type": "object",
			}
		}
		out = append(out, provider.ToolDefinition{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  schema,
		})
	}
	return out
}

func toProviderMessages(msgs []ChatMessage) []provider.Message {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]provider.Message, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, provider.Message{
			Role:    string(m.Role),
			Content: m.Content,
		})
	}
	return out
}
