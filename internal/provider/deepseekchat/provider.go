package deepseekchat

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/frediansh/rcopenai/internal/provider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

type chatCompletionsAPI interface {
	New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
}

type Provider struct {
	client openai.Client
	api    chatCompletionsAPI

	mu       sync.Mutex
	messages []openai.ChatCompletionMessageParamUnion
}

func New(apiKey, baseURL string) (*Provider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("openai api key is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if strings.TrimSpace(baseURL) != "" {
		opts = append(opts, option.WithBaseURL(strings.TrimSpace(baseURL)))
	}

	client := openai.NewClient(opts...)
	p := &Provider{client: client}
	p.api = &p.client.Chat.Completions
	return p, nil
}

func (p *Provider) Close() error {
	return nil
}

func (p *Provider) RunTurn(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if in.PreviousResponseID == "" {
		p.messages = p.messages[:0]
		if strings.TrimSpace(in.Instructions) != "" {
			p.messages = append(p.messages, openai.SystemMessage(strings.TrimSpace(in.Instructions)))
		}
		for _, m := range in.Messages {
			switch m.Role {
			case "assistant":
				p.messages = append(p.messages, openai.AssistantMessage(m.Content))
			case "system":
				p.messages = append(p.messages, openai.SystemMessage(m.Content))
			case "developer":
				p.messages = append(p.messages, openai.DeveloperMessage(m.Content))
			default:
				p.messages = append(p.messages, openai.UserMessage(m.Content))
			}
		}
	} else {
		for _, fo := range in.FunctionOutputs {
			p.messages = append(p.messages, openai.ToolMessage(fo.Output, fo.CallID))
		}
	}

	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(in.Model),
		Messages: append([]openai.ChatCompletionMessageParamUnion(nil), p.messages...),
	}

	if len(in.Tools) > 0 {
		tools := make([]openai.ChatCompletionToolUnionParam, 0, len(in.Tools))
		for _, t := range in.Tools {
			fn := shared.FunctionDefinitionParam{
				Name:       t.Name,
				Parameters: t.Parameters,
				Strict:     openai.Bool(true),
			}
			if strings.TrimSpace(t.Description) != "" {
				fn.Description = openai.String(strings.TrimSpace(t.Description))
			}
			tools = append(tools, openai.ChatCompletionFunctionTool(fn))
		}
		params.Tools = tools
	}

	resp, err := p.api.New(ctx, params)
	if err != nil {
		return provider.TurnOutput{}, fmt.Errorf("chat.completions.new: %w", err)
	}
	if len(resp.Choices) == 0 {
		return provider.TurnOutput{}, fmt.Errorf("chat.completions.new: empty choices")
	}

	choice := resp.Choices[0]
	msg := choice.Message

	assistant := openai.ChatCompletionAssistantMessageParam{}
	if strings.TrimSpace(msg.Content) != "" {
		assistant.Content.OfString = openai.String(msg.Content)
	}
	if len(msg.ToolCalls) > 0 {
		assistant.ToolCalls = make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(msg.ToolCalls))
	}

	out := provider.TurnOutput{
		ResponseID: resp.ID,
		Text:       msg.Content,
		Usage: provider.Usage{
			InputTokens:  int(resp.Usage.PromptTokens),
			OutputTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		},
	}

	for _, tc := range msg.ToolCalls {
		variant := tc.AsAny()
		fnCall, ok := variant.(openai.ChatCompletionMessageFunctionToolCall)
		if !ok {
			continue
		}

		assistant.ToolCalls = append(assistant.ToolCalls, tc.ToParam())
		out.ToolCalls = append(out.ToolCalls, provider.FunctionCall{
			CallID:    fnCall.ID,
			Name:      fnCall.Function.Name,
			Arguments: fnCall.Function.Arguments,
		})
	}

	p.messages = append(p.messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant})

	return out, nil
}
