package openairesponses

import (
	"context"
	"encoding/base64"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/frediansh/rcopenai/internal/provider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

type Provider struct {
	client openai.Client
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
	return &Provider{client: client}, nil
}

func (p *Provider) Close() error {
	return nil
}

func (p *Provider) RunTurn(ctx context.Context, in provider.TurnInput) (provider.TurnOutput, error) {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(in.Model),
	}
	if in.Instructions != "" {
		params.Instructions = openai.String(in.Instructions)
	}

	if len(in.Tools) > 0 {
		tools := make([]responses.ToolUnionParam, 0, len(in.Tools))
		for _, t := range in.Tools {
			fn := responses.FunctionToolParam{
				Name:       t.Name,
				Parameters: t.Parameters,
				Strict:     openai.Bool(true),
			}
			if t.Description != "" {
				fn.Description = openai.String(t.Description)
			}
			tools = append(tools, responses.ToolUnionParam{OfFunction: &fn})
		}
		params.Tools = tools
	}

	if in.PreviousResponseID == "" {
		items := make([]responses.ResponseInputItemUnionParam, 0, len(in.Messages))
		for _, m := range in.Messages {
			role := responses.EasyInputMessageRoleUser
			switch m.Role {
			case "assistant":
				role = responses.EasyInputMessageRoleAssistant
			case "system":
				role = responses.EasyInputMessageRoleSystem
			case "developer":
				role = responses.EasyInputMessageRoleDeveloper
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(m.Content, role))
		}
		params.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: items}
	} else {
		params.PreviousResponseID = openai.String(in.PreviousResponseID)
		items := make([]responses.ResponseInputItemUnionParam, 0, len(in.FunctionOutputs))
		for _, fo := range in.FunctionOutputs {
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: fo.CallID,
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{OfString: openai.String(fo.Output)},
				},
			})
		}
		params.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: items}
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return provider.TurnOutput{}, fmt.Errorf("responses.new: %w", err)
	}

	return toTurnOutput(resp), nil
}

func (p *Provider) DescribeImage(ctx context.Context, model, instructions, prompt, imageSource string) (provider.TurnOutput, error) {
	imageURL, err := resolveImageSource(imageSource)
	if err != nil {
		return provider.TurnOutput{}, err
	}

	content := responses.ResponseInputMessageContentListParam{
		responses.ResponseInputContentParamOfInputText(prompt),
	}
	img := responses.ResponseInputContentParamOfInputImage(responses.ResponseInputImageDetailAuto)
	img.OfInputImage.ImageURL = openai.String(imageURL)
	content = append(content, img)

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser),
			},
		},
	}
	if instructions != "" {
		params.Instructions = openai.String(instructions)
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return provider.TurnOutput{}, fmt.Errorf("responses.new: %w", err)
	}

	return toTurnOutput(resp), nil
}

func toTurnOutput(resp *responses.Response) provider.TurnOutput {

	out := provider.TurnOutput{
		ResponseID: resp.ID,
		Text:       resp.OutputText(),
		Usage: provider.Usage{
			InputTokens:  int(resp.Usage.InputTokens),
			OutputTokens: int(resp.Usage.OutputTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		},
	}

	for _, item := range resp.Output {
		if item.Type != "function_call" {
			continue
		}
		fc := item.AsFunctionCall()
		out.ToolCalls = append(out.ToolCalls, provider.FunctionCall{
			CallID:    fc.CallID,
			Name:      fc.Name,
			Arguments: fc.Arguments,
		})
	}

	return out
}

func resolveImageSource(imageSource string) (string, error) {
	s := strings.TrimSpace(imageSource)
	if s == "" {
		return "", fmt.Errorf("image source is required")
	}

	u, err := url.Parse(s)
	if err == nil && u.Scheme != "" {
		switch strings.ToLower(u.Scheme) {
		case "http", "https", "data":
			return s, nil
		case "file":
			return localPathToDataURL(u.Path)
		}
	}

	return localPathToDataURL(s)
}

func localPathToDataURL(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("image path is required")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve image path: %w", err)
	}

	b, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("read image file %q: %w", absPath, err)
	}
	if len(b) == 0 {
		return "", fmt.Errorf("image file %q is empty", absPath)
	}

	contentType := http.DetectContentType(b)
	if contentType == "application/octet-stream" {
		ext := strings.ToLower(filepath.Ext(absPath))
		if guessed := mime.TypeByExtension(ext); guessed != "" {
			contentType = guessed
		}
	}
	if !strings.HasPrefix(contentType, "image/") {
		return "", fmt.Errorf("unsupported image mime type %q for %q", contentType, absPath)
	}

	encoded := base64.StdEncoding.EncodeToString(b)
	return fmt.Sprintf("data:%s;base64,%s", contentType, encoded), nil
}
