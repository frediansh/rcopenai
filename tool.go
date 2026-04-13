package glopenai

import "context"

// Tool defines the local tool contract used by AgentClient.
type Tool interface {
	Name() string
	Description() string
	JSONSchema() map[string]any
	Call(ctx context.Context, argumentsJSON string) (string, error)
}

func indexTools(tools []Tool) map[string]Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make(map[string]Tool, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		out[t.Name()] = t
	}
	return out
}
