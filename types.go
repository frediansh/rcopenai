package rcopenai

// Role represents a chat message role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// ChatMessage is a provider-agnostic chat message representation.
type ChatMessage struct {
	Role    Role
	Content string
}

// TokenUsage represents token accounting for one model call.
type TokenUsage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}
