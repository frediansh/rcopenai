package provider_test

import (
	"github.com/frediansh/rcopenai/internal/provider"
	"github.com/frediansh/rcopenai/internal/provider/deepseekchat"
	"github.com/frediansh/rcopenai/internal/provider/openairesponses"
)

// Compile-time assertions: both concrete providers must satisfy the Provider interface.
var _ provider.Provider = (*openairesponses.Provider)(nil)
var _ provider.Provider = (*deepseekchat.Provider)(nil)
