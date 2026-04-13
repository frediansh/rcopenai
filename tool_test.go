package rcopenai

import (
	"context"
	"testing"
)

// stubTool is a minimal Tool implementation for testing.
type stubTool struct {
	name string
}

func (s *stubTool) Name() string                                              { return s.name }
func (s *stubTool) Description() string                                       { return "stub: " + s.name }
func (s *stubTool) JSONSchema() map[string]any                                { return map[string]any{"type": "object"} }
func (s *stubTool) Call(_ context.Context, _ string) (string, error)         { return "ok", nil }

func TestIndexTools_Basic(t *testing.T) {
	tools := []Tool{
		&stubTool{name: "search"},
		&stubTool{name: "calc"},
		&stubTool{name: "weather"},
	}
	m := indexTools(tools)
	if len(m) != 3 {
		t.Errorf("len = %d, want 3", len(m))
	}
	for _, name := range []string{"search", "calc", "weather"} {
		if _, ok := m[name]; !ok {
			t.Errorf("tool %q not found in index", name)
		}
	}
}

func TestIndexTools_EmptySlice(t *testing.T) {
	m := indexTools([]Tool{})
	if m != nil {
		t.Errorf("expected nil for empty slice, got %v", m)
	}
}

func TestIndexTools_NilSlice(t *testing.T) {
	m := indexTools(nil)
	if m != nil {
		t.Errorf("expected nil for nil slice, got %v", m)
	}
}

func TestIndexTools_DuplicateNames_LastWins(t *testing.T) {
	first := &stubTool{name: "dup"}
	second := &stubTool{name: "dup"}
	m := indexTools([]Tool{first, second})

	if len(m) != 1 {
		t.Errorf("len = %d, want 1 for duplicate names", len(m))
	}
	if m["dup"] != second {
		t.Error("expected last tool to win for duplicate name")
	}
}

func TestIndexTools_NilEntrySkipped(t *testing.T) {
	tools := []Tool{
		&stubTool{name: "a"},
		nil,
		&stubTool{name: "b"},
	}
	m := indexTools(tools)
	if len(m) != 2 {
		t.Errorf("len = %d, want 2 (nil skipped)", len(m))
	}
	if _, ok := m["a"]; !ok {
		t.Error("tool 'a' not found")
	}
	if _, ok := m["b"]; !ok {
		t.Error("tool 'b' not found")
	}
}

func TestIndexTools_AllNil(t *testing.T) {
	tools := []Tool{nil, nil, nil}
	m := indexTools(tools)
	// all nil entries skipped — map is empty but not nil (len(tools)=3 > 0)
	if len(m) != 0 {
		t.Errorf("len = %d, want 0 for all-nil tools", len(m))
	}
}

func TestIndexTools_SingleTool(t *testing.T) {
	tool := &stubTool{name: "only"}
	m := indexTools([]Tool{tool})
	if len(m) != 1 {
		t.Errorf("len = %d, want 1", len(m))
	}
	if m["only"] != tool {
		t.Error("tool not found by name")
	}
}
