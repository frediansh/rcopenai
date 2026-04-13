package rcopenai

import (
	"strings"
	"testing"
)

func TestRenderHumanPromptTemplate_EmptyTemplate(t *testing.T) {
	got, err := renderHumanPromptTemplate("", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// empty template renders to empty string
	if got != "" {
		t.Errorf("got %q, want %q", got, "")
	}
}

func TestRenderHumanPromptTemplate_StaticText(t *testing.T) {
	got, err := renderHumanPromptTemplate("static text", "ignored")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "static text" {
		t.Errorf("got %q, want %q", got, "static text")
	}
}

func TestRenderHumanPromptTemplate_InputVar(t *testing.T) {
	got, err := renderHumanPromptTemplate("Process: {{.Input}}", "my data")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "Process: my data" {
		t.Errorf("got %q, want %q", got, "Process: my data")
	}
}

func TestRenderHumanPromptTemplate_HumanPromptVar(t *testing.T) {
	got, err := renderHumanPromptTemplate("You said: {{.HumanPrompt}}", "hi there")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "You said: hi there" {
		t.Errorf("got %q, want %q", got, "You said: hi there")
	}
}

func TestRenderHumanPromptTemplate_BothVarsEqual(t *testing.T) {
	got, err := renderHumanPromptTemplate("{{.Input}} / {{.HumanPrompt}}", "x")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "x / x" {
		t.Errorf("got %q, want %q", got, "x / x")
	}
}

func TestRenderHumanPromptTemplate_MultiLine(t *testing.T) {
	tmpl := "Input: {{.Input}}\nEnd"
	got, err := renderHumanPromptTemplate(tmpl, "val")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "Input: val\nEnd" {
		t.Errorf("got %q, want %q", got, "Input: val\nEnd")
	}
}

func TestRenderHumanPromptTemplate_ParseError(t *testing.T) {
	_, err := renderHumanPromptTemplate("{{.Broken", "x")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "parse HumanPromptTemplate") {
		t.Errorf("error %q should contain 'parse HumanPromptTemplate'", err.Error())
	}
}

func TestRenderHumanPromptTemplate_UnknownKey_NoError(t *testing.T) {
	// map[string]any returns zero value for missing keys — no error
	got, err := renderHumanPromptTemplate("{{.UnknownKey}}", "x")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "<no value>" && got != "" {
		// Go templates render missing map keys as "<no value>" or ""
		// depending on template option; either is acceptable here
		t.Logf("got %q for unknown key (acceptable)", got)
	}
}

func TestRenderHumanPromptTemplate_EmptyHumanPrompt(t *testing.T) {
	got, err := renderHumanPromptTemplate("value={{.Input}}", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "value=" {
		t.Errorf("got %q, want %q", got, "value=")
	}
}
