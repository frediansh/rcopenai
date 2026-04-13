package glopenai

import (
	"bytes"
	"fmt"
	"text/template"
)

func renderHumanPromptTemplate(tmpl string, humanPrompt string) (string, error) {
	t, err := template.New("human_prompt").Parse(tmpl)
	if err != nil {
		return "", fmt.Errorf("parse HumanPromptTemplate: %w", err)
	}

	var b bytes.Buffer
	err = t.Execute(&b, map[string]any{
		"Input":       humanPrompt,
		"HumanPrompt": humanPrompt,
	})
	if err != nil {
		return "", fmt.Errorf("render HumanPromptTemplate: %w", err)
	}
	return b.String(), nil
}
