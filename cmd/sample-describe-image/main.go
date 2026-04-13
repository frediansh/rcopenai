package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/frediansh/rcopenai"
	"github.com/joho/godotenv"
)

func main() {
	ctx := context.Background()

	defaultImage := filepath.Join("cmd", "sample-describe-image", "sample-image.jpeg")
	imagePath := flag.String("image", defaultImage, "Local image path / file:// path / http(s) URL")
	flag.Parse()

	loadDotEnv()

	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if apiKey == "" {
		log.Fatal("missing env OPENAI_API_KEY")
	}

	model := strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
	if model == "" {
		model = "gpt-4.1-mini"
	}

	tokenHandler := &rcopenai.TokenUsageHandler{}

	client, err := rcopenai.NewAgentClient(ctx, rcopenai.AgentClientConfig{
		OpenAIToken: apiKey,
		OpenAIModel: model,
		Callback:    tokenHandler,
	})
	if err != nil {
		log.Fatalf("NewAgentClient: %v", err)
	}
	defer client.Close()

	prompt := `You are an OCR and structured data extraction specialist for textile warehouse labels.

Your task is:
1. Read all visible text from the product label image.
2. Extract only the required structured fields.
3. Do not guess missing data.
4. If a field is not visible, return null.
5. Output MUST be valid JSON only.
6. Do not include explanations.
7. Normalize units where applicable.

Field Mapping Rules:
- Nama Produk = top-most product name text.
- DESIGN = value after "DESIGN".
- COL = value after "COL".
- ROLL = value after "ROLL".
- QTY_YDS = numeric value before "YDS".
- QTY_MTS = numeric value before "MTS".
- NET_WEIGHT_KGS = numeric value before "KGS".

If QTY contains both YDS and MTS in one line (e.g., 100YDS/91.4MTS), split them correctly.
Return numeric values without unit text.
Return string values trimmed.`

	out, err := client.DescribeImage(ctx, strings.TrimSpace(*imagePath), prompt)
	if err != nil {
		log.Fatalf("DescribeImage: %v", err)
	}

	normalized, err := normalizeJSONObject(out)
	if err != nil {
		log.Fatalf("invalid model JSON output: %v\nraw: %s", err, out)
	}

	pretty, err := json.MarshalIndent(normalized, "", "  ")
	if err != nil {
		log.Fatalf("json marshal: %v", err)
	}

	fmt.Println(string(pretty))
	fmt.Printf("usage: tokens_in=%d tokens_out=%d tokens_total=%d llm_calls=%d\n",
		tokenHandler.PromptTokens,
		tokenHandler.CompletionTokens,
		tokenHandler.TotalTokens,
		tokenHandler.CallCount,
	)
}

func normalizeJSONObject(s string) (map[string]string, error) {
	raw := strings.TrimSpace(s)
	if raw == "" {
		return nil, errors.New("empty response")
	}

	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start >= 0 && end > start {
		raw = raw[start : end+1]
	}

	obj := map[string]string{}
	if err := json.Unmarshal([]byte(raw), &obj); err == nil {
		return obj, nil
	}

	generic := map[string]any{}
	if err := json.Unmarshal([]byte(raw), &generic); err != nil {
		return nil, err
	}

	normalized := make(map[string]string, len(generic))
	for k, v := range generic {
		normalized[k] = fmt.Sprint(v)
	}
	return normalized, nil
}

func loadDotEnv() {
	paths := []string{".env", "rcopenai/.env"}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			if err := godotenv.Load(p); err != nil {
				log.Printf("warning: failed loading %s: %v", p, err)
			}
			return
		}
	}
}
