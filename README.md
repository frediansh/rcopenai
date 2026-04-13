# glopenai

Library Go untuk agent LLM berbasis `openai-go/v3`, dengan target pengganti `glaichain` tanpa ketergantungan ke `langchaingo`.

## Module name

```go
module github.com/frediansh/rcopenai
```

## Dokumen plan

Rencana implementasi besar ada di:

- `glopenai/PLAN.md`

Dokumen tersebut mencakup:

- target kompatibilitas terhadap pola `main.go` di `glaichain`
- desain wrapper agar tidak terikat vendor SDK
- pemetaan config lama (`AgentClientConfig`) ke implementasi baru
- tahapan delivery dan test plan

## Status implementasi

Yang sudah tersedia:

- `AgentClient` (`NewAgentClient`, `Chat`, `Close`)
- in-memory conversation history antar turn
- `SystemPrompt`, `HumanPromptTemplate`, `InitialHistory`
- tool calling via OpenAI Responses API
- `MaxIterations`, warning callback, max-iteration callback
- `TokenUsageHandler` untuk statistik token/call
- built-in tools: `http-get`, `db-query`
- sample CLI: `cmd/glopenai-sample/main.go`

## Quickstart

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/frediansh/rcopenai"
)

func main() {
	client, err := glopenai.NewAgentClient(context.Background(), glopenai.AgentClientConfig{
		OpenAIToken: os.Getenv("OPENAI_API_KEY"),
		OpenAIModel: "gpt-5.3-codex",
		SystemPrompt: "Kamu asisten data yang ringkas.",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	out, err := client.Chat(context.Background(), "Halo")
	if err != nil {
		log.Fatal(err)
	}
	log.Println(out)
}
```

Jalankan sample interaktif:

```bash
go run ./cmd/glopenai-sample
```
