# OVERVIEW rcopenai

`rcopenai` adalah library helper Go untuk call OpenAI Responses API dengan pola agent loop (LLM + tool calling), mendukung:

- `system prompt`
- `human prompt`
- seed `history`
- custom tools (plus built-in tools)
- callback observability (token/tool events)
- guard iterasi agar tidak loop tanpa akhir

Module:

```go
module github.com/frediansh/rcopenai
```

## 1. Requirement

- Go `1.25.4` (sesuai `go.mod`)
- OpenAI API key valid
- (Opsional) PostgreSQL jika ingin pakai built-in tool `db-query`

## 2. Instalasi

```bash
go get github.com/frediansh/rcopenai
```

Import utama:

```go
import "github.com/frediansh/rcopenai"
```

Import built-in tools:

```go
import "github.com/frediansh/rcopenai/internal/builtin"
```

## 3. Konsep Arsitektur Singkat

Alur `Chat(...)`:

1. Prompt user (opsional dibungkus `HumanPromptTemplate`) dimasukkan ke turn.
2. Library call OpenAI Responses API.
3. Jika model minta function call, library eksekusi tool lokal berdasarkan nama.
4. Output tool dikirim balik ke OpenAI (`function_call_output`) pada iterasi berikutnya.
5. Saat model mengembalikan teks final (tanpa tool call), hasil dipulangkan ke caller.
6. History in-memory diperbarui (`user` + `assistant`) untuk chat berikutnya.

## 4. API Utama

### 4.1 `AgentClientConfig`

```go
type AgentClientConfig struct {
    OpenAIToken string
    OpenAIModel string

    SystemPrompt string

    // Go template vars: {{.Input}}, {{.HumanPrompt}}
    HumanPromptTemplate string

    InitialHistory []ChatMessage

    MaxIterations int

    Tools []Tool

    Callback Handler

    IterationWarningThreshold float64
    OnIterationWarning     func(current, max int)
    OnMaxIterationsReached func(current, max int)
}
```

Catatan penting:

- `OpenAIToken` wajib ada.
- Jika `OpenAIModel` kosong, default library: `gpt-4.1-mini`.
- Jika `MaxIterations <= 0`, default: `8`.
- `IterationWarningThreshold` aktif jika `> 0`; monitor internal memaksa range valid `(0,1)`, default efektif `0.75`.

### 4.2 `AgentClient`

```go
client, err := rcopenai.NewAgentClient(ctx, cfg)
out, err := client.Chat(ctx, "...prompt...")
client.Close()
```

Kontrak:

- `Chat(ctx, humanPrompt)` return `(string, error)`.
- Return error jika client belum init, provider gagal, template invalid, tool error fatal di provider path, atau max iterasi habis.
- Thread-safety: `Chat` diserialisasi via mutex internal (aman dipanggil concurrent, tapi eksekusi one-at-a-time per client instance).

### 4.3 Message Types

```go
type Role string
const (
    RoleSystem    Role = "system"
    RoleDeveloper Role = "developer"
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
)

type ChatMessage struct {
    Role    Role
    Content string
}
```

### 4.4 Tool Contract

```go
type Tool interface {
    Name() string
    Description() string
    JSONSchema() map[string]any
    Call(ctx context.Context, argumentsJSON string) (string, error)
}
```

Tool dipilih berdasarkan `Name()`. Jika nama duplicate, entri terakhir di slice `Tools` akan overwrite sebelumnya.

### 4.5 Callback Contract

```go
type Handler interface {
    HandleLLMCallStart(ctx context.Context, iteration int)
    HandleLLMCallEnd(ctx context.Context, usage TokenUsage)
    HandleToolStart(ctx context.Context, name, arguments string)
    HandleToolEnd(ctx context.Context, name, output string)
    HandleToolError(ctx context.Context, name string, err error)
}
```

Default callback jika tidak diisi adalah `NopHandler`.

## 5. Input dan Output

### 5.1 Input ke `Chat`

Input utama berupa `humanPrompt string`.

Jika `HumanPromptTemplate` diisi, maka input akhir dirender dari template dengan data:

- `{{.Input}}`
- `{{.HumanPrompt}}`

Keduanya berisi nilai prompt user yang sama.

### 5.2 Output dari `Chat`

- `string`: final text dari model (`trim-space`)
- `error`: jika ada kegagalan

Contoh error yang mungkin:

- `OpenAIToken is required`
- `parse HumanPromptTemplate: ...`
- `render HumanPromptTemplate: ...`
- `max iterations reached before final response`

## 6. Contoh Penggunaan

### 6.1 Basic one-shot

```go
package main

import (
    "context"
    "log"
    "os"

    "github.com/frediansh/rcopenai"
)

func main() {
    client, err := rcopenai.NewAgentClient(context.Background(), rcopenai.AgentClientConfig{
        OpenAIToken: os.Getenv("OPENAI_API_KEY"),
        OpenAIModel: "gpt-5.3-codex",
        SystemPrompt: "Kamu asisten data yang ringkas.",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    out, err := client.Chat(context.Background(), "Tolong jelaskan ringkas ETL")
    if err != nil {
        log.Fatal(err)
    }
    log.Println(out)
}
```

### 6.2 Dengan history dan human template

```go
cfg := rcopenai.AgentClientConfig{
    OpenAIToken: os.Getenv("OPENAI_API_KEY"),
    OpenAIModel: "gpt-5.3-codex",
    SystemPrompt: "Jawab bahasa Indonesia.",
    HumanPromptTemplate: "[USER_REQUEST]\n{{.Input}}\n[/USER_REQUEST]",
    InitialHistory: []rcopenai.ChatMessage{
        {Role: rcopenai.RoleUser, Content: "Halo"},
        {Role: rcopenai.RoleAssistant, Content: "Halo, saya siap bantu."},
    },
}
```

### 6.3 Custom tool

```go
type EchoTool struct{}

func (EchoTool) Name() string        { return "echo" }
func (EchoTool) Description() string { return "Kembalikan argumen apa adanya" }
func (EchoTool) JSONSchema() map[string]any {
    return map[string]any{
        "type": "object",
        "properties": map[string]any{
            "text": map[string]any{"type": "string"},
        },
        "required": []string{"text"},
        "additionalProperties": false,
    }
}
func (EchoTool) Call(ctx context.Context, argumentsJSON string) (string, error) {
    return argumentsJSON, nil
}

client, _ := rcopenai.NewAgentClient(ctx, rcopenai.AgentClientConfig{
    OpenAIToken: os.Getenv("OPENAI_API_KEY"),
    OpenAIModel: "gpt-5.3-codex",
    Tools: []rcopenai.Tool{EchoTool{}},
})
```

### 6.4 Built-in tool: `http-get`

```go
httpClient := &http.Client{Timeout: 10 * time.Second}
httpTool := builtin.NewHTTPGetTool(builtin.HTTPGetToolConfig{
    Client: httpClient,
    // AllowHosts: []string{"api.example.com"}, // opsional
    MaxBytes: builtin.DefaultHTTPGetMaxBodyBytes,
})
```

Input yang diterima `http-get`:

- JSON: `{"url":"https://example.com"}`
- atau string URL langsung: `https://example.com`

Output format:

```text
status=<code>
content-type=<type>
body(truncated-to-<max>-bytes)=<body>
```

### 6.5 Built-in tool: `db-query`

```go
pool, _ := pgxpool.New(ctx, os.Getenv("DATABASE_URL"))
dbTool := builtin.NewDBQueryTool(builtin.DBQueryToolConfig{Pool: pool})
```

Input JSON:

```json
{"query":"SELECT id, name FROM users WHERE id = $1","params":[1]}
```

Output: JSON array maksimal 20 row.

### 6.6 Contoh lengkap (system prompt + tools + callback)

```go
httpClient := &http.Client{Timeout: 10 * time.Second}
tokenHandler := &rcopenai.TokenUsageHandler{}

tools := []rcopenai.Tool{
    builtin.NewHTTPGetTool(builtin.HTTPGetToolConfig{
        Client:   httpClient,
        MaxBytes: builtin.DefaultHTTPGetMaxBodyBytes,
    }),
    builtin.NewDBQueryTool(builtin.DBQueryToolConfig{Pool: dbPool}),
}

client, err := rcopenai.NewAgentClient(ctx, rcopenai.AgentClientConfig{
    OpenAIToken:  os.Getenv("OPENAI_API_KEY"),
    OpenAIModel:  "gpt-5.3-codex",
    SystemPrompt: "Kamu asisten data yang ringkas.",
    MaxIterations: 15,
    Tools:        tools,
    Callback:     tokenHandler,
    IterationWarningThreshold: 0.75,
    OnIterationWarning: func(current, max int) {
        log.Printf("iteration warning %d/%d", current, max)
        log.Printf("stats: %s", tokenHandler.GetStats())
    },
    OnMaxIterationsReached: func(current, max int) {
        log.Printf("max iterations reached %d/%d", current, max)
    },
})
if err != nil {
    log.Fatal(err)
}
defer client.Close()

answer, err := client.Chat(ctx, "Ambil 10 data terbaru dari tm_task_template")
if err != nil {
    log.Fatal(err)
}
fmt.Println(answer)
```

## 7. Built-in Tool Behavior dan Validasi

### 7.1 `http-get`

Validasi/guard:

- Hanya `http` / `https`
- Blok `localhost`
- Blok private/loopback/link-local IP
- Opsional allowlist host (`AllowHosts`)
- Body di-truncate via `MaxBytes` (default 1MB)

Error umum:

- `http-get not configured`
- `http-get input is empty`
- `unsupported scheme ...`
- `host ... not in allowlist`

### 7.2 `db-query`

Validasi/guard:

- Wajib JSON object input `{query, params}`
- Query kosong ditolak
- Transaksi `ReadOnly`
- Keyword DML/DDL tertentu diblok regex (mis. `INSERT`, `UPDATE`, `DELETE`, `DROP`, dst)
- Output row dibatasi `maxRows = 20`

Catatan:

- Filter keyword bersifat sederhana (regex denylist), bukan SQL parser penuh.

## 8. Token Usage dan Observability

`TokenUsageHandler` sudah disediakan:

```go
tokenHandler := &rcopenai.TokenUsageHandler{}

client, _ := rcopenai.NewAgentClient(ctx, rcopenai.AgentClientConfig{
    OpenAIToken: os.Getenv("OPENAI_API_KEY"),
    OpenAIModel: "gpt-5.3-codex",
    Callback: tokenHandler,
})

_, _ = client.Chat(ctx, "Halo")
fmt.Println(tokenHandler.GetStats())
```

Data yang dikumpulkan:

- `CallCount`
- `PromptTokens`
- `CompletionTokens`
- `TotalTokens`

## 9. Iteration Control

Konfigurasi:

- `MaxIterations`
- `IterationWarningThreshold`
- `OnIterationWarning`
- `OnMaxIterationsReached`

Perilaku:

- Warning callback dipanggil sekali saat threshold tercapai.
- Jika sampai batas iterasi tanpa jawaban final, `Chat` return error.

## 10. Limitasi Saat Ini

- Provider saat ini fixed ke OpenAI Responses API (`openai-go/v3`).
- Belum ada streaming output token-by-token.
- History hanya in-memory per instance `AgentClient` (belum ada persistence built-in).
- Eksekusi `Chat` diserialisasi per instance (throughput paralel per client terbatas).
- Mapping tool berdasarkan nama, sehingga nama tool harus unik.
- `db-query` read-only guard menggunakan regex denylist, jadi bukan proteksi SQL parser level penuh.

## 11. Validasi Konfigurasi yang Disarankan

Sebelum buat client:

1. Pastikan `OPENAI_API_KEY` ada.
2. Tetapkan model eksplisit (hindari bergantung default bila environment lintas tim).
3. Jika pakai `HumanPromptTemplate`, lakukan test render minimal 1 kali.
4. Pastikan schema tool valid JSON Schema object.
5. Pastikan timeout HTTP client/tool sudah sesuai SLA.

## 12. Testing

Status saat ini di repo:

- Belum ada file unit test (`*_test.go`) di package.
- `go test ./...` berhasil compile semua package.

Checklist test yang direkomendasikan:

1. Unit test `renderHumanPromptTemplate` (valid template, parse error, execute error).
2. Unit test `IterationMonitor` (warning threshold dan max reached).
3. Unit test `indexTools` untuk duplicate name behavior.
4. Unit test `http-get` validation (scheme, localhost, private IP, allowlist).
5. Unit test `db-query` validation (`validateSelectOnly`, input parsing, row cap 20).
6. Integration test provider dengan mocked OpenAI responses (tool-call loop + final response).

Contoh command:

```bash
go test ./...
go test -race ./...
```

## 13. Troubleshooting

- Error `OpenAIToken is required`
  - Isi `OpenAIToken` di config.
- Error `max iterations reached before final response`
  - Naikkan `MaxIterations`, perbaiki `SystemPrompt`, atau rapikan tool schema/output agar model cepat konvergen.
- Tool tidak pernah dipanggil
  - Cek `Tools` terdaftar, nama unik, dan deskripsi/schema cukup jelas untuk model.
- `db-query` unavailable
  - Pastikan pool DB berhasil dibuat dan dipass ke `NewDBQueryTool`.

## 14. Referensi File Implementasi

- Core client: `agent_client.go`
- Message/token types: `types.go`
- Callback contract: `callbacks.go`
- Prompt template renderer: `prompt_template.go`
- Iteration monitor: `iteration_monitor.go`
- Token usage handler: `token_usage_handler.go`
- Tool contract: `tool.go`
- OpenAI provider: `internal/provider/openairesponses/provider.go`
- Built-in tools: `internal/builtin/http_get.go`, `internal/builtin/db_query.go`
- Sample app: `cmd/rcopenai-sample/main.go`
