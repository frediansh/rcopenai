package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/frediansh/rcopenai"
	"github.com/frediansh/rcopenai/internal/builtin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
)

func main() {
	ctx := context.Background()
	promptFlag := flag.String("prompt", "", "Prompt user for one-shot mode")
	flag.Parse()

	if err := godotenv.Load(".env"); err != nil && !os.IsNotExist(err) {
		log.Printf("warning: failed loading .env: %v", err)
	}

	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if apiKey == "" {
		log.Fatal("missing env OPENAI_API_KEY")
	}

	dbURL := strings.TrimSpace(os.Getenv("DATABASE_URL"))
	if dbURL == "" {
		log.Println("warning: missing env DATABASE_URL (db-query tool will be unavailable)")
	}

	systemPrompt := strings.TrimSpace(os.Getenv("SYSTEM_PROMPT"))
	if systemPrompt == "" {
		systemPrompt = defaultSystemPrompt
	}
	openAIModel := strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
	if openAIModel == "" {
		openAIModel = "gpt-5.3-codex"
	}
	openAIBaseURL := strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	openAIProvider := strings.TrimSpace(os.Getenv("OPENAI_PROVIDER"))
	if openAIProvider == "" {
		openAIProvider = "openai"
	}

	var dbPool *pgxpool.Pool
	var err error
	if dbURL != "" {
		dbPool, err = pgxpool.New(ctx, dbURL)
		if err != nil {
			log.Fatalf("pgxpool.New: %v", err)
		}
		if err := dbPool.Ping(ctx); err != nil {
			dbPool.Close()
			log.Fatalf("postgres ping: %v", err)
		}
		defer dbPool.Close()
	}

	httpClient := &http.Client{Timeout: 10 * time.Second}

	toolList := make([]glopenai.Tool, 0, 2)
	toolList = append(toolList, builtin.NewHTTPGetTool(builtin.HTTPGetToolConfig{
		Client:   httpClient,
		MaxBytes: builtin.DefaultHTTPGetMaxBodyBytes,
	}))
	if dbPool != nil {
		toolList = append(toolList, builtin.NewDBQueryTool(builtin.DBQueryToolConfig{
			Pool: dbPool,
		}))
	}

	tokenHandler := &glopenai.TokenUsageHandler{}

	client, err := glopenai.NewAgentClient(ctx, glopenai.AgentClientConfig{
		OpenAIToken:               apiKey,
		OpenAIModel:               openAIModel,
		OpenAIBaseURL:             openAIBaseURL,
		Provider:                  openAIProvider,
		SystemPrompt:              systemPrompt,
		MaxIterations:             15,
		Tools:                     toolList,
		Callback:                  tokenHandler,
		IterationWarningThreshold: 0.75,
		OnIterationWarning: func(current, max int) {
			log.Printf("Iteration warning: %d/%d iterations used", current, max)
			log.Printf("Token usage so far: %s", tokenHandler.GetStats())
		},
		OnMaxIterationsReached: func(current, max int) {
			log.Printf("Max iterations reached: %d/%d", current, max)
			log.Printf("Final token usage: %s", tokenHandler.GetStats())
		},
	})
	if err != nil {
		log.Fatalf("NewAgentClient: %v", err)
	}
	defer client.Close()

	fmt.Println("glopenai sample")
	fmt.Println("- one-shot mode (tanpa input stream)")
	fmt.Println()

	userPrompt := strings.TrimSpace(*promptFlag)
	if userPrompt == "" {
		userPrompt = strings.TrimSpace(os.Getenv("USER_PROMPT"))
	}
	if userPrompt == "" {
		userPrompt = "Berikan lastest data 10 dari table tm_task_template"
	}
	fmt.Printf("You> %s\n", userPrompt)

	beforeTotal := tokenHandler.TotalTokens
	beforePrompt := tokenHandler.PromptTokens
	beforeCompletion := tokenHandler.CompletionTokens
	beforeCalls := tokenHandler.CallCount

	start := time.Now()
	done := make(chan bool)
	go showProgress(done)

	out, err := client.Chat(ctx, userPrompt)
	elapsed := time.Since(start)

	done <- true
	fmt.Print("\r\033[K")

	if err != nil {
		fmt.Printf("Err> %v\n\n", err)
	} else {
		fmt.Printf("AI> %s\n\n", strings.TrimSpace(out))
	}

	usedTotal := tokenHandler.TotalTokens - beforeTotal
	usedPrompt := tokenHandler.PromptTokens - beforePrompt
	usedCompletion := tokenHandler.CompletionTokens - beforeCompletion
	usedCalls := tokenHandler.CallCount - beforeCalls

	fmt.Printf("Stats> tokens_in=%d tokens_out=%d tokens_total=%d llm_calls=%d time=%s\n\n",
		usedPrompt, usedCompletion, usedTotal, usedCalls, elapsed.Round(time.Millisecond))

	fmt.Println("\n" + strings.Repeat("-", 50))
	fmt.Printf("Token Usage Statistics:\n%s\n", tokenHandler.GetStats())
	fmt.Println(strings.Repeat("-", 50))
}

func showProgress(done chan bool) {
	dots := []string{"   ", ".  ", ".. ", "..."}
	i := 0
	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			fmt.Printf("\rAI sedang berpikir%s", dots[i%len(dots)])
			i++
		}
	}
}

const defaultSystemPrompt = `You are a helpful data assistant.
Use clear, short Indonesian language by default.

If a request needs database data, follow this simple flow:
1. Find relevant tables from information_schema.tables. Do not guess table names.
2. Check columns from information_schema.columns before writing final SQL.
3. Run a small preview query first (LIMIT 3) to validate assumptions.
4. Build and run the final SQL query.
5. Explain the result briefly and mention which tables were used.

If the user request is ambiguous, ask 1 short clarification question before running a complex query.
Use the available 'db-query' tool for SQL execution.`
