package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type DBQueryToolConfig struct {
	Pool *pgxpool.Pool
}

func NewDBQueryTool(cfg DBQueryToolConfig) *DBQueryTool {
	return &DBQueryTool{pool: cfg.Pool}
}

type DBQueryTool struct {
	pool *pgxpool.Pool
}

func (t *DBQueryTool) Name() string {
	return "db-query"
}

func (t *DBQueryTool) Description() string {
	return "Run a read-only SQL query on PostgreSQL (SELECT-only). Input: JSON {query}."
}

func (t *DBQueryTool) JSONSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"query": map[string]any{
				"type": "string",
			},
		},
		"required": []string{"query"},
	}
}

type DBQueryInput struct {
	Query  string `json:"query"`
	Params []any  `json:"params"`
}

func parseQueryInput(input string) (string, []any, error) {
	var in DBQueryInput
	if err := json.Unmarshal([]byte(input), &in); err != nil {
		return "", nil, fmt.Errorf("input must be a JSON object {query}: %w", err)
	}
	return in.Query, in.Params, nil
}

func (t *DBQueryTool) Call(ctx context.Context, input string) (string, error) {
	if t == nil || t.pool == nil {
		return "", errors.New("db-query not configured (missing pool)")
	}

	sqlText, params, err := parseQueryInput(input)
	if err != nil {
		return "", err
	}

	sqlText = strings.TrimSpace(sqlText)
	if sqlText == "" {
		return "", errors.New("db-query input query is empty")
	}
	if err := validateSelectOnly(sqlText); err != nil {
		return "", err
	}

	tx, err := t.pool.BeginTx(ctx, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		return "", fmt.Errorf("postgres begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	rows, err := tx.Query(ctx, sqlText, params...)
	if err != nil {
		return "", fmt.Errorf("postgres query: %w", err)
	}
	defer rows.Close()

	fields := rows.FieldDescriptions()
	colNames := make([]string, 0, len(fields))
	for _, fd := range fields {
		colNames = append(colNames, string(fd.Name))
	}

	const maxRows = 20
	out := make([]map[string]any, 0, maxRows)
	for rows.Next() {
		vals, err := rows.Values()
		if err != nil {
			return "", fmt.Errorf("rows.Values: %w", err)
		}
		rowObj := make(map[string]any, len(colNames))
		for i, name := range colNames {
			if i < len(vals) {
				rowObj[name] = vals[i]
			}
		}
		out = append(out, rowObj)
		if len(out) >= maxRows {
			break
		}
	}
	if err := rows.Err(); err != nil {
		return "", fmt.Errorf("rows.Err: %w", err)
	}

	b, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("json.Marshal: %w", err)
	}
	return string(b), nil
}

func validateSelectOnly(sqlText string) error {
	deny := []string{
		"INSERT", "UPDATE", "DELETE", "UPSERT",
		"CREATE", "ALTER", "DROP", "TRUNCATE",
		"GRANT", "REVOKE",
		"VACUUM", "ANALYZE",
		"COPY",
		"CALL", "DO", "EXECUTE",
	}
	pattern := "(?i)\\b(" + strings.Join(deny, "|") + ")\\b"
	re := regexp.MustCompile(pattern)
	if match := re.FindString(sqlText); match != "" {
		return fmt.Errorf("keyword %q is not allowed in read-only mode", strings.ToUpper(match))
	}
	return nil
}
