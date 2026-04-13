package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// ---- mock types ----

type mockPool struct {
	beginTxFn func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error)
}

func (m *mockPool) BeginTx(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
	return m.beginTxFn(ctx, opts)
}

// mockTx embeds pgx.Tx to satisfy the full interface; only overrides used methods.
type mockTx struct {
	pgx.Tx
	queryFn func(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

func (m *mockTx) Rollback(ctx context.Context) error { return nil }
func (m *mockTx) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	return m.queryFn(ctx, sql, args...)
}

// mockRows embeds pgx.Rows to satisfy the full interface; only overrides used methods.
type mockRows struct {
	pgx.Rows
	cols []pgconn.FieldDescription
	data [][]any
	idx  int
	err  error
}

func (r *mockRows) Close()                                       {}
func (r *mockRows) Err() error                                   { return r.err }
func (r *mockRows) FieldDescriptions() []pgconn.FieldDescription { return r.cols }
func (r *mockRows) Next() bool {
	r.idx++
	return r.idx <= len(r.data)
}
func (r *mockRows) Values() ([]any, error) { return r.data[r.idx-1], nil }

// errValRows is a mockRows variant where Values() returns an error.
type errValRows struct {
	pgx.Rows
	cols    []pgconn.FieldDescription
	hasNext bool
	err     error
}

func (r *errValRows) Close()                                       {}
func (r *errValRows) Err() error                                   { return nil }
func (r *errValRows) FieldDescriptions() []pgconn.FieldDescription { return r.cols }
func (r *errValRows) Next() bool {
	v := !r.hasNext
	r.hasNext = true
	return v
}
func (r *errValRows) Values() ([]any, error) { return nil, r.err }

// ---- parseQueryInput ----

func TestParseQueryInput_ValidJSON(t *testing.T) {
	q, params, err := parseQueryInput(`{"query":"SELECT 1","params":[1,"two"]}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if q != "SELECT 1" {
		t.Errorf("query=%q, want SELECT 1", q)
	}
	if len(params) != 2 {
		t.Errorf("params len=%d, want 2", len(params))
	}
}

func TestParseQueryInput_NoParams(t *testing.T) {
	q, params, err := parseQueryInput(`{"query":"SELECT 2"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if q != "SELECT 2" {
		t.Errorf("query=%q", q)
	}
	if params != nil {
		t.Errorf("params=%v, want nil", params)
	}
}

func TestParseQueryInput_InvalidJSON(t *testing.T) {
	_, _, err := parseQueryInput(`not json`)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestParseQueryInput_EmptyObject(t *testing.T) {
	q, params, err := parseQueryInput(`{}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if q != "" {
		t.Errorf("query=%q, want empty", q)
	}
	if params != nil {
		t.Errorf("params=%v, want nil", params)
	}
}

// ---- validateSelectOnly ----

func TestValidateSelectOnly_SelectAllowed(t *testing.T) {
	queries := []string{
		"SELECT * FROM users",
		"select id, name from products where id = $1",
		"SELECT u.id FROM users u JOIN products p ON u.id = p.user_id",
	}
	for _, q := range queries {
		if err := validateSelectOnly(q); err != nil {
			t.Errorf("validateSelectOnly(%q) = %v, want nil", q, err)
		}
	}
}

func TestValidateSelectOnly_ForbiddenKeywords(t *testing.T) {
	forbidden := []string{
		"INSERT INTO users VALUES (1)",
		"UPDATE users SET name = 'x'",
		"DELETE FROM users",
		"DROP TABLE users",
		"CREATE TABLE foo (id int)",
		"ALTER TABLE foo ADD COLUMN bar int",
		"TRUNCATE users",
		"GRANT SELECT ON users TO role",
		"REVOKE SELECT ON users FROM role",
		"VACUUM users",
		"ANALYZE users",
		"COPY users TO '/tmp/out.csv'",
		"CALL my_proc()",
		"DO $$ BEGIN END $$",
		"EXECUTE my_prepared",
	}
	for _, q := range forbidden {
		if err := validateSelectOnly(q); err == nil {
			t.Errorf("validateSelectOnly(%q) = nil, want error", q)
		}
	}
}

func TestValidateSelectOnly_CaseInsensitive(t *testing.T) {
	if err := validateSelectOnly("insert into users values (1)"); err == nil {
		t.Error("expected error for lowercase insert")
	}
}

// ---- DBQueryTool.Call ----

func TestCall_NilPool(t *testing.T) {
	tool := &DBQueryTool{pool: nil}
	_, err := tool.Call(context.Background(), `{"query":"SELECT 1"}`)
	if err == nil {
		t.Fatal("expected error for nil pool")
	}
}

func TestCall_InvalidJSON(t *testing.T) {
	tool := &DBQueryTool{pool: &mockPool{}}
	_, err := tool.Call(context.Background(), `not json`)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestCall_EmptyQuery(t *testing.T) {
	tool := &DBQueryTool{pool: &mockPool{}}
	_, err := tool.Call(context.Background(), `{"query":"   "}`)
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestCall_ForbiddenKeyword(t *testing.T) {
	tool := &DBQueryTool{pool: &mockPool{}}
	_, err := tool.Call(context.Background(), `{"query":"DELETE FROM users"}`)
	if err == nil {
		t.Fatal("expected error for forbidden keyword")
	}
}

func TestCall_BeginTxError(t *testing.T) {
	beginErr := errors.New("connection refused")
	tool := &DBQueryTool{
		pool: &mockPool{
			beginTxFn: func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
				return nil, beginErr
			},
		},
	}
	_, err := tool.Call(context.Background(), `{"query":"SELECT 1"}`)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, beginErr) {
		t.Errorf("err=%v, want to wrap %v", err, beginErr)
	}
}

func TestCall_QueryError(t *testing.T) {
	queryErr := errors.New("syntax error")
	tool := &DBQueryTool{
		pool: &mockPool{
			beginTxFn: func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
				return &mockTx{
					queryFn: func(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
						return nil, queryErr
					},
				}, nil
			},
		},
	}
	_, err := tool.Call(context.Background(), `{"query":"SELECT 1"}`)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, queryErr) {
		t.Errorf("err=%v, want to wrap %v", err, queryErr)
	}
}

func TestCall_RowsValuesError(t *testing.T) {
	valErr := errors.New("scan error")
	tool := &DBQueryTool{
		pool: &mockPool{
			beginTxFn: func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
				r := &errValRows{
					cols: []pgconn.FieldDescription{{Name: "id"}},
					err:  valErr,
				}
				return &mockTx{
					queryFn: func(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
						return r, nil
					},
				}, nil
			},
		},
	}
	_, err := tool.Call(context.Background(), `{"query":"SELECT id FROM users"}`)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, valErr) {
		t.Errorf("err=%v, want to wrap %v", err, valErr)
	}
}

func TestCall_RowsErrAfterIteration(t *testing.T) {
	iterErr := errors.New("iteration error")
	tool := &DBQueryTool{
		pool: &mockPool{
			beginTxFn: func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
				r := &mockRows{
					cols: []pgconn.FieldDescription{{Name: "id"}},
					data: nil,
					err:  iterErr,
				}
				return &mockTx{
					queryFn: func(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
						return r, nil
					},
				}, nil
			},
		},
	}
	_, err := tool.Call(context.Background(), `{"query":"SELECT id FROM users"}`)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, iterErr) {
		t.Errorf("err=%v, want to wrap %v", err, iterErr)
	}
}

func TestCall_Success(t *testing.T) {
	rows := &mockRows{
		cols: []pgconn.FieldDescription{
			{Name: "id"},
			{Name: "name"},
		},
		data: [][]any{
			{1, "Alice"},
			{2, "Bob"},
		},
	}
	tool := &DBQueryTool{
		pool: &mockPool{
			beginTxFn: func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
				return &mockTx{
					queryFn: func(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
						return rows, nil
					},
				}, nil
			},
		},
	}
	out, err := tool.Call(context.Background(), `{"query":"SELECT id, name FROM users"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "Alice") || !strings.Contains(out, "Bob") {
		t.Errorf("output=%q, want Alice and Bob", out)
	}
}

func TestCall_MaxRowsLimit(t *testing.T) {
	data := make([][]any, 25)
	for i := range data {
		data[i] = []any{i + 1}
	}
	rows := &mockRows{
		cols: []pgconn.FieldDescription{{Name: "n"}},
		data: data,
	}
	tool := &DBQueryTool{
		pool: &mockPool{
			beginTxFn: func(ctx context.Context, opts pgx.TxOptions) (pgx.Tx, error) {
				return &mockTx{
					queryFn: func(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
						return rows, nil
					},
				}, nil
			},
		},
	}
	out, err := tool.Call(context.Background(), `{"query":"SELECT n FROM t"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var result []map[string]any
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatalf("json parse: %v", err)
	}
	if len(result) != 20 {
		t.Errorf("result len=%d, want 20", len(result))
	}
}

// ---- static/metadata tests ----

func TestDBQueryTool_Name(t *testing.T) {
	tool := NewDBQueryTool(DBQueryToolConfig{})
	if tool.Name() != "db-query" {
		t.Errorf("Name()=%q, want db-query", tool.Name())
	}
}

func TestDBQueryTool_Description(t *testing.T) {
	tool := NewDBQueryTool(DBQueryToolConfig{})
	if tool.Description() == "" {
		t.Error("Description() is empty")
	}
}

func TestDBQueryTool_JSONSchema(t *testing.T) {
	tool := NewDBQueryTool(DBQueryToolConfig{})
	schema := tool.JSONSchema()
	if schema == nil {
		t.Fatal("JSONSchema() is nil")
	}
	if schema["type"] != "object" {
		t.Errorf("schema type=%v, want object", schema["type"])
	}
}
