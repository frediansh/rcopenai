package builtin

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

// roundTripFunc allows building a mock http.Client from a plain function.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func newMockClient(statusCode int, body, contentType string) *http.Client {
	return &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			h := make(http.Header)
			if contentType != "" {
				h.Set("Content-Type", contentType)
			}
			return &http.Response{
				StatusCode: statusCode,
				Body:       io.NopCloser(strings.NewReader(body)),
				Header:     h,
			}, nil
		}),
	}
}

func newErrorClient(err error) *http.Client {
	return &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return nil, err
		}),
	}
}

// ---------- parseURLFromToolInput ----------

func TestParseURL_JSONObject(t *testing.T) {
	u, err := parseURLFromToolInput(`{"url":"https://example.com/path"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u.Host != "example.com" {
		t.Errorf("host = %q, want 'example.com'", u.Host)
	}
}

func TestParseURL_RawString(t *testing.T) {
	u, err := parseURLFromToolInput("https://example.com/path")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u.Host != "example.com" {
		t.Errorf("host = %q, want 'example.com'", u.Host)
	}
}

func TestParseURL_EmptyInput(t *testing.T) {
	_, err := parseURLFromToolInput("")
	if err == nil {
		t.Fatal("expected error for empty input, got nil")
	}
}

func TestParseURL_WhitespaceOnly(t *testing.T) {
	_, err := parseURLFromToolInput("   ")
	if err == nil {
		t.Fatal("expected error for whitespace input, got nil")
	}
}

func TestParseURL_JSONWithEmptyURL(t *testing.T) {
	// JSON with empty "url" value — fallback to the raw string which is also the JSON itself
	// parseURLFromToolInput parses JSON but if url is empty, s stays as the JSON string
	// This results in parsing the JSON string itself as a URL — should succeed (url.Parse is lenient)
	// but will fail validateHTTPGetURL later due to missing scheme
	u, err := parseURLFromToolInput(`{"url":""}`)
	// url.Parse on a JSON-like string won't error; it just returns weird URL
	// The important thing is no panic
	_ = u
	_ = err
}

func TestParseURL_JSONObject_HTTPWithPath(t *testing.T) {
	u, err := parseURLFromToolInput(`{"url":"http://api.example.com/v1/data"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u.Scheme != "http" {
		t.Errorf("scheme = %q, want 'http'", u.Scheme)
	}
	if u.Path != "/v1/data" {
		t.Errorf("path = %q, want '/v1/data'", u.Path)
	}
}

// ---------- validateHTTPGetURL ----------

func TestValidate_HTTPSAllowed(t *testing.T) {
	u := mustParseURL(t, "https://example.com")
	if err := validateHTTPGetURL(u, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_HTTPAllowed(t *testing.T) {
	u := mustParseURL(t, "http://example.com")
	if err := validateHTTPGetURL(u, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_FTPBlocked(t *testing.T) {
	u := mustParseURL(t, "ftp://example.com")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for ftp scheme, got nil")
	}
	if !strings.Contains(err.Error(), "scheme") {
		t.Errorf("error = %q, want to mention 'scheme'", err.Error())
	}
}

func TestValidate_FileBlocked(t *testing.T) {
	u := mustParseURL(t, "file:///etc/passwd")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for file scheme, got nil")
	}
}

func TestValidate_Localhost_Name(t *testing.T) {
	u := mustParseURL(t, "http://localhost/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for localhost, got nil")
	}
	if !strings.Contains(err.Error(), "localhost") {
		t.Errorf("error = %q, want to mention 'localhost'", err.Error())
	}
}

func TestValidate_Localhost_CaseInsensitive(t *testing.T) {
	u := mustParseURL(t, "http://LOCALHOST/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for LOCALHOST, got nil")
	}
}

func TestValidate_Loopback_IP_127_0_0_1(t *testing.T) {
	u := mustParseURL(t, "http://127.0.0.1/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for 127.0.0.1, got nil")
	}
}

func TestValidate_Private_10x(t *testing.T) {
	u := mustParseURL(t, "http://10.0.0.1/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for 10.x private IP, got nil")
	}
}

func TestValidate_Private_172_16(t *testing.T) {
	u := mustParseURL(t, "http://172.16.0.1/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for 172.16.x private IP, got nil")
	}
}

func TestValidate_Private_192_168(t *testing.T) {
	u := mustParseURL(t, "http://192.168.1.1/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for 192.168.x private IP, got nil")
	}
}

func TestValidate_LinkLocal_169_254(t *testing.T) {
	u := mustParseURL(t, "http://169.254.1.1/")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for 169.254.x link-local IP, got nil")
	}
}

func TestValidate_Nil_URL(t *testing.T) {
	err := validateHTTPGetURL(nil, nil)
	if err == nil {
		t.Fatal("expected error for nil URL, got nil")
	}
}

func TestValidate_MissingHost(t *testing.T) {
	u := mustParseURL(t, "https://")
	err := validateHTTPGetURL(u, nil)
	if err == nil {
		t.Fatal("expected error for missing host, got nil")
	}
}

func TestValidate_AllowHosts_Pass(t *testing.T) {
	u := mustParseURL(t, "https://api.internal.com/data")
	err := validateHTTPGetURL(u, []string{"api.internal.com"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_AllowHosts_CaseInsensitive(t *testing.T) {
	u := mustParseURL(t, "https://API.INTERNAL.COM/data")
	err := validateHTTPGetURL(u, []string{"api.internal.com"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_AllowHosts_Block(t *testing.T) {
	u := mustParseURL(t, "https://blocked.com/")
	err := validateHTTPGetURL(u, []string{"allowed.com"})
	if err == nil {
		t.Fatal("expected error for host not in allowlist, got nil")
	}
	if !strings.Contains(err.Error(), "allowlist") {
		t.Errorf("error = %q, want to mention 'allowlist'", err.Error())
	}
}

func TestValidate_AllowHosts_EmptyList_NotBlocked(t *testing.T) {
	u := mustParseURL(t, "https://example.com/")
	err := validateHTTPGetURL(u, []string{})
	if err != nil {
		t.Errorf("unexpected error with empty allowlist: %v", err)
	}
}

// ---------- isPrivateIP ----------

func TestIsPrivate_10_0_0_0(t *testing.T) {
	assertPrivate(t, "10.0.0.0", true)
}

func TestIsPrivate_10_255_255_255(t *testing.T) {
	assertPrivate(t, "10.255.255.255", true)
}

func TestIsPrivate_172_16_0_0(t *testing.T) {
	assertPrivate(t, "172.16.0.0", true)
}

func TestIsPrivate_172_31_255_255(t *testing.T) {
	assertPrivate(t, "172.31.255.255", true)
}

func TestIsPrivate_172_32_0_0_NotPrivate(t *testing.T) {
	assertPrivate(t, "172.32.0.0", false)
}

func TestIsPrivate_192_168_0_0(t *testing.T) {
	assertPrivate(t, "192.168.0.0", true)
}

func TestIsPrivate_192_168_255_255(t *testing.T) {
	assertPrivate(t, "192.168.255.255", true)
}

func TestIsPrivate_127_0_0_1(t *testing.T) {
	assertPrivate(t, "127.0.0.1", true)
}

func TestIsPrivate_127_128_0_1(t *testing.T) {
	assertPrivate(t, "127.128.0.1", true)
}

func TestIsPrivate_169_254_1_1(t *testing.T) {
	assertPrivate(t, "169.254.1.1", true)
}

func TestIsPrivate_0_0_0_0(t *testing.T) {
	assertPrivate(t, "0.0.0.0", true)
}

func TestIsPrivate_Public_8_8_8_8(t *testing.T) {
	assertPrivate(t, "8.8.8.8", false)
}

func TestIsPrivate_Public_1_1_1_1(t *testing.T) {
	assertPrivate(t, "1.1.1.1", false)
}

func TestIsPrivate_Public_93_184_216_34(t *testing.T) {
	assertPrivate(t, "93.184.216.34", false)
}

// ---------- HTTPGetTool.Call (end-to-end) ----------

func TestHTTPGetTool_Call_Success(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(200, "hello world", "text/plain"),
	})
	out, err := tool.Call(context.Background(), `{"url":"https://example.com"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "status=200") {
		t.Errorf("output %q missing 'status=200'", out)
	}
	if !strings.Contains(out, "text/plain") {
		t.Errorf("output %q missing 'text/plain'", out)
	}
	if !strings.Contains(out, "hello world") {
		t.Errorf("output %q missing body 'hello world'", out)
	}
}

func TestHTTPGetTool_Call_404_NotAnError(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(404, "not found", "text/plain"),
	})
	out, err := tool.Call(context.Background(), `{"url":"https://example.com"}`)
	if err != nil {
		t.Fatalf("unexpected error for 404: %v", err)
	}
	if !strings.Contains(out, "status=404") {
		t.Errorf("output %q missing 'status=404'", out)
	}
}

func TestHTTPGetTool_Call_EmptyContentType(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(200, "body", ""),
	})
	out, err := tool.Call(context.Background(), `{"url":"https://example.com"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "content-type=unknown") {
		t.Errorf("output %q should have 'content-type=unknown'", out)
	}
}

func TestHTTPGetTool_Call_TransportError(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newErrorClient(errors.New("connection refused")),
	})
	_, err := tool.Call(context.Background(), `{"url":"https://example.com"}`)
	if err == nil {
		t.Fatal("expected error for transport failure, got nil")
	}
}

func TestHTTPGetTool_Call_BlockedLocalhost(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(200, "body", "text/plain"),
	})
	_, err := tool.Call(context.Background(), `{"url":"http://localhost/"}`)
	if err == nil {
		t.Fatal("expected error for localhost URL, got nil")
	}
}

func TestHTTPGetTool_Call_BlockedPrivateIP(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(200, "body", "text/plain"),
	})
	_, err := tool.Call(context.Background(), `{"url":"http://192.168.1.1/"}`)
	if err == nil {
		t.Fatal("expected error for private IP, got nil")
	}
}

func TestHTTPGetTool_Call_MaxBytesEnforced(t *testing.T) {
	body := strings.Repeat("A", 100)
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client:   newMockClient(200, body, "text/plain"),
		MaxBytes: 10,
	})
	out, err := tool.Call(context.Background(), `{"url":"https://example.com"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Body should be truncated — output contains at most 10 'A' chars
	if strings.Count(out, "A") > 10 {
		t.Errorf("body not truncated: output has more than 10 'A' chars")
	}
}

func TestHTTPGetTool_Call_AllowList_Pass(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client:     newMockClient(200, "ok", "text/plain"),
		AllowHosts: []string{"example.com"},
	})
	_, err := tool.Call(context.Background(), `{"url":"https://example.com/"}`)
	if err != nil {
		t.Errorf("unexpected error for allowed host: %v", err)
	}
}

func TestHTTPGetTool_Call_AllowList_Blocked(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client:     newMockClient(200, "ok", "text/plain"),
		AllowHosts: []string{"allowed.com"},
	})
	_, err := tool.Call(context.Background(), `{"url":"https://blocked.com/"}`)
	if err == nil {
		t.Fatal("expected error for host not in allowlist, got nil")
	}
}

func TestHTTPGetTool_Call_NilClient(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{Client: nil})
	_, err := tool.Call(context.Background(), `{"url":"https://example.com"}`)
	if err == nil {
		t.Fatal("expected error for nil client, got nil")
	}
}

func TestHTTPGetTool_Call_InvalidInput(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(200, "ok", "text/plain"),
	})
	_, err := tool.Call(context.Background(), "")
	if err == nil {
		t.Fatal("expected error for empty input, got nil")
	}
}

func TestHTTPGetTool_Call_RawURLInput(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{
		Client: newMockClient(200, "raw ok", "text/html"),
	})
	out, err := tool.Call(context.Background(), "https://example.com/page")
	if err != nil {
		t.Fatalf("unexpected error for raw URL input: %v", err)
	}
	if !strings.Contains(out, "raw ok") {
		t.Errorf("output %q missing expected body", out)
	}
}

// ---------- Constructor & Interface ----------

func TestHTTPGetTool_DefaultMaxBytes(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{Client: newMockClient(200, "", "")})
	if tool.maxBytes != DefaultHTTPGetMaxBodyBytes {
		t.Errorf("maxBytes = %d, want %d", tool.maxBytes, DefaultHTTPGetMaxBodyBytes)
	}
}

func TestHTTPGetTool_CustomMaxBytes(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{Client: newMockClient(200, "", ""), MaxBytes: 500})
	if tool.maxBytes != 500 {
		t.Errorf("maxBytes = %d, want 500", tool.maxBytes)
	}
}

func TestHTTPGetTool_Name(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{Client: newMockClient(200, "", "")})
	if tool.Name() != "http-get" {
		t.Errorf("Name() = %q, want 'http-get'", tool.Name())
	}
}

func TestHTTPGetTool_Description_NotEmpty(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{Client: newMockClient(200, "", "")})
	if tool.Description() == "" {
		t.Error("Description() is empty, want non-empty")
	}
}

func TestHTTPGetTool_JSONSchema_HasURL(t *testing.T) {
	tool := NewHTTPGetTool(HTTPGetToolConfig{Client: newMockClient(200, "", "")})
	schema := tool.JSONSchema()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("JSONSchema() missing 'properties' map")
	}
	if _, ok := props["url"]; !ok {
		t.Error("JSONSchema() properties missing 'url'")
	}
}

// ---------- helpers ----------

func mustParseURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := parseURLFromToolInput(raw)
	if err != nil {
		t.Fatalf("mustParseURL(%q): %v", raw, err)
	}
	return u
}

func assertPrivate(t *testing.T, ipStr string, want bool) {
	t.Helper()
	ip := net.ParseIP(ipStr)
	if ip == nil {
		t.Fatalf("net.ParseIP(%q) returned nil", ipStr)
	}
	if got := isPrivateIP(ip); got != want {
		t.Errorf("isPrivateIP(%q) = %v, want %v", ipStr, got, want)
	}
}
