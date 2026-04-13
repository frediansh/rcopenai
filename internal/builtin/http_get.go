package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
)

const DefaultHTTPGetMaxBodyBytes = int64(1_000_000) // 1MB

type HTTPGetToolConfig struct {
	Client     *http.Client
	AllowHosts []string
	MaxBytes   int64
}

func NewHTTPGetTool(cfg HTTPGetToolConfig) *HTTPGetTool {
	maxBytes := cfg.MaxBytes
	if maxBytes <= 0 {
		maxBytes = DefaultHTTPGetMaxBodyBytes
	}
	return &HTTPGetTool{
		client:     cfg.Client,
		allowHosts: cfg.AllowHosts,
		maxBytes:   maxBytes,
	}
}

type HTTPGetTool struct {
	client     *http.Client
	allowHosts []string
	maxBytes   int64
}

func (t *HTTPGetTool) Name() string {
	return "http-get"
}

func (t *HTTPGetTool) Description() string {
	return "HTTP GET a URL and return status, content-type, and a truncated body."
}

func (t *HTTPGetTool) JSONSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"url": map[string]any{
				"type":        "string",
				"description": "HTTP or HTTPS URL",
			},
		},
		"required": []string{"url"},
	}
}

func (t *HTTPGetTool) Call(ctx context.Context, input string) (string, error) {
	if t == nil || t.client == nil {
		return "", errors.New("http-get not configured")
	}

	u, err := parseURLFromToolInput(input)
	if err != nil {
		return "", err
	}
	if err := validateHTTPGetURL(u, t.allowHosts); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", fmt.Errorf("http.NewRequest: %w", err)
	}
	req.Header.Set("User-Agent", "rcopenai/http-get")
	req.Header.Set("Accept", "*/*")

	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http-get request failed: %w", err)
	}
	defer resp.Body.Close()

	limited := io.LimitReader(resp.Body, t.maxBytes)
	bodyBytes, readErr := io.ReadAll(limited)
	if readErr != nil {
		return "", fmt.Errorf("read body: %w", readErr)
	}

	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "unknown"
	}

	return fmt.Sprintf(
		"status=%d\ncontent-type=%s\nbody(truncated-to-%d-bytes)=%s",
		resp.StatusCode,
		ct,
		t.maxBytes,
		string(bodyBytes),
	), nil
}

func parseURLFromToolInput(input string) (*url.URL, error) {
	s := strings.TrimSpace(input)
	if s == "" {
		return nil, errors.New("http-get input is empty")
	}

	if strings.HasPrefix(s, "{") {
		var payload struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal([]byte(s), &payload); err == nil && strings.TrimSpace(payload.URL) != "" {
			s = strings.TrimSpace(payload.URL)
		}
	}

	u, err := url.Parse(s)
	if err != nil {
		return nil, fmt.Errorf("invalid url: %w", err)
	}
	return u, nil
}

func validateHTTPGetURL(u *url.URL, allowHosts []string) error {
	if u == nil {
		return errors.New("nil url")
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported scheme %q (only http/https)", u.Scheme)
	}
	if u.Host == "" {
		return errors.New("missing host")
	}
	host := u.Hostname()
	if host == "" {
		return errors.New("missing hostname")
	}

	if strings.EqualFold(host, "localhost") {
		return errors.New("blocked host: localhost")
	}
	if ip := net.ParseIP(host); ip != nil {
		if isPrivateIP(ip) || ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
			return fmt.Errorf("blocked ip: %s", ip.String())
		}
	}

	if len(allowHosts) > 0 {
		ok := false
		for _, allowed := range allowHosts {
			if strings.EqualFold(host, strings.TrimSpace(allowed)) {
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("host %q not in allowlist", host)
		}
	}

	return nil
}

func isPrivateIP(ip net.IP) bool {
	ip4 := ip.To4()
	if ip4 == nil {
		return ip.IsPrivate()
	}
	if ip4[0] == 10 {
		return true
	}
	if ip4[0] == 172 && ip4[1] >= 16 && ip4[1] <= 31 {
		return true
	}
	if ip4[0] == 192 && ip4[1] == 168 {
		return true
	}
	if ip4[0] == 127 {
		return true
	}
	if ip4[0] == 169 && ip4[1] == 254 {
		return true
	}
	if ip4[0] == 0 {
		return true
	}
	return false
}
