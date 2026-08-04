package provider

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
)

// TestSendWithRetryWithMaxRetriesZeroPinsOneRequest proves the vision retry
// contract at the transport layer: with WithMaxRetries(ctx, 0) a retryable
// status (429/5xx) and a transient auth rejection both fail after exactly one
// HTTP attempt — the outer router owns retries, never SendWithRetry.
func TestSendWithRetryWithMaxRetriesZeroPinsOneRequest(t *testing.T) {
	for _, tc := range []struct {
		name    string
		status  int
		retry   bool
		keyOK   bool
		wantErr string
	}{
		{"retryable-status", 429, false, false, "status 429"},
		{"server-error", 500, false, false, "status 500"},
		{"auth-rejected", 401, true, true, "401"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			cl := &http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				return statusResp(tc.status, nil), nil
			})}
			ctx := WithMaxRetries(context.Background(), 0)
			ctx = WithRequestAttemptCounter(ctx)
			_, err := SendWithRetry(ctx, cl, SendOptions{Provider: "p", KeyEnv: "KEY", RetryAuth: tc.retry}, newDummyReq)
			if err == nil {
				t.Fatal("expected error")
			}
			if calls != 1 {
				t.Fatalf("HTTP calls = %d, want exactly 1", calls)
			}
			if got := RequestAttemptCount(ctx); got != 1 {
				t.Fatalf("RequestAttemptCount = %d, want 1", got)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

// TestSendWithRetryHonoursBoundedRetryBudget proves an explicit budget of 2
// retries yields at most 3 HTTP attempts even for a perpetually retryable
// upstream.
func TestSendWithRetryHonoursBoundedRetryBudget(t *testing.T) {
	calls := 0
	cl := &http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		return statusResp(429, nil), nil
	})}
	ctx := WithMaxRetries(context.Background(), 2)
	ctx = WithRequestAttemptCounter(ctx)
	if _, err := SendWithRetry(ctx, cl, SendOptions{Provider: "p", KeyEnv: "KEY"}, newDummyReq); err == nil {
		t.Fatal("expected terminal failure after budget exhausted")
	}
	if calls != 3 {
		t.Fatalf("HTTP calls = %d, want 3 (initial + 2 retries)", calls)
	}
	if got := RequestAttemptCount(ctx); got != 3 {
		t.Fatalf("RequestAttemptCount = %d, want 3", got)
	}
}

// TestSendWithRetryDefaultBudgetUnaffected proves plain callers keep the full
// package default: a retryable status is retried far past a single attempt.
func TestSendWithRetryDefaultBudgetUnaffected(t *testing.T) {
	var calls atomic.Int64
	cl := &http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		return statusResp(429, nil), nil
	})}
	// Every attempt fails; the default MaxRetries loop would take up to
	// MaxRetries+1 calls and ~seconds of backoff. Bound the test by cancelling
	// after the first retry — the point is that more than one attempt happens.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		for calls.Load() < 2 {
		}
		cancel()
	}()
	SendWithRetry(ctx, cl, SendOptions{Provider: "p", KeyEnv: "KEY"}, newDummyReq)
	if calls.Load() < 2 {
		t.Fatalf("HTTP calls = %d, want >= 2 (default budget must still retry)", calls.Load())
	}
}

// TestMaxRetriesFromContextAbsent proves a plain context reports no override,
// so SendWithRetry falls back to the package default.
func TestMaxRetriesFromContextAbsent(t *testing.T) {
	if max, ok := MaxRetriesFromContext(context.Background()); ok || max != 0 {
		t.Fatalf("plain ctx: max=%d ok=%v, want 0/false", max, ok)
	}
	if max, ok := MaxRetriesFromContext(WithMaxRetries(context.Background(), 0)); !ok || max != 0 {
		t.Fatalf("budget ctx: max=%d ok=%v, want 0/true", max, ok)
	}
}

// TestRequestRetryLimitFallback proves the resolved budget defaults to the
// package constant when no override is attached.
func TestRequestRetryLimitFallback(t *testing.T) {
	if got := requestRetryLimit(context.Background()); got != MaxRetries {
		t.Fatalf("default limit = %d, want %d", got, MaxRetries)
	}
	if got := requestRetryLimit(WithMaxRetries(context.Background(), 4)); got != 4 {
		t.Fatalf("overridden limit = %d, want 4", got)
	}
	if got := requestRetryLimit(WithMaxRetries(context.Background(), -1)); got != MaxRetries {
		t.Fatalf("negative override limit = %d, want default %d", got, MaxRetries)
	}
}

var _ = io.NopCloser // keep io imported for parity with sibling tests
