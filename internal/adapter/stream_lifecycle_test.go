package adapter

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

func TestEventStreamFrames(t *testing.T) {
	for _, tc := range []struct {
		name, wire string
		want       []string
	}{
		{"multiline", "event: chunk\nid: 1\ndata: {\ndata:   \"answer\": \"ok\"\ndata: }\n\n", []string{"{\n  \"answer\": \"ok\"\n}"}},
		{"comments", ": heartbeat\n\ndata: first\n: still here\ndata: second\n\n", []string{"first\nsecond"}},
		{"whitespace", "data:  keep both sides \n\n", []string{" keep both sides "}},
		{"empty fields", "data\ndata:\n\n", []string{"\n"}},
		{"CRLF", "data: one\r\n\r\ndata: two\r\n\r\n", []string{"one", "two"}},
		{"CR", "data: one\r\rdata: two\r\r", []string{"one", "two"}},
		{"BOM", "\xef\xbb\xbfdata: one\n\n", []string{"one"}},
		{"incomplete EOF", "data: complete\n\ndata: unfinished\n", []string{"complete"}},
		{"terminal", "data: one\n\ndata: [DONE]\n\ndata: too late\n\n", []string{"one"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got []string
			// Splitting every byte also exercises CRLF across network reads.
			err := readEventStream(iotest.OneByteReader(strings.NewReader(tc.wire)), func(data []byte) error {
				got = append(got, string(data))
				return nil
			})
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("events = %q, err = %v; want %q", got, err, tc.want)
			}
		})
	}
}

func TestEventStreamBoundsTheWholeEvent(t *testing.T) {
	// Each line fits the scanner, but combining them must still be bounded.
	wire := strings.Repeat("data: "+strings.Repeat("x", 1024)+"\n", 1024) + "\n"
	called := false
	err := readEventStream(strings.NewReader(wire), func([]byte) error {
		called = true
		return nil
	})
	if err == nil || called {
		t.Fatalf("oversized event dispatched = %v, err = %v", called, err)
	}
}

func TestEventStreamPropagatesReadAndSinkErrors(t *testing.T) {
	stop := errors.New("reader disconnected")
	calls := 0
	err := readEventStream(strings.NewReader("data: first\n\ndata: second\n\n"), func([]byte) error {
		calls++
		return stop
	})
	if err != stop || calls != 1 {
		t.Fatalf("sink calls = %d, err = %v", calls, err)
	}
	readErr := errors.New("upstream connection broke")
	err = readEventStream(io.MultiReader(strings.NewReader("data: unfinished"), iotest.ErrReader(readErr)), func([]byte) error {
		t.Fatal("incomplete event was dispatched")
		return nil
	})
	if !errors.Is(err, readErr) {
		t.Fatalf("read error = %v", err)
	}
}

func TestStreamTerminalClosesAnOpenResponse(t *testing.T) {
	for _, tc := range []struct {
		kind   Kind
		frames []string
		finish string
	}{
		{KindOpenAI, []string{
			`{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}`,
			`[DONE]`,
		}, "stop"},
		{KindAnthropic, []string{
			`{"type":"message_start","message":{"usage":{"input_tokens":7}}}`,
			`{"type":"content_block_delta","delta":{"type":"text_delta","text":"answer"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
			`{"type":"message_stop"}`,
		}, "end_turn"},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			closed := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				for _, frame := range tc.frames {
					fmt.Fprintf(w, "data: %s\n\n", frame)
				}
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
					close(closed)
				case <-release:
				}
			}))
			defer server.Close()
			defer close(release)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			var out collected
			result, err := testRegistry().Chat(ctx, Provider{Kind: tc.kind, BaseURL: server.URL},
				ChatRequest{Model: testModel(), Stream: true}, out.sink)
			if err != nil {
				t.Fatalf("completed stream waited for HTTP EOF: %v", err)
			}
			if result.Text != "answer" || out.answer.String() != "answer" || result.FinishReason != tc.finish ||
				result.Usage != (Usage{InputTokens: 7, OutputTokens: 3}) {
				t.Fatalf("terminal lost answer or final usage: %#v", result)
			}
			select {
			case <-closed:
			case <-time.After(2 * time.Second):
				t.Fatal("completed response body was left open")
			}
		})
	}
}

func TestAnthropicIgnoresFramesAfterMessageStop(t *testing.T) {
	server := sseServer(t, []string{
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":"answer"}}`,
		`{"type":"message_stop"}`,
		`{"type":"error","error":{"message":"after completion"}}`,
	}, nil)
	var out collected
	result, err := testRegistry().Chat(context.Background(), Provider{Kind: KindAnthropic, BaseURL: server.URL},
		ChatRequest{Model: testModel(), Stream: true}, out.sink)
	if err != nil || result.Text != "answer" {
		t.Fatalf("terminal did not end the message: %#v, %v", result, err)
	}
}

func TestOpenAIMultilineStreamKeepsTextAndUsage(t *testing.T) {
	server := sseServer(t, []string{
		"{\ndata: \"choices\":[{\"delta\":{\"content\":\"answer\"}}],\ndata: \"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}\ndata: }",
		"[DONE]",
	}, nil)
	var out collected
	result, err := testRegistry().Chat(context.Background(), Provider{Kind: KindOpenAI, BaseURL: server.URL},
		ChatRequest{Model: testModel(), Stream: true}, out.sink)
	if err != nil || result.Text != "answer" || result.Usage.Total() != 10 {
		t.Fatalf("multiline event was lost: %#v, %v", result, err)
	}
}

func TestEventStreamSizeBoundary(t *testing.T) {
	payload := strings.Repeat("x", 512*1024-1)
	wire := "data: " + payload + "\ndata: " + payload + "\n\n"
	calls := 0
	err := readEventStream(strings.NewReader(wire), func(data []byte) error {
		calls++
		if len(data) != 1024*1024-1 {
			t.Fatalf("payload length = %d", len(data))
		}
		return nil
	})
	if err != nil || calls != 1 {
		t.Fatalf("bounded event rejected: calls = %d, err = %v", calls, err)
	}
	err = readEventStream(strings.NewReader("data: "+strings.Repeat("x", 1024*1024)+"\n\n"), func([]byte) error {
		t.Fatal("oversized line dispatched")
		return nil
	})
	if err == nil {
		t.Fatal("oversized line was accepted")
	}
}

func TestStreamSinkFailureClosesTheUpstream(t *testing.T) {
	for _, tc := range []struct {
		kind  Kind
		frame string
	}{
		{KindOpenAI, `{"choices":[{"delta":{"content":"partial"}}]}`},
		{KindAnthropic, `{"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}`},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			closed := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: %s\n\n", tc.frame)
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
					close(closed)
				case <-release:
				}
			}))
			defer server.Close()
			defer close(release)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			stop := errors.New("client disconnected")
			calls := 0
			_, err := testRegistry().Chat(ctx, Provider{Kind: tc.kind, BaseURL: server.URL},
				ChatRequest{Model: testModel(), Stream: true}, func(Event) error {
					calls++
					return stop
				})
			if err != stop || calls != 1 {
				t.Fatalf("sink calls = %d, err = %v", calls, err)
			}
			select {
			case <-closed:
			case <-time.After(2 * time.Second):
				t.Fatal("client disconnection left the upstream running")
			}
		})
	}
}
