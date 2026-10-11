package adapter

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestFallbackSinkFailureKeepsCompletedResult(t *testing.T) {
	for _, tc := range []struct {
		kind Kind
		body string
	}{
		{KindOpenAI, `{"choices":[{"message":{"content":"answer","reasoning_content":"thought","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":3}}`},
		{KindAnthropic, `{"content":[{"type":"thinking","thinking":"thought"},{"type":"text","text":"answer"},{"type":"tool_use","id":"call_1","name":"lookup","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":7,"output_tokens":3}}`},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			registry := testRegistry()
			provider := Provider{Kind: tc.kind, BaseURL: server.URL}
			req := ChatRequest{Model: testModel(), Stream: true}
			want, err := registry.Chat(context.Background(), provider, req, func(Event) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			for failAt := 1; failAt <= 4; failAt++ {
				t.Run(fmt.Sprintf("event_%d", failAt), func(t *testing.T) {
					stop := errors.New("client disconnected")
					calls := 0
					got, err := registry.Chat(context.Background(), provider, req, func(Event) error {
						calls++
						if calls == failAt {
							return stop
						}
						return nil
					})
					if err != stop || calls != failAt {
						t.Fatalf("delivery continued: calls = %d, err = %v", calls, err)
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("received result lost on delivery failure: got %#v; want %#v", got, want)
					}
				})
			}
		})
	}
}

func TestOpenAIStreamSinkFailureKeepsReceivedContent(t *testing.T) {
	for _, content := range []string{"answer", "<think>thought</think>answer"} {
		t.Run(content, func(t *testing.T) {
			server := sseServer(t, []string{
				fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}],"usage":{"prompt_tokens":7,"completion_tokens":3}}`, content),
			}, nil)
			stop := errors.New("client disconnected")
			result, err := testRegistry().Chat(context.Background(), Provider{Kind: KindOpenAI, BaseURL: server.URL},
				ChatRequest{Model: testModel(), Stream: true}, func(event Event) error {
					if event.Type != EventUsage {
						return stop
					}
					return nil
				})
			thought, answer := splitThinking(content)
			if err != stop || result.Text != answer || result.Reasoning != thought || result.Usage.Total() != 10 {
				t.Fatalf("received content lost on sink failure: %#v, %v", result, err)
			}
		})
	}
}

func TestOpenAIMixedReasoningSurvivesLaterSinkCancellation(t *testing.T) {
	for _, field := range []string{"reasoning", "reasoning_content"} {
		for _, earlierField := range []string{"", "earlier field "} {
			t.Run(field+"/"+earlierField, func(t *testing.T) {
				var frames []string
				if earlierField != "" {
					frames = append(frames, fmt.Sprintf(`{"choices":[{"delta":{%q:%q}}]}`, field, earlierField))
				}
				frames = append(frames,
					`{"choices":[{"delta":{"content":"<think>inline</think>answer"}}],"usage":{"prompt_tokens":7,"completion_tokens":3}}`,
					fmt.Sprintf(`{"choices":[{"delta":{%q:"later field"}}]}`, field),
					`{"choices":[{"delta":{"content":"too late"}}]}`,
					`[DONE]`,
				)
				server := sseServer(t, frames, nil)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				var out collected
				result, err := testRegistry().Chat(ctx, Provider{Kind: KindOpenAI, BaseURL: server.URL},
					ChatRequest{Model: testModel(), Stream: true}, func(event Event) error {
						_ = out.sink(event)
						if event.Type == EventReasoning && event.Text == "later field" {
							cancel()
							return context.Canceled
						}
						return nil
					})
				if err != context.Canceled || ctx.Err() != context.Canceled {
					t.Fatalf("cancellation was lost: err = %v, ctx = %v", err, ctx.Err())
				}
				if result.Reasoning != earlierField+"later fieldinline" || result.Text != "answer" || result.Usage.Total() != 10 {
					t.Fatalf("mixed reasoning lost after later sink cancellation: %#v", result)
				}
				if out.answer.String() != "answer" || out.reasoning.String() != earlierField+"inlinelater field" {
					t.Fatalf("delivery changed or continued after cancellation: answer = %q, reasoning = %q", out.answer.String(), out.reasoning.String())
				}
			})
		}
	}
}
