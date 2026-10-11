package chat

import (
	"context"
	"testing"

	"github.com/OnyxAxisOwO/ObsidianArc/internal/adapter"
)

func TestCancelledFallbackKeepsReportedUsage(t *testing.T) {
	for _, cancelAt := range []string{EventReasoning, EventDelta} {
		t.Run(cancelAt, func(t *testing.T) {
			f := newFixture(t)
			f.upstream.reply(`{"choices":[{"message":{"content":"answer","reasoning_content":"thought"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3}}`)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			req := TurnRequest{User: f.account, ModelID: f.model.ID, Content: "hello", Stream: true}
			resolved, release, err := f.service.Prepare(ctx, &req)
			if err != nil {
				t.Fatal(err)
			}
			defer release()
			var records []TurnRecord
			f.service.OnTurn = func(saveCtx context.Context, record TurnRecord) {
				if saveCtx.Err() != nil {
					t.Errorf("accounting inherited cancelled context: %v", saveCtx.Err())
				}
				records = append(records, record)
			}
			conversationID := ""
			usageEmitted := false
			err = f.service.Run(ctx, req, resolved, func(event string, payload any) error {
				if event == EventStart {
					conversationID = payload.(StartPayload).ConversationID
				}
				if event == EventUsage {
					usageEmitted = true
				}
				if event == cancelAt {
					cancel()
					return context.Canceled
				}
				return nil
			})
			if err != nil {
				t.Fatalf("cancelled turn: %v", err)
			}
			if ctx.Err() == nil || usageEmitted {
				t.Fatalf("fixture did not cancel before usage delivery: ctx = %v, usage = %v", ctx.Err(), usageEmitted)
			}
			if len(records) != 1 {
				t.Fatalf("recorded %d turns, want 1", len(records))
			}
			record := records[0]
			if record.Status != StatusAborted || record.Usage != (adapter.Usage{InputTokens: 7, OutputTokens: 3}) || record.Credits <= 0 {
				t.Fatalf("completed upstream usage disappeared on disconnect: %#v", record)
			}
			messages := f.messages(t, conversationID)
			if len(messages) != 2 || messages[1].Reasoning != "thought" {
				t.Fatalf("partial transcript not saved: %#v", messages)
			}
			if cancelAt == EventReasoning && messages[1].Content != "" {
				t.Fatalf("undelivered answer saved as visible text: %q", messages[1].Content)
			}
			stats := messages[1].Stats
			if stats == nil || stats.InputTokens == nil || *stats.InputTokens != 7 || stats.OutputTokens == nil || *stats.OutputTokens != 3 {
				t.Fatalf("saved partial answer lost token counts: %#v", stats)
			}
		})
	}
}
