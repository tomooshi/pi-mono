import { describe, expect, it } from "vitest";
import { transformMessages } from "../src/providers/transform-messages.js";
import type { AssistantMessage, Message, Model, ToolResultMessage } from "../src/types.js";

function makeAnthropicModel(): Model<"anthropic-messages"> {
	return {
		id: "claude-sonnet-4",
		name: "Claude Sonnet 4",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 16000,
	};
}

function makeAssistant(toolCalls: { id: string; name: string }[]): AssistantMessage {
	return {
		role: "assistant",
		content: toolCalls.map((tc) => ({
			type: "toolCall" as const,
			id: tc.id,
			name: tc.name,
			arguments: {},
		})),
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

function makeToolResult(toolCallId: string, toolName: string): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName,
		content: [{ type: "text", text: "result" }],
		isError: false,
		timestamp: Date.now(),
	};
}

describe("orphaned toolResult handling after compaction", () => {
	const model = makeAnthropicModel();

	it("drops toolResult when parent assistant was pruned by compaction", () => {
		// After compaction, the history might start with a compaction summary (user msg)
		// followed by toolResults whose parent assistant was summarized away
		const messages: Message[] = [
			{ role: "user", content: "compaction summary...", timestamp: Date.now() },
			// This toolResult's parent assistant (with toolu_orphaned) was compacted away
			makeToolResult("toolu_orphaned", "Read"),
			// This assistant and its toolResult are intact
			makeAssistant([{ id: "toolu_valid", name: "Bash" }]),
			makeToolResult("toolu_valid", "Bash"),
		];

		const result = transformMessages(messages, model);

		// The orphaned toolResult should be dropped
		const toolResults = result.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(1);
		expect((toolResults[0] as ToolResultMessage).toolCallId).toBe("toolu_valid");

		// Total messages: user + assistant + toolResult (orphaned one dropped)
		expect(result).toHaveLength(3);
	});

	it("keeps toolResult when parent assistant exists", () => {
		const messages: Message[] = [
			{ role: "user", content: "hello", timestamp: Date.now() },
			makeAssistant([
				{ id: "toolu_1", name: "Read" },
				{ id: "toolu_2", name: "Bash" },
			]),
			makeToolResult("toolu_1", "Read"),
			makeToolResult("toolu_2", "Bash"),
		];

		const result = transformMessages(messages, model);

		const toolResults = result.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(2);
	});

	it("drops multiple orphaned toolResults from different pruned assistants", () => {
		// Simulates compaction cutting deeply — multiple assistant+toolResult pairs pruned
		// but some toolResults survive in the kept portion
		const messages: Message[] = [
			{ role: "user", content: "compaction summary", timestamp: Date.now() },
			makeToolResult("toolu_orphan_1", "Read"),
			makeToolResult("toolu_orphan_2", "Bash"),
			{ role: "user", content: "next question", timestamp: Date.now() },
			makeAssistant([{ id: "toolu_kept", name: "Edit" }]),
			makeToolResult("toolu_kept", "Edit"),
		];

		const result = transformMessages(messages, model);

		const toolResults = result.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(1);
		expect((toolResults[0] as ToolResultMessage).toolCallId).toBe("toolu_kept");
	});

	it("handles mixed orphaned and valid toolResults in the same turn", () => {
		// After compaction mid-turn: some toolResults are orphaned, some are valid
		const messages: Message[] = [
			{ role: "user", content: "compaction summary", timestamp: Date.now() },
			// Orphaned — parent assistant was compacted
			makeToolResult("toolu_orphan", "Read"),
			// Valid — this assistant exists
			makeAssistant([
				{ id: "toolu_a", name: "Bash" },
				{ id: "toolu_b", name: "Edit" },
			]),
			makeToolResult("toolu_a", "Bash"),
			makeToolResult("toolu_b", "Edit"),
		];

		const result = transformMessages(messages, model);

		const toolResults = result.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(2);
		const ids = toolResults.map((r) => (r as ToolResultMessage).toolCallId);
		expect(ids).toEqual(["toolu_a", "toolu_b"]);
	});

	it("still inserts synthetic results for orphaned toolCalls (existing behavior)", () => {
		// Assistant has toolCalls but no toolResults follow — interrupted by user
		const messages: Message[] = [
			{ role: "user", content: "hello", timestamp: Date.now() },
			makeAssistant([{ id: "toolu_1", name: "Read" }]),
			// No toolResult for toolu_1 — user interrupted
			{ role: "user", content: "stop", timestamp: Date.now() },
		];

		const result = transformMessages(messages, model);

		// Synthetic toolResult should be inserted
		const toolResults = result.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(1);
		expect((toolResults[0] as ToolResultMessage).toolCallId).toBe("toolu_1");
		expect((toolResults[0] as ToolResultMessage).isError).toBe(true);
	});
});
