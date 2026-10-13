import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildEmbeddedRunnerAssistant,
  makeEmbeddedRunnerAttempt,
} from "../../test-helpers/embedded-agent-runner-e2e-fixtures.js";
import { buildEmbeddedRunPayloads } from "./payloads.js";
import { resolveEmbeddedRunTerminal } from "./terminal-resolution.js";
import { makeTerminalInput, type TerminalInput } from "./terminal-resolution.test-support.js";
import { createEmbeddedRunTerminalRetryState } from "./terminal-retry-state.js";

// mock-isolation: terminal recovery must not update persistent auth-profile state.
vi.mock("./auth-profile-success.js", () => ({
  markEmbeddedRunAuthProfileSuccess: vi.fn(),
  reportEmbeddedRunSuccessfulAuthBinding: vi.fn(),
}));

const REJECTED_TOOL_CALL_MESSAGE = "Provider returned an incomplete or malformed tool call";

// The reported turn: an exec call completed, then the provider streamed a second
// call whose arguments were not valid JSON. The transport removed that call
// before dispatch and ended the turn with stopReason=error.
function rejectedCallAfterSettledTool(
  overrides: Parameters<typeof makeEmbeddedRunnerAttempt>[0] = {},
  assistantOverrides: Parameters<typeof buildEmbeddedRunnerAssistant>[0] = {},
) {
  const assistant = buildEmbeddedRunnerAssistant({
    api: "openai-completions",
    provider: "meterkey",
    model: "deepseek/deepseek-v4-flash",
    stopReason: "error",
    errorMessage: REJECTED_TOOL_CALL_MESSAGE,
    content: [],
    ...assistantOverrides,
  });
  const attempt = makeEmbeddedRunnerAttempt({
    assistantTexts: [],
    lastAssistant: assistant,
    currentAttemptAssistant: assistant,
    toolMetas: [{ toolName: "exec", toolCallId: "call_exec", meta: "git log" }],
    itemLifecycle: { startedCount: 1, completedCount: 1, activeCount: 0 },
    ...overrides,
  });
  return {
    attempt,
    attemptAssistant: assistant,
    activeErrorContext: { provider: "meterkey", model: "deepseek/deepseek-v4-flash" },
    modelApi: "openai-completions",
    payloadsWithToolMedia: buildEmbeddedRunPayloads({
      assistantTexts: attempt.assistantTexts,
      lastAssistant: assistant,
      currentAssistant: assistant,
      sessionKey: "session:rejected-tool-call",
    }),
  } satisfies Partial<TerminalInput>;
}

describe("terminal resolution for a tool call rejected before dispatch", () => {
  beforeEach(() => vi.clearAllMocks());

  it("continues once after settled tools instead of replaying the attempt", async () => {
    const turn = rejectedCallAfterSettledTool();
    // The completed exec makes the attempt unsafe to replay.
    expect(turn.attempt.replayMetadata.hadPotentialSideEffects).toBe(true);
    const activateInternalPrompt = vi.fn();

    await expect(
      resolveEmbeddedRunTerminal(
        makeTerminalInput({ ...turn, sessionPromptState: { activateInternalPrompt } }),
      ),
    ).resolves.toEqual({ action: "retry" });
    expect(activateInternalPrompt).toHaveBeenCalledOnce();
    const instruction = activateInternalPrompt.mock.calls[0]?.[0] as string;
    expect(instruction).toContain("rejected before it ran");
    expect(instruction).toContain("Do not repeat completed tool calls");

    const exhausted = await resolveEmbeddedRunTerminal(
      makeTerminalInput({
        ...turn,
        retryState: { ...createEmbeddedRunTerminalRetryState(), emptyResponseAttempts: 1 },
      }),
    );
    expect(exhausted.action).toBe("complete");
    if (exhausted.action !== "complete") {
      return;
    }
    expect(exhausted.result.meta.error).toMatchObject({ kind: "incomplete_turn" });
  });

  it("also covers the malformed-arguments error code", async () => {
    const activateInternalPrompt = vi.fn();
    await expect(
      resolveEmbeddedRunTerminal(
        makeTerminalInput({
          ...rejectedCallAfterSettledTool(
            {},
            {
              errorMessage: "Provider completed tool call with malformed JSON arguments",
              errorCode: "malformed_tool_call_arguments",
            },
          ),
          sessionPromptState: { activateInternalPrompt },
        }),
      ),
    ).resolves.toEqual({ action: "retry" });
  });

  // The follow-up report: an exec call was refused by preflight, the model read
  // that error and answered with another call whose JSON was malformed.
  function rejectedCallAfterConsumedToolError(params: { modelSawFailure: boolean }) {
    const rejected = buildEmbeddedRunnerAssistant({
      api: "openai-completions",
      provider: "meterkey",
      model: "deepseek/deepseek-v4-flash",
      stopReason: "error",
      errorMessage: REJECTED_TOOL_CALL_MESSAGE,
      content: [],
    });
    const refusedExec = {
      role: "toolResult",
      toolCallId: "call_exec",
      toolName: "exec",
      isError: true,
      content: [{ type: "text", text: "exec preflight: complex interpreter invocation detected" }],
    };
    const execCall = buildEmbeddedRunnerAssistant({
      stopReason: "toolUse",
      content: [{ type: "toolCall", id: "call_exec", name: "exec", arguments: {} }],
    });
    const messagesSnapshot = [
      { role: "user", content: "summarize the repo" },
      execCall,
      refusedExec,
      ...(params.modelSawFailure ? [rejected] : []),
    ] as unknown as Parameters<typeof makeEmbeddedRunnerAttempt>[0]["messagesSnapshot"];
    return rejectedCallAfterSettledTool({
      lastAssistant: rejected,
      currentAttemptAssistant: rejected,
      lastToolError: { toolName: "exec", error: "exec preflight refused" },
      messagesSnapshot,
    });
  }

  it("continues after an earlier tool failure the model already answered", async () => {
    const activateInternalPrompt = vi.fn();
    await expect(
      resolveEmbeddedRunTerminal(
        makeTerminalInput({
          ...rejectedCallAfterConsumedToolError({ modelSawFailure: true }),
          sessionPromptState: { activateInternalPrompt },
        }),
      ),
    ).resolves.toEqual({ action: "retry" });
    expect(activateInternalPrompt.mock.calls[0]?.[0]).toContain("rejected before it ran");
  });

  it("does not continue when no model response followed the tool failure", async () => {
    const activateInternalPrompt = vi.fn();
    const result = await resolveEmbeddedRunTerminal(
      makeTerminalInput({
        ...rejectedCallAfterConsumedToolError({ modelSawFailure: false }),
        sessionPromptState: { activateInternalPrompt },
      }),
    );

    expect(result.action).toBe("complete");
    expect(activateInternalPrompt).not.toHaveBeenCalled();
  });

  it("does not continue after an unrelated provider error", async () => {
    const overrides = {};
    const assistant = { errorMessage: "upstream connect error" };
    const activateInternalPrompt = vi.fn();
    const result = await resolveEmbeddedRunTerminal(
      makeTerminalInput({
        ...rejectedCallAfterSettledTool(overrides, assistant),
        sessionPromptState: { activateInternalPrompt },
      }),
    );

    expect(result.action).toBe("complete");
    expect(activateInternalPrompt).not.toHaveBeenCalled();
  });
});
