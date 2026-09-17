import type { DecisionBatch, DecisionOutcome } from "openclaw/plugin-sdk/decisions";
import { expect, it, type Mock } from "vitest";
import {
  ACTIVE_MEMORY_ESCALATION_DECISION_PURPOSE,
  ACTIVE_MEMORY_ESCALATION_RUBRIC_VERSION,
} from "./decision-escalation.js";

type EvaluateDecision = (
  batch: DecisionBatch,
  options: {
    agentId?: string;
    purpose: string;
    rubricVersion: string;
    timeoutMs: number;
    signal: AbortSignal;
  },
) => Promise<DecisionOutcome>;

type ActiveMemoryEscalationIntegrationTestHarness = {
  evaluateDecision: Mock<EvaluateDecision>;
  expectPrependContextContains: (result: unknown, text: string) => void;
  hasDebugLine: (needle: string) => boolean;
  hasInfoLine: (needle: string) => boolean;
  registerPluginConfig: (overrides: Record<string, unknown>) => void;
  runEmbeddedAgent: unknown;
  runPromptBuild: (
    event: Record<string, unknown>,
    context?: Record<string, unknown>,
  ) => Promise<unknown>;
  updateConfigFile: (update: (config: Record<string, unknown>) => Record<string, unknown>) => void;
  skippedRecallContext: string;
};

const operatorContext = {
  sessionKey: "agent:main:webchat:direct:operator",
  messageProvider: "webchat",
  channelId: "operator",
};

function recallAnswer(probabilityTrue: number): DecisionOutcome {
  return {
    status: "ok",
    result: {
      model: "test/decider",
      answers: { deepRecall: { type: "boolean", probabilityTrue } },
    },
    provenance: {
      providerId: "test",
      rubricVersion: ACTIVE_MEMORY_ESCALATION_RUBRIC_VERSION,
      runtimeGeneration: "generation-1",
    },
  };
}

export function registerActiveMemoryEscalationIntegrationTests({
  evaluateDecision,
  expectPrependContextContains,
  hasDebugLine,
  hasInfoLine,
  registerPluginConfig,
  runEmbeddedAgent,
  runPromptBuild,
  updateConfigFile,
  skippedRecallContext,
}: ActiveMemoryEscalationIntegrationTestHarness): void {
  const setDecisionAssistance = (enabled: boolean) =>
    updateConfigFile((config) => {
      const agents = (config.agents ?? {}) as Record<string, unknown>;
      const defaults = (agents.defaults ?? {}) as Record<string, unknown>;
      return {
        ...config,
        agents: {
          ...agents,
          defaults: { ...defaults, experimental: { decisionAssistance: enabled } },
        },
      };
    });
  const enableDecisionEscalation = (outcome: DecisionOutcome) => {
    evaluateDecision.mockReset();
    evaluateDecision.mockResolvedValue(outcome);
    registerPluginConfig({ mode: "escalate", escalationDecision: true });
    setDecisionAssistance(true);
  };

  it("recalls an ordinary turn when the Decision model asks for deep recall", async () => {
    enableDecisionEscalation(recallAnswer(0.9));

    const result = await runPromptBuild({ prompt: "Continue with that" }, operatorContext);

    expect(evaluateDecision).toHaveBeenCalledOnce();
    const [batch, options] = evaluateDecision.mock.calls[0] ?? [];
    expect(batch?.state).toEqual({
      latestUserMessage: "Continue with that",
      searchQuery: "Continue with that",
    });
    expect(Object.keys(batch?.questions ?? {})).toEqual(["deepRecall"]);
    expect(options).toMatchObject({
      agentId: "main",
      purpose: ACTIVE_MEMORY_ESCALATION_DECISION_PURPOSE,
      rubricVersion: ACTIVE_MEMORY_ESCALATION_RUBRIC_VERSION,
      signal: expect.any(AbortSignal),
    });
    expect(options?.timeoutMs).toBeGreaterThan(0);
    expect(options?.timeoutMs).toBeLessThanOrEqual(1_000);
    expect(runEmbeddedAgent).toHaveBeenCalledOnce();
    expectPrependContextContains(result, "lemon pepper wings");
  });

  it("sends only bounded normalized text to the Decision model", async () => {
    enableDecisionEscalation(recallAnswer(0.1));

    await runPromptBuild({ prompt: `  ${"context ".repeat(100)}  ` }, operatorContext);

    const state = evaluateDecision.mock.calls[0]?.[0]?.state as
      | { latestUserMessage: string; searchQuery: string }
      | undefined;
    expect(state?.latestUserMessage.length).toBeGreaterThan(0);
    expect(state?.latestUserMessage.length).toBeLessThanOrEqual(480);
    expect(state?.searchQuery.length).toBeGreaterThan(0);
    expect(state?.searchQuery.length).toBeLessThanOrEqual(480);
    expect(state?.latestUserMessage).not.toContain("  ");
    expect(state?.searchQuery).not.toContain("  ");
  });

  it("skips a built-in recall match when the Decision model declines deep recall", async () => {
    enableDecisionEscalation(recallAnswer(0.1));

    const result = await runPromptBuild(
      { prompt: "What did we decide last time?" },
      operatorContext,
    );

    expect(evaluateDecision).toHaveBeenCalledOnce();
    expect(runEmbeddedAgent).not.toHaveBeenCalled();
    expectPrependContextContains(result, skippedRecallContext);
    expect(hasDebugLine("active-memory: recall skipped reason=decision-skip")).toBe(true);
    expect(hasInfoLine("active-memory: recall skipped reason=decision-skip")).toBe(false);
  });

  it("keeps the built-in matcher when no Decision model is available", async () => {
    enableDecisionEscalation({ status: "unavailable", reason: "disabled" });

    const result = await runPromptBuild(
      { prompt: "What did we decide last time?" },
      operatorContext,
    );

    expect(evaluateDecision).toHaveBeenCalledOnce();
    expect(runEmbeddedAgent).toHaveBeenCalledOnce();
    expectPrependContextContains(result, "lemon pepper wings");
    expect(
      hasDebugLine(
        "active-memory: escalation decision unavailable reason=disabled; using built-in matcher",
      ),
    ).toBe(true);
  });

  it("keeps the built-in matcher when the Decision runtime rejects", async () => {
    enableDecisionEscalation(recallAnswer(0.1));
    evaluateDecision.mockRejectedValue(new Error("Decision consumer authority closed."));

    const result = await runPromptBuild(
      { prompt: "What did we decide last time?" },
      operatorContext,
    );

    expect(runEmbeddedAgent).toHaveBeenCalledOnce();
    expectPrependContextContains(result, "lemon pepper wings");
    expect(hasDebugLine("active-memory: escalation decision fallback reason=error")).toBe(true);
  });

  it("does not dispatch when Decision assistance is revoked while the runtime prepares", async () => {
    enableDecisionEscalation(recallAnswer(0.1));
    let dispatched = false;
    evaluateDecision.mockImplementation(async (_batch, options) => {
      setDecisionAssistance(false);
      // Preparation awaits before the runtime's final pre-dispatch signal check.
      await new Promise((resolve) => {
        setTimeout(resolve, 60);
      });
      options.signal.throwIfAborted();
      dispatched = true;
      return recallAnswer(0.1);
    });

    const result = await runPromptBuild(
      { prompt: "What did we decide last time?" },
      operatorContext,
    );

    expect(evaluateDecision).toHaveBeenCalledOnce();
    expect(dispatched).toBe(false);
    expect(runEmbeddedAgent).toHaveBeenCalledOnce();
    expectPrependContextContains(result, "lemon pepper wings");
    expect(
      hasDebugLine(
        "active-memory: escalation decision unavailable reason=revoked; using built-in matcher",
      ),
    ).toBe(true);
  });

  it("does not dispatch when the chat stops being targeted while the runtime prepares", async () => {
    enableDecisionEscalation(recallAnswer(0.1));
    let dispatched = false;
    evaluateDecision.mockImplementation(async (_batch, options) => {
      registerPluginConfig({
        mode: "escalate",
        escalationDecision: true,
        deniedChatIds: ["operator"],
      });
      await new Promise((resolve) => {
        setTimeout(resolve, 60);
      });
      options.signal.throwIfAborted();
      dispatched = true;
      return recallAnswer(0.1);
    });

    await runPromptBuild({ prompt: "What did we decide last time?" }, operatorContext);

    expect(evaluateDecision).toHaveBeenCalledOnce();
    expect(dispatched).toBe(false);
    expect(
      hasDebugLine(
        "active-memory: escalation decision unavailable reason=revoked; using built-in matcher",
      ),
    ).toBe(true);
  });

  it("ignores the answer when escalationDecision is withdrawn during inference", async () => {
    enableDecisionEscalation(recallAnswer(0.1));
    evaluateDecision.mockImplementation(async () => {
      // Dispatched already; the opt-in is withdrawn before the skip answer arrives.
      registerPluginConfig({ mode: "escalate" });
      await new Promise((resolve) => {
        setTimeout(resolve, 60);
      });
      return recallAnswer(0.1);
    });

    const result = await runPromptBuild(
      { prompt: "What did we decide last time?" },
      operatorContext,
    );

    expect(evaluateDecision).toHaveBeenCalledOnce();
    expect(runEmbeddedAgent).toHaveBeenCalledOnce();
    expectPrependContextContains(result, "lemon pepper wings");
    expect(hasDebugLine("active-memory: recall skipped reason=decision-skip")).toBe(false);
  });

  it("does not ask the Decision model without Decision assistance", async () => {
    enableDecisionEscalation(recallAnswer(0.1));
    setDecisionAssistance(false);

    await runPromptBuild({ prompt: "What did we decide last time?" }, operatorContext);

    expect(evaluateDecision).not.toHaveBeenCalled();
    expect(runEmbeddedAgent).toHaveBeenCalledOnce();
    expect(
      hasDebugLine(
        "active-memory: escalation decision requires Decision assistance; using built-in matcher",
      ),
    ).toBe(true);
  });

  it("does not ask the Decision model unless escalationDecision is enabled", async () => {
    enableDecisionEscalation(recallAnswer(0.9));
    registerPluginConfig({ mode: "escalate" });

    const result = await runPromptBuild(
      { prompt: "Explain the current configuration" },
      operatorContext,
    );

    expect(evaluateDecision).not.toHaveBeenCalled();
    expect(runEmbeddedAgent).not.toHaveBeenCalled();
    expectPrependContextContains(result, skippedRecallContext);
  });

  it.each(["always", "off"] as const)(
    "does not ask the Decision model in mode=%s",
    async (mode) => {
      enableDecisionEscalation(recallAnswer(0.1));
      registerPluginConfig({ mode, escalationDecision: true });

      await runPromptBuild({ prompt: "Explain the current configuration" }, operatorContext);

      expect(evaluateDecision).not.toHaveBeenCalled();
    },
  );
}
