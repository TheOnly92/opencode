import { describe, expect, test, mock, beforeEach } from "bun:test"
import { Instance } from "../src/project/instance"
import { randomBytes } from "crypto"
import { $ } from "bun"
import { MessageV2 } from "../src/session/message-v2"

async function withApp<T>(cb: () => Promise<T>) {
  const testId = randomBytes(8).toString("hex")
  const testCwd = `/tmp/opencode-test-${testId}`
  await $`mkdir -p ${testCwd}`.quiet()

  try {
  return await Instance.provide({ directory: testCwd, fn: cb })
  } finally {
    await $`rm -rf ${testCwd}`.quiet().nothrow()
  }
}

let generateTextCallCount = 0

function assertAssistant(message: MessageV2.WithParts): asserts message is MessageV2.WithParts & {
  info: MessageV2.Assistant
} {
  if (message.info.role !== "assistant") {
    throw new Error("expected assistant message")
  }
}

const mockModelInfo = {
  id: "test-model",
  temperature: 0,
  tool_call: true,
  options: {},
  cost: { input: 0, output: 0 },
  limit: { input: 32000, output: 32000 },
}

const originalFetch = globalThis.fetch
if (!Object.prototype.hasOwnProperty.call(originalFetch, "__opencodePatched")) {
  const patched = async (input: any, init?: any) => {
    if (typeof input === "string" && input.includes("models.dev")) {
      return new Response(JSON.stringify({ providers: {} }), { status: 200 })
    }
    return originalFetch(input, init)
  }
  Object.defineProperty(patched, "__opencodePatched", { value: true })
  // @ts-ignore annotate patched fetch
  globalThis.fetch = patched as typeof globalThis.fetch
}

mock.module("ai", () => {
  function streamText() {
    async function* gen() {
      yield {
        type: "error",
        error: { name: "UnknownError", data: { message: "Your organization must be verified to stream" } },
      }
    }
    return { fullStream: gen() } as any
  }

  async function generateText() {
    generateTextCallCount++
    return { 
      text: "This is a test response", 
      toolCalls: [], 
      usage: { inputTokens: 10, outputTokens: 5 } 
    }
  }

  function wrapLanguageModel({ model }: any) {
    return model
  }

  return { streamText, generateText, wrapLanguageModel }
})

mock.module("../src/provider/models", () => {
  const provider = {
    api: undefined,
    name: "Mock",
    env: [],
    id: "mock",
    npm: undefined,
    models: {
      [mockModelInfo.id]: {
        ...mockModelInfo,
        provider: { npm: "" },
      },
    },
  }
  const ModelsDev = {
    get: async () => ({ mock: provider }),
    refresh: () => {},
    Model: { safeParse: (value: unknown) => ({ success: true, data: value }) },
    Provider: { safeParse: (value: unknown) => ({ success: true, data: value }) },
  }
  return { ModelsDev }
})

import { ModelsDev } from "../src/provider/models"
// @ts-ignore override network calls for tests
ModelsDev.get = async () => ({
  mock: {
    api: undefined,
    name: "Mock",
    env: [],
    id: "mock",
    npm: undefined,
    models: {
      [mockModelInfo.id]: {
        ...mockModelInfo,
        provider: { npm: "" },
      },
    },
  },
}) as any
// @ts-ignore override refresh
ModelsDev.refresh = () => {}

import * as ProviderMod from "../src/provider/provider"
const mockModel = {
  providerID: "mock",
  modelID: mockModelInfo.id,
  info: mockModelInfo,
  language: {},
} as any

// @ts-ignore monkey patch
ProviderMod.Provider.getModel = async () => mockModel
// @ts-ignore monkey patch
ProviderMod.Provider.getProvider = async () => ({ id: "mock" } as any)
// @ts-ignore monkey patch
ProviderMod.Provider.defaultModel = async () => ({ providerID: "mock", modelID: mockModel.info.id })
// @ts-ignore monkey patch
ProviderMod.Provider.getSmallModel = async () => mockModel

import { Session } from "../src/session/index"
import { SessionPrompt } from "../src/session/prompt"

describe("Streaming duplicate text prevention", () => {
  beforeEach(() => {
    generateTextCallCount = 0
  })

  test("streaming fallback should not create duplicate text parts", async () => {
    await withApp(async () => {
      const { id: sessionID } = await Session.create(undefined)
      const result = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "test-1", type: "text", text: "Generate a summary" }],
      })

      assertAssistant(result)
      expect(result.info.error).toBeUndefined()
      const textParts = result.parts.filter(part => part.type === "text")
      expect(textParts.length).toBe(1)
      expect(textParts[0].text).toBe("This is a test response")
      expect(generateTextCallCount).toBeGreaterThan(0)
    })
  })

  test("direct fallback (streamingCapable=false) should not create duplicate text parts", async () => {
    await withApp(async () => {
      const { id: sessionID } = await Session.create(undefined)
      await Session.setStreamingCapable(sessionID, false)
      const result = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "test-2", type: "text", text: "Another test message" }],
      })

      assertAssistant(result)
      expect(result.info.error).toBeUndefined()
      const textParts = result.parts.filter(part => part.type === "text")
      expect(textParts.length).toBe(1)
      expect(textParts[0].text).toBe("This is a test response")
      expect(generateTextCallCount).toBeGreaterThan(0)
    })
  })

  test("multiple consecutive messages should each have only one text part", async () => {
    await withApp(async () => {
      const { id: sessionID } = await Session.create(undefined)
      await Session.setStreamingCapable(sessionID, false)
      const result1 = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "msg-1", type: "text", text: "First message" }],
      })
      const result2 = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "msg-2", type: "text", text: "Second message" }],
      })

      assertAssistant(result1)
      assertAssistant(result2)
      expect(result1.info.error).toBeUndefined()
      expect(result2.info.error).toBeUndefined()
      const textParts1 = result1.parts.filter(part => part.type === "text")
      const textParts2 = result2.parts.filter(part => part.type === "text")
      expect(textParts1.length).toBe(1)
      expect(textParts2.length).toBe(1)
      expect(textParts1[0].text).toBe("This is a test response")
      expect(textParts2[0].text).toBe("This is a test response")
      expect(generateTextCallCount).toBeGreaterThanOrEqual(2)
    })
  })
  test("fallback after streaming rejection drains queued prompts", async () => {
    await withApp(async () => {
      const { id: sessionID } = await Session.create(undefined)

      const first = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "queued-1", type: "text", text: "First fallback request" }],
      })

      await new Promise((resolve) => setTimeout(resolve, 10))

      const second = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "queued-2", type: "text", text: "Second fallback request" }],
      })

      const [firstResult, secondResult] = await Promise.all([first, second])

      assertAssistant(firstResult)
      assertAssistant(secondResult)
      expect(firstResult.info.error).toBeUndefined()
      expect(secondResult.info.error).toBeUndefined()

      const firstText = firstResult.parts.filter(part => part.type === "text")
      expect(firstText.length).toBe(1)
      expect(firstText[0].text).toBe("This is a test response")

      const secondText = secondResult.parts.filter(part => part.type === "text")
      expect(secondText.length).toBe(1)
      expect(secondText[0].text).toBe("This is a test response")

      expect(generateTextCallCount).toBeGreaterThanOrEqual(2)
    })
  }, 8000)

})
