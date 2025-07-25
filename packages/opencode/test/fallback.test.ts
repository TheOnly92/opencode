import { describe, expect, test, mock, beforeEach } from "bun:test"
import { MessageV2 } from "../src/session/message-v2"

let generateTextCalls = 0
let streamTextCalls = 0
let streamMode: "verify" | "timeout" = "verify"
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

mock.module("ai", () => {
  function streamText() {
    async function* gen() {
      streamTextCalls++
      if (streamMode === "timeout") {
        yield {
          type: "error",
          error: new DOMException("The operation timed out.", "TimeoutError"),
        }
        return
      }
      yield {
        type: "error",
        error: {
          name: "UnknownError",
          data: { message: "Your organization must be verified to stream" },
        },
      }
    }

    return {
      fullStream: gen(),
    } as any
  }

  async function generateText() {
    const stack = new Error().stack ?? ""
    const skip = stack.includes("ensureTitle")
    if (!skip) generateTextCalls++
    return {
      text: streamMode === "timeout" ? "Timeout fallback" : "🐱 Fallback response",
      toolCalls: [],
      usage: {},
    }
  }

  function wrapLanguageModel({ model }: any) {
    return model
  }

  return {
    streamText,
    generateText,
    wrapLanguageModel,
  }
})

const dummyModel = {
  providerID: "mock",
  modelID: "mock-model",
  info: {
    id: "mock-model",
    temperature: 0,
    tool_call: false,
    options: {},
    cost: { input: 0, output: 0 },
    limit: { input: 32000, output: 32000 },
  },
  language: {},
} as any

mock.module("../src/provider/provider", () => ({
  Provider: {
    async getModel() {
      return dummyModel
    },
    async getSmallModel() {
      return dummyModel
    },
    async defaultModel() {
      return { providerID: "mock", modelID: dummyModel.info.id }
    },
    async getProvider() {
      return { id: "mock" } as any
    },
  },
}))

mock.module("../src/provider/models", () => ({
  ModelsDev: {
    async get() {
      return {
        mock: {
          api: undefined,
          name: "Mock",
          env: [],
          id: "mock",
          npm: undefined,
          models: {
            [dummyModel.info.id]: {
              ...dummyModel.info,
              provider: { npm: "" },
            },
          },
        },
        opencode: {
          api: undefined,
          name: "Opencode",
          env: [],
          id: "opencode",
          npm: undefined,
          models: {},
        },
      }
    },
    refresh() {},
    Model: {
      safeParse(value: unknown) {
        return { success: true, data: value }
      },
    },
    Provider: {
      safeParse(value: unknown) {
        return { success: true, data: value }
      },
    },
  },
}))

import { Session } from "../src/session/index"
import { SessionPrompt } from "../src/session/prompt"
import { Instance } from "../src/project/instance"

async function withApp<T>(cb: () => Promise<T>) {
  return Instance.provide({ directory: process.cwd(), fn: cb })
}

function assertAssistant(message: MessageV2.WithParts): asserts message is MessageV2.WithParts & {
  info: MessageV2.Assistant
} {
  if (message.info.role !== "assistant") {
    throw new Error("expected assistant message")
  }
}

describe("Session streaming fallback", () => {
  beforeEach(() => {
    generateTextCalls = 0
    streamTextCalls = 0
    streamMode = "verify"
  })

  test("falls back to generateText when streaming verification fails", async () => {
    await withApp(async () => {
      const { id: sessionID } = await Session.create(undefined)
      const result = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [
          {
            id: "p1",
            type: "text",
            text: "Hello",
          },
        ],
      })

      expect(streamTextCalls).toBeGreaterThanOrEqual(1)
      expect(generateTextCalls).toBeGreaterThanOrEqual(1)
      assertAssistant(result)
      expect(result.info.error).toBeUndefined()
      const textPart = result.parts.find((p) => p.type === "text") as any
      expect(textPart?.text).toContain("Fallback response")
      const capable = await Session.getStreamingCapable(sessionID)
      expect(capable).toBe(false)
      const timeline = await Session.messages({ sessionID })
      const assistants = timeline.filter(
        (msg): msg is MessageV2.WithParts & { info: MessageV2.Assistant } => msg.info.role === "assistant",
      )
      expect(assistants.length).toBe(1)
      expect(assistants[0]?.info.error).toBeUndefined()
    })
  })

  test("does not fallback when streaming times out", async () => {
    await withApp(async () => {
      streamMode = "timeout"
      const { id: sessionID } = await Session.create(undefined)
      const result = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [
          {
            id: "p-timeout",
            type: "text",
            text: "Hello again",
          },
        ],
      })

      expect(streamTextCalls).toBeGreaterThanOrEqual(1)
      expect(generateTextCalls).toBe(0)
      assertAssistant(result)
      const error = result.info.error
      let message: string | undefined
      if (isRecord(error)) {
        const record: Record<string, unknown> = error
        const maybeMessage = record.message
        if (typeof maybeMessage === "string") message = maybeMessage
        if (!message && "data" in record && isRecord(record.data)) {
          const data = record.data
          const dataMessage = (data as Record<string, unknown>).message
          if (typeof dataMessage === "string") message = dataMessage
        }
      }
      expect(message).toContain("timed out")
      const textPart = result.parts.find((p) => p.type === "text")
      expect(textPart).toBeUndefined()
      const capable = await Session.getStreamingCapable(sessionID)
      expect(capable).toBeUndefined()
    })
  })

  test("subsequent chats skip streaming and use generateText directly", async () => {
    await withApp(async () => {
      const { id: sessionID } = await Session.create(undefined)
      await Session.setStreamingCapable(sessionID, false)

      for (let i = 0; i < 2; i++) {
        await SessionPrompt.prompt({
          sessionID,
          model: { providerID: "mock", modelID: "mock-model" },
          parts: [{ id: `p${i}`, type: "text", text: `Hi ${i}` }],
        })
      }

      expect(streamTextCalls).toBe(0)
      expect(generateTextCalls).toBeGreaterThanOrEqual(2)
    })
  })
})
