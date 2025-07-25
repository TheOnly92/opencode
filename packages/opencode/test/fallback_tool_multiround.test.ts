import { describe, expect, test, mock, beforeEach, afterEach } from "bun:test"
import { Instance } from "../src/project/instance"
import { randomBytes } from "crypto"
import { $ } from "bun"
import { ToolRegistry } from "../src/tool/registry"
import z from "zod"
import { SessionPrompt } from "../src/session/prompt"
import { MessageV2 } from "../src/session/message-v2"

async function withApp<T>(cb: () => Promise<T>) {
  const id = randomBytes(8).toString("hex")
  const cwd = `/tmp/opencode-fallback-tool-${id}`
  await $`mkdir -p ${cwd}`.quiet()
  try {
  return await Instance.provide({ directory: cwd, fn: cb })
  } finally {
    await $`rm -rf ${cwd}`.quiet().nothrow()
  }
}

let generateTextCalls = 0
const providerModuleURL = new URL("../src/provider/provider.ts", import.meta.url).href
const pluginModuleURL = new URL("../src/plugin/index.ts", import.meta.url).href
mock.module("ai", () => {
  function streamText() {
    async function* gen() {
      yield { type: "error", error: { name: "UnknownError", data: { message: "Your organization must be verified to stream" } } } as any
    }
    return { fullStream: gen() } as any
  }

  async function generateText() {
    generateTextCalls++
    if (generateTextCalls === 1) {
      return {
        text: null,
        toolCalls: [
          { toolCallId: "tc-ping", toolName: "ping", input: { value: "ping" } },
        ],
        usage: {},
      }
    }
    return { text: "done after tool", toolCalls: [], usage: {} }
  }

  function wrapLanguageModel({ model }: any) { return model }
  function tool(def: any) { return def }
  async function generateObject<T>() { return { object: {} as T } }
  function convertToModelMessages(messages: unknown) { return messages as any }
  function jsonSchema(schema: unknown) { return schema }
  function stepCountIs(_count: number) { return () => false }
  class LoadAPIKeyError extends Error {
    static isInstance(value: unknown): value is LoadAPIKeyError { return value instanceof LoadAPIKeyError }
  }
  async function experimental_createMCPClient(_options: any) {
    return { tools: async () => ({}), close() {} }
  }

  return {
    streamText,
    generateText,
    wrapLanguageModel,
    tool,
    generateObject,
    convertToModelMessages,
    jsonSchema,
    stepCountIs,
    LoadAPIKeyError,
    experimental_createMCPClient,
  }
})

const model = {
  providerID: "mock",
  modelID: "test-model",
  info: {
    id: "test-model",
    temperature: 0,
    tool_call: true,
    options: {},
    cost: { input: 0, output: 0 },
    limit: { input: 32000, output: 32000 },
  },
  language: {},
} as any
mock.module("../provider/provider", () => {
  return {
    Provider: {
      getModel: async () => model,
      getSmallModel: async () => model,
      defaultModel: async () => ({ providerID: "mock", modelID: model.info.id }),
      getProvider: async () => ({ id: "mock" } as any),
    },
  }
})

mock.module("../plugin", () => {
  const Plugin = {
    async trigger(_name: string, _i: any, o: any) { return o },
    async list() { return [] },
    init() {},
  }
  return { Plugin }
})

mock.module(providerModuleURL, () => {
  return {
    Provider: {
      getModel: async () => model,
      getSmallModel: async () => model,
      defaultModel: async () => ({ providerID: "mock", modelID: model.info.id }),
      getProvider: async () => ({ id: "mock" } as any),
    },
  }
})
mock.module(pluginModuleURL, () => {
  const Plugin = {
    async trigger(_name: string, _i: any, o: any) { return o },
    async list() { return [] },
    init() {},
  }
  return { Plugin }
})

mock.module("../src/session/summary", () => {
  return {
    SessionSummary: {
      async summarize() {},
    },
  }
})

function assertAssistant(message: MessageV2.WithParts): asserts message is MessageV2.WithParts & {
  info: MessageV2.Assistant
} {
  if (message.info.role !== "assistant") {
    throw new Error("expected assistant message")
  }
}

describe("Fallback multi-round tool-calls", () => {
  beforeEach(() => {
    generateTextCalls = 0
  })

  afterEach(() => {
  })

  test("non-streaming fallback executes tools across rounds", async () => {
    await withApp(async () => {
      const { Session } = await import("../src/session/index")
      // @ts-ignore monkey patch ToolRegistry
      ToolRegistry.tools = async () => [
        {
          id: "ping",
          description: "Return pong",
          parameters: z.object({ value: z.string() }),
          execute: async (_args: any) => ({ output: "pong" }),
        },
      ]
      // @ts-ignore enable tool
      ToolRegistry.enabled = () => ({ ping: true })

      const { id: sessionID } = await Session.create(undefined)
      await Session.update(sessionID, (d: any) => {
        d.title = "Custom"
      })
      generateTextCalls = 0

      const res = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "test-model" },
        parts: [{ id: "u1", type: "text", text: "please call ping" }],
      })

      assertAssistant(res)
      expect(generateTextCalls).toBeGreaterThanOrEqual(2)

      const toolParts = res.parts.filter((p: any) => p.type === "tool")
      expect(toolParts.length).toBeGreaterThan(0)
      const ping = toolParts.find((p: any) => p.tool === "ping") as any
      expect(ping?.state?.status).toBe("completed")

      const textParts = res.parts.filter((p: any) => p.type === "text")
      expect(textParts.length).toBeGreaterThan(0)
      const lastText = textParts[textParts.length - 1] as any
      expect(lastText.text).toBe("done after tool")

      expect(res.info.error).toBeUndefined()
  })
})
})
