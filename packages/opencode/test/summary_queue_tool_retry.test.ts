import { describe, expect, test, mock, beforeEach, afterEach } from "bun:test"
import { randomBytes } from "crypto"
import path from "path"
import fs from "fs/promises"
import z from "zod"
import { MessageV2 } from "../src/session/message-v2"

const sandboxRoot = path.join(process.cwd(), ".sandbox-test-env")
const envRoot = path.join(sandboxRoot, `home-${randomBytes(8).toString("hex")}`)
const dataHome = path.join(envRoot, ".local", "share")
const cacheHome = path.join(envRoot, ".cache")
const configHome = path.join(envRoot, ".config")
const stateHome = path.join(envRoot, ".local", "state")
const dataPath = path.join(dataHome, "opencode")
const cachePath = path.join(cacheHome, "opencode")
const configPath = path.join(configHome, "opencode")
const statePath = path.join(stateHome, "opencode")
const logPath = path.join(dataPath, "log")
const binPath = path.join(dataPath, "bin")
const projectPath = path.join(dataPath, "project")
const storagePath = path.join(dataPath, "storage")

await Promise.all([
  fs.mkdir(sandboxRoot, { recursive: true }),
  fs.mkdir(dataHome, { recursive: true }),
  fs.mkdir(cacheHome, { recursive: true }),
  fs.mkdir(configHome, { recursive: true }),
  fs.mkdir(stateHome, { recursive: true }),
  fs.mkdir(dataPath, { recursive: true }),
  fs.mkdir(cachePath, { recursive: true }),
  fs.mkdir(configPath, { recursive: true }),
  fs.mkdir(statePath, { recursive: true }),
  fs.mkdir(logPath, { recursive: true }),
  fs.mkdir(binPath, { recursive: true }),
  fs.mkdir(projectPath, { recursive: true }),
  fs.mkdir(storagePath, { recursive: true }),
])

const globalModule = await import("../src/global/index.ts")
const originalGlobalPath = { ...globalModule.Global.Path }

async function withApp<T>(cb: (deps: {
  Session: typeof import("../src/session/index").Session
  SessionPrompt: typeof import("../src/session/prompt").SessionPrompt
  ToolRegistry: typeof import("../src/tool/registry").ToolRegistry
}) => Promise<T>) {
  const testId = randomBytes(8).toString("hex")
  const testCwd = path.join(envRoot, "workspace", testId)
  await fs.mkdir(testCwd, { recursive: true })

  const { Instance } = await import("../src/project/instance")
  const originalEnv = {
    HOME: process.env["HOME"],
    XDG_DATA_HOME: process.env["XDG_DATA_HOME"],
    XDG_CACHE_HOME: process.env["XDG_CACHE_HOME"],
    XDG_CONFIG_HOME: process.env["XDG_CONFIG_HOME"],
    XDG_STATE_HOME: process.env["XDG_STATE_HOME"],
  }

  try {
    process.env["HOME"] = envRoot
    process.env["XDG_DATA_HOME"] = dataHome
    process.env["XDG_CACHE_HOME"] = cacheHome
    process.env["XDG_CONFIG_HOME"] = configHome
    process.env["XDG_STATE_HOME"] = stateHome
    Object.assign(globalModule.Global.Path as any, {
      home: envRoot,
      data: dataPath,
      bin: binPath,
      log: logPath,
      cache: cachePath,
      config: configPath,
      state: statePath,
    })
    return await Instance.provide({
      directory: testCwd,
      fn: async () => {
        const [{ Session }, { SessionPrompt }, { ToolRegistry }] = await Promise.all([
          import("../src/session/index"),
          import("../src/session/prompt"),
          import("../src/tool/registry"),
        ])
        return cb({ Session, SessionPrompt, ToolRegistry })
      },
    })
  } finally {
    Object.assign(globalModule.Global.Path as any, originalGlobalPath)
    process.env["HOME"] = originalEnv.HOME
    process.env["XDG_DATA_HOME"] = originalEnv.XDG_DATA_HOME
    process.env["XDG_CACHE_HOME"] = originalEnv.XDG_CACHE_HOME
    process.env["XDG_CONFIG_HOME"] = originalEnv.XDG_CONFIG_HOME
    process.env["XDG_STATE_HOME"] = originalEnv.XDG_STATE_HOME
    await Instance.dispose().catch(() => {})
    await fs.rm(testCwd, { recursive: true, force: true }).catch(() => {})
  }
}

let generateTextCalls = 0
let streamTextCalls = 0

// Scenario selector for generateText behavior
// "normal" -> returns plain text, no tool calls
// "toolCall" -> requests the "explode" tool
// "errorLoop" -> throws server_error to trigger retry exhaustion
let scenario: "normal" | "toolCall" | "errorLoop" = "normal"

function setScenario(s: typeof scenario) {
  scenario = s
}

function resetCounters() {
  generateTextCalls = 0
  streamTextCalls = 0
}

function resetScenario() {
  scenario = "normal"
}

function assertAssistant(message: MessageV2.WithParts): asserts message is MessageV2.WithParts & {
  info: MessageV2.Assistant
} {
  if (message.info.role !== "assistant") {
    throw new Error("expected assistant message")
  }
}

mock.module("ai", () => {
  function streamText() {
    async function* gen() {
      streamTextCalls++
      yield {
        type: "error",
        error: { name: "UnknownError", data: { message: "Your organization must be verified to stream" } },
      }
    }
    return { fullStream: gen() } as any
  }
  async function generateText() {
    generateTextCalls++
    if (scenario === "toolCall") {
      return {
        text: null,
        toolCalls: [
          {
            toolCallId: "tc1",
            toolName: "explode",
            input: {},
          },
        ],
        usage: {},
      }
    }
    if (scenario === "errorLoop") {
      const err: any = new Error("Internal server error")
      err.type = "server_error"
      throw err
    }
    return { text: "assistant text", toolCalls: [], usage: {} }
  }
  function wrapLanguageModel({ model }: any) {
    return model
  }
  async function generateObject<T>() {
    return { object: {} as T }
  }
  function tool(def: any) {
    return def
  }
  function convertToModelMessages(messages: unknown) {
    return messages as any
  }
  function jsonSchema(schema: unknown) {
    return schema
  }
  function stepCountIs(_count: number) {
    return () => false
  }
  class LoadAPIKeyError extends Error {
    static isInstance(value: unknown): value is LoadAPIKeyError {
      return value instanceof LoadAPIKeyError
    }
  }
  class APICallError extends Error {
    statusCode?: number
    isRetryable: boolean
    responseHeaders?: Record<string, string>
    responseBody?: string
    static isInstance(value: unknown): value is APICallError {
      return value instanceof APICallError
    }
    constructor(options?: { statusCode?: number; isRetryable?: boolean; responseHeaders?: Record<string, string>; responseBody?: string }) {
      super("mock api error")
      this.statusCode = options?.statusCode
      this.isRetryable = options?.isRetryable ?? false
      this.responseHeaders = options?.responseHeaders
      this.responseBody = options?.responseBody
    }
  }
  class NoSuchModelError extends Error {
    static isInstance(value: unknown): value is NoSuchModelError {
      return value instanceof NoSuchModelError
    }
  }
  async function experimental_createMCPClient(_options: any) {
    return {
      tools: async () => ({}),
      close() {},
    }
  }
  return {
    streamText,
    generateText,
    wrapLanguageModel,
    generateObject,
    tool,
    convertToModelMessages,
    jsonSchema,
    stepCountIs,
    LoadAPIKeyError,
    APICallError,
    NoSuchModelError,
    experimental_createMCPClient,
  }
})

const dummyModel = {
  providerID: "mock",
  modelID: "mock-model",
  info: {
    id: "mock-model",
    temperature: 0,
    tool_call: true,
    options: {},
    cost: { input: 0, output: 0 },
    limit: { input: 32000, output: 32000 },
  },
  language: {},
} as any
const logModuleURL = new URL("../src/util/log.ts", import.meta.url).href
const mcpModuleURL = new URL("../src/mcp/index.ts", import.meta.url).href

const configModule = await import("../src/config/config")
const originalConfigGet = configModule.Config.get
const originalConfigDirectories = configModule.Config.directories
const originalConfigUpdate = configModule.Config.update

const providerModule = await import("../src/provider/provider")
const originalProviderGetModel = providerModule.Provider.getModel
const originalProviderGetSmallModel = providerModule.Provider.getSmallModel
const originalProviderDefaultModel = providerModule.Provider.defaultModel
const originalProviderGetProvider = providerModule.Provider.getProvider

const modelsModule = await import("../src/provider/models")
const originalModelsGet = modelsModule.ModelsDev.get
const originalModelsRefresh = modelsModule.ModelsDev.refresh

const toolRegistryModule = await import("../src/tool/registry")
const originalToolRegistryTools = toolRegistryModule.ToolRegistry.tools
const originalToolRegistryEnabled = toolRegistryModule.ToolRegistry.enabled

const bunModule = await import("../src/bun/index.ts")
const originalBunProcInstall = bunModule.BunProc.install
const originalBunProcRun = bunModule.BunProc.run
const originalBunProcWhich = bunModule.BunProc.which
type BunProcRunResult = Awaited<ReturnType<typeof bunModule.BunProc.run>>

const pluginModule = await import("../src/plugin/index.ts")
const originalPluginInit = pluginModule.Plugin.init
const originalPluginList = pluginModule.Plugin.list
const originalPluginTrigger = pluginModule.Plugin.trigger

const agentModule = await import("../src/agent/agent")
const originalAgentGet = agentModule.Agent.get
const originalAgentList = agentModule.Agent.list

const logMock = () => {
  const create = () => ({
    info() {},
    error() {},
    warn() {},
    debug() {},
    tag() {
      return this
    },
    clone() {
      return create()
    },
    time() {
      return {
        stop() {},
        [Symbol.dispose]() {},
      }
    },
  })
  return {
    Log: {
      Level: { DEBUG: "DEBUG", INFO: "INFO", WARN: "WARN", ERROR: "ERROR" },
      init: async () => {},
      file: () => "",
      create,
      Default: create(),
    },
  }
}

mock.module("../src/util/log", logMock)
mock.module(logModuleURL, logMock)
mock.module("../src/util/log.ts", logMock)
mock.module("../util/log", logMock)
mock.module("../util/log.ts", logMock)

await import("../src/util/log")

const mcpMock = () => ({
  MCP: {
    async tools() {
      return {}
    },
    async clients() {
      return {}
    },
  },
})

mock.module("../src/mcp", mcpMock)
mock.module(mcpModuleURL, mcpMock)

describe("Session summarise, queue flush, tool failure, retry exhaustion", () => {
  beforeEach(() => {
    resetCounters()
    resetScenario()
    configModule.Config.get = async () => ({})
    configModule.Config.directories = async () => []
    configModule.Config.update = async () => {}
    agentModule.Agent.get = async () => ({
      name: "build",
      tools: {} as Record<string, boolean>,
      temperature: undefined,
      topP: undefined,
      options: {},
      mode: "primary",
      builtIn: false,
      permission: {
        edit: "allow",
        bash: {
          "*": "allow",
        },
        webfetch: "ask",
      },
    })
    agentModule.Agent.list = async () => []
    providerModule.Provider.getModel = async () => dummyModel
    providerModule.Provider.getSmallModel = async () => dummyModel
    providerModule.Provider.defaultModel = async () => ({ providerID: "mock", modelID: dummyModel.info.id })
    providerModule.Provider.getProvider = async () => ({ id: "mock" } as any)
    modelsModule.ModelsDev.get = async () => ({
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
    })
    modelsModule.ModelsDev.refresh = async () => {}
    toolRegistryModule.ToolRegistry.tools = async () => []
    toolRegistryModule.ToolRegistry.enabled = async () => ({})
    bunModule.BunProc.install = async () => path.join(cachePath, "node_modules", "mock-package")
    bunModule.BunProc.run = async () =>
      ({
        exitCode: 0,
        exited: Promise.resolve(0),
        stdin: undefined,
        stdout: undefined,
        stderr: undefined,
        kill: () => true,
      } as unknown as BunProcRunResult)
    bunModule.BunProc.which = () => "bun"
    pluginModule.Plugin.init = async () => {}
    pluginModule.Plugin.list = async () => []
    const trigger: typeof pluginModule.Plugin.trigger = async (_name, _input, output) => output
    pluginModule.Plugin.trigger = trigger
  })

  afterEach(() => {
    resetScenario()
    configModule.Config.get = originalConfigGet
    configModule.Config.directories = originalConfigDirectories
    configModule.Config.update = originalConfigUpdate
    agentModule.Agent.get = originalAgentGet
    agentModule.Agent.list = originalAgentList
    providerModule.Provider.getModel = originalProviderGetModel
    providerModule.Provider.getSmallModel = originalProviderGetSmallModel
    providerModule.Provider.defaultModel = originalProviderDefaultModel
    providerModule.Provider.getProvider = originalProviderGetProvider
    modelsModule.ModelsDev.get = originalModelsGet
    modelsModule.ModelsDev.refresh = originalModelsRefresh
    toolRegistryModule.ToolRegistry.tools = originalToolRegistryTools
    toolRegistryModule.ToolRegistry.enabled = originalToolRegistryEnabled
    bunModule.BunProc.install = originalBunProcInstall
    bunModule.BunProc.run = originalBunProcRun
    bunModule.BunProc.which = originalBunProcWhich
    pluginModule.Plugin.init = originalPluginInit
    pluginModule.Plugin.list = originalPluginList
    pluginModule.Plugin.trigger = originalPluginTrigger
  })
  test("queue flush processes queued message after current one", async () => {
    await withApp(async ({ Session, SessionPrompt }) => {
      const { id: sessionID } = await Session.create(undefined)

      await Session.setStreamingCapable(sessionID, false)
      const r1 = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "p1", type: "text", text: "one" }],
      })

      const r2 = await SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "p2", type: "text", text: "two" }],
      })

      assertAssistant(r1)
      assertAssistant(r2)
      expect(r1.info.error).toBeUndefined()
      expect(r2.info.error).toBeUndefined()

      expect(generateTextCalls).toBeGreaterThanOrEqual(2)

      expect(r1.parts.length).toBeGreaterThan(0)
      expect(r2.parts.length).toBeGreaterThan(0)
    })
  }, 10000)

  test("concurrent queue processing resolves promises correctly", async () => {
    await withApp(async ({ Session, SessionPrompt }) => {
      const { id: sessionID } = await Session.create(undefined)

      await Session.setStreamingCapable(sessionID, false)
      const r1 = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "c1", type: "text", text: "concurrent one" }],
      })

      await new Promise(resolve => setTimeout(resolve, 10))

      const r2 = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "c2", type: "text", text: "concurrent two" }],
      })

      const [res1, res2] = await Promise.all([r1, r2])

      assertAssistant(res1)
      assertAssistant(res2)
      expect(res1.info.error).toBeUndefined()
      expect(res2.info.error).toBeUndefined()

      expect(res1.parts.length).toBeGreaterThan(0)
      expect(res2.parts.length).toBeGreaterThan(0)

      const textParts1 = res1.parts.filter(part => part.type === "text")
      const textParts2 = res2.parts.filter(part => part.type === "text")
      expect(textParts1.length).toBe(1)
      expect(textParts2.length).toBe(1)
      expect(textParts1[0].text).toBe("assistant text")
      expect(textParts2[0].text).toBe("assistant text")

      expect(res1.info.id).not.toBe(res2.info.id)
    })
  }, 8000)

  test("queue drains multiple messages (>2) concurrently", async () => {
    await withApp(async ({ Session, SessionPrompt }) => {
      const { id: sessionID } = await Session.create(undefined)

      await Session.setStreamingCapable(sessionID, false)

      const r1 = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "msg1", type: "text", text: "message 1" }],
      })

      await new Promise(resolve => setTimeout(resolve, 10))

      const r2 = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "msg2", type: "text", text: "message 2" }],
      })

      await new Promise(resolve => setTimeout(resolve, 10))

      const r3 = SessionPrompt.prompt({
        sessionID,
        model: { providerID: "mock", modelID: "mock-model" },
        parts: [{ id: "msg3", type: "text", text: "message 3" }],
      })

      const results = await Promise.all([r1, r2, r3])

      for (const result of results) {
        assertAssistant(result)
        expect(result.info.error).toBeUndefined()
        expect(result.parts.length).toBeGreaterThan(0)
        const textParts = result.parts.filter(part => part.type === "text")
        expect(textParts.length).toBe(1)
        expect(textParts[0].text).toBe("assistant text")
      }

      expect(generateTextCalls).toBeGreaterThanOrEqual(3)

      expect(results.length).toBe(3)
      const messageIds = results.map(r => r.info.id)
      const uniqueIds = new Set(messageIds)
      expect(uniqueIds.size).toBe(3)
    })
  }, 8000)

  test("tool failure marks tool part as error", async () => {
    let retries = 3
    let lastError: any

    while (retries > 0) {
      try {
        await withApp(async ({ Session, SessionPrompt, ToolRegistry }) => {
          const originalTools = ToolRegistry.tools
          const originalEnabled = ToolRegistry.enabled
          // @ts-ignore monkey patch ToolRegistry
          ToolRegistry.tools = async () => [
            {
              id: "explode",
              description: "boom",
              parameters: z.object({}),
              execute: async () => {
                throw new Error("Kaboom")
              },
            },
          ]
          // @ts-ignore enable tool
          ToolRegistry.enabled = () => ({ explode: true })

          setScenario("toolCall")

          try {
            const { id: sessionID } = await Session.create(undefined)
            await Session.setStreamingCapable(sessionID, false)

            await new Promise((resolve) => setTimeout(resolve, 300))

            const result = await SessionPrompt.prompt({
              sessionID,
              model: { providerID: "mock", modelID: "mock-model" },
              parts: [{ id: "pt", type: "text", text: "use explode" }],
              tools: { explode: {} } as any,
            })

            assertAssistant(result)
            const hasErrorTool = result.parts.some(
              (p: any) => p.type === "tool" && (p.state?.status === "error" || p.state?.metadata?.error === true),
            )
            const hasError = result.info.error !== undefined

            expect(hasErrorTool || hasError).toBe(true)
            const toolParts = result.parts.filter((p: any) => p.type === "tool")
            expect(toolParts.length).toBeGreaterThan(0)
          } finally {
            resetScenario()
            ToolRegistry.tools = originalTools
            ToolRegistry.enabled = originalEnabled
          }
        })

        return
      } catch (error: any) {
        lastError = error
        if (error.message?.includes("index.lock") && retries > 1) {
          retries--
          await new Promise((resolve) => setTimeout(resolve, 1000))
        } else {
          throw error
        }
      }
    }

    throw lastError
  }, 20000)

  test("retry exhaustion surfaces error", async () => {
    await withApp(async ({ Session, SessionPrompt }) => {
      setScenario("errorLoop")

      try {
        const { id: sessionID } = await Session.create(undefined)
        await Session.setStreamingCapable(sessionID, false)

        const result = await SessionPrompt.prompt({
          sessionID,
          model: { providerID: "mock", modelID: "mock-model" },
          parts: [{ id: "p", type: "text", text: "hi" }],
        })
        assertAssistant(result)
        expect(result.info.error).toBeDefined()
        expect(generateTextCalls).toBeGreaterThan(0)
      } finally {
        resetScenario()
      }
    })
  }, 15000)
})
