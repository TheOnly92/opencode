import path from "path"
import os from "os"
import fs from "fs/promises"
import z from "zod"
import { Identifier } from "../id/id"
import { MessageV2 } from "./message-v2"
import { Log } from "../util/log"
import { SessionRevert } from "./revert"
import { Session } from "."
import { Agent } from "../agent/agent"
import { Provider } from "../provider/provider"
import {
  generateText,
  streamText,
  type JSONValue,
  type ModelMessage,
  type GenerateTextResult,
  type JSONSchema7,
  type ProviderMetadata,
  type Tool as AITool,
  tool,
  wrapLanguageModel,
  type StreamTextResult,
  LoadAPIKeyError,
  stepCountIs,
  jsonSchema,
} from "ai"
import { SessionCompaction } from "./compaction"
import { SessionLock } from "./lock"
import { Instance } from "../project/instance"
import { Bus } from "../bus"
import { ProviderTransform } from "../provider/transform"
import { SystemPrompt } from "./system"
import { Plugin } from "../plugin"
import { SessionRetry } from "./retry"

import PROMPT_PLAN from "../session/prompt/plan.txt"
import BUILD_SWITCH from "../session/prompt/build-switch.txt"
import { ModelsDev } from "../provider/models"
import { defer } from "../util/defer"
import { mergeDeep, pipe } from "remeda"
import { ToolRegistry } from "../tool/registry"
import { Wildcard } from "../util/wildcard"
import { MCP } from "../mcp"
import { LSP } from "../lsp"
import { ReadTool } from "../tool/read"
import { Storage } from "../storage/storage"
import { ListTool } from "../tool/ls"
import { TaskTool } from "../tool/task"
import { FileTime } from "../file/time"
import { Permission } from "../permission"
import { Snapshot } from "../snapshot"
import { NamedError } from "../util/error"
import { ulid } from "ulid"
import { spawn } from "child_process"
import { Command } from "../command"
import { $, fileURLToPath } from "bun"
import { ConfigMarkdown } from "../config/markdown"
import { SessionSummary } from "./summary"
import { Config } from "@/config/config"

export namespace SessionPrompt {
  const log = Log.create({ service: "session.prompt" })
  export const OUTPUT_TOKEN_MAX = 32_000
  const MAX_RETRIES = 10
  const DOOM_LOOP_THRESHOLD = 3

  const isRecord = (value: unknown): value is Record<string, unknown> =>
    typeof value === "object" && value !== null

  const readMessageField = (value: unknown): string | undefined => {
    if (typeof value === "string") return value
    if (!isRecord(value)) return undefined
    const fromMessage = value["message"]
    if (typeof fromMessage === "string") return fromMessage
    const fromData = value["data"]
    if (isRecord(fromData) && typeof fromData["message"] === "string") return fromData["message"]
    return undefined
  }

  const toErrorString = (value: unknown) => {
    if (value instanceof Error) return value.message
    if (typeof value === "string") return value
    if (isRecord(value) && typeof value.message === "string") return value.message
    return String(value)
  }

  type MCPTextPart = {
    type: "text"
    text: string
  }

  type MCPImagePart = {
    type: "image"
    data: string
    mimeType: string
  }

  const isMCPTextPart = (value: unknown): value is MCPTextPart => {
    if (!isRecord(value)) return false
    if (value["type"] !== "text") return false
    return typeof value["text"] === "string"
  }

  const isMCPImagePart = (value: unknown): value is MCPImagePart => {
    if (!isRecord(value)) return false
    if (value["type"] !== "image") return false
    if (typeof value["data"] !== "string") return false
    return typeof value["mimeType"] === "string"
  }

  function isStreamingVerificationError(input: unknown) {
    if (!input) return false
    const msg = readMessageField(input)
    if (!msg) return false
    const m = msg.toLowerCase()
    return m.includes("must be verified to stream")
  }

  function isStreamingTimeoutError(input: unknown) {
    if (!input) return false
    const message = readMessageField(input)
    if (typeof message === "string") return message.toLowerCase().includes("timed out")
    return false
  }

  export class BusyError extends Error {
    constructor(readonly sessionID: string) {
      super(`Session ${sessionID} is busy`)
      this.name = "BusyError"
    }
  }

  export const Event = {
    Idle: Bus.event(
      "session.idle",
      z.object({
        sessionID: z.string(),
      }),
    ),
  }

  const state = Instance.state(
    () => {
      const pending = new Map<string, AbortController>()
      const queued = new Map<
        string,
        {
          messageID: string
          callback: (input: MessageV2.WithParts) => void
        }[]
      >()
      const tracked = new Set<Promise<void>>()
      const track = (promise: Promise<void>) => {
        tracked.add(promise)
        promise.finally(() => tracked.delete(promise))
      }
      return {
        pending,
        queued,
        track,
        tracked,
      }
    },
    async (current) => {
      for (const controller of current.pending.values()) {
        controller.abort()
      }
      current.pending.clear()
      current.queued.clear()
      await Promise.allSettled([...current.tracked])
      current.tracked.clear()
    },
  )

  type ToolValue = {
    output: string
    metadata?: unknown
    title?: string
  }

  type ToolExecutor = {
    execute: (input: unknown, context: unknown) => Promise<ToolValue>
    onInputAvailable?: (input: {
      input: unknown
      toolCallId: string
      messages?: ModelMessage[]
      abortSignal: AbortSignal
    }) => Promise<void> | void
  }

  type ToolContextFactory = (input: {
    update: (state: { metadata?: unknown; title?: string }) => Promise<MessageV2.ToolPart>
  }) => Record<string, unknown>

  type ToolResult =
    | { status: "ok"; value: ToolValue; part: MessageV2.ToolPart }
    | { status: "error"; message: string; metadata?: unknown; part: MessageV2.ToolPart }

  async function executeToolCall(options: {
    tool: ToolExecutor
    name: string
    callId: string
    input: unknown
    sessionID: string
    messageID: string
    abortSignal: AbortSignal
    messages?: ModelMessage[]
    context?: Record<string, unknown> | ToolContextFactory
    stateInput?: unknown
  }): Promise<ToolResult> {
    const startedAt = Date.now()
    const displayInput = options.stateInput ?? options.input
    const inputRecord =
      typeof displayInput === "object" && displayInput !== null
        ? (displayInput as Record<string, unknown>)
        : { value: displayInput }
    const created = await Session.updatePart({
      id: Identifier.ascending("part"),
      messageID: options.messageID,
      sessionID: options.sessionID,
      type: "tool",
      tool: options.name,
      callID: options.callId,
      state: {
        status: "running",
        input: inputRecord,
        time: {
          start: startedAt,
        },
      },
    })
    const holder = {
      part: created as MessageV2.ToolPart,
    }
    const updateRunning = async (state: { metadata?: unknown; title?: string }) => {
      if (holder.part.state.status !== "running") return holder.part
      const next = (await Session.updatePart({
        ...holder.part,
        state: {
          ...holder.part.state,
          metadata: state.metadata ?? holder.part.state.metadata,
          title: state.title ?? holder.part.state.title,
        },
      })) as MessageV2.ToolPart
      holder.part = next
      return next
    }
    const ctxBase = {
      toolCallId: options.callId,
      messages: options.messages,
      abortSignal: options.abortSignal,
      metadata: async (meta: { metadata?: unknown; title?: string }) => {
        await updateRunning({
          title: meta.title,
          metadata: meta.metadata,
        })
      },
    }
    const ctxInput =
      typeof options.context === "function"
        ? options.context({ update: updateRunning })
        : options.context ?? {}
    const ctx = {
      ...ctxBase,
      ...ctxInput,
    }
    if (typeof options.tool.onInputAvailable === "function") {
      await options.tool.onInputAvailable({
        input: options.input,
        toolCallId: options.callId,
        messages: options.messages,
        abortSignal: options.abortSignal,
      })
    }
    try {
      const value = await options.tool.execute(options.input, ctx)
      const metadataRecord =
        typeof value.metadata === "object" && value.metadata !== null
          ? (value.metadata as Record<string, unknown>)
          : {}
      const title = value.title ?? ""
      const next = (await Session.updatePart({
        ...holder.part,
        state: {
          status: "completed",
          input: inputRecord,
          output: value.output,
          metadata: metadataRecord,
          title,
          time: {
            start: startedAt,
            end: Date.now(),
          },
        },
      })) as MessageV2.ToolPart
      holder.part = next
      return {
        status: "ok",
        value,
        part: holder.part,
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      const metadata = err instanceof Permission.RejectedError ? err.metadata : undefined
      const errorMetadata =
        typeof metadata === "object" && metadata !== null
          ? (metadata as Record<string, unknown>)
          : undefined
      const next = (await Session.updatePart({
        ...holder.part,
        state: {
          status: "error",
          input: inputRecord,
          error: message,
          metadata: errorMetadata,
          time: {
            start: startedAt,
            end: Date.now(),
          },
        },
      })) as MessageV2.ToolPart
      holder.part = next
      return {
        status: "error",
        message,
        metadata,
        part: holder.part,
      }
    }
  }

  export const PromptInput = z.object({
    sessionID: Identifier.schema("session"),
    messageID: Identifier.schema("message").optional(),
    model: z
      .object({
        providerID: z.string(),
        modelID: z.string(),
      })
      .optional(),
    agent: z.string().optional(),
    noReply: z.boolean().optional(),
    system: z.string().optional(),
    tools: z.record(z.string(), z.boolean()).optional(),
    acpConnection: z
      .object({
        connection: z.any(),
        sessionId: z.string(),
      })
      .optional(),
    parts: z.array(
      z.discriminatedUnion("type", [
        MessageV2.TextPart.omit({
          messageID: true,
          sessionID: true,
        })
          .partial({
            id: true,
          })
          .meta({
            ref: "TextPartInput",
          }),
        MessageV2.FilePart.omit({
          messageID: true,
          sessionID: true,
        })
          .partial({
            id: true,
          })
          .meta({
            ref: "FilePartInput",
          }),
        MessageV2.AgentPart.omit({
          messageID: true,
          sessionID: true,
        })
          .partial({
            id: true,
          })
          .meta({
            ref: "AgentPartInput",
          }),
      ]),
    ),
  })
  export type PromptInput = z.infer<typeof PromptInput>

  export async function resolvePromptParts(template: string): Promise<PromptInput["parts"]> {
    const parts: PromptInput["parts"] = [
      {
        type: "text",
        text: template,
      },
    ]
    const files = ConfigMarkdown.files(template)
    await Promise.all(
      files.map(async (match) => {
        const name = match[1]
        const filepath = name.startsWith("~/")
          ? path.join(os.homedir(), name.slice(2))
          : path.resolve(Instance.worktree, name)

        const stats = await fs.stat(filepath).catch(() => undefined)
        if (!stats) {
          const agent = await Agent.get(name)
          if (agent) {
            parts.push({
              type: "agent",
              name: agent.name,
            })
          }
          return
        }

        if (stats.isDirectory()) {
          parts.push({
            type: "file",
            url: `file://${filepath}`,
            filename: name,
            mime: "application/x-directory",
          })
          return
        }

        parts.push({
          type: "file",
          url: `file://${filepath}`,
          filename: name,
          mime: "text/plain",
        })
      }),
    )
    return parts
  }
  export async function prompt(input: PromptInput): Promise<MessageV2.WithParts> {
    const l = log.clone().tag("session", input.sessionID)
    l.info("prompt")

    const session = await Session.get(input.sessionID)
    await SessionRevert.cleanup(session)

    const userMsg = await createUserMessage(input)
    await Session.touch(input.sessionID)

    // Early return for context-only messages (no AI inference)
    if (input.noReply) {
      return userMsg
    }

    if (isBusy(input.sessionID)) {
      return new Promise((resolve) => {
        const queue = state().queued.get(input.sessionID) ?? []
        queue.push({
          messageID: userMsg.info.id,
          callback: resolve,
        })
        state().queued.set(input.sessionID, queue)
      })
    }
    using abort = lock(input.sessionID)

    const agent = await Agent.get(input.agent ?? "build")
    const model = await resolveModel({
      agent,
      model: input.model,
    }).then((x) => Provider.getModel(x.providerID, x.modelID))
    const outputLimit = Math.min(model.info.limit.output, OUTPUT_TOKEN_MAX) || OUTPUT_TOKEN_MAX

    const system = await resolveSystemPrompt({
      providerID: model.providerID,
      modelID: model.info.id,
      agent,
      system: input.system,
    })

    const processor = await createProcessor({
      sessionID: input.sessionID,
      model: model.info,
      providerID: model.providerID,
      agent: agent.name,
      system,
      abort: abort.signal,
      acpConnection: input.acpConnection,
    })

    const tools = await resolveTools({
      agent,
      sessionID: input.sessionID,
      modelID: model.modelID,
      providerID: model.providerID,
      tools: input.tools,
      processor,
    })

    const params = await Plugin.trigger(
      "chat.params",
      {
        sessionID: input.sessionID,
        agent: agent.name,
        model: model.info,
        provider: await Provider.getProvider(model.providerID),
        message: userMsg,
      },
      {
        temperature: model.info.temperature
          ? (agent.temperature ?? ProviderTransform.temperature(model.providerID, model.modelID))
          : undefined,
        topP: agent.topP ?? ProviderTransform.topP(model.providerID, model.modelID),
        options: {
          ...ProviderTransform.options(model.providerID, model.modelID, model.npm ?? "", input.sessionID),
          ...model.info.options,
          ...agent.options,
        },
      },
    )
    const providerDefaults = {
      ...ProviderTransform.options(model.providerID, model.modelID, model.npm ?? "", input.sessionID),
      ...model.info.options,
      ...agent.options,
    }
    const providerBaseOptions = (() => {
      const source =
        params.options && typeof params.options === "object"
          ? { ...providerDefaults, ...(params.options as Record<string, unknown>) }
          : { ...providerDefaults }
      return Object.entries(source).reduce<Record<string, JSONValue>>((acc, [key, value]) => {
        if (value === undefined) return acc
        acc[key] = value as JSONValue
        return acc
      }, {})
    })()
    type ProviderOptionMap = ReturnType<typeof ProviderTransform.providerOptions>
    const buildProviderOptions = (stream: boolean): ProviderOptionMap => {
      const options: Record<string, JSONValue> = { ...providerBaseOptions }
      if (!stream) options.stream = false
      if (options["store"] === false) options["store"] = true
      return ProviderTransform.providerOptions(model.npm, model.providerID, options)
    }

    const isThenable = (value: unknown): value is PromiseLike<unknown> =>
      isRecord(value) && typeof value.then === "function"

    const nestedString = (value: unknown, keys: string[]): string | undefined => {
      let current: unknown = value
      for (const key of keys) {
        if (!isRecord(current)) return undefined
        current = current[key]
      }
      return typeof current === "string" ? current : undefined
    }

    const isLengthReason = (reason?: string) => {
      if (!reason) return false
      const normalized = reason.toLowerCase()
      return normalized === "length" || normalized === "max_tokens"
    }

    const providerLengthStop = (finishReason: unknown, metadata: ProviderMetadata | undefined) => {
      if (typeof finishReason === "string" && finishReason.toLowerCase().includes("length")) return true
      if (!metadata || typeof metadata !== "object") return false
      if (isLengthReason(nestedString(metadata, ["openai", "finish_reason"]))) return true
      if (nestedString(metadata, ["anthropic", "stop_reason"]) === "max_tokens") return true
      if (isLengthReason(nestedString(metadata, ["bedrock", "stopReason"]))) return true
      return false
    }

    type ToolCallInfo = {
      id: string
      name: string
      input: unknown
    }

    const toPassiveTools = (source: typeof tools): Record<string, AITool> => {
      const entries: Array<[string, AITool]> = []
      for (const [name, toolEntry] of Object.entries(source)) {
        const clone = { ...toolEntry } as Record<string, unknown>
        delete clone.execute
        entries.push([name, clone as AITool])
      }
      return Object.fromEntries(entries)
    }

    const toToolCall = (value: unknown): ToolCallInfo | undefined => {
      if (!isRecord(value)) return
      const rawName = value.toolName
      if (typeof rawName !== "string" || !rawName.length) return
      const rawId = value.toolCallId
      const id = typeof rawId === "string" && rawId.length ? rawId : Identifier.ascending("part")
      const inputValue = Reflect.get(value, "input")
      return {
        id,
        name: rawName,
        input: inputValue,
      }
    }

    const normalizeToolCallArray = async (value: unknown): Promise<ToolCallInfo[]> => {
      if (!value) return []
      if (Array.isArray(value)) {
        return value
          .map((entry) => toToolCall(entry))
          .filter((entry): entry is ToolCallInfo => Boolean(entry))
      }
      if (isThenable(value)) return normalizeToolCallArray(await value)
      return []
    }

    const collectToolCalls = async (output: unknown): Promise<ToolCallInfo[]> => {
      if (!isRecord(output)) return []
      const direct = await normalizeToolCallArray(output.toolCalls)
      if (direct.length) return direct
      const response = output.response
      if (!isRecord(response)) return []
      const messages = response.messages
      if (!Array.isArray(messages)) return []
      const result: ToolCallInfo[] = []
      for (const message of messages) {
        if (!isRecord(message)) continue
        if (message.role !== "assistant") continue
        const content = message.content
        if (!Array.isArray(content)) continue
        for (const part of content) {
          if (!isRecord(part)) continue
          if (part.type !== "tool-call") continue
          const mapped = toToolCall(part)
          if (mapped) result.push(mapped)
        }
      }
      return result
    }

    const toolCallContent = (calls: ToolCallInfo[]) =>
      calls.map((call) => ({
        type: "tool-call" as const,
        toolCallId: call.id,
        toolName: call.name,
        input: call.input,
      }))

    const runToolCall = async (input: {
      call: ToolCallInfo
      toolEntry?: ToolExecutor
      failureCounts: Map<string, number>
      failureLimit: number
      sessionID: string
      messageID: string
      abortSignal: AbortSignal
      messages: ModelMessage[]
      conv: ModelMessage[]
    }): Promise<{ finalize?: string }> => {
      const { call } = input
      if (!input.toolEntry) {
        const record =
          typeof call.input === "object" && call.input !== null
            ? (call.input as Record<string, unknown>)
            : { value: call.input }
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: input.messageID,
          sessionID: input.sessionID,
          type: "tool",
          tool: call.name,
          callID: call.id,
          state: {
            status: "error",
            input: record,
            error: "Tool unavailable",
            time: {
              start: Date.now(),
              end: Date.now(),
            },
          },
        })
        input.conv.push({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: call.id,
              toolName: call.name,
              output: { type: "text", value: `Error: Tool ${call.name} is unavailable.` },
            },
          ],
        })
        const count = (input.failureCounts.get(call.name) ?? 0) + 1
        input.failureCounts.set(call.name, count)
        if (count >= input.failureLimit) {
          return {
            finalize: `Tool ${call.name} is unavailable. Please update the request or register the tool.`,
          }
        }
        return {}
      }
      const outcome = await executeToolCall({
        tool: input.toolEntry,
        name: call.name,
        callId: call.id,
        input: call.input,
        sessionID: input.sessionID,
        messageID: input.messageID,
        abortSignal: input.abortSignal,
        messages: input.messages,
      })
      if (outcome.status === "ok") {
        if (input.failureCounts.has(call.name)) input.failureCounts.delete(call.name)
        input.conv.push({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: call.id,
              toolName: call.name,
              output: { type: "text", value: outcome.value.output },
            },
          ],
        })
        return {}
      }
      input.conv.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: call.id,
            toolName: call.name,
            output: { type: "text", value: `Error: ${outcome.message}` },
          },
        ],
      })
      const nextCount = (input.failureCounts.get(call.name) ?? 0) + 1
      input.failureCounts.set(call.name, nextCount)
      if (nextCount >= input.failureLimit) {
        return {
          finalize: `Tool ${call.name} failed ${nextCount} times. Please check the request and try again.`,
        }
      }
      return {}
    }

    const settle = async (result: MessageV2.WithParts & { blocked?: boolean }) => {
      const queued = state().queued.get(input.sessionID) ?? []
      for (const item of queued) item.callback(result)
      state().queued.delete(input.sessionID)
      SessionCompaction.prune(input)
      return result
    }

    const loadConversation = async (options?: { signal?: AbortSignal }) => {
      const history = await getMessages({
        sessionID: input.sessionID,
        model: model.info,
        providerID: model.providerID,
        signal: options?.signal,
      })
      const augmented = insertReminders({ messages: history, agent })
      const systemMessages = system.map(
        (entry): ModelMessage => ({
          role: "system",
          content: entry,
        }),
      )
      const conversation = [...systemMessages, ...toModelConversation(augmented)]
      const parentID = augmented.findLast((msg) => msg.info.role === "user")?.info.id ?? userMsg.info.id
      return { history: augmented, conversation, parentID }
    }

    const upsertAssistantText = async (inputText: {
      messageID: string
      sessionID: string
      text: string
    }) => {
      const content = inputText.text.trim()
      if (!content.length) return
      const parts = await MessageV2.parts(inputText.messageID)
      const match = parts.find((part) => part.type === "text") as MessageV2.TextPart | undefined
      if (!match) {
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: inputText.messageID,
          sessionID: inputText.sessionID,
          type: "text",
          text: content,
        })
        return
      }
      if (match.text === content) return
      await Session.updatePart({ ...match, text: content })
    }

    const finalizeResponse = async (text?: string) => {
      const msg = processor.message
      const content = text?.trim()
      if (content && content.length) {
        await upsertAssistantText({
          messageID: msg.id,
          sessionID: msg.sessionID,
          text: content,
        })
      }
      await processor.end()
      await Session.setStreamingCapable(input.sessionID, false)
      const partsFinal = await MessageV2.parts(msg.id)
      return settle({ info: msg, parts: partsFinal })
    }

    const clearAssistantMessage = async (msg: MessageV2.Assistant) => {
      const parts = await MessageV2.parts(msg.id)
      for (const part of parts) {
        await Storage.remove(["part", msg.id, part.id])
        await Bus.publish(MessageV2.Event.PartRemoved, {
          sessionID: msg.sessionID,
          messageID: msg.id,
          partID: part.id,
        })
      }
    }

    async function nonStreamingFallback(options?: { replaceMessageID?: string }) {
      if (options?.replaceMessageID) {
        await Session.removeMessage({
          sessionID: input.sessionID,
          messageID: options.replaceMessageID,
        }).catch((err) => {
          log.warn("failed to remove streaming placeholder message", {
            sessionID: input.sessionID,
            messageID: options.replaceMessageID,
            error: err,
          })
        })
      }
      const seed = await loadConversation({ signal: abort.signal })
      const conv: ModelMessage[] = [...seed.conversation]
      if (!processor.hasMessage()) {
        await processor.next(seed.parentID)
      }
      const passiveTools = toPassiveTools(tools)
      type PassiveToolSet = typeof passiveTools
      type GenerateResult = GenerateTextResult<PassiveToolSet, unknown>
      let wroteText = false
      let awaiting = false
      let hadTools = false
      const failureCounts = new Map<string, number>()
      const failureLimit = 3
      const idleLimit = 8
      let idleRounds = 0
      while (true) {
        wroteText = false
        await processor.startStep()
        hadTools = false
        const request = {
          maxOutputTokens: ProviderTransform.maxOutputTokens(
            model.providerID,
            params.options,
            model.info.limit.output,
            outputLimit,
          ),
          providerOptions: buildProviderOptions(false),
          messages: ProviderTransform.message(conv, model.providerID, model.modelID),
          temperature: params.temperature,
          topP: params.topP,
          tools: model.info.tool_call === false ? undefined : passiveTools,
            model: wrapLanguageModel({
              model: model.language,
              middleware: [
                {
                  // @ts-expect-error: AI SDK does not currently export the middleware payload type.
                  async transformParams(input: unknown) {
                    const payload = input as {
                      type: string
                      params: { prompt?: ModelMessage[] | string }
                    }
                  if (payload.type !== "generate") return payload.params
                  const promptValue = payload.params.prompt
                  if (Array.isArray(promptValue)) {
                    payload.params.prompt = ProviderTransform.message(
                      promptValue as ModelMessage[],
                      model.providerID,
                      model.modelID,
                    )
                  }
                  return payload.params
                },
              },
            ],
          }),
        }
        let out: GenerateResult | undefined
        try {
          out = await generateText<PassiveToolSet>(request)
        } catch (err) {
          const error = err instanceof Error ? err : new Error(String(err))
          const named = LoadAPIKeyError.isInstance(error)
            ? new MessageV2.AuthError({ providerID: model.providerID, message: error.message }, { cause: error }).toObject()
            : new NamedError.Unknown({ message: error.message }, { cause: error }).toObject()
          processor.message.error = named
          return finalizeResponse()
        }
        const usage = out?.usage
          ? Session.getUsage({ model: model.info, usage: out.usage, metadata: out.providerMetadata })
          : undefined
        if (usage) {
          await processor.finishStep(usage)
        }
        const toolCalls = await collectToolCalls(out)
        if (toolCalls.length) {
          hadTools = true
          conv.push({ role: "assistant", content: toolCallContent(toolCalls) })
          const promptMessages = conv.slice()
          awaiting = true

          for (const call of toolCalls) {
            const result = await runToolCall({
              call,
              toolEntry: tools[call.name] as ToolExecutor | undefined,
              failureCounts,
              failureLimit,
              sessionID: processor.message.sessionID,
              messageID: processor.message.id,
              abortSignal: abort.signal,
              messages: promptMessages,
              conv,
            })
            if (result.finalize) return finalizeResponse(result.finalize)
          }
          awaiting = false
          await processor.flushSnapshot()
          idleRounds = 0
          continue
        }

        const txt = typeof out?.text === "string" ? out.text : ""
        if (txt) {
          await upsertAssistantText({
            messageID: processor.message.id,
            sessionID: processor.message.sessionID,
            text: txt,
          })
          conv.push({ role: "assistant", content: [{ type: "text", text: txt }] })
          awaiting = false
          wroteText = true
        }

        if (wroteText) idleRounds = 0
        if (!wroteText) {
          idleRounds += 1
          if (idleRounds >= idleLimit) {
            const text = "Unable to produce a response after multiple attempts. Please try again or simplify the request."
            return finalizeResponse(text)
          }
        }

        const lengthStop = providerLengthStop(out?.finishReason, out?.providerMetadata)
        if (lengthStop) continue
        if (awaiting) continue
        if (hadTools && !wroteText) continue
        const partsNow = await MessageV2.parts(processor.message.id)
        const hasText = partsNow.some((pp) => pp.type === "text")
        if (!hasText) continue
        return finalizeResponse()
      }
    }

    // If streaming is known to be unsupported for this session, go straight to fallback
    const knownCapable = await Session.getStreamingCapable(input.sessionID)
    if (knownCapable === false) {
      return nonStreamingFallback()
    }

    let step = 0
    while (true) {
      const seed = await loadConversation({ signal: abort.signal })
      const msgs = seed.history
      const conversation = seed.conversation
      const parentID = seed.parentID
      step += 1
      await processor.next(parentID)
      if (step === 1) {
        state().track(
          ensureTitle({
            session,
            history: msgs,
            message: userMsg,
            providerID: model.providerID,
            modelID: model.info.id,
          }),
        )
        SessionSummary.summarize({
          sessionID: input.sessionID,
          messageID: userMsg.info.id,
        })
      }
      await using _ = defer(async () => {
        await processor.end()
      })
      const doStream = () =>
        streamText({
          onError(error) {
            log.error("stream error", {
              error,
            })
          },
          async experimental_repairToolCall(input) {
            const lower = input.toolCall.toolName.toLowerCase()
            if (lower !== input.toolCall.toolName && tools[lower]) {
              log.info("repairing tool call", {
                tool: input.toolCall.toolName,
                repaired: lower,
              })
              return {
                ...input.toolCall,
                toolName: lower,
              }
            }
            return {
              ...input.toolCall,
              input: JSON.stringify({
                tool: input.toolCall.toolName,
                error: input.error.message,
              }),
              toolName: "invalid",
            }
          },
          headers: {
            ...(model.providerID === "opencode"
              ? {
                  "x-opencode-session": input.sessionID,
                  "x-opencode-request": userMsg.info.id,
                }
              : undefined),
            ...model.info.headers,
          },
          activeTools: Object.keys(tools).filter((x) => x !== "invalid"),
          maxOutputTokens: ProviderTransform.maxOutputTokens(
            model.npm ?? "",
            params.options,
            model.info.limit.output,
            OUTPUT_TOKEN_MAX,
          ),
          abortSignal: abort.signal,
          // set to 0, we handle retry loop manually
          maxRetries: 0,
          providerOptions: buildProviderOptions(true),
          stopWhen: stepCountIs(1),
          temperature: params.temperature,
          topP: params.topP,
          messages: conversation,
          tools: model.info.tool_call === false ? undefined : tools,
          model: wrapLanguageModel({
            model: model.language,
            middleware: [
              {
                async transformParams(args) {
                  if (args.type === "stream") {
                    // @ts-expect-error
                    args.params.prompt = ProviderTransform.message(args.params.prompt, model.providerID, model.modelID)
                  }
                  return args.params
                },
              },
            ],
          }),
        })

      let stream = doStream()
      const cfg = await Config.get()
      const maxRetries = cfg.experimental?.chatMaxRetries ?? MAX_RETRIES
      let result = await processor.process(stream, {
        count: 0,
        max: maxRetries,
      })
      if (result.shouldRetry) {
        const start = Date.now()
        for (let retry = 1; retry < maxRetries; retry++) {
          const lastRetryPart = result.parts.findLast((p): p is MessageV2.RetryPart => p.type === "retry")

          if (lastRetryPart) {
            const delayMs = SessionRetry.getBoundedDelay({
              error: lastRetryPart.error,
              attempt: retry,
              startTime: start,
            })
            if (!delayMs) {
              break
            }

            log.info("retrying with backoff", {
              attempt: retry,
              delayMs,
              elapsed: Date.now() - start,
            })

            const stop = await SessionRetry.sleep(delayMs, abort.signal)
              .then(() => false)
              .catch((error) => {
                if (error instanceof DOMException && error.name === "AbortError") {
                  const err = new MessageV2.AbortedError(
                    { message: error.message },
                    {
                      cause: error,
                    },
                  ).toObject()
                  result.info.error = err
                  Bus.publish(Session.Event.Error, {
                    sessionID: result.info.sessionID,
                    error: result.info.error,
                  })
                  return true
                }
                throw error
              })

            if (stop) break
          }

          stream = doStream()
          result = await processor.process(stream, {
            count: retry,
            max: maxRetries,
          })
          if (!result.shouldRetry) {
            break
          }
        }
      }
      const infoError = result.info.role === "assistant" ? result.info.error : undefined
      const verify = isStreamingVerificationError(infoError)
      if (verify) {
        await Session.setStreamingCapable(input.sessionID, false)
        const msg = processor.message
        await clearAssistantMessage(msg)
        msg.error = undefined
        msg.time.completed = undefined
        msg.cost = 0
        msg.tokens = {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        }
        await Session.updateMessage(msg)
        const replaceMessageID = msg.id
        processor.reset()
        return nonStreamingFallback({ replaceMessageID })
      }
      const timeout = isStreamingTimeoutError(infoError)
      if (timeout) {
        log.info("streaming timed out, preserving streaming capability", {
          sessionID: input.sessionID,
        })
      }
      await processor.end()

      const queued = state().queued.get(input.sessionID) ?? []

      if (!result.blocked && !result.info.error) {
        if ((await stream.finishReason) === "tool-calls") {
          continue
        }

        const unprocessed = queued.filter((x) => x.messageID > result.info.id)
        if (unprocessed.length) {
          continue
        }
      }
      return settle(result)
    }
  }

  async function getMessages(input: {
    sessionID: string
    model: ModelsDev.Model
    providerID: string
    signal?: AbortSignal
  }) {
    let msgs = await MessageV2.filterCompacted(MessageV2.stream(input.sessionID))
    const lastAssistant = msgs.findLast((msg) => msg.info.role === "assistant")
    if (
      lastAssistant?.info.role === "assistant" &&
      SessionCompaction.isOverflow({
        tokens: lastAssistant.info.tokens,
        model: input.model,
      })
    ) {
      const summaryMsg = await SessionCompaction.run({
        sessionID: input.sessionID,
        providerID: input.providerID,
        modelID: input.model.id,
        signal: input.signal,
      })
      const resumeMsgID = Identifier.ascending("message")
      const resumeMsg = {
        info: await Session.updateMessage({
          id: resumeMsgID,
          role: "user",
          sessionID: input.sessionID,
          time: {
            created: Date.now(),
          },
        }),
        parts: [
          await Session.updatePart({
            type: "text",
            sessionID: input.sessionID,
            messageID: resumeMsgID,
            id: Identifier.ascending("part"),
            text: "Use the above summary generated from your last session to resume from where you left off.",
            time: {
              start: Date.now(),
              end: Date.now(),
            },
            synthetic: true,
          }),
        ],
      }
      msgs = [summaryMsg, resumeMsg]
    }
    return msgs
  }

  async function resolveModel(input: { model: PromptInput["model"]; agent: Agent.Info }) {
    if (input.model) {
      return input.model
    }
    if (input.agent.model) {
      return input.agent.model
    }
    return Provider.defaultModel()
  }

  async function resolveSystemPrompt(input: {
    system?: string
    agent: Agent.Info
    providerID: string
    modelID: string
  }) {
    let system = SystemPrompt.header(input.providerID)
    system.push(
      ...(() => {
        if (input.system) return [input.system]
        if (input.agent.prompt) return [input.agent.prompt]
        return SystemPrompt.provider(input.modelID)
      })(),
    )
    system.push(...(await SystemPrompt.environment()))
    system.push(...(await SystemPrompt.custom()))
    // max 2 system prompt messages for caching purposes
    const [first, ...rest] = system
    system = [first, rest.join("\n")]
    return system
  }

  async function resolveTools(input: {
    agent: Agent.Info
    sessionID: string
    modelID: string
    providerID: string
    tools?: Record<string, boolean>
    processor: Processor
  }) {
    const tools: Record<string, AITool> = {}
    const enabledTools = pipe(
      input.agent.tools,
      mergeDeep(await ToolRegistry.enabled(input.providerID, input.modelID, input.agent)),
      mergeDeep(input.tools ?? {}),
    )
      for (const item of await ToolRegistry.tools(input.providerID, input.modelID)) {
        if (Wildcard.all(item.id, enabledTools) === false) continue
        const schema = ProviderTransform.schema(input.providerID, input.modelID, z.toJSONSchema(item.parameters))
        const schemaJSON = schema as JSONSchema7
        tools[item.id] = tool({
          description: item.description,
          inputSchema: jsonSchema(schemaJSON),
          async execute(args, options) {
          await Plugin.trigger(
            "tool.execute.before",
            {
              tool: item.id,
              sessionID: input.sessionID,
              callID: options.toolCallId,
            },
            {
              args,
            },
          )
          const result = await item.execute(args, {
            sessionID: input.sessionID,
            abort: options.abortSignal!,
            messageID: input.processor.message.id,
            callID: options.toolCallId,
            extra: {
              modelID: input.modelID,
              providerID: input.providerID,
            },
            agent: input.agent.name,
            metadata: async (val) => {
              const match = input.processor.partFromToolCall(options.toolCallId)
              if (match && match.state.status === "running") {
                await Session.updatePart({
                  ...match,
                  state: {
                    title: val.title,
                    metadata: val.metadata,
                    status: "running",
                    input: args,
                    time: {
                      start: Date.now(),
                    },
                  },
                })
              }
            },
          })
          await Plugin.trigger(
            "tool.execute.after",
            {
              tool: item.id,
              sessionID: input.sessionID,
              callID: options.toolCallId,
            },
            result,
          )
          return result
        },
        toModelOutput(result) {
          return {
            type: "text",
            value: result.output,
          }
        },
      })
    }

    for (const [key, item] of Object.entries(await MCP.tools())) {
      if (Wildcard.all(key, enabledTools) === false) continue
      const execute = item.execute
      if (!execute) continue
      item.execute = async (args, opts) => {
        await Plugin.trigger(
          "tool.execute.before",
          {
            tool: key,
            sessionID: input.sessionID,
            callID: opts.toolCallId,
          },
          {
            args,
          },
        )
        const result = await execute(args, opts)

        await Plugin.trigger(
          "tool.execute.after",
          {
            tool: key,
            sessionID: input.sessionID,
            callID: opts.toolCallId,
          },
          result,
        )

        const textParts: string[] = []
        const attachments: MessageV2.FilePart[] = []
        const content = Array.isArray(result.content) ? result.content : []

        for (const entry of content) {
          if (isMCPTextPart(entry)) {
            textParts.push(entry.text)
            continue
          }
          if (isMCPImagePart(entry)) {
            attachments.push({
              id: Identifier.ascending("part"),
              sessionID: input.sessionID,
              messageID: input.processor.message.id,
              type: "file",
              mime: entry.mimeType,
              url: `data:${entry.mimeType};base64,${entry.data}`,
            })
          }
        }

        return {
          title: "",
          metadata: result.metadata ?? {},
          output: textParts.join("\n\n"),
          attachments,
          content,
        }
      }
      item.toModelOutput = (result) => {
        return {
          type: "text",
          value: result.output,
        }
      }
      tools[key] = item
    }
    return tools
  }

  async function createUserMessage(input: PromptInput) {
    const info: MessageV2.Info = {
      id: input.messageID ?? Identifier.ascending("message"),
      role: "user",
      sessionID: input.sessionID,
      time: {
        created: Date.now(),
      },
    }

    const parts = await Promise.all(
      input.parts.map(async (part): Promise<MessageV2.Part[]> => {
        if (part.type === "file") {
          const url = new URL(part.url)
          switch (url.protocol) {
            case "data:":
              if (part.mime === "text/plain") {
                return [
                  {
                    id: Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify({ filePath: part.filename })}`,
                  },
                  {
                    id: Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: Buffer.from(part.url, "base64url").toString(),
                  },
                  {
                    ...part,
                    id: part.id ?? Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                  },
                ]
              }
              break
            case "file:":
              log.info("file", { mime: part.mime })
              // have to normalize, symbol search returns absolute paths
              // Decode the pathname since URL constructor doesn't automatically decode it
              const filepath = fileURLToPath(part.url)
              const stat = await Bun.file(filepath).stat()

              if (stat.isDirectory()) {
                part.mime = "application/x-directory"
              }

              if (part.mime === "text/plain") {
                let offset: number | undefined = undefined
                let limit: number | undefined = undefined
                const range = {
                  start: url.searchParams.get("start"),
                  end: url.searchParams.get("end"),
                }
                if (range.start != null) {
                  const filePathURI = part.url.split("?")[0]
                  let start = parseInt(range.start)
                  let end = range.end ? parseInt(range.end) : undefined
                  // some LSP servers (eg, gopls) don't give full range in
                  // workspace/symbol searches, so we'll try to find the
                  // symbol in the document to get the full range
                  if (start === end) {
                    const symbols = await LSP.documentSymbol(filePathURI)
                    for (const symbol of symbols) {
                      let range: LSP.Range | undefined
                      if ("range" in symbol) {
                        range = symbol.range
                      } else if ("location" in symbol) {
                        range = symbol.location.range
                      }
                      if (range?.start?.line && range?.start?.line === start) {
                        start = range.start.line
                        end = range?.end?.line ?? start
                        break
                      }
                    }
                  }
                  offset = Math.max(start - 1, 0)
                  if (end) {
                    limit = end - offset
                  }
                }
                const args = { filePath: filepath, offset, limit }

                const pieces: MessageV2.Part[] = [
                  {
                    id: Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                ]

                await ReadTool.init()
                  .then(async (t) => {
                    const result = await t.execute(args, {
                      sessionID: input.sessionID,
                      abort: new AbortController().signal,
                      agent: input.agent!,
                      messageID: info.id,
                      extra: { bypassCwdCheck: true },
                      metadata: async () => {},
                    })
                    pieces.push(
                      {
                        id: Identifier.ascending("part"),
                        messageID: info.id,
                        sessionID: input.sessionID,
                        type: "text",
                        synthetic: true,
                        text: result.output,
                      },
                      {
                        ...part,
                        id: part.id ?? Identifier.ascending("part"),
                        messageID: info.id,
                        sessionID: input.sessionID,
                      },
                    )
                  })
                  .catch((error) => {
                    log.error("failed to read file", { error })
                    const message = error instanceof Error ? error.message : error.toString()
                    Bus.publish(Session.Event.Error, {
                      sessionID: input.sessionID,
                      error: new NamedError.Unknown({
                        message,
                      }).toObject(),
                    })
                    pieces.push({
                      id: Identifier.ascending("part"),
                      messageID: info.id,
                      sessionID: input.sessionID,
                      type: "text",
                      synthetic: true,
                      text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                    })
                  })

                return pieces
              }

              if (part.mime === "application/x-directory") {
                const args = { path: filepath }
                const result = await ListTool.init().then((t) =>
                  t.execute(args, {
                    sessionID: input.sessionID,
                    abort: new AbortController().signal,
                    agent: input.agent!,
                    messageID: info.id,
                    extra: { bypassCwdCheck: true },
                    metadata: async () => {},
                  }),
                )
                return [
                  {
                    id: Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the list tool with the following input: ${JSON.stringify(args)}`,
                  },
                  {
                    id: Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: result.output,
                  },
                  {
                    ...part,
                    id: part.id ?? Identifier.ascending("part"),
                    messageID: info.id,
                    sessionID: input.sessionID,
                  },
                ]
              }

              const file = Bun.file(filepath)
              FileTime.read(input.sessionID, filepath)
              return [
                {
                  id: Identifier.ascending("part"),
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "text",
                  text: `Called the Read tool with the following input: {\"filePath\":\"${filepath}\"}`,
                  synthetic: true,
                },
                {
                  id: part.id ?? Identifier.ascending("part"),
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "file",
                  url: `data:${part.mime};base64,` + Buffer.from(await file.bytes()).toString("base64"),
                  mime: part.mime,
                  filename: part.filename!,
                  source: part.source,
                },
              ]
          }
        }

        if (part.type === "agent") {
          return [
            {
              id: Identifier.ascending("part"),
              ...part,
              messageID: info.id,
              sessionID: input.sessionID,
            },
            {
              id: Identifier.ascending("part"),
              messageID: info.id,
              sessionID: input.sessionID,
              type: "text",
              synthetic: true,
              text:
                "Use the above message and context to generate a prompt and call the task tool with subagent: " +
                part.name,
            },
          ]
        }

        return [
          {
            id: Identifier.ascending("part"),
            ...part,
            messageID: info.id,
            sessionID: input.sessionID,
          },
        ]
      }),
    ).then((x) => x.flat())

    await Plugin.trigger(
      "chat.message",
      {
        sessionID: input.sessionID,
        agent: input.agent,
        model: input.model,
        messageID: input.messageID,
      },
      {
        message: info,
        parts,
      },
    )
    await Session.updateMessage(info)
    for (const part of parts) {
      await Session.updatePart(part)
    }

    return {
      info,
      parts,
    }
  }

  function includeAssistantMessage(message: MessageV2.WithParts) {
    if (message.info.role !== "assistant" || message.info.error === undefined) return true
    if (
      MessageV2.AbortedError.isInstance(message.info.error) &&
      message.parts.some((part) => part.type !== "step-start" && part.type !== "reasoning")
    )
      return true
    return false
  }

  function toModelConversation(messages: MessageV2.WithParts[]) {
    return MessageV2.toModelMessage(messages.filter(includeAssistantMessage))
  }

  function insertReminders(input: { messages: MessageV2.WithParts[]; agent: Agent.Info }) {
    const userMessage = input.messages.findLast((msg) => msg.info.role === "user")
    if (!userMessage) return input.messages
    if (input.agent.name === "plan") {
      userMessage.parts.push({
        id: Identifier.ascending("part"),
        messageID: userMessage.info.id,
        sessionID: userMessage.info.sessionID,
        type: "text",
        text: PROMPT_PLAN,
        synthetic: true,
      })
    }
    const wasPlan = input.messages.some((msg) => msg.info.role === "assistant" && msg.info.mode === "plan")
    if (wasPlan && input.agent.name === "build") {
      userMessage.parts.push({
        id: Identifier.ascending("part"),
        messageID: userMessage.info.id,
        sessionID: userMessage.info.sessionID,
        type: "text",
        text: BUILD_SWITCH,
        synthetic: true,
      })
    }
    return input.messages
  }

  function determineToolKind(toolName: string): "read" | "edit" | "other" {
    const readTools = [
      "read",
      "glob",
      "grep",
      "list",
      "webfetch",
      "context7_resolve_library_id",
      "context7_get_library_docs",
    ]
    const editTools = ["edit", "write", "bash"]

    if (readTools.includes(toolName.toLowerCase())) return "read"
    if (editTools.includes(toolName.toLowerCase())) return "edit"
    return "other"
  }

  function extractLocations(toolName: string, input: Record<string, any>): { path: string }[] {
    try {
      switch (toolName.toLowerCase()) {
        case "read":
        case "edit":
        case "write":
          return input["filePath"] ? [{ path: input["filePath"] }] : []
        case "glob":
        case "grep":
          return input["path"] ? [{ path: input["path"] }] : []
        case "bash":
          return []
        case "list":
          return input["path"] ? [{ path: input["path"] }] : []
        default:
          return []
      }
    } catch {
      return []
    }
  }

  export type Processor = Awaited<ReturnType<typeof createProcessor>>
  export async function createProcessor(input: {
    sessionID: string
    providerID: string
    model: ModelsDev.Model
    system: string[]
    agent: string
    abort: AbortSignal
    acpConnection?: {
      connection: any
      sessionId: string
    }
  }) {
    const toolcalls: Record<string, MessageV2.ToolPart> = {}
    let snapshot: string | undefined
    let blocked = false

    async function createMessage(parentID: string) {
      const msg: MessageV2.Assistant = {
        id: Identifier.ascending("message"),
        role: "assistant",
        mode: input.agent,
        path: {
          cwd: Instance.directory,
          root: Instance.worktree,
        },
        cost: 0,
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: input.model.id,
        providerID: input.providerID,
        time: {
          created: Date.now(),
        },
        sessionID: input.sessionID,
        parentID,
      }
      await Session.updateMessage(msg)
      return msg
    }

    let assistantMsg: MessageV2.Assistant | undefined

    async function flushSnapshot() {
      if (!snapshot || !assistantMsg) return
      const patch = await Snapshot.patch(snapshot)
      if (patch.files.length) {
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: assistantMsg.id,
          sessionID: assistantMsg.sessionID,
          type: "patch",
          hash: patch.hash,
          files: patch.files,
        })
      }
      snapshot = undefined
    }

    type UsageInfo = ReturnType<typeof Session.getUsage>

    async function startStepInternal() {
      if (!assistantMsg) throw new Error("call next() first before starting step")
      if (snapshot) await flushSnapshot()
      await Session.updatePart({
        id: Identifier.ascending("part"),
        messageID: assistantMsg.id,
        sessionID: assistantMsg.sessionID,
        type: "step-start",
      })
      snapshot = await Snapshot.track()
    }

    async function finishStepInternal(usage?: UsageInfo, options?: { reason?: string }) {
      if (!assistantMsg) throw new Error("call next() first before finishing step")
      if (!usage) return
      assistantMsg.cost += usage.cost
      assistantMsg.tokens = usage.tokens
      await Session.updatePart({
        id: Identifier.ascending("part"),
        messageID: assistantMsg.id,
        sessionID: assistantMsg.sessionID,
        type: "step-finish",
        tokens: usage.tokens,
        cost: usage.cost,
        reason: options?.reason ?? "completed",
        snapshot: await Snapshot.track(),
      })
      await Session.updateMessage(assistantMsg)
    }

    const result = {
      async end() {
        if (assistantMsg) {
          await flushSnapshot()
          assistantMsg.time.completed = Date.now()
          await Session.updateMessage(assistantMsg)
          assistantMsg = undefined
        }
      },
      async next(parentID: string) {
        if (assistantMsg) {
          throw new Error("end previous assistant message first")
        }
        assistantMsg = await createMessage(parentID)
        return assistantMsg
      },
      get message() {
        if (!assistantMsg) throw new Error("call next() first before accessing message")
        return assistantMsg
      },
      hasMessage() {
        return !!assistantMsg
      },
      partFromToolCall(toolCallID: string) {
        return toolcalls[toolCallID]
      },
      async startStep() {
        await startStepInternal()
      },
      async finishStep(usage?: UsageInfo) {
        await finishStepInternal(usage)
      },
      async flushSnapshot() {
        await flushSnapshot()
      },
      reset() {
        for (const key of Object.keys(toolcalls)) delete toolcalls[key]
        snapshot = undefined
        blocked = false
      },
      async process(
        stream: StreamTextResult<Record<string, AITool>, never>,
        retries?: { count: number; max: number },
      ) {
        log.info("process")
        if (!assistantMsg) throw new Error("call next() first before processing")
        let shouldRetry = false
        const retryState = retries ?? { count: 0, max: 0 }
        try {
          let currentText: MessageV2.TextPart | undefined
          let reasoningMap: Record<string, MessageV2.ReasoningPart> = {}

          for await (const value of stream.fullStream) {
            input.abort.throwIfAborted()
            switch (value.type) {
              case "start":
                break

              case "reasoning-start":
                if (value.id in reasoningMap) {
                  continue
                }
                reasoningMap[value.id] = {
                  id: Identifier.ascending("part"),
                  messageID: assistantMsg.id,
                  sessionID: assistantMsg.sessionID,
                  type: "reasoning",
                  text: "",
                  time: {
                    start: Date.now(),
                  },
                  metadata: value.providerMetadata,
                }
                break

              case "reasoning-delta":
                if (value.id in reasoningMap) {
                  const part = reasoningMap[value.id]
                  part.text += value.text
                  if (value.providerMetadata) part.metadata = value.providerMetadata
                  if (part.text) await Session.updatePart({ part, delta: value.text })
                }
                break

              case "reasoning-end":
                if (value.id in reasoningMap) {
                  const part = reasoningMap[value.id]
                  part.text = part.text.trimEnd()

                  part.time = {
                    ...part.time,
                    end: Date.now(),
                  }
                  if (value.providerMetadata) part.metadata = value.providerMetadata
                  await Session.updatePart(part)
                  delete reasoningMap[value.id]
                }
                break

              case "tool-input-start":
                const part = await Session.updatePart({
                  id: toolcalls[value.id]?.id ?? Identifier.ascending("part"),
                  messageID: assistantMsg.id,
                  sessionID: assistantMsg.sessionID,
                  type: "tool",
                  tool: value.toolName,
                  callID: value.id,
                  state: {
                    status: "pending",
                    input: {},
                    raw: "",
                  },
                })
                toolcalls[value.id] = part as MessageV2.ToolPart
                if (input.acpConnection) {
                  await input.acpConnection.connection
                    .sessionUpdate({
                      sessionId: input.acpConnection.sessionId,
                      update: {
                        sessionUpdate: "tool_call",
                        toolCallId: value.id,
                        title: value.toolName,
                        kind: determineToolKind(value.toolName),
                        status: "pending",
                        locations: [],
                        rawInput: {},
                      },
                    })
                    .catch((err: Error) => {
                      log.error("failed to send tool pending to ACP", { error: err })
                    })
                }
                break

              case "tool-input-delta":
                break

              case "tool-input-end":
                break

              case "tool-call": {
                const match = toolcalls[value.toolCallId]
                if (match) {
                  const part = await Session.updatePart({
                    ...match,
                    tool: value.toolName,
                    state: {
                      status: "running",
                      input: value.input,
                      time: {
                        start: Date.now(),
                      },
                    },
                    metadata: value.providerMetadata,
                  })
                  toolcalls[value.toolCallId] = part as MessageV2.ToolPart
                  const recent = await MessageV2.parts(assistantMsg.id)
                  const lastThree = recent.slice(-DOOM_LOOP_THRESHOLD)
                  if (
                    lastThree.length === DOOM_LOOP_THRESHOLD &&
                    lastThree.every(
                      (p) =>
                        p.type === "tool" &&
                        p.tool === value.toolName &&
                        p.state.status !== "pending" &&
                        JSON.stringify(p.state.input) === JSON.stringify(value.input),
                    )
                  ) {
                    const permission = await Agent.get(input.agent).then((x) => x.permission)
                    if (permission.doom_loop === "ask") {
                      await Permission.ask({
                        type: "doom_loop",
                        pattern: value.toolName,
                        sessionID: assistantMsg.sessionID,
                        messageID: assistantMsg.id,
                        callID: value.toolCallId,
                        title: `Possible doom loop: "${value.toolName}" called ${DOOM_LOOP_THRESHOLD} times with identical arguments`,
                        metadata: {
                          tool: value.toolName,
                          input: value.input,
                        },
                      })
                    }
                  }
                  if (input.acpConnection) {
                    await input.acpConnection.connection
                      .sessionUpdate({
                        sessionId: input.acpConnection.sessionId,
                        update: {
                          sessionUpdate: "tool_call_update",
                          toolCallId: value.toolCallId,
                          status: "in_progress",
                          locations: extractLocations(value.toolName, value.input),
                          rawInput: value.input,
                        },
                      })
                      .catch((err: Error) => {
                        log.error("failed to send tool in_progress to ACP", { error: err })
                      })
                  }
                }
                break
              }
              case "tool-result": {
                const match = toolcalls[value.toolCallId]
                if (match && match.state.status === "running") {
                  await Session.updatePart({
                    ...match,
                    state: {
                      status: "completed",
                      input: value.input,
                      output: value.output.output,
                      metadata: value.output.metadata,
                      title: value.output.title,
                      time: {
                        start: match.state.time.start,
                        end: Date.now(),
                      },
                      attachments: value.output.attachments,
                    },
                  })

                  if (input.acpConnection) {
                    await input.acpConnection.connection
                      .sessionUpdate({
                        sessionId: input.acpConnection.sessionId,
                        update: {
                          sessionUpdate: "tool_call_update",
                          toolCallId: value.toolCallId,
                          status: "completed",
                          content: [
                            {
                              type: "content",
                              content: {
                                type: "text",
                                text: value.output.output,
                              },
                            },
                          ],
                          rawOutput: value.output,
                        },
                      })
                      .catch((err: Error) => {
                        log.error("failed to send tool completed to ACP", { error: err })
                      })
                  }

                  delete toolcalls[value.toolCallId]
                }
                break
              }

              case "tool-error": {
                const match = toolcalls[value.toolCallId]
                if (match && match.state.status === "running") {
                  await Session.updatePart({
                    ...match,
                    state: {
                      status: "error",
                      input: value.input,
                      error: toErrorString(value.error),
                      metadata: value.error instanceof Permission.RejectedError ? value.error.metadata : undefined,
                      time: {
                        start: match.state.time.start,
                        end: Date.now(),
                      },
                    },
                  })

                  if (input.acpConnection) {
                    await input.acpConnection.connection
                      .sessionUpdate({
                        sessionId: input.acpConnection.sessionId,
                        update: {
                          sessionUpdate: "tool_call_update",
                          toolCallId: value.toolCallId,
                          status: "failed",
                          content: [
                            {
                              type: "content",
                              content: {
                                type: "text",
                                text: `Error: ${toErrorString(value.error)}`,
                              },
                            },
                          ],
                          rawOutput: {
                            error: toErrorString(value.error),
                          },
                        },
                      })
                      .catch((err: Error) => {
                        log.error("failed to send tool error to ACP", { error: err })
                      })
                  }

                  if (value.error instanceof Permission.RejectedError) {
                    blocked = true
                  }
                  delete toolcalls[value.toolCallId]
                }
                break
              }
              case "error":
                throw value.error

              case "start-step":
                await startStepInternal()
                break

              case "finish-step":
                const usage = Session.getUsage({
                  model: input.model,
                  usage: value.usage,
                  metadata: value.providerMetadata,
                })
                await finishStepInternal(usage, { reason: value.finishReason })
                await flushSnapshot()
                SessionSummary.summarize({
                  sessionID: input.sessionID,
                  messageID: assistantMsg.parentID,
                })
                break

              case "text-start":
                currentText = {
                  id: Identifier.ascending("part"),
                  messageID: assistantMsg.id,
                  sessionID: assistantMsg.sessionID,
                  type: "text",
                  text: "",
                  time: {
                    start: Date.now(),
                  },
                  metadata: value.providerMetadata,
                }
                break

              case "text-delta":
                if (currentText) {
                  currentText.text += value.text
                  if (value.providerMetadata) currentText.metadata = value.providerMetadata
                  if (currentText.text)
                    await Session.updatePart({
                      part: currentText,
                      delta: value.text,
                    })
                  if (input.acpConnection && value.text) {
                    await input.acpConnection.connection
                      .sessionUpdate({
                        sessionId: input.acpConnection.sessionId,
                        update: {
                          sessionUpdate: "agent_message_chunk",
                          content: {
                            type: "text",
                            text: value.text,
                          },
                        },
                      })
                      .catch((err: Error) => {
                        log.error("failed to send text delta to ACP", { error: err })
                      })
                  }
                }
                break

              case "text-end":
                if (currentText) {
                  currentText.text = currentText.text.trimEnd()
                  currentText.time = {
                    start: Date.now(),
                    end: Date.now(),
                  }
                  if (value.providerMetadata) currentText.metadata = value.providerMetadata
                  await Session.updatePart(currentText)
                }
                currentText = undefined
                break

              case "finish":
                assistantMsg.time.completed = Date.now()
                await Session.updateMessage(assistantMsg)
                break

              default:
                log.info("unhandled", {
                  ...value,
                })
                continue
            }
          }
        } catch (e) {
          log.error("process", {
            error: e,
          })
          const error = MessageV2.fromError(e, { providerID: input.providerID })
          if (retryState.count < retryState.max && MessageV2.APIError.isInstance(error) && error.data.isRetryable) {
            shouldRetry = true
            await Session.updatePart({
              id: Identifier.ascending("part"),
              messageID: assistantMsg.id,
              sessionID: assistantMsg.sessionID,
              type: "retry",
              attempt: retryState.count + 1,
              time: {
                created: Date.now(),
              },
              error,
            })
          } else {
            assistantMsg.error = error
            if (isStreamingVerificationError(assistantMsg.error)) {
              await Session.setStreamingCapable(assistantMsg.sessionID, false)
            }
            Bus.publish(Session.Event.Error, {
              sessionID: assistantMsg.sessionID,
              error: assistantMsg.error,
            })
          }
        }
        const p = await MessageV2.parts(assistantMsg.id)
        for (const part of p) {
          if (part.type === "tool" && part.state.status !== "completed" && part.state.status !== "error") {
            await Session.updatePart({
              ...part,
              state: {
                status: "error",
                error: "Tool execution aborted",
                time: {
                  start: Date.now(),
                  end: Date.now(),
                },
                input: {},
              },
            })
          }
        }
        if (!shouldRetry) {
          assistantMsg.time.completed = Date.now()
        }
        await Session.updateMessage(assistantMsg)
        return { info: assistantMsg, parts: p, blocked, shouldRetry }
      },
    }
    return result
  }

  function isBusy(sessionID: string) {
    if (SessionLock.isLocked(sessionID)) return true
    return state().pending.has(sessionID)
  }

  export function abort(sessionID: string) {
    const controller = state().pending.get(sessionID)
    if (!controller) return SessionLock.abort(sessionID)
    log.info("aborting", { sessionID })
    if (!controller.signal.aborted) controller.abort()
    if (state().pending.get(sessionID) === controller) state().pending.delete(sessionID)
    SessionLock.abort(sessionID)
    return true
  }

  function lock(sessionID: string) {
    const handle = SessionLock.acquire({
      sessionID,
    })
    log.info("locking", { sessionID })
    if (state().pending.has(sessionID)) {
      handle[Symbol.dispose]()
      throw new BusyError(sessionID)
    }
    const controller = new AbortController()
    const clear = () => {
      if (state().pending.get(sessionID) === controller) state().pending.delete(sessionID)
    }
    state().pending.set(sessionID, controller)
    handle.signal.addEventListener(
      "abort",
      () => {
        if (!controller.signal.aborted) controller.abort()
        clear()
      },
      { once: true },
    )
    return {
      signal: controller.signal,
      abort() {
        if (!controller.signal.aborted) controller.abort()
        clear()
        handle.abort()
      },
      async [Symbol.dispose]() {
        if (!controller.signal.aborted) controller.abort()
        clear()
        handle[Symbol.dispose]()
        log.info("unlocking", { sessionID })

        const session = await Session.get(sessionID)
        if (session.parentID) return

        Bus.publish(Event.Idle, {
          sessionID,
        })
      },
    }
  }

  export const ShellInput = z.object({
    sessionID: Identifier.schema("session"),
    agent: z.string(),
    command: z.string(),
  })
  export type ShellInput = z.infer<typeof ShellInput>
  export async function shell(input: ShellInput) {
    using abort = lock(input.sessionID)
    const session = await Session.get(input.sessionID)
    if (session.revert) {
      SessionRevert.cleanup(session)
    }
    const userMsg: MessageV2.User = {
      id: Identifier.ascending("message"),
      sessionID: input.sessionID,
      time: {
        created: Date.now(),
      },
      role: "user",
    }
    await Session.updateMessage(userMsg)
    const userPart: MessageV2.Part = {
      type: "text",
      id: Identifier.ascending("part"),
      messageID: userMsg.id,
      sessionID: input.sessionID,
      text: "The following tool was executed by the user",
      synthetic: true,
    }
    await Session.updatePart(userPart)

    const msg: MessageV2.Assistant = {
      id: Identifier.ascending("message"),
      sessionID: input.sessionID,
      parentID: userMsg.id,
      mode: input.agent,
      cost: 0,
      path: {
        cwd: Instance.directory,
        root: Instance.worktree,
      },
      time: {
        created: Date.now(),
      },
      role: "assistant",
      tokens: {
        input: 0,
        output: 0,
        reasoning: 0,
        cache: { read: 0, write: 0 },
      },
      modelID: "",
      providerID: "",
    }
    await Session.updateMessage(msg)
    const part: MessageV2.Part = {
      type: "tool",
      id: Identifier.ascending("part"),
      messageID: msg.id,
      sessionID: input.sessionID,
      tool: "bash",
      callID: ulid(),
      state: {
        status: "running",
        time: {
          start: Date.now(),
        },
        input: {
          command: input.command,
        },
      },
    }
    await Session.updatePart(part)
    const shell = process.env["SHELL"] ?? "bash"
    const shellName = path.basename(shell)

    const invocations: Record<string, { args: string[] }> = {
      nu: {
        args: ["-c", input.command],
      },
      fish: {
        args: ["-c", input.command],
      },
      zsh: {
        args: [
          "-c",
          "-l",
          `
            [[ -f ~/.zshenv ]] && source ~/.zshenv >/dev/null 2>&1 || true
            [[ -f "\${ZDOTDIR:-$HOME}/.zshrc" ]] && source "\${ZDOTDIR:-$HOME}/.zshrc" >/dev/null 2>&1 || true
            ${input.command}
          `,
        ],
      },
      bash: {
        args: [
          "-c",
          "-l",
          `
            [[ -f ~/.bashrc ]] && source ~/.bashrc >/dev/null 2>&1 || true
            ${input.command}
          `,
        ],
      },
      // Fallback: any shell that doesn't match those above
      "": {
        args: ["-c", "-l", `${input.command}`],
      },
    }

    const matchingInvocation = invocations[shellName] ?? invocations[""]
    const args = matchingInvocation?.args

    const proc = spawn(shell, args, {
      cwd: Instance.directory,
      signal: abort.signal,
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        TERM: "dumb",
      },
    })

    abort.signal.addEventListener("abort", () => {
      if (!proc.pid) return
      process.kill(-proc.pid)
    })

    let output = ""

    proc.stdout?.on("data", (chunk) => {
      output += chunk.toString()
      if (part.state.status === "running") {
        part.state.metadata = {
          output: output,
          description: "",
        }
        Session.updatePart(part)
      }
    })

    proc.stderr?.on("data", (chunk) => {
      output += chunk.toString()
      if (part.state.status === "running") {
        part.state.metadata = {
          output: output,
          description: "",
        }
        Session.updatePart(part)
      }
    })

    await new Promise<void>((resolve) => {
      proc.on("close", () => {
        resolve()
      })
    })
    msg.time.completed = Date.now()
    await Session.updateMessage(msg)
    if (part.state.status === "running") {
      part.state = {
        status: "completed",
        time: {
          ...part.state.time,
          end: Date.now(),
        },
        input: part.state.input,
        title: "",
        metadata: {
          output,
          description: "",
        },
        output,
      }
      await Session.updatePart(part)
    }
    return { info: msg, parts: [part] }
  }

  export const CommandInput = z.object({
    messageID: Identifier.schema("message").optional(),
    sessionID: Identifier.schema("session"),
    agent: z.string().optional(),
    model: z.string().optional(),
    arguments: z.string(),
    command: z.string(),
  })
  export type CommandInput = z.infer<typeof CommandInput>
  const bashRegex = /!`([^`]+)`/g
  const argsRegex = /(?:[^\s"']+|"[^"]*"|'[^']*')+/g
  const placeholderRegex = /\$(\d+)/g
  const quoteTrimRegex = /^["']|["']$/g
  /**
   * Regular expression to match @ file references in text
   * Matches @ followed by file paths, excluding commas, periods at end of sentences, and backticks
   * Does not match when preceded by word characters or backticks (to avoid email addresses and quoted references)
   */

  export async function command(input: CommandInput) {
    log.info("command", input)
    const command = await Command.get(input.command)
    const agentName = command.agent ?? input.agent ?? "build"

    const raw = input.arguments.match(argsRegex) ?? []
    const args = raw.map((arg) => arg.replace(quoteTrimRegex, ""))

    const placeholders = command.template.match(placeholderRegex) ?? []
    let last = 0
    for (const item of placeholders) {
      const value = Number(item.slice(1))
      if (value > last) last = value
    }

    const withArgs = command.template.replaceAll(placeholderRegex, (_, index) => {
      const position = Number(index)
      const argIndex = position - 1
      if (argIndex >= args.length) return ""
      if (position === last) return args.slice(argIndex).join(" ")
      return args[argIndex]
    })
    let template = withArgs.replaceAll("$ARGUMENTS", input.arguments)

    const shell = ConfigMarkdown.shell(template)
    if (shell.length > 0) {
      const results = await Promise.all(
        shell.map(async ([, cmd]) => {
          try {
            return await $`${{ raw: cmd }}`.nothrow().text()
          } catch (error) {
            return `Error executing command: ${error instanceof Error ? error.message : String(error)}`
          }
        }),
      )
      let index = 0
      template = template.replace(bashRegex, () => results[index++])
    }
    template = template.trim()

    const parts = await resolvePromptParts(template)

    const model = await (async () => {
      if (command.model) {
        return Provider.parseModel(command.model)
      }
      if (command.agent) {
        const cmdAgent = await Agent.get(command.agent)
        if (cmdAgent.model) {
          return cmdAgent.model
        }
      }
      if (input.model) {
        return Provider.parseModel(input.model)
      }
      return await Provider.defaultModel()
    })()

    const agent = await Agent.get(agentName)
    let result: MessageV2.WithParts
    if ((agent.mode === "subagent" && command.subtask !== false) || command.subtask === true) {
      using abort = lock(input.sessionID)

      const userMsg: MessageV2.User = {
        id: Identifier.ascending("message"),
        sessionID: input.sessionID,
        time: {
          created: Date.now(),
        },
        role: "user",
      }
      await Session.updateMessage(userMsg)
      const userPart: MessageV2.Part = {
        type: "text",
        id: Identifier.ascending("part"),
        messageID: userMsg.id,
        sessionID: input.sessionID,
        text: "The following tool was executed by the user",
        synthetic: true,
      }
      await Session.updatePart(userPart)

      const assistantMsg: MessageV2.Assistant = {
        id: Identifier.ascending("message"),
        sessionID: input.sessionID,
        parentID: userMsg.id,
        mode: agentName,
        cost: 0,
        path: {
          cwd: Instance.directory,
          root: Instance.worktree,
        },
        time: {
          created: Date.now(),
        },
        role: "assistant",
        tokens: {
          input: 0,
          output: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: model.modelID,
        providerID: model.providerID,
      }
      await Session.updateMessage(assistantMsg)

      const args = {
        description: "Consulting " + agent.name,
        subagent_type: agent.name,
        prompt: template,
      }
      const callId = ulid()
      const tool = ((await TaskTool.init()) as unknown) as ToolExecutor
      const truncated = args.prompt.length > 100 ? args.prompt.substring(0, 97) + "..." : args.prompt
      const outcome = await executeToolCall({
        tool,
        name: "task",
        callId,
        input: args,
        sessionID: input.sessionID,
        messageID: assistantMsg.id,
        abortSignal: abort.signal,
        stateInput: {
          description: args.description,
          subagent_type: args.subagent_type,
          prompt: truncated,
        },
        context: ({
          update,
        }: {
          update: (state: { metadata?: unknown; title?: string }) => Promise<MessageV2.ToolPart>
        }) => ({
          sessionID: input.sessionID,
          abort: abort.signal,
          agent: agent.name,
          messageID: assistantMsg.id,
          extra: {},
          metadata: async (meta: { title?: string; metadata?: unknown }) => {
            await update({
              title: meta.title,
              metadata: meta.metadata,
            })
          },
        }),
      })
      assistantMsg.time.completed = Date.now()
      await Session.updateMessage(assistantMsg)
      result = { info: assistantMsg, parts: [outcome.part] }
    } else {
      result = await prompt({
        sessionID: input.sessionID,
        messageID: input.messageID,
        model,
        agent: agentName,
        parts,
      })
    }

    Bus.publish(Command.Event.Executed, {
      name: input.command,
      sessionID: input.sessionID,
      arguments: input.arguments,
      messageID: result.info.id,
    })

    return result
  }

  async function ensureTitle(input: {
    session: Session.Info
    message: MessageV2.WithParts
    history: MessageV2.WithParts[]
    providerID: string
    modelID: string
  }) {
    if (input.session.parentID) return
    if (!Session.isDefaultTitle(input.session.title)) return
    const isFirst =
      input.history.filter((m) => m.info.role === "user" && !m.parts.every((p) => "synthetic" in p && p.synthetic))
        .length === 1
    if (!isFirst) return
    const small =
      (await Provider.getSmallModel(input.providerID)) ?? (await Provider.getModel(input.providerID, input.modelID))
    const options = {
      ...ProviderTransform.options(small.providerID, small.modelID, small.npm ?? "", input.session.id),
      ...small.info.options,
    }
    if (small.providerID === "openai" || small.modelID.includes("gpt-5")) {
      options["reasoningEffort"] = "minimal"
    }
    if (small.providerID === "google") {
      options["thinkingConfig"] = {
        thinkingBudget: 0,
      }
    }
    const abortSignal = AbortSignal.timeout(15_000)
    await generateText({
      maxOutputTokens: small.info.reasoning ? 1500 : 20,
      providerOptions: ProviderTransform.providerOptions(small.npm, small.providerID, options),
      messages: [
        ...SystemPrompt.title(small.providerID).map(
          (x): ModelMessage => ({
            role: "system",
            content: x,
          }),
        ),
        {
          role: "user" as const,
          content: `
              The following is the text to summarize:
            `,
        },
        ...MessageV2.toModelMessage([
          {
            info: {
              id: Identifier.ascending("message"),
              role: "user",
              sessionID: input.session.id,
              time: {
                created: Date.now(),
              },
            },
            parts: input.message.parts,
          },
        ]),
      ],
      headers: small.info.headers,
      model: small.language,
      abortSignal,
    })
      .then((result) => {
        if (result.text)
          return Session.update(input.session.id, (draft) => {
            const cleaned = result.text
              .replace(/<think>[\s\S]*?<\/think>\s*/g, "")
              .split("\n")
              .map((line) => line.trim())
              .find((line) => line.length > 0)
            if (!cleaned) return

            const title = cleaned.length > 100 ? cleaned.substring(0, 97) + "..." : cleaned
            draft.title = title
          })
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "TimeoutError") return
        log.error("failed to generate title", { error, model: small.info.id })
      })
  }
}
