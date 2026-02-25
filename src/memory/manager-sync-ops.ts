import { randomUUID } from "node:crypto";
import fsSync from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import chokidar, { FSWatcher } from "chokidar";
import { resolveAgentDir } from "../agents/agent-scope.js";
import { ResolvedMemorySearchConfig } from "../agents/memory-search.js";
import { type OpenClawConfig } from "../config/config.js";
import { resolveSessionTranscriptsDirForAgent } from "../config/sessions/paths.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { onSessionTranscriptUpdate } from "../sessions/transcript-events.js";
import { resolveUserPath } from "../utils.js";
import { DEFAULT_GEMINI_EMBEDDING_MODEL } from "./embeddings-gemini.js";
import { DEFAULT_MISTRAL_EMBEDDING_MODEL } from "./embeddings-mistral.js";
import { DEFAULT_OPENAI_EMBEDDING_MODEL } from "./embeddings-openai.js";
import { DEFAULT_VOYAGE_EMBEDDING_MODEL } from "./embeddings-voyage.js";
import {
  createEmbeddingProvider,
  type EmbeddingProvider,
  type GeminiEmbeddingClient,
  type MistralEmbeddingClient,
  type OpenAiEmbeddingClient,
  type VoyageEmbeddingClient,
} from "./embeddings.js";
import { isFileMissingError } from "./fs-utils.js";
import {
  buildFileEntry,
  ensureDir,
  listMemoryFiles,
  normalizeExtraMemoryPaths,
  runWithConcurrency,
} from "./internal.js";
import { type MemoryFileEntry } from "./internal.js";
import { ensureMemoryIndexSchema } from "./memory-schema.js";
import type { SessionFileEntry } from "./session-files.js";
import {
  buildSessionEntry,
  listSessionFilesForAgent,
  sessionPathForFile,
} from "./session-files.js";
import { loadSqliteVecExtension } from "./sqlite-vec.js";
import { requireNodeSqlite } from "./sqlite.js";
import type { MemorySource, MemorySyncProgressUpdate } from "./types.js";

type MemoryIndexMeta = {
  model: string;
  provider: string;
  providerKey?: string;
  sources?: MemorySource[];
  chunkTokens: number;
  chunkOverlap: number;
  vectorDims?: number;
};

type MemorySyncProgressState = {
  completed: number;
  total: number;
  label?: string;
  report: (update: MemorySyncProgressUpdate) => void;
};

const META_KEY = "memory_index_meta_v1";
const VECTOR_TABLE = "chunks_vec";
const FTS_TABLE = "chunks_fts";
const EMBEDDING_CACHE_TABLE = "embedding_cache";
const SESSION_DIRTY_DEBOUNCE_MS = 5000;
const SESSION_DELTA_READ_CHUNK_BYTES = 64 * 1024;
const VECTOR_LOAD_TIMEOUT_MS = 30_000;
const IGNORED_MEMORY_WATCH_DIR_NAMES = new Set([
  ".git",
  "node_modules",
  ".pnpm-store",
  ".venv",
  "venv",
  ".tox",
  "__pycache__",
]);

const log = createSubsystemLogger("memory");

function shouldIgnoreMemoryWatchPath(watchPath: string): boolean {
  const normalized = path.normalize(watchPath);
  const parts = normalized.split(path.sep).map((segment) => segment.trim().toLowerCase());
  return parts.some((segment) => IGNORED_MEMORY_WATCH_DIR_NAMES.has(segment));
}

export abstract class MemoryManagerSyncOps {
  protected abstract readonly cfg: OpenClawConfig;
  protected abstract readonly agentId: string;
  protected abstract readonly workspaceDir: string;
  protected abstract readonly settings: ResolvedMemorySearchConfig;
  protected provider: EmbeddingProvider | null = null;
  protected fallbackFrom?: "openai" | "local" | "gemini" | "voyage" | "mistral";
  protected openAi?: OpenAiEmbeddingClient;
  protected gemini?: GeminiEmbeddingClient;
  protected voyage?: VoyageEmbeddingClient;
  protected mistral?: MistralEmbeddingClient;
  protected abstract batch: {
    enabled: boolean;
    wait: boolean;
    concurrency: number;
    pollIntervalMs: number;
    timeoutMs: number;
  };
  protected readonly sources: Set<MemorySource> = new Set();
  protected providerKey: string | null = null;
  protected abstract readonly vector: {
    enabled: boolean;
    available: boolean | null;
    extensionPath?: string;
    loadError?: string;
    dims?: number;
  };
  protected readonly fts: {
    enabled: boolean;
    available: boolean;
    loadError?: string;
  } = { enabled: false, available: false };
  protected vectorReady: Promise<boolean> | null = null;
  protected watcher: FSWatcher | null = null;
  protected watchTimer: NodeJS.Timeout | null = null;
  protected sessionWatchTimer: NodeJS.Timeout | null = null;
  protected sessionUnsubscribe: (() => void) | null = null;
  protected fallbackReason?: string;
  protected intervalTimer: NodeJS.Timeout | null = null;
  protected backgroundSyncTimer: NodeJS.Timeout | null = null;
  protected backgroundSyncReasons = new Set<string>();
  protected closed = false;
  protected dirty = false;
  protected sessionsDirty = false;
  protected sessionsDirtyFiles = new Set<string>();
  protected sessionPendingFiles = new Set<string>();
  protected sessionDeltas = new Map<
    string,
    { lastSize: number; pendingBytes: number; pendingMessages: number }
  >();

  protected abstract syncInProgress: boolean;
  protected abstract syncDeferred: boolean;
  protected abstract syncDeferredReason?: string;
  protected abstract readonly cache: { enabled: boolean; maxEntries?: number };
  protected abstract db: DatabaseSync;
  protected abstract computeProviderKey(): string;
  protected abstract sync(params?: {
    reason?: string;
    force?: boolean;
    progress?: (update: MemorySyncProgressUpdate) => void;
  }): Promise<void>;
  protected abstract withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number,
    message: string,
  ): Promise<T>;
  protected abstract getIndexConcurrency(): number;
  protected abstract pruneEmbeddingCacheIfNeeded(): void;
  protected abstract indexFile(
    entry: MemoryFileEntry | SessionFileEntry,
    options: { source: MemorySource; content?: string },
  ): Promise<void>;
  protected abstract indexFileToDb(
    entry: MemoryFileEntry | SessionFileEntry,
    options: { source: MemorySource; content?: string },
    db: DatabaseSync,
    runtime?: { vectorEnabled?: boolean },
  ): Promise<void>;

  protected logPerf(functionName: string, fields: Record<string, unknown>): void {
    log.debug("memory lifecycle perf", {
      function: functionName,
      ...fields,
    });
  }

  protected async withPerf<T>(
    functionName: string,
    fields: Record<string, unknown>,
    run: () => Promise<T>,
  ): Promise<T> {
    const startedAt = Date.now();
    try {
      return await run();
    } finally {
      this.logPerf(functionName, {
        ...fields,
        total_ms: Date.now() - startedAt,
      });
    }
  }

  protected async ensureVectorReady(dimensions?: number): Promise<boolean> {
    const startedAt = Date.now();
    let ready = false;
    try {
      if (!this.vector.enabled) {
        return false;
      }
      if (!this.vectorReady) {
        this.vectorReady = this.withTimeout(
          this.loadVectorExtension(),
          VECTOR_LOAD_TIMEOUT_MS,
          `sqlite-vec load timed out after ${Math.round(VECTOR_LOAD_TIMEOUT_MS / 1000)}s`,
        );
      }
      try {
        ready = (await this.vectorReady) || false;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this.vector.available = false;
        this.vector.loadError = message;
        this.vectorReady = null;
        log.warn(`sqlite-vec unavailable: ${message}`);
        return false;
      }
      if (ready && typeof dimensions === "number" && dimensions > 0) {
        this.ensureVectorTable(dimensions);
      }
      return ready;
    } finally {
      this.logPerf("ensureVectorReady", {
        dims: dimensions,
        ready,
        total_ms: Date.now() - startedAt,
      });
    }
  }

  private async loadVectorExtension(): Promise<boolean> {
    const startedAt = Date.now();
    let loadedOk = false;
    try {
      if (this.vector.available !== null) {
        loadedOk = this.vector.available;
        return this.vector.available;
      }
      if (!this.vector.enabled) {
        this.vector.available = false;
        loadedOk = false;
        return false;
      }
      try {
        const resolvedPath = this.vector.extensionPath?.trim()
          ? resolveUserPath(this.vector.extensionPath)
          : undefined;
        const loaded = await loadSqliteVecExtension({ db: this.db, extensionPath: resolvedPath });
        if (!loaded.ok) {
          throw new Error(loaded.error ?? "unknown sqlite-vec load error");
        }
        this.vector.extensionPath = loaded.extensionPath;
        this.vector.available = true;
        loadedOk = true;
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this.vector.available = false;
        this.vector.loadError = message;
        log.warn(`sqlite-vec unavailable: ${message}`);
        loadedOk = false;
        return false;
      }
    } finally {
      this.logPerf("loadVectorExtension", {
        available: loadedOk,
        total_ms: Date.now() - startedAt,
      });
    }
  }

  private ensureVectorTable(dimensions: number): void {
    if (this.vector.dims === dimensions) {
      return;
    }
    if (this.vector.dims && this.vector.dims !== dimensions) {
      this.dropVectorTable();
    }
    this.db.exec(
      `CREATE VIRTUAL TABLE IF NOT EXISTS ${VECTOR_TABLE} USING vec0(\n` +
        `  id TEXT PRIMARY KEY,\n` +
        `  embedding FLOAT[${dimensions}]\n` +
        `)`,
    );
    this.vector.dims = dimensions;
  }

  private dropVectorTable(): void {
    try {
      this.db.exec(`DROP TABLE IF EXISTS ${VECTOR_TABLE}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log.debug(`Failed to drop ${VECTOR_TABLE}: ${message}`);
    }
  }

  protected buildSourceFilter(alias?: string): { sql: string; params: MemorySource[] } {
    const sources = Array.from(this.sources);
    if (sources.length === 0) {
      return { sql: "", params: [] };
    }
    const column = alias ? `${alias}.source` : "source";
    const placeholders = sources.map(() => "?").join(", ");
    return { sql: ` AND ${column} IN (${placeholders})`, params: sources };
  }

  private normalizeProjectId(input: string): string {
    return (
      input
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "") || "project"
    );
  }

  private resolveProjectIdForMemoryFile(absPath: string): string | null {
    const relPath = path.relative(this.workspaceDir, absPath).replace(/\\/g, "/");

    // Fix: Match .openclaw_kb at any depth under Projects/, extracting the first segment
    // as projectId. e.g. "Projects/a/b/c/.openclaw_kb/file.md" → projectId "a"
    const projectMatch =
      relPath.match(/^Projects\/([^/]+)\/.+\/\.openclaw_kb\//i) ??
      relPath.match(/^Projects\/([^/]+)\/\.openclaw_kb\//i);
    if (projectMatch?.[1]) {
      return this.normalizeProjectId(projectMatch[1]);
    }

    // Fix: Route skills/<name>/.openclaw_kb/ content to skill-specific DBs
    // to prevent skills KB content from landing in main.sqlite
    const skillMatch = relPath.match(/^skills\/([^/]+)\/\.openclaw_kb\//i);
    if (skillMatch?.[1]) {
      return this.normalizeProjectId(`skill-${skillMatch[1]}`);
    }

    return null;
  }

  private resolveProjectDbPathForSync(projectId: string): string {
    const template = this.settings.store.projectPathTemplate;
    const withProject = template.includes("{projectId}")
      ? template.replaceAll("{projectId}", projectId)
      : template;
    return resolveUserPath(withProject);
  }

  protected openDatabase(): DatabaseSync {
    const dbPath = resolveUserPath(this.settings.store.corePath || this.settings.store.path);
    return this.openDatabaseAtPath(dbPath);
  }

  protected openDatabaseAtPath(dbPath: string): DatabaseSync {
    const dir = path.dirname(dbPath);
    ensureDir(dir);
    const { DatabaseSync } = requireNodeSqlite();
    const db = new DatabaseSync(dbPath, { allowExtension: this.settings.store.vector.enabled });
    try {
      // Improve resilience under concurrent sync/search access.
      db.exec("PRAGMA journal_mode = WAL");
      db.exec("PRAGMA busy_timeout = 5000");
      db.exec("PRAGMA synchronous = NORMAL");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log.debug(`memory sqlite pragma setup skipped: ${message}`);
    }
    return db;
  }

  private seedEmbeddingCache(sourceDb: DatabaseSync): void {
    if (!this.cache.enabled) {
      return;
    }
    try {
      const rows = sourceDb
        .prepare(
          `SELECT provider, model, provider_key, hash, embedding, dims, updated_at FROM ${EMBEDDING_CACHE_TABLE}`,
        )
        .all() as Array<{
        provider: string;
        model: string;
        provider_key: string;
        hash: string;
        embedding: string;
        dims: number | null;
        updated_at: number;
      }>;
      if (!rows.length) {
        return;
      }
      const insert = this.db.prepare(
        `INSERT INTO ${EMBEDDING_CACHE_TABLE} (provider, model, provider_key, hash, embedding, dims, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(provider, model, provider_key, hash) DO UPDATE SET
           embedding=excluded.embedding,
           dims=excluded.dims,
           updated_at=excluded.updated_at`,
      );
      this.db.exec("BEGIN");
      for (const row of rows) {
        insert.run(
          row.provider,
          row.model,
          row.provider_key,
          row.hash,
          row.embedding,
          row.dims,
          row.updated_at,
        );
      }
      this.db.exec("COMMIT");
    } catch (err) {
      try {
        this.db.exec("ROLLBACK");
      } catch {}
      throw err;
    }
  }

  private async swapIndexFiles(targetPath: string, tempPath: string): Promise<void> {
    const backupPath = `${targetPath}.backup-${randomUUID()}`;
    await this.moveIndexFiles(targetPath, backupPath);
    try {
      await this.moveIndexFiles(tempPath, targetPath);
    } catch (err) {
      await this.moveIndexFiles(backupPath, targetPath);
      throw err;
    }
    await this.removeIndexFiles(backupPath);
  }

  private async moveIndexFiles(sourceBase: string, targetBase: string): Promise<void> {
    const suffixes = ["", "-wal", "-shm"];
    for (const suffix of suffixes) {
      const source = `${sourceBase}${suffix}`;
      const target = `${targetBase}${suffix}`;
      try {
        await fs.rename(source, target);
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
          throw err;
        }
      }
    }
  }

  private async removeIndexFiles(basePath: string): Promise<void> {
    const suffixes = ["", "-wal", "-shm"];
    await Promise.all(suffixes.map((suffix) => fs.rm(`${basePath}${suffix}`, { force: true })));
  }

  protected ensureSchema() {
    const result = ensureMemoryIndexSchema({
      db: this.db,
      embeddingCacheTable: EMBEDDING_CACHE_TABLE,
      ftsTable: FTS_TABLE,
      ftsEnabled: this.fts.enabled,
    });
    this.fts.available = result.ftsAvailable;
    if (result.ftsError) {
      this.fts.loadError = result.ftsError;
      log.warn(`fts unavailable: ${result.ftsError}`);
    }
  }

  protected ensureWatcher() {
    if (!this.sources.has("memory") || !this.settings.sync.watch || this.watcher) {
      return;
    }
    const watchPaths = new Set<string>([
      path.join(this.workspaceDir, "MEMORY.md"),
      path.join(this.workspaceDir, "memory.md"),
      path.join(this.workspaceDir, "memory", "**", "*.md"),
    ]);
    const additionalPaths = normalizeExtraMemoryPaths(this.workspaceDir, this.settings.extraPaths);
    for (const entry of additionalPaths) {
      try {
        const stat = fsSync.lstatSync(entry);
        if (stat.isSymbolicLink()) {
          continue;
        }
        if (stat.isDirectory()) {
          watchPaths.add(path.join(entry, "**", "*.md"));
          continue;
        }
        if (stat.isFile() && entry.toLowerCase().endsWith(".md")) {
          watchPaths.add(entry);
        }
      } catch {
        // Skip missing/unreadable additional paths.
      }
    }
    this.watcher = chokidar.watch(Array.from(watchPaths), {
      ignoreInitial: true,
      ignored: (watchPath) => shouldIgnoreMemoryWatchPath(String(watchPath)),
      awaitWriteFinish: {
        stabilityThreshold: this.settings.sync.watchDebounceMs,
        pollInterval: 100,
      },
    });
    const markDirty = () => {
      this.dirty = true;
      this.scheduleWatchSync();
    };
    this.watcher.on("add", markDirty);
    this.watcher.on("change", markDirty);
    this.watcher.on("unlink", markDirty);
  }

  protected ensureSessionListener() {
    if (!this.sources.has("sessions") || this.sessionUnsubscribe) {
      return;
    }
    this.sessionUnsubscribe = onSessionTranscriptUpdate((update) => {
      if (this.closed) {
        return;
      }
      const sessionFile = update.sessionFile;
      if (!this.isSessionFileForAgent(sessionFile)) {
        return;
      }
      this.scheduleSessionDirty(sessionFile);
    });
  }

  private scheduleSessionDirty(sessionFile: string) {
    this.sessionPendingFiles.add(sessionFile);
    if (this.sessionWatchTimer) {
      return;
    }
    this.sessionWatchTimer = setTimeout(() => {
      this.sessionWatchTimer = null;
      void this.processSessionDeltaBatch().catch((err) => {
        log.warn(`memory session delta failed: ${String(err)}`);
      });
    }, SESSION_DIRTY_DEBOUNCE_MS);
  }

  private async processSessionDeltaBatch(): Promise<void> {
    if (this.sessionPendingFiles.size === 0) {
      return;
    }
    const pending = Array.from(this.sessionPendingFiles);
    this.sessionPendingFiles.clear();
    let shouldSync = false;
    for (const sessionFile of pending) {
      const delta = await this.updateSessionDelta(sessionFile);
      if (!delta) {
        continue;
      }
      const bytesThreshold = delta.deltaBytes;
      const messagesThreshold = delta.deltaMessages;
      const bytesHit =
        bytesThreshold <= 0 ? delta.pendingBytes > 0 : delta.pendingBytes >= bytesThreshold;
      const messagesHit =
        messagesThreshold <= 0
          ? delta.pendingMessages > 0
          : delta.pendingMessages >= messagesThreshold;
      if (!bytesHit && !messagesHit) {
        continue;
      }
      this.sessionsDirtyFiles.add(sessionFile);
      this.sessionsDirty = true;
      delta.pendingBytes =
        bytesThreshold > 0 ? Math.max(0, delta.pendingBytes - bytesThreshold) : 0;
      delta.pendingMessages =
        messagesThreshold > 0 ? Math.max(0, delta.pendingMessages - messagesThreshold) : 0;
      shouldSync = true;
    }
    if (shouldSync) {
      this.scheduleBackgroundSync("session-delta");
    }
  }

  private async updateSessionDelta(sessionFile: string): Promise<{
    deltaBytes: number;
    deltaMessages: number;
    pendingBytes: number;
    pendingMessages: number;
  } | null> {
    const thresholds = this.settings.sync.sessions;
    if (!thresholds) {
      return null;
    }
    let stat: { size: number };
    try {
      stat = await fs.stat(sessionFile);
    } catch {
      return null;
    }
    const size = stat.size;
    let state = this.sessionDeltas.get(sessionFile);
    if (!state) {
      state = { lastSize: 0, pendingBytes: 0, pendingMessages: 0 };
      this.sessionDeltas.set(sessionFile, state);
    }
    const deltaBytes = Math.max(0, size - state.lastSize);
    if (deltaBytes === 0 && size === state.lastSize) {
      return {
        deltaBytes: thresholds.deltaBytes,
        deltaMessages: thresholds.deltaMessages,
        pendingBytes: state.pendingBytes,
        pendingMessages: state.pendingMessages,
      };
    }
    if (size < state.lastSize) {
      state.lastSize = size;
      state.pendingBytes += size;
      const shouldCountMessages =
        thresholds.deltaMessages > 0 &&
        (thresholds.deltaBytes <= 0 || state.pendingBytes < thresholds.deltaBytes);
      if (shouldCountMessages) {
        state.pendingMessages += await this.countNewlines(sessionFile, 0, size);
      }
    } else {
      state.pendingBytes += deltaBytes;
      const shouldCountMessages =
        thresholds.deltaMessages > 0 &&
        (thresholds.deltaBytes <= 0 || state.pendingBytes < thresholds.deltaBytes);
      if (shouldCountMessages) {
        state.pendingMessages += await this.countNewlines(sessionFile, state.lastSize, size);
      }
      state.lastSize = size;
    }
    this.sessionDeltas.set(sessionFile, state);
    return {
      deltaBytes: thresholds.deltaBytes,
      deltaMessages: thresholds.deltaMessages,
      pendingBytes: state.pendingBytes,
      pendingMessages: state.pendingMessages,
    };
  }

  private async countNewlines(absPath: string, start: number, end: number): Promise<number> {
    if (end <= start) {
      return 0;
    }
    let handle;
    try {
      handle = await fs.open(absPath, "r");
    } catch (err) {
      if (isFileMissingError(err)) {
        return 0;
      }
      throw err;
    }
    try {
      let offset = start;
      let count = 0;
      const buffer = Buffer.alloc(SESSION_DELTA_READ_CHUNK_BYTES);
      while (offset < end) {
        const toRead = Math.min(buffer.length, end - offset);
        const { bytesRead } = await handle.read(buffer, 0, toRead, offset);
        if (bytesRead <= 0) {
          break;
        }
        for (let i = 0; i < bytesRead; i += 1) {
          if (buffer[i] === 10) {
            count += 1;
          }
        }
        offset += bytesRead;
      }
      return count;
    } finally {
      await handle.close();
    }
  }

  private resetSessionDelta(absPath: string, size: number): void {
    const state = this.sessionDeltas.get(absPath);
    if (!state) {
      return;
    }
    state.lastSize = size;
    state.pendingBytes = 0;
    state.pendingMessages = 0;
  }

  private isSessionFileForAgent(sessionFile: string): boolean {
    if (!sessionFile) {
      return false;
    }
    const sessionsDir = resolveSessionTranscriptsDirForAgent(this.agentId);
    const resolvedFile = path.resolve(sessionFile);
    const resolvedDir = path.resolve(sessionsDir);
    return resolvedFile.startsWith(`${resolvedDir}${path.sep}`);
  }

  protected scheduleBackgroundSync(reason: string, delayMs = 250): void {
    if (this.closed) {
      return;
    }
    this.backgroundSyncReasons.add(reason);
    if (this.backgroundSyncTimer) {
      return;
    }
    this.backgroundSyncTimer = setTimeout(() => {
      this.backgroundSyncTimer = null;
      const reasons = Array.from(this.backgroundSyncReasons);
      this.backgroundSyncReasons.clear();
      const reasonLabel = reasons.length ? `bg:${reasons.join(",")}` : "bg";
      void this.sync({ reason: reasonLabel }).catch((err) => {
        log.warn(`memory sync failed (${reasonLabel}): ${String(err)}`);
      });
    }, delayMs);
  }

  protected ensureIntervalSync() {
    const minutes = this.settings.sync.intervalMinutes;
    if (!minutes || minutes <= 0 || this.intervalTimer) {
      return;
    }
    const ms = minutes * 60 * 1000;
    this.intervalTimer = setInterval(() => {
      this.scheduleBackgroundSync("interval", 0);
    }, ms);
  }

  private scheduleWatchSync() {
    if (!this.sources.has("memory") || !this.settings.sync.watch) {
      return;
    }
    if (this.watchTimer) {
      clearTimeout(this.watchTimer);
    }
    this.watchTimer = setTimeout(() => {
      this.watchTimer = null;
      this.scheduleBackgroundSync("watch", 0);
    }, this.settings.sync.watchDebounceMs);
  }

  private shouldSyncSessions(
    params?: { reason?: string; force?: boolean },
    needsFullReindex = false,
  ) {
    if (!this.sources.has("sessions")) {
      return false;
    }
    if (params?.force) {
      return true;
    }
    const reason = params?.reason;
    if (reason === "session-start" || reason === "watch") {
      return false;
    }
    if (needsFullReindex) {
      return true;
    }
    return this.sessionsDirty && this.sessionsDirtyFiles.size > 0;
  }

  private async syncMemoryFiles(params: {
    needsFullReindex: boolean;
    progress?: MemorySyncProgressState;
    scope?: "all" | "core-only" | "projects-only";
  }) {
    // FTS-only mode: skip embedding sync (no provider)
    if (!this.provider) {
      log.debug("Skipping memory file sync in FTS-only mode (no embedding provider)");
      return;
    }

    const discoverStart = Date.now();
    const files = await listMemoryFiles(this.workspaceDir, this.settings.extraPaths);
    const coreFiles: string[] = [];
    const projectFiles = new Map<string, string[]>();
    for (const absPath of files) {
      const projectId = this.resolveProjectIdForMemoryFile(absPath);
      if (!projectId) {
        coreFiles.push(absPath);
        continue;
      }
      const list = projectFiles.get(projectId) ?? [];
      list.push(absPath);
      projectFiles.set(projectId, list);
    }

    // Fix: Warn when project routing exists but most files land in core DB,
    // which suggests the routing regex isn't matching nested .openclaw_kb paths.
    if (
      projectFiles.size > 0 &&
      coreFiles.length > 5000 &&
      !this.settings.suppressCoreFileWarning
    ) {
      const projectFileCount = [...projectFiles.values()].reduce(
        (sum, files) => sum + files.length,
        0,
      );
      const totalFiles = coreFiles.length + projectFileCount;
      const coreRatio = coreFiles.length / totalFiles;

      if (coreRatio > 0.5) {
        log.warn(
          `memory sync: ${coreFiles.length} files (${(coreRatio * 100).toFixed(0)}%) routed to core DB ` +
            `with ${projectFiles.size} project DB(s) active. ` +
            `Imbalanced distribution may indicate misconfigured routing. ` +
            `Verify .openclaw_kb paths are under Projects/ or skills/ directories. ` +
            `(Set suppressCoreFileWarning: true to dismiss.)`,
        );
      }
    }

    const scope = params.scope ?? "all";
    const includeCore = scope !== "projects-only";
    const includeProjects = scope !== "core-only";
    const projectFileCount = Array.from(projectFiles.values()).reduce(
      (acc, items) => acc + items.length,
      0,
    );
    const selectedFileCount =
      (includeCore ? coreFiles.length : 0) + (includeProjects ? projectFileCount : 0);

    const discoverMs = Date.now() - discoverStart;
    log.debug("memory sync: indexing memory files", {
      scope,
      files: files.length,
      selected_files: selectedFileCount,
      coreFiles: coreFiles.length,
      projectFiles: projectFileCount,
      projects: projectFiles.size,
      discover_ms: discoverMs,
      needsFullReindex: params.needsFullReindex,
      batch: this.batch.enabled,
      concurrency: this.getIndexConcurrency(),
    });

    if (params.progress) {
      params.progress.total += selectedFileCount;
      params.progress.report({
        completed: params.progress.completed,
        total: params.progress.total,
        label:
          scope === "core-only"
            ? "Indexing core memory…"
            : scope === "projects-only"
              ? "Indexing project memory…"
              : this.batch.enabled
                ? "Indexing memory files (batch)..."
                : "Indexing memory files…",
      });
    }

    if (includeCore) {
      const coreActivePaths = new Set(
        coreFiles.map((file) => path.relative(this.workspaceDir, file).replace(/\\/g, "/")),
      );
      await this.syncMemoryFilesForDb({
        files: coreFiles,
        activePaths: coreActivePaths,
        db: this.db,
        needsFullReindex: params.needsFullReindex,
        progress: params.progress,
        target: "core",
        vectorEnabled: true,
      });
    }

    if (includeProjects) {
      for (const [projectId, projectAbsFiles] of projectFiles.entries()) {
        const projectDbPath = this.resolveProjectDbPathForSync(projectId);
        const projectDb = this.openDatabaseAtPath(projectDbPath);
        try {
          ensureMemoryIndexSchema({
            db: projectDb,
            embeddingCacheTable: EMBEDDING_CACHE_TABLE,
            ftsTable: FTS_TABLE,
            ftsEnabled: this.fts.enabled,
          });
          const projectActivePaths = new Set(
            projectAbsFiles.map((file) =>
              path.relative(this.workspaceDir, file).replace(/\\/g, "/"),
            ),
          );
          await this.syncMemoryFilesForDb({
            files: projectAbsFiles,
            activePaths: projectActivePaths,
            db: projectDb,
            needsFullReindex: params.needsFullReindex,
            progress: params.progress,
            target: "project",
            projectId,
            vectorEnabled: true,
          });
        } finally {
          try {
            projectDb.close();
          } catch {}
        }
      }
    }

    this.logPerf("syncMemoryFiles", {
      scope,
      discover_ms: discoverMs,
      selected_files: selectedFileCount,
      core_files: includeCore ? coreFiles.length : 0,
      project_files: includeProjects ? projectFileCount : 0,
      projects: includeProjects ? projectFiles.size : 0,
      needs_full_reindex: params.needsFullReindex,
      total_ms: Date.now() - discoverStart,
    });
  }

  private async syncMemoryFilesForDb(params: {
    files: string[];
    activePaths: Set<string>;
    db: DatabaseSync;
    needsFullReindex: boolean;
    progress?: MemorySyncProgressState;
    target: "core" | "project";
    projectId?: string;
    vectorEnabled: boolean;
  }): Promise<void> {
    const syncStart = Date.now();
    let hashCheckMs = 0;
    let indexMs = 0;
    const cachedRows = params.db
      .prepare(`SELECT path, hash, mtime, size FROM files WHERE source = ?`)
      .all("memory") as Array<{ path: string; hash: string; mtime: number; size: number }>;
    const cachedByPath = new Map(cachedRows.map((row) => [row.path, row]));

    const tasks = params.files.map((absPath) => async () => {
      const relPath = path.relative(this.workspaceDir, absPath).replace(/\\/g, "/");
      const cached = cachedByPath.get(relPath);

      if (!params.needsFullReindex) {
        const statStart = Date.now();
        try {
          const stat = await fs.stat(absPath);
          if (cached && cached.mtime === stat.mtimeMs && cached.size === stat.size) {
            if (params.progress) {
              params.progress.completed += 1;
              params.progress.report({
                completed: params.progress.completed,
                total: params.progress.total,
              });
            }
            return;
          }
        } catch (err) {
          if (isFileMissingError(err)) {
            if (params.progress) {
              params.progress.completed += 1;
              params.progress.report({
                completed: params.progress.completed,
                total: params.progress.total,
              });
            }
            return;
          }
          throw err;
        } finally {
          hashCheckMs += Date.now() - statStart;
        }
      }

      const buildStart = Date.now();
      const entry = await buildFileEntry(absPath, this.workspaceDir);
      hashCheckMs += Date.now() - buildStart;
      if (!entry) {
        if (params.progress) {
          params.progress.completed += 1;
          params.progress.report({
            completed: params.progress.completed,
            total: params.progress.total,
          });
        }
        return;
      }

      if (!params.needsFullReindex && cached?.hash === entry.hash) {
        if (params.progress) {
          params.progress.completed += 1;
          params.progress.report({
            completed: params.progress.completed,
            total: params.progress.total,
          });
        }
        return;
      }

      const indexStart = Date.now();
      await this.indexFileToDb(entry, { source: "memory" }, params.db, {
        vectorEnabled: params.vectorEnabled,
      });
      indexMs += Date.now() - indexStart;
      if (params.progress) {
        params.progress.completed += 1;
        params.progress.report({
          completed: params.progress.completed,
          total: params.progress.total,
        });
      }
    });
    await runWithConcurrency(tasks, this.getIndexConcurrency());

    const cleanupStart = Date.now();
    const staleRows = params.db
      .prepare(`SELECT path FROM files WHERE source = ?`)
      .all("memory") as Array<{ path: string }>;
    let staleProcessed = 0;
    for (const stale of staleRows) {
      staleProcessed += 1;
      if (staleProcessed % 50 === 0) {
        await new Promise<void>((resolve) => setImmediate(resolve));
      }
      if (params.activePaths.has(stale.path)) {
        continue;
      }
      params.db
        .prepare(`DELETE FROM files WHERE path = ? AND source = ?`)
        .run(stale.path, "memory");
      try {
        params.db
          .prepare(
            `DELETE FROM ${VECTOR_TABLE} WHERE id IN (SELECT id FROM chunks WHERE path = ? AND source = ?)`,
          )
          .run(stale.path, "memory");
      } catch {}
      params.db
        .prepare(`DELETE FROM chunks WHERE path = ? AND source = ?`)
        .run(stale.path, "memory");
      if (this.fts.enabled && this.fts.available) {
        try {
          params.db
            .prepare(`DELETE FROM ${FTS_TABLE} WHERE path = ? AND source = ? AND model = ?`)
            .run(stale.path, "memory", this.provider?.model ?? "");
        } catch {}
      }
    }

    const cleanupMs = Date.now() - cleanupStart;
    log.debug("memory sync: db target complete", {
      db_target: params.target,
      project_id: params.projectId,
      indexed_files: params.files.length,
      stale_rows_scanned: staleRows.length,
      discover_ms: 0,
      hash_check_ms: hashCheckMs,
      index_ms: indexMs,
      cleanup_ms: cleanupMs,
      total_ms: Date.now() - syncStart,
      vector_enabled: params.vectorEnabled,
    });
  }

  private async syncSessionFiles(params: {
    needsFullReindex: boolean;
    progress?: MemorySyncProgressState;
  }) {
    const startedAt = Date.now();
    let hashCheckMs = 0;
    let indexMs = 0;

    // FTS-only mode: skip embedding sync (no provider)
    if (!this.provider) {
      log.debug("Skipping session file sync in FTS-only mode (no embedding provider)");
      return;
    }

    const files = await listSessionFilesForAgent(this.agentId);
    const activePaths = new Set(files.map((file) => sessionPathForFile(file)));
    const indexAll = params.needsFullReindex || this.sessionsDirtyFiles.size === 0;
    log.debug("memory sync: indexing session files", {
      files: files.length,
      indexAll,
      dirtyFiles: this.sessionsDirtyFiles.size,
      batch: this.batch.enabled,
      concurrency: this.getIndexConcurrency(),
    });
    if (params.progress) {
      params.progress.total += files.length;
      params.progress.report({
        completed: params.progress.completed,
        total: params.progress.total,
        label: this.batch.enabled ? "Indexing session files (batch)..." : "Indexing session files…",
      });
    }

    const cachedRows = this.db
      .prepare(`SELECT path, hash, mtime, size FROM files WHERE source = ?`)
      .all("sessions") as Array<{ path: string; hash: string; mtime: number; size: number }>;
    const cachedByPath = new Map(cachedRows.map((row) => [row.path, row]));

    const tasks = files.map((absPath) => async () => {
      const relPath = sessionPathForFile(absPath);
      const cached = cachedByPath.get(relPath);

      if (!indexAll && !this.sessionsDirtyFiles.has(absPath)) {
        if (params.progress) {
          params.progress.completed += 1;
          params.progress.report({
            completed: params.progress.completed,
            total: params.progress.total,
          });
        }
        return;
      }

      if (!params.needsFullReindex) {
        const statStart = Date.now();
        try {
          const stat = await fs.stat(absPath);
          if (cached && cached.mtime === stat.mtimeMs && cached.size === stat.size) {
            this.resetSessionDelta(absPath, stat.size);
            if (params.progress) {
              params.progress.completed += 1;
              params.progress.report({
                completed: params.progress.completed,
                total: params.progress.total,
              });
            }
            return;
          }
        } catch (err) {
          if (isFileMissingError(err)) {
            if (params.progress) {
              params.progress.completed += 1;
              params.progress.report({
                completed: params.progress.completed,
                total: params.progress.total,
              });
            }
            return;
          }
          throw err;
        } finally {
          hashCheckMs += Date.now() - statStart;
        }
      }

      const buildStart = Date.now();
      const entry = await buildSessionEntry(absPath);
      hashCheckMs += Date.now() - buildStart;
      if (!entry) {
        if (params.progress) {
          params.progress.completed += 1;
          params.progress.report({
            completed: params.progress.completed,
            total: params.progress.total,
          });
        }
        return;
      }

      if (!params.needsFullReindex && cached?.hash === entry.hash) {
        if (params.progress) {
          params.progress.completed += 1;
          params.progress.report({
            completed: params.progress.completed,
            total: params.progress.total,
          });
        }
        this.resetSessionDelta(absPath, entry.size);
        return;
      }

      const indexStart = Date.now();
      await this.indexFileToDb(entry, { source: "sessions", content: entry.content }, this.db, {
        vectorEnabled: true,
      });
      indexMs += Date.now() - indexStart;
      this.resetSessionDelta(absPath, entry.size);
      if (params.progress) {
        params.progress.completed += 1;
        params.progress.report({
          completed: params.progress.completed,
          total: params.progress.total,
        });
      }
    });
    await runWithConcurrency(tasks, this.getIndexConcurrency());

    const cleanupStart = Date.now();
    const staleRows = this.db
      .prepare(`SELECT path FROM files WHERE source = ?`)
      .all("sessions") as Array<{ path: string }>;
    let staleProcessed = 0;
    for (const stale of staleRows) {
      staleProcessed += 1;
      if (staleProcessed % 50 === 0) {
        await new Promise<void>((resolve) => setImmediate(resolve));
      }
      if (activePaths.has(stale.path)) {
        continue;
      }
      this.db
        .prepare(`DELETE FROM files WHERE path = ? AND source = ?`)
        .run(stale.path, "sessions");
      try {
        this.db
          .prepare(
            `DELETE FROM ${VECTOR_TABLE} WHERE id IN (SELECT id FROM chunks WHERE path = ? AND source = ?)`,
          )
          .run(stale.path, "sessions");
      } catch {}
      this.db
        .prepare(`DELETE FROM chunks WHERE path = ? AND source = ?`)
        .run(stale.path, "sessions");
      if (this.fts.enabled && this.fts.available) {
        try {
          this.db
            .prepare(`DELETE FROM ${FTS_TABLE} WHERE path = ? AND source = ? AND model = ?`)
            .run(stale.path, "sessions", this.provider.model);
        } catch {}
      }
    }

    this.logPerf("syncSessionFiles", {
      files: files.length,
      dirty_files: this.sessionsDirtyFiles.size,
      index_all: indexAll,
      hash_check_ms: hashCheckMs,
      index_ms: indexMs,
      cleanup_ms: Date.now() - cleanupStart,
      total_ms: Date.now() - startedAt,
    });
  }

  private createSyncProgress(
    onProgress: (update: MemorySyncProgressUpdate) => void,
  ): MemorySyncProgressState {
    const state: MemorySyncProgressState = {
      completed: 0,
      total: 0,
      label: undefined,
      report: (update) => {
        if (update.label) {
          state.label = update.label;
        }
        const label =
          update.total > 0 && state.label
            ? `${state.label} ${update.completed}/${update.total}`
            : state.label;
        onProgress({
          completed: update.completed,
          total: update.total,
          label,
        });
      },
    };
    return state;
  }

  protected async runSync(params?: {
    reason?: string;
    force?: boolean;
    progress?: (update: MemorySyncProgressUpdate) => void;
  }) {
    const startedAt = Date.now();
    let vectorReady = false;
    let needsFullReindex = false;
    let shouldSyncMemory = false;
    let shouldSyncSessions = false;
    const progress = params?.progress ? this.createSyncProgress(params.progress) : undefined;

    // Mark sync as in progress and clear any previous deferral
    this.syncInProgress = true;
    this.syncDeferred = false;
    this.syncDeferredReason = undefined;

    if (progress) {
      progress.report({
        completed: progress.completed,
        total: progress.total,
        label: "Loading vector extension…",
      });
    }
    vectorReady = await this.ensureVectorReady();
    const meta = this.readMeta();
    const configuredSources = this.resolveConfiguredSourcesForMeta();
    needsFullReindex =
      params?.force ||
      !meta ||
      (this.provider && meta.model !== this.provider.model) ||
      (this.provider && meta.provider !== this.provider.id) ||
      meta.providerKey !== this.providerKey ||
      this.metaSourcesDiffer(meta, configuredSources) ||
      meta.chunkTokens !== this.settings.chunking.tokens ||
      meta.chunkOverlap !== this.settings.chunking.overlap ||
      (vectorReady && !meta?.vectorDims);
    try {
      if (needsFullReindex) {
        if (
          process.env.OPENCLAW_TEST_FAST === "1" &&
          process.env.OPENCLAW_TEST_MEMORY_UNSAFE_REINDEX === "1"
        ) {
          await this.runUnsafeReindex({
            reason: params?.reason,
            force: params?.force,
            progress: progress ?? undefined,
          });
        } else {
          await this.runSafeReindex({
            reason: params?.reason,
            force: params?.force,
            progress: progress ?? undefined,
          });
        }
        return;
      }

      shouldSyncMemory =
        this.sources.has("memory") && (params?.force || needsFullReindex || this.dirty);
      shouldSyncSessions = this.shouldSyncSessions(params, needsFullReindex);

      if (shouldSyncMemory) {
        await this.syncMemoryFiles({ needsFullReindex, progress: progress ?? undefined });
        this.dirty = false;
      }

      if (shouldSyncSessions) {
        await this.syncSessionFiles({ needsFullReindex, progress: progress ?? undefined });
        this.sessionsDirty = false;
        this.sessionsDirtyFiles.clear();
      } else if (this.sessionsDirtyFiles.size > 0) {
        this.sessionsDirty = true;
      } else {
        this.sessionsDirty = false;
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      const activated =
        this.shouldFallbackOnError(reason) && (await this.activateFallbackProvider(reason));
      if (activated) {
        await this.runSafeReindex({
          reason: params?.reason ?? "fallback",
          force: true,
          progress: progress ?? undefined,
        });
        return;
      }
      throw err;
    } finally {
      this.logPerf("runSync", {
        reason: params?.reason,
        force: Boolean(params?.force),
        vector_ready: vectorReady,
        needs_full_reindex: needsFullReindex,
        sync_memory: shouldSyncMemory,
        sync_sessions: shouldSyncSessions,
        total_ms: Date.now() - startedAt,
      });
    }
  }

  private shouldFallbackOnError(message: string): boolean {
    return /embedding|embeddings|batch/i.test(message);
  }

  protected resolveBatchConfig(): {
    enabled: boolean;
    wait: boolean;
    concurrency: number;
    pollIntervalMs: number;
    timeoutMs: number;
  } {
    const batch = this.settings.remote?.batch;
    const enabled = Boolean(
      batch?.enabled &&
      this.provider &&
      ((this.openAi && this.provider.id === "openai") ||
        (this.gemini && this.provider.id === "gemini") ||
        (this.voyage && this.provider.id === "voyage")),
    );
    return {
      enabled,
      wait: batch?.wait ?? true,
      concurrency: Math.max(1, batch?.concurrency ?? 2),
      pollIntervalMs: batch?.pollIntervalMs ?? 2000,
      timeoutMs: (batch?.timeoutMinutes ?? 60) * 60 * 1000,
    };
  }

  private async activateFallbackProvider(reason: string): Promise<boolean> {
    const fallback = this.settings.fallback;
    if (!fallback || fallback === "none" || !this.provider || fallback === this.provider.id) {
      return false;
    }
    if (this.fallbackFrom) {
      return false;
    }
    const fallbackFrom = this.provider.id as "openai" | "gemini" | "local" | "voyage" | "mistral";

    const fallbackModel =
      fallback === "gemini"
        ? DEFAULT_GEMINI_EMBEDDING_MODEL
        : fallback === "openai"
          ? DEFAULT_OPENAI_EMBEDDING_MODEL
          : fallback === "voyage"
            ? DEFAULT_VOYAGE_EMBEDDING_MODEL
            : fallback === "mistral"
              ? DEFAULT_MISTRAL_EMBEDDING_MODEL
              : this.settings.model;

    const fallbackResult = await createEmbeddingProvider({
      config: this.cfg,
      agentDir: resolveAgentDir(this.cfg, this.agentId),
      provider: fallback,
      remote: this.settings.remote,
      model: fallbackModel,
      fallback: "none",
      local: this.settings.local,
    });

    this.fallbackFrom = fallbackFrom;
    this.fallbackReason = reason;
    this.provider = fallbackResult.provider;
    this.openAi = fallbackResult.openAi;
    this.gemini = fallbackResult.gemini;
    this.voyage = fallbackResult.voyage;
    this.mistral = fallbackResult.mistral;
    this.providerKey = this.computeProviderKey();
    this.batch = this.resolveBatchConfig();
    log.warn(`memory embeddings: switched to fallback provider (${fallback})`, { reason });
    return true;
  }

  private async runSafeReindex(params: {
    reason?: string;
    force?: boolean;
    progress?: MemorySyncProgressState;
  }): Promise<void> {
    const startedAt = Date.now();
    const dbPath = resolveUserPath(this.settings.store.corePath || this.settings.store.path);
    const tempDbPath = `${dbPath}.tmp-${randomUUID()}`;
    const tempDb = this.openDatabaseAtPath(tempDbPath);

    const originalDb = this.db;
    let originalDbClosed = false;
    const originalState = {
      ftsAvailable: this.fts.available,
      ftsError: this.fts.loadError,
      vectorAvailable: this.vector.available,
      vectorLoadError: this.vector.loadError,
      vectorDims: this.vector.dims,
      vectorReady: this.vectorReady,
    };

    const restoreOriginalState = () => {
      if (originalDbClosed) {
        this.db = this.openDatabaseAtPath(dbPath);
      } else {
        this.db = originalDb;
      }
      this.fts.available = originalState.ftsAvailable;
      this.fts.loadError = originalState.ftsError;
      this.vector.available = originalDbClosed ? null : originalState.vectorAvailable;
      this.vector.loadError = originalState.vectorLoadError;
      this.vector.dims = originalState.vectorDims;
      this.vectorReady = originalDbClosed ? null : originalState.vectorReady;
    };

    this.db = tempDb;
    this.vectorReady = null;
    this.vector.available = null;
    this.vector.loadError = undefined;
    this.vector.dims = undefined;
    this.fts.available = false;
    this.fts.loadError = undefined;
    this.ensureSchema();

    let nextMeta: MemoryIndexMeta | null = null;

    try {
      this.seedEmbeddingCache(originalDb);
      const shouldSyncMemory = this.sources.has("memory");
      const shouldSyncSessions = this.shouldSyncSessions(
        { reason: params.reason, force: params.force },
        true,
      );
      const publishStart = Date.now();

      // Two-stage reindex: core memory is published first so the agent has
      // continuity/recall while the (potentially large) project corpus indexes
      // in stage 2.  If stage 2 fails, the catch block restores the original
      // DB and throws — this.dirty remains true (it's only set to false after
      // stage 2 succeeds at the end of the try block), so the next sync cycle
      // will retry a full reindex rather than leaving a partial project index.
      //
      // Stage 1: build and publish core continuity memory first.
      if (shouldSyncMemory) {
        await this.syncMemoryFiles({
          needsFullReindex: true,
          progress: params.progress,
          scope: "core-only",
        });
      }

      if (shouldSyncSessions) {
        await this.syncSessionFiles({ needsFullReindex: true, progress: params.progress });
        this.sessionsDirty = false;
        this.sessionsDirtyFiles.clear();
      } else if (this.sessionsDirtyFiles.size > 0) {
        this.sessionsDirty = true;
      } else {
        this.sessionsDirty = false;
      }

      nextMeta = {
        model: this.provider?.model ?? "fts-only",
        provider: this.provider?.id ?? "none",
        providerKey: this.providerKey!,
        sources: this.resolveConfiguredSourcesForMeta(),
        chunkTokens: this.settings.chunking.tokens,
        chunkOverlap: this.settings.chunking.overlap,
      };
      if (!nextMeta) {
        throw new Error("Failed to compute memory index metadata for reindexing.");
      }

      if (this.vector.available && this.vector.dims) {
        nextMeta.vectorDims = this.vector.dims;
      }

      this.writeMeta(nextMeta);
      this.pruneEmbeddingCacheIfNeeded?.();

      this.db.close();
      originalDb.close();
      originalDbClosed = true;

      await this.swapIndexFiles(dbPath, tempDbPath);

      this.db = this.openDatabaseAtPath(dbPath);
      this.vectorReady = null;
      this.vector.available = null;
      this.vector.loadError = undefined;
      this.ensureSchema();
      this.vector.dims = nextMeta?.vectorDims;

      this.logPerf("runSafeReindex_corePublish", {
        reason: params.reason,
        force: Boolean(params.force),
        total_ms: Date.now() - publishStart,
      });

      // Stage 2: continue project corpus indexing without blocking core publication.
      if (shouldSyncMemory) {
        await this.syncMemoryFiles({
          needsFullReindex: true,
          progress: params.progress,
          scope: "projects-only",
        });
      }
      this.dirty = false;
    } catch (err) {
      try {
        this.db.close();
      } catch {}
      await this.removeIndexFiles(tempDbPath);
      restoreOriginalState();
      throw err;
    } finally {
      this.logPerf("runSafeReindex", {
        reason: params.reason,
        force: Boolean(params.force),
        total_ms: Date.now() - startedAt,
      });
    }
  }

  private async runUnsafeReindex(params: {
    reason?: string;
    force?: boolean;
    progress?: MemorySyncProgressState;
  }): Promise<void> {
    const startedAt = Date.now();
    // Perf: for test runs, skip atomic temp-db swapping. The index is isolated
    // under the per-test HOME anyway, and this cuts substantial fs+sqlite churn.
    this.resetIndex();

    const shouldSyncMemory = this.sources.has("memory");
    const shouldSyncSessions = this.shouldSyncSessions(
      { reason: params.reason, force: params.force },
      true,
    );

    if (shouldSyncMemory) {
      await this.syncMemoryFiles({ needsFullReindex: true, progress: params.progress });
      this.dirty = false;
    }

    if (shouldSyncSessions) {
      await this.syncSessionFiles({ needsFullReindex: true, progress: params.progress });
      this.sessionsDirty = false;
      this.sessionsDirtyFiles.clear();
    } else if (this.sessionsDirtyFiles.size > 0) {
      this.sessionsDirty = true;
    } else {
      this.sessionsDirty = false;
    }

    const nextMeta: MemoryIndexMeta = {
      model: this.provider?.model ?? "fts-only",
      provider: this.provider?.id ?? "none",
      providerKey: this.providerKey!,
      sources: this.resolveConfiguredSourcesForMeta(),
      chunkTokens: this.settings.chunking.tokens,
      chunkOverlap: this.settings.chunking.overlap,
    };
    if (this.vector.available && this.vector.dims) {
      nextMeta.vectorDims = this.vector.dims;
    }

    this.writeMeta(nextMeta);
    this.pruneEmbeddingCacheIfNeeded?.();

    this.logPerf("runUnsafeReindex", {
      reason: params.reason,
      force: Boolean(params.force),
      total_ms: Date.now() - startedAt,
    });
  }

  private resetIndex() {
    this.db.exec(`DELETE FROM files`);
    this.db.exec(`DELETE FROM chunks`);
    if (this.fts.enabled && this.fts.available) {
      try {
        this.db.exec(`DELETE FROM ${FTS_TABLE}`);
      } catch {}
    }
    this.dropVectorTable();
    this.vector.dims = undefined;
    this.sessionsDirtyFiles.clear();
  }

  protected readMeta(): MemoryIndexMeta | null {
    const row = this.db.prepare(`SELECT value FROM meta WHERE key = ?`).get(META_KEY) as
      | { value: string }
      | undefined;
    if (!row?.value) {
      return null;
    }
    try {
      return JSON.parse(row.value) as MemoryIndexMeta;
    } catch {
      return null;
    }
  }

  protected writeMeta(meta: MemoryIndexMeta) {
    const value = JSON.stringify(meta);
    this.db
      .prepare(
        `INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
      )
      .run(META_KEY, value);
  }

  private resolveConfiguredSourcesForMeta(): MemorySource[] {
    const normalized = Array.from(this.sources)
      .filter((source): source is MemorySource => source === "memory" || source === "sessions")
      .toSorted();
    return normalized.length > 0 ? normalized : ["memory"];
  }

  private normalizeMetaSources(meta: MemoryIndexMeta): MemorySource[] {
    if (!Array.isArray(meta.sources)) {
      // Backward compatibility for older indexes that did not persist sources.
      return ["memory"];
    }
    const normalized = Array.from(
      new Set(
        meta.sources.filter(
          (source): source is MemorySource => source === "memory" || source === "sessions",
        ),
      ),
    ).toSorted();
    return normalized.length > 0 ? normalized : ["memory"];
  }

  private metaSourcesDiffer(meta: MemoryIndexMeta, configuredSources: MemorySource[]): boolean {
    const metaSources = this.normalizeMetaSources(meta);
    if (metaSources.length !== configuredSources.length) {
      return true;
    }
    return metaSources.some((source, index) => source !== configuredSources[index]);
  }
}
