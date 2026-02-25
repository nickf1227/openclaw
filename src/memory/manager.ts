import fs from "node:fs/promises";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { type FSWatcher } from "chokidar";
import { resolveAgentDir, resolveAgentWorkspaceDir } from "../agents/agent-scope.js";
import type { ResolvedMemorySearchConfig } from "../agents/memory-search.js";
import { resolveMemorySearchConfig } from "../agents/memory-search.js";
import type { OpenClawConfig } from "../config/config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import {
  createEmbeddingProvider,
  type EmbeddingProvider,
  type EmbeddingProviderResult,
  type GeminiEmbeddingClient,
  type MistralEmbeddingClient,
  type OpenAiEmbeddingClient,
  type VoyageEmbeddingClient,
} from "./embeddings.js";
import { isFileMissingError, statRegularFile } from "./fs-utils.js";
import { bm25RankToScore, buildFtsQuery, mergeHybridResults } from "./hybrid.js";
import { isMemoryPath, normalizeExtraMemoryPaths } from "./internal.js";
import { MemoryManagerEmbeddingOps } from "./manager-embedding-ops.js";
import { searchKeyword, searchVector } from "./manager-search.js";
import { extractKeywords } from "./query-expansion.js";
import type {
  MemoryEmbeddingProbeResult,
  MemoryProviderStatus,
  MemorySearchManager,
  MemorySearchResult,
  MemorySource,
  MemorySyncProgressUpdate,
} from "./types.js";
const SNIPPET_MAX_CHARS = 700;
const VECTOR_TABLE = "chunks_vec";
const FTS_TABLE = "chunks_fts";
const EMBEDDING_CACHE_TABLE = "embedding_cache";
const BATCH_FAILURE_LIMIT = 2;
const SEARCH_SYNC_MIN_INTERVAL_MS = 60_000;
const VECTOR_FALLBACK_WARN_INTERVAL_MS = 60_000;
const REPO_BG_ENRICH_MAX_INFLIGHT = 2;
const REPO_BG_ENRICH_BUDGET_MS = 12_000;
const REPO_BG_DROP_LOG_INTERVAL_MS = 30_000;

const log = createSubsystemLogger("memory");

const INDEX_CACHE = new Map<string, MemoryIndexManager>();

type ProjectRoute = {
  id: string;
  tokens: Set<string>;
  pathBits: Set<string>;
};

type ProjectRouteMatch = {
  route: ProjectRoute;
  score: number;
};

type QueryTier = "core" | "repo" | "mixed";

export class MemoryIndexManager extends MemoryManagerEmbeddingOps implements MemorySearchManager {
  private readonly cacheKey: string;
  protected readonly cfg: OpenClawConfig;
  protected readonly agentId: string;
  protected readonly workspaceDir: string;
  protected readonly settings: ResolvedMemorySearchConfig;
  protected provider: EmbeddingProvider | null;
  private readonly requestedProvider: "openai" | "local" | "gemini" | "voyage" | "mistral" | "auto";
  protected fallbackFrom?: "openai" | "local" | "gemini" | "voyage" | "mistral";
  protected fallbackReason?: string;
  private readonly providerUnavailableReason?: string;
  protected openAi?: OpenAiEmbeddingClient;
  protected gemini?: GeminiEmbeddingClient;
  protected voyage?: VoyageEmbeddingClient;
  protected mistral?: MistralEmbeddingClient;
  protected batch: {
    enabled: boolean;
    wait: boolean;
    concurrency: number;
    pollIntervalMs: number;
    timeoutMs: number;
  };
  protected batchFailureCount = 0;
  protected batchFailureLastError?: string;
  protected batchFailureLastProvider?: string;
  protected batchFailureLock: Promise<void> = Promise.resolve();
  protected db: DatabaseSync;
  protected readonly sources: Set<MemorySource>;
  protected providerKey: string;
  protected readonly cache: { enabled: boolean; maxEntries?: number };
  protected readonly vector: {
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
  };
  protected vectorReady: Promise<boolean> | null = null;
  protected watcher: FSWatcher | null = null;
  protected watchTimer: NodeJS.Timeout | null = null;
  protected sessionWatchTimer: NodeJS.Timeout | null = null;
  protected sessionUnsubscribe: (() => void) | null = null;
  protected intervalTimer: NodeJS.Timeout | null = null;
  protected closed = false;
  protected dirty = false;
  protected sessionsDirty = false;
  protected sessionsDirtyFiles = new Set<string>();
  protected sessionPendingFiles = new Set<string>();
  protected sessionDeltas = new Map<
    string,
    { lastSize: number; pendingBytes: number; pendingMessages: number }
  >();
  private sessionWarm = new Set<string>();
  private syncing: Promise<void> | null = null;
  private lastSearchTriggeredSyncAt = 0;
  private lastLargeCorpusSkipLogAt = 0;
  private lastVectorFallbackWarnAt = 0;
  private indexedFileCountCache: { value: number; at: number } = { value: 0, at: 0 };
  private readonly queryPathHints: Map<string, string[]>;
  private readonly projectRoutes: ProjectRoute[];
  private readonly repoQueryCache = new Map<
    string,
    { at: number; results: MemorySearchResult[] }
  >();
  private readonly repoQueryJobs = new Map<string, Promise<void>>();
  private readonly projectDbs = new Map<string, DatabaseSync>();
  /** Track access order for LRU eviction of project DB handles. */
  private readonly projectDbAccess = new Map<string, number>();
  /** Maximum number of project DB handles to keep open simultaneously. */
  private static readonly MAX_PROJECT_DB_HANDLES = 16;
  private lastRepoBgDropLogAt = 0;

  static async get(params: {
    cfg: OpenClawConfig;
    agentId: string;
    purpose?: "default" | "status";
  }): Promise<MemoryIndexManager | null> {
    const { cfg, agentId } = params;
    const settings = resolveMemorySearchConfig(cfg, agentId);
    if (!settings) {
      return null;
    }
    const workspaceDir = resolveAgentWorkspaceDir(cfg, agentId);
    const key = `${agentId}:${workspaceDir}:${JSON.stringify(settings)}`;
    const statusOnly = params.purpose === "status";
    if (!statusOnly) {
      const existing = INDEX_CACHE.get(key);
      if (existing) {
        return existing;
      }
    }
    const providerResult = await createEmbeddingProvider({
      config: cfg,
      agentDir: resolveAgentDir(cfg, agentId),
      provider: settings.provider,
      remote: settings.remote,
      model: settings.model,
      fallback: settings.fallback,
      local: settings.local,
    });
    const manager = new MemoryIndexManager({
      cacheKey: key,
      cfg,
      agentId,
      workspaceDir,
      settings,
      providerResult,
      purpose: params.purpose,
    });
    if (!statusOnly) {
      INDEX_CACHE.set(key, manager);
    }
    return manager;
  }

  private constructor(params: {
    cacheKey: string;
    cfg: OpenClawConfig;
    agentId: string;
    workspaceDir: string;
    settings: ResolvedMemorySearchConfig;
    providerResult: EmbeddingProviderResult;
    purpose?: "default" | "status";
  }) {
    super();
    this.cacheKey = params.cacheKey;
    this.cfg = params.cfg;
    this.agentId = params.agentId;
    this.workspaceDir = params.workspaceDir;
    this.settings = params.settings;
    this.provider = params.providerResult.provider;
    this.requestedProvider = params.providerResult.requestedProvider;
    this.fallbackFrom = params.providerResult.fallbackFrom;
    this.fallbackReason = params.providerResult.fallbackReason;
    this.providerUnavailableReason = params.providerResult.providerUnavailableReason;
    this.openAi = params.providerResult.openAi;
    this.gemini = params.providerResult.gemini;
    this.voyage = params.providerResult.voyage;
    this.mistral = params.providerResult.mistral;
    this.sources = new Set(params.settings.sources);
    this.queryPathHints = this.buildQueryPathHints();
    this.projectRoutes = this.buildProjectRoutes();

    // Fix: Warn if split-DB routing is active but corePath defaults to path,
    // which causes project query results to mix with core memory.
    if (
      this.projectRoutes.length > 0 &&
      (!params.settings.store.corePath ||
        params.settings.store.corePath === params.settings.store.path)
    ) {
      log.warn(
        `memory: ${this.projectRoutes.length} project route(s) active but corePath is not explicitly set. ` +
          `Core and project memories share the same DB path, which may cause mixed storage. ` +
          `Set store.corePath to a dedicated path to enable proper split-DB routing.`,
      );
    }

    this.db = this.openDatabase();
    this.providerKey = this.computeProviderKey();
    this.cache = {
      enabled: params.settings.cache.enabled,
      maxEntries: params.settings.cache.maxEntries,
    };
    this.fts = { enabled: params.settings.query.hybrid.enabled, available: false };
    this.ensureSchema();
    this.vector = {
      enabled: params.settings.store.vector.enabled,
      available: null,
      extensionPath: params.settings.store.vector.extensionPath,
    };
    const meta = this.readMeta();
    if (meta?.vectorDims) {
      this.vector.dims = meta.vectorDims;
    }
    this.ensureWatcher();
    this.ensureSessionListener();
    this.ensureIntervalSync();
    const statusOnly = params.purpose === "status";
    this.dirty = this.sources.has("memory") && (statusOnly ? !meta : true);
    this.batch = this.resolveBatchConfig();
  }

  private logSyncFailure(reason: string, err: unknown): void {
    const message = err instanceof Error ? err.message : String(err);
    if (/database is locked/i.test(message)) {
      log.debug(`memory sync skipped (${reason}): ${message}`);
      return;
    }
    log.warn(`memory sync failed (${reason}): ${message}`);
  }

  async warmSession(sessionKey?: string): Promise<void> {
    if (!this.settings.sync.onSessionStart) {
      return;
    }
    const key = sessionKey?.trim() || "";
    if (key && this.sessionWarm.has(key)) {
      return;
    }
    void this.sync({ reason: "session-start" }).catch((err) => {
      this.logSyncFailure("session-start", err);
    });
    if (key) {
      this.sessionWarm.add(key);
    }
  }

  async search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      sessionKey?: string;
    },
  ): Promise<MemorySearchResult[]> {
    void this.warmSession(opts?.sessionKey);
    if (this.settings.sync.onSearch && (this.dirty || this.sessionsDirty)) {
      if (!this.shouldSkipSearchTriggeredSyncForLargeCorpus()) {
        const now = Date.now();
        if (now - this.lastSearchTriggeredSyncAt >= SEARCH_SYNC_MIN_INTERVAL_MS) {
          this.lastSearchTriggeredSyncAt = now;
          void this.sync({ reason: "search" }).catch((err) => {
            this.logSyncFailure("search", err);
          });
        }
      }
    }
    const cleaned = query.trim();
    if (!cleaned) {
      return [];
    }
    const minScore = opts?.minScore ?? this.settings.query.minScore;
    const maxResults = opts?.maxResults ?? this.settings.query.maxResults;
    const hybrid = this.settings.query.hybrid;
    const t0 = Date.now();
    let keywordMs = 0;
    let embedQueryMs = 0;
    let vectorMs = 0;
    let fusionMs = 0;
    const candidates = Math.min(
      200,
      Math.max(1, Math.floor(maxResults * hybrid.candidateMultiplier)),
    );
    const queryTier = this.detectQueryTier(cleaned);
    const pathFilter = this.resolvePathFilter(cleaned);
    const searchDb = this.getSearchDatabase(cleaned, queryTier);

    // FTS-only mode: no embedding provider available
    if (!this.provider) {
      if (!this.fts.enabled || !this.fts.available) {
        log.warn("memory search: no provider and FTS unavailable");
        return [];
      }

      // Extract keywords for better FTS matching on conversational queries
      // e.g., "that thing we discussed about the API" → ["discussed", "API"]
      const keywords = extractKeywords(cleaned);
      const searchTerms = keywords.length > 0 ? keywords : [cleaned];

      // Search with each keyword and merge results
      const resultSets = await Promise.all(
        searchTerms.map((term) =>
          this.searchKeyword(term, candidates, pathFilter, searchDb.db).catch(() => []),
        ),
      );

      // Merge and deduplicate results, keeping highest score for each chunk
      const seenIds = new Map<string, (typeof resultSets)[0][0]>();
      for (const results of resultSets) {
        for (const result of results) {
          const existing = seenIds.get(result.id);
          if (!existing || result.score > existing.score) {
            seenIds.set(result.id, result);
          }
        }
      }

      const merged = [...seenIds.values()]
        .toSorted((a, b) => b.score - a.score)
        .filter((entry) => entry.score >= minScore)
        .slice(0, maxResults);

      return merged;
    }

    const keywordStart = Date.now();
    const keywordResults = hybrid.enabled
      ? await this.searchKeyword(cleaned, candidates, pathFilter, searchDb.db).catch(() => [])
      : [];
    keywordMs = Date.now() - keywordStart;

    const keywordOnlyResults = keywordResults
      .map((item) => ({
        path: item.path,
        startLine: item.startLine,
        endLine: item.endLine,
        score: item.textScore,
        snippet: item.snippet,
        source: item.source,
      }))
      .filter((item) => item.score >= minScore)
      .slice(0, maxResults);

    if (queryTier === "repo" && keywordOnlyResults.length === 0) {
      const coreFallbackFilter = { sql: " AND path NOT LIKE ?", params: ["%/.openclaw_kb/%"] };
      const coreFallback = await this.searchKeyword(
        cleaned,
        candidates,
        coreFallbackFilter,
        this.db,
      ).catch(() => []);
      const coreFallbackResults = coreFallback
        .map((item) => ({
          path: item.path,
          startLine: item.startLine,
          endLine: item.endLine,
          score: item.textScore,
          snippet: item.snippet,
          source: item.source,
        }))
        .filter((item) => item.score >= minScore)
        .slice(0, maxResults);
      if (coreFallbackResults.length > 0) {
        log.debug("memory search: repo-empty fallback to core", {
          query: cleaned.slice(0, 120),
          results: coreFallbackResults.length,
        });
        return coreFallbackResults;
      }
    }

    const indexedFiles = this.getIndexedFileCount();
    const topKeywordScore = keywordResults[0]?.textScore ?? 0;
    const shouldUseKeywordOnly =
      hybrid.enabled &&
      indexedFiles >= this.settings.query.routing.keywordOnlyLargeCorpusFileThreshold &&
      keywordResults.length >=
        Math.max(1, Math.min(maxResults, this.settings.query.routing.keywordOnlyMinResults)) &&
      topKeywordScore >= this.settings.query.routing.keywordOnlyMinScore;

    if (shouldUseKeywordOnly) {
      log.debug("memory search: keyword-only fast path", {
        indexedFiles,
        topKeywordScore,
        results: keywordResults.length,
      });
      log.debug("memory search perf", {
        tier: queryTier,
        db_target: searchDb.target,
        project_id: searchDb.projectId,
        mode: "keyword-fast-path",
        keyword_ms: keywordMs,
        total_ms: Date.now() - t0,
        results: keywordOnlyResults.length,
      });
      return keywordOnlyResults;
    }

    if (queryTier === "repo" && hybrid.enabled) {
      const cacheKey = this.getRepoCacheKey(cleaned, maxResults, minScore);
      const cached = this.getRepoCachedResults(cacheKey);
      if (cached) {
        return cached;
      }

      if (!this.settings.query.routing.foregroundVectorEnabled) {
        if (!this.repoQueryJobs.has(cacheKey)) {
          if (this.repoQueryJobs.size >= REPO_BG_ENRICH_MAX_INFLIGHT) {
            const now = Date.now();
            if (now - this.lastRepoBgDropLogAt >= REPO_BG_DROP_LOG_INTERVAL_MS) {
              this.lastRepoBgDropLogAt = now;
              log.debug("memory repo enrich skipped (backpressure)", {
                inflight: this.repoQueryJobs.size,
                limit: REPO_BG_ENRICH_MAX_INFLIGHT,
              });
            }
          } else {
            const job = (async () => {
              let vectorResults: Array<MemorySearchResult & { id: string }> = [];
              try {
                const queryVec = await this.embedQueryWithBudget(cleaned, REPO_BG_ENRICH_BUDGET_MS);
                const hasVector = queryVec.some((v) => v !== 0);
                vectorResults = hasVector
                  ? await this.searchVector(queryVec, candidates, pathFilter, searchDb.db).catch(
                      () => [],
                    )
                  : [];
                const merged = await this.mergeHybridResults({
                  vector: vectorResults,
                  keyword: keywordResults,
                  vectorWeight: hybrid.vectorWeight,
                  textWeight: hybrid.textWeight,
                  mmr: hybrid.mmr,
                  temporalDecay: hybrid.temporalDecay,
                });
                this.repoQueryCache.set(cacheKey, {
                  at: Date.now(),
                  results: merged.filter((entry) => entry.score >= minScore).slice(0, maxResults),
                });
              } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                const now = Date.now();
                if (now - this.lastVectorFallbackWarnAt >= VECTOR_FALLBACK_WARN_INTERVAL_MS) {
                  this.lastVectorFallbackWarnAt = now;
                  log.warn(`memory vector query fallback to keyword-only: ${message}`);
                } else {
                  log.debug(`memory vector query fallback to keyword-only: ${message}`);
                }
              } finally {
                this.repoQueryJobs.delete(cacheKey);
              }
            })();
            this.repoQueryJobs.set(cacheKey, job);
          }
        }
        return keywordOnlyResults;
      }
    }

    if (!this.settings.query.routing.foregroundVectorEnabled) {
      return keywordOnlyResults;
    }

    let vectorResults: Array<MemorySearchResult & { id: string }> = [];
    try {
      const embedStart = Date.now();
      const queryVec = await this.embedQueryWithTimeout(cleaned);
      embedQueryMs = Date.now() - embedStart;
      const hasVector = queryVec.some((v) => v !== 0);
      if (hasVector) {
        const vectorStart = Date.now();
        vectorResults = await this.searchVector(
          queryVec,
          candidates,
          pathFilter,
          searchDb.db,
        ).catch(() => []);
        vectorMs = Date.now() - vectorStart;
      } else {
        vectorResults = [];
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const now = Date.now();
      if (now - this.lastVectorFallbackWarnAt >= VECTOR_FALLBACK_WARN_INTERVAL_MS) {
        this.lastVectorFallbackWarnAt = now;
        log.warn(`memory vector query fallback to keyword-only: ${message}`);
      } else {
        log.debug(`memory vector query fallback to keyword-only: ${message}`);
      }
      vectorResults = [];
    }

    if (!hybrid.enabled) {
      const out = vectorResults.filter((entry) => entry.score >= minScore).slice(0, maxResults);
      log.debug("memory search perf", {
        tier: queryTier,
        db_target: searchDb.target,
        project_id: searchDb.projectId,
        mode: "vector-only",
        keyword_ms: keywordMs,
        embed_query_ms: embedQueryMs,
        vector_ms: vectorMs,
        total_ms: Date.now() - t0,
        results: out.length,
      });
      return out;
    }

    const fusionStart = Date.now();
    const merged = await this.mergeHybridResults({
      vector: vectorResults,
      keyword: keywordResults,
      vectorWeight: hybrid.vectorWeight,
      textWeight: hybrid.textWeight,
      mmr: hybrid.mmr,
      temporalDecay: hybrid.temporalDecay,
    });

    fusionMs = Date.now() - fusionStart;
    const out = merged.filter((entry) => entry.score >= minScore).slice(0, maxResults);
    log.debug("memory search perf", {
      tier: queryTier,
      db_target: searchDb.target,
      project_id: searchDb.projectId,
      mode: "hybrid",
      keyword_ms: keywordMs,
      embed_query_ms: embedQueryMs,
      vector_ms: vectorMs,
      fusion_ms: fusionMs,
      total_ms: Date.now() - t0,
      results: out.length,
    });
    return out;
  }

  private buildQueryPathHints(): Map<string, string[]> {
    const hints = new Map<string, string[]>();
    for (const raw of this.settings.extraPaths) {
      const abs = path.resolve(this.workspaceDir, raw);
      const parts = abs.split(path.sep).filter(Boolean);
      const candidates = [parts.at(-1), parts.at(-2)].filter((v): v is string => Boolean(v));
      for (const candidate of candidates) {
        const tokens = new Set<string>();
        const compact = candidate.toLowerCase().replace(/[^a-z0-9]+/g, "");
        if (compact.length >= 4) {
          tokens.add(compact);
        }
        for (const part of candidate.toLowerCase().split(/[^a-z0-9]+/g)) {
          if (part.length >= 4) {
            tokens.add(part);
          }
        }
        for (const token of tokens) {
          const list = hints.get(token) ?? [];
          list.push(candidate);
          hints.set(token, list);
        }
      }
    }
    return hints;
  }

  private buildProjectRoutes(): ProjectRoute[] {
    const routes = new Map<string, ProjectRoute>();
    for (const raw of this.settings.extraPaths) {
      const abs = path.resolve(this.workspaceDir, raw);
      const parts = abs.split(path.sep).filter(Boolean);
      const projectName = parts.includes("Projects")
        ? (parts[parts.indexOf("Projects") + 1] ?? parts.at(-2) ?? parts.at(-1) ?? "project")
        : (parts.at(-2) ?? parts.at(-1) ?? "project");
      const id = projectName.toLowerCase().replace(/[^a-z0-9]+/g, "-");
      const entry =
        routes.get(id) ??
        ({ id, tokens: new Set<string>(), pathBits: new Set<string>() } satisfies ProjectRoute);

      const candidates = [projectName, parts.at(-2), parts.at(-1)].filter((v): v is string =>
        Boolean(v),
      );
      for (const candidate of candidates) {
        const cleaned = candidate.toLowerCase();
        const compact = cleaned.replace(/[^a-z0-9]+/g, "");
        if (compact.length >= 4) {
          entry.tokens.add(compact);
        }
        for (const part of cleaned.split(/[^a-z0-9]+/g)) {
          if (part.length >= 4) {
            entry.tokens.add(part);
          }
        }
        entry.pathBits.add(candidate);
      }
      routes.set(id, entry);
    }
    return Array.from(routes.values());
  }

  /**
   * Signals indicating a query targets personal/continuity memory (dates,
   * people, decisions).  Extend via subclass override if the default
   * heuristics don't fit your deployment.  `protected static` allows
   * subclass customization; runtime config is not yet supported.
   */
  protected static readonly CORE_QUERY_SIGNALS: readonly string[] = [
    "yesterday",
    "today",
    "remember",
    "memory",
    "todo",
    "decision",
    "preference",
    "we did",
    "what did we",
  ];

  /**
   * Signals indicating a query targets code/repository knowledge.
   * Same extensibility model as CORE_QUERY_SIGNALS (subclass override).
   */
  protected static readonly REPO_QUERY_SIGNALS: readonly string[] = [
    "code",
    "repo",
    "middleware",
    "function",
    "class",
    "method",
    "api",
    "query",
    "python",
    "typescript",
    "import",
    "module",
  ];

  private detectQueryTier(query: string): QueryTier {
    const normalized = query.toLowerCase().replace(/[^a-z0-9]+/g, " ");
    const hasCore = MemoryIndexManager.CORE_QUERY_SIGNALS.some((s) => normalized.includes(s));
    const hasRepo = MemoryIndexManager.REPO_QUERY_SIGNALS.some((s) => normalized.includes(s));
    if (hasCore && !hasRepo) {
      return "core";
    }
    if (hasRepo && !hasCore) {
      return "repo";
    }
    return hasCore && hasRepo ? "mixed" : "core";
  }

  private findBestProjectRouteMatch(query: string): ProjectRouteMatch | null {
    const normalized = query.toLowerCase().replace(/[^a-z0-9]+/g, " ");
    let best: ProjectRouteMatch | null = null;
    for (const route of this.projectRoutes) {
      let score = 0;
      for (const token of route.tokens) {
        if (normalized.includes(token)) {
          score += token.length >= 8 ? 2 : 1;
        }
      }
      if (!best || score > best.score) {
        best = { route, score };
      }
    }
    if (!best || best.score < this.settings.query.routing.projectRouteMinScore) {
      return null;
    }
    return best;
  }

  private resolvePathFilter(query: string): { sql: string; params: string[] } {
    const normalized = query.toLowerCase().replace(/[^a-z0-9]+/g, " ");
    const params = new Set<string>();
    const tier = this.detectQueryTier(query);

    if (tier === "core") {
      return { sql: " AND path NOT LIKE ?", params: ["%/.openclaw_kb/%"] };
    }

    const best = this.findBestProjectRouteMatch(query);
    if (best) {
      for (const bit of best.route.pathBits) {
        params.add(`%${bit}%`);
      }
    }

    // Fallback: legacy token-to-path-hint mapping.
    for (const [token, pathBits] of this.queryPathHints.entries()) {
      if (!normalized.includes(token)) {
        continue;
      }
      for (const bit of pathBits) {
        params.add(`%${bit}%`);
      }
    }

    if (params.size === 0) {
      return tier === "mixed"
        ? { sql: "", params: [] }
        : { sql: " AND path LIKE ?", params: ["%/.openclaw_kb/%"] };
    }
    const clauses = Array.from(params, () => "path LIKE ?").join(" OR ");
    return { sql: ` AND (${clauses})`, params: Array.from(params) };
  }

  private resolveProjectDbPath(projectId: string): string {
    const template = this.settings.store.projectPathTemplate;
    const withProject = template.includes("{projectId}")
      ? template.replaceAll("{projectId}", projectId)
      : template;
    return path.resolve(withProject);
  }

  private getSearchDatabase(
    query: string,
    tier: QueryTier,
  ): {
    db: DatabaseSync;
    target: "core" | "project";
    projectId?: string;
  } {
    if (tier !== "repo") {
      return { db: this.db, target: "core" };
    }
    const best = this.findBestProjectRouteMatch(query);
    if (!best) {
      return { db: this.db, target: "core" };
    }

    try {
      const dbPath = this.resolveProjectDbPath(best.route.id);
      let projectDb = this.projectDbs.get(dbPath);
      if (!projectDb) {
        this.evictProjectDbIfNeeded();
        projectDb = this.openDatabaseAtPath(dbPath);
        this.projectDbs.set(dbPath, projectDb);
      }
      this.projectDbAccess.set(dbPath, Date.now());
      return { db: projectDb, target: "project", projectId: best.route.id };
    } catch {
      return { db: this.db, target: "core" };
    }
  }

  /**
   * Evict least-recently-used project DB handle when the pool exceeds
   * MAX_PROJECT_DB_HANDLES.  Prevents unbounded handle accumulation for
   * deployments with many indexed projects.
   */
  private evictProjectDbIfNeeded(): void {
    if (this.projectDbs.size < MemoryIndexManager.MAX_PROJECT_DB_HANDLES) {
      return;
    }
    let oldestPath: string | undefined;
    let oldestTime = Infinity;
    for (const [dbPath, accessTime] of this.projectDbAccess) {
      if (!this.projectDbs.has(dbPath)) {
        continue;
      }
      // Skip DBs with active per-DB locks (in-flight queries).
      const db = this.projectDbs.get(dbPath);
      if (db && this.perDbLocks.has(db)) {
        continue;
      }
      if (accessTime < oldestTime) {
        oldestTime = accessTime;
        oldestPath = dbPath;
      }
    }
    if (oldestPath) {
      try {
        this.projectDbs.get(oldestPath)?.close();
      } catch {
        // DB handle may already be closed; safe to ignore.
      }
      this.projectDbs.delete(oldestPath);
      this.projectDbAccess.delete(oldestPath);
    }
  }

  private getRepoCacheKey(query: string, maxResults: number, minScore: number): string {
    return `${query}::${maxResults}::${minScore}`;
  }

  private async embedQueryWithBudget(query: string, budgetMs: number): Promise<number[]> {
    return await Promise.race([
      this.embedQueryWithTimeout(query),
      new Promise<number[]>((_, reject) =>
        setTimeout(() => reject(new Error(`embed budget exceeded (${budgetMs}ms)`)), budgetMs),
      ),
    ]);
  }

  private getRepoCachedResults(key: string): MemorySearchResult[] | null {
    const CACHE_TTL_MS = 5 * 60_000;
    const hit = this.repoQueryCache.get(key);
    if (!hit) {
      return null;
    }
    if (Date.now() - hit.at > CACHE_TTL_MS) {
      this.repoQueryCache.delete(key);
      return null;
    }
    return hit.results;
  }

  private getIndexedFileCount(): number {
    const now = Date.now();
    const CACHE_TTL_MS = 60_000;
    if (now - this.indexedFileCountCache.at < CACHE_TTL_MS) {
      return this.indexedFileCountCache.value;
    }
    try {
      const row = this.db.prepare(`SELECT COUNT(*) as c FROM files`).get() as { c?: number };
      const value = row?.c ?? 0;
      this.indexedFileCountCache = { value, at: now };
      return value;
    } catch {
      return this.indexedFileCountCache.value;
    }
  }

  private shouldSkipSearchTriggeredSyncForLargeCorpus(): boolean {
    try {
      const count = this.getIndexedFileCount();
      const threshold = this.settings.query.routing.onSearchSyncSkipFileThreshold;
      if (count <= threshold) {
        return false;
      }
      const now = Date.now();
      if (now - this.lastLargeCorpusSkipLogAt >= 60_000) {
        this.lastLargeCorpusSkipLogAt = now;
        log.warn(
          "Using synchronous onSearch indexing with large codebases can lead to request timeouts. For corpora >10K files, recommend: set sync.onSearch=false and enable intervalMinutes (60-120) and/or watch: true for background indexing.",
        );
        log.debug("memory search: skipping on-search sync for large corpus", {
          indexedFiles: count,
          threshold,
        });
      }
      return true;
    } catch {
      return false;
    }
  }

  private async searchVector(
    queryVec: number[],
    limit: number,
    pathFilter?: { sql: string; params: string[] },
    dbOverride?: DatabaseSync,
  ): Promise<Array<MemorySearchResult & { id: string }>> {
    const startedAt = Date.now();
    let resultsCount = 0;
    // This method should never be called without a provider
    if (!this.provider) {
      return [];
    }

    const run = async (): Promise<Array<MemorySearchResult & { id: string }>> => {
      const results = await searchVector({
        db: this.db,
        vectorTable: VECTOR_TABLE,
        providerModel: this.provider?.model ?? "",
        queryVec,
        limit,
        snippetMaxChars: SNIPPET_MAX_CHARS,
        ensureVectorReady: async (dimensions) => await this.ensureVectorReady(dimensions),
        sourceFilterVec: this.buildSourceFilter("c"),
        sourceFilterChunks: this.buildSourceFilter(),
        pathFilter,
      });
      const mapped = results.map((entry) => entry as MemorySearchResult & { id: string });
      resultsCount = mapped.length;
      return mapped;
    };

    try {
      if (dbOverride && dbOverride !== this.db) {
        return await this.withDbContext({
          db: dbOverride,
          vectorEnabled: true,
          fn: run,
        });
      }

      return await run();
    } finally {
      log.debug("memory lifecycle perf", {
        function: "searchVector",
        limit,
        results: resultsCount,
        used_db_override: Boolean(dbOverride && dbOverride !== this.db),
        total_ms: Date.now() - startedAt,
      });
    }
  }

  private buildFtsQuery(raw: string): string | null {
    return buildFtsQuery(raw);
  }

  private async searchKeyword(
    query: string,
    limit: number,
    pathFilter?: { sql: string; params: string[] },
    dbOverride?: DatabaseSync,
  ): Promise<Array<MemorySearchResult & { id: string; textScore: number }>> {
    const startedAt = Date.now();
    let resultsCount = 0;
    if (!this.fts.enabled || !this.fts.available) {
      log.debug("memory lifecycle perf", {
        function: "searchKeyword",
        skipped: true,
        reason: "fts-unavailable",
        total_ms: Date.now() - startedAt,
      });
      return [];
    }
    const sourceFilter = this.buildSourceFilter();
    // In FTS-only mode (no provider), search all models; otherwise filter by current provider's model
    const providerModel = this.provider?.model;
    const results = await searchKeyword({
      db: dbOverride ?? this.db,
      ftsTable: FTS_TABLE,
      providerModel,
      query,
      limit,
      snippetMaxChars: SNIPPET_MAX_CHARS,
      sourceFilter,
      pathFilter,
      buildFtsQuery: (raw) => this.buildFtsQuery(raw),
      bm25RankToScore,
    });
    const mapped = results.map(
      (entry) => entry as MemorySearchResult & { id: string; textScore: number },
    );
    resultsCount = mapped.length;
    log.debug("memory lifecycle perf", {
      function: "searchKeyword",
      limit,
      results: resultsCount,
      used_db_override: Boolean(dbOverride && dbOverride !== this.db),
      total_ms: Date.now() - startedAt,
    });
    return mapped;
  }

  private async mergeHybridResults(params: {
    vector: Array<MemorySearchResult & { id: string }>;
    keyword: Array<MemorySearchResult & { id: string; textScore: number }>;
    vectorWeight: number;
    textWeight: number;
    mmr?: { enabled: boolean; lambda: number };
    temporalDecay?: { enabled: boolean; halfLifeDays: number };
  }): Promise<MemorySearchResult[]> {
    const startedAt = Date.now();
    const entries = await mergeHybridResults({
      vector: params.vector.map((r) => ({
        id: r.id,
        path: r.path,
        startLine: r.startLine,
        endLine: r.endLine,
        source: r.source,
        snippet: r.snippet,
        vectorScore: r.score,
      })),
      keyword: params.keyword.map((r) => ({
        id: r.id,
        path: r.path,
        startLine: r.startLine,
        endLine: r.endLine,
        source: r.source,
        snippet: r.snippet,
        textScore: r.textScore,
      })),
      vectorWeight: params.vectorWeight,
      textWeight: params.textWeight,
      mmr: params.mmr,
      temporalDecay: params.temporalDecay,
      workspaceDir: this.workspaceDir,
    });
    const out = entries.map((entry) => entry as MemorySearchResult);
    log.debug("memory lifecycle perf", {
      function: "mergeHybridResults",
      vector_candidates: params.vector.length,
      keyword_candidates: params.keyword.length,
      results: out.length,
      total_ms: Date.now() - startedAt,
    });
    return out;
  }

  async sync(params?: {
    reason?: string;
    force?: boolean;
    progress?: (update: MemorySyncProgressUpdate) => void;
  }): Promise<void> {
    if (this.closed) {
      return;
    }
    if (this.syncing) {
      return this.syncing;
    }
    this.syncing = this.runSync(params).finally(() => {
      this.syncing = null;
    });
    return this.syncing ?? Promise.resolve();
  }

  async readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<{ text: string; path: string }> {
    const rawPath = params.relPath.trim();
    if (!rawPath) {
      throw new Error("path required");
    }
    const absPath = path.isAbsolute(rawPath)
      ? path.resolve(rawPath)
      : path.resolve(this.workspaceDir, rawPath);
    const relPath = path.relative(this.workspaceDir, absPath).replace(/\\/g, "/");
    const inWorkspace =
      relPath.length > 0 && !relPath.startsWith("..") && !path.isAbsolute(relPath);
    const allowedWorkspace = inWorkspace && isMemoryPath(relPath);
    let allowedAdditional = false;
    if (!allowedWorkspace && this.settings.extraPaths.length > 0) {
      const additionalPaths = normalizeExtraMemoryPaths(
        this.workspaceDir,
        this.settings.extraPaths,
      );
      for (const additionalPath of additionalPaths) {
        try {
          const stat = await fs.lstat(additionalPath);
          if (stat.isSymbolicLink()) {
            continue;
          }
          if (stat.isDirectory()) {
            if (absPath === additionalPath || absPath.startsWith(`${additionalPath}${path.sep}`)) {
              allowedAdditional = true;
              break;
            }
            continue;
          }
          if (stat.isFile()) {
            if (absPath === additionalPath && absPath.endsWith(".md")) {
              allowedAdditional = true;
              break;
            }
          }
        } catch {}
      }
    }
    if (!allowedWorkspace && !allowedAdditional) {
      throw new Error("path required");
    }
    if (!absPath.endsWith(".md")) {
      throw new Error("path required");
    }
    const statResult = await statRegularFile(absPath);
    if (statResult.missing) {
      return { text: "", path: relPath };
    }
    let content: string;
    try {
      content = await fs.readFile(absPath, "utf-8");
    } catch (err) {
      if (isFileMissingError(err)) {
        return { text: "", path: relPath };
      }
      throw err;
    }
    if (!params.from && !params.lines) {
      return { text: content, path: relPath };
    }
    const lines = content.split("\n");
    const start = Math.max(1, params.from ?? 1);
    const count = Math.max(1, params.lines ?? lines.length);
    const slice = lines.slice(start - 1, start - 1 + count);
    return { text: slice.join("\n"), path: relPath };
  }

  status(): MemoryProviderStatus {
    const sourceFilter = this.buildSourceFilter();
    const files = this.db
      .prepare(`SELECT COUNT(*) as c FROM files WHERE 1=1${sourceFilter.sql}`)
      .get(...sourceFilter.params) as {
      c: number;
    };
    const chunks = this.db
      .prepare(`SELECT COUNT(*) as c FROM chunks WHERE 1=1${sourceFilter.sql}`)
      .get(...sourceFilter.params) as {
      c: number;
    };
    const sourceCounts = (() => {
      const sources = Array.from(this.sources);
      if (sources.length === 0) {
        return [];
      }
      const bySource = new Map<MemorySource, { files: number; chunks: number }>();
      for (const source of sources) {
        bySource.set(source, { files: 0, chunks: 0 });
      }
      const fileRows = this.db
        .prepare(
          `SELECT source, COUNT(*) as c FROM files WHERE 1=1${sourceFilter.sql} GROUP BY source`,
        )
        .all(...sourceFilter.params) as Array<{ source: MemorySource; c: number }>;
      for (const row of fileRows) {
        const entry = bySource.get(row.source) ?? { files: 0, chunks: 0 };
        entry.files = row.c ?? 0;
        bySource.set(row.source, entry);
      }
      const chunkRows = this.db
        .prepare(
          `SELECT source, COUNT(*) as c FROM chunks WHERE 1=1${sourceFilter.sql} GROUP BY source`,
        )
        .all(...sourceFilter.params) as Array<{ source: MemorySource; c: number }>;
      for (const row of chunkRows) {
        const entry = bySource.get(row.source) ?? { files: 0, chunks: 0 };
        entry.chunks = row.c ?? 0;
        bySource.set(row.source, entry);
      }
      return sources.map((source) => Object.assign({ source }, bySource.get(source)!));
    })();

    // Determine search mode: "fts-only" if no provider, "hybrid" otherwise
    const searchMode = this.provider ? "hybrid" : "fts-only";
    const providerInfo = this.provider
      ? { provider: this.provider.id, model: this.provider.model }
      : { provider: "none", model: undefined };

    return {
      backend: "builtin",
      files: files?.c ?? 0,
      chunks: chunks?.c ?? 0,
      dirty: this.dirty || this.sessionsDirty,
      workspaceDir: this.workspaceDir,
      dbPath: this.settings.store.path,
      provider: providerInfo.provider,
      model: providerInfo.model,
      requestedProvider: this.requestedProvider,
      sources: Array.from(this.sources),
      extraPaths: this.settings.extraPaths,
      sourceCounts,
      cache: this.cache.enabled
        ? {
            enabled: true,
            entries:
              (
                this.db.prepare(`SELECT COUNT(*) as c FROM ${EMBEDDING_CACHE_TABLE}`).get() as
                  | { c: number }
                  | undefined
              )?.c ?? 0,
            maxEntries: this.cache.maxEntries,
          }
        : { enabled: false, maxEntries: this.cache.maxEntries },
      fts: {
        enabled: this.fts.enabled,
        available: this.fts.available,
        error: this.fts.loadError,
      },
      fallback: this.fallbackReason
        ? { from: this.fallbackFrom ?? "local", reason: this.fallbackReason }
        : undefined,
      vector: {
        enabled: this.vector.enabled,
        available: this.vector.available ?? undefined,
        extensionPath: this.vector.extensionPath,
        loadError: this.vector.loadError,
        dims: this.vector.dims,
      },
      batch: {
        enabled: this.batch.enabled,
        failures: this.batchFailureCount,
        limit: BATCH_FAILURE_LIMIT,
        wait: this.batch.wait,
        concurrency: this.batch.concurrency,
        pollIntervalMs: this.batch.pollIntervalMs,
        timeoutMs: this.batch.timeoutMs,
        lastError: this.batchFailureLastError,
        lastProvider: this.batchFailureLastProvider,
      },
      custom: {
        searchMode,
        providerUnavailableReason: this.providerUnavailableReason,
      },
    };
  }

  async probeVectorAvailability(): Promise<boolean> {
    // FTS-only mode: vector search not available
    if (!this.provider) {
      return false;
    }
    if (!this.vector.enabled) {
      return false;
    }
    return this.ensureVectorReady();
  }

  async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
    // FTS-only mode: embeddings not available but search still works
    if (!this.provider) {
      return {
        ok: false,
        error: this.providerUnavailableReason ?? "No embedding provider available (FTS-only mode)",
      };
    }
    try {
      await this.embedBatchWithRetry(["ping"]);
      return { ok: true };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { ok: false, error: message };
    }
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    const pendingSync = this.syncing;
    if (this.watchTimer) {
      clearTimeout(this.watchTimer);
      this.watchTimer = null;
    }
    if (this.sessionWatchTimer) {
      clearTimeout(this.sessionWatchTimer);
      this.sessionWatchTimer = null;
    }
    if (this.intervalTimer) {
      clearInterval(this.intervalTimer);
      this.intervalTimer = null;
    }
    if (this.backgroundSyncTimer) {
      clearTimeout(this.backgroundSyncTimer);
      this.backgroundSyncTimer = null;
    }
    this.backgroundSyncReasons.clear();
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = null;
    }
    if (this.sessionUnsubscribe) {
      this.sessionUnsubscribe();
      this.sessionUnsubscribe = null;
    }
    if (pendingSync) {
      try {
        await pendingSync;
      } catch {}
    }
    this.repoQueryCache.clear();
    this.repoQueryJobs.clear();
    for (const db of this.projectDbs.values()) {
      try {
        db.close();
      } catch {}
    }
    this.projectDbs.clear();
    this.db.close();
    INDEX_CACHE.delete(this.cacheKey);
  }
}
