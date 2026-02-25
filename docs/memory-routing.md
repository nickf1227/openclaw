# Memory Routing

OpenClaw supports splitting your memory index across multiple databases based on the user's workspace context. This keeps massive project corpora (like the Linux kernel) from bogging down your personal assistant queries, and allows dropping a project database without losing core history.

## How Routing Works

Every memory file path is evaluated against routing rules to determine a `projectId`.

- If a `projectId` is found, the file goes to the project database.
- If no `projectId` is found, the file goes to the core database (`corePath`).

### Project Boundaries

The router looks for `.openclaw_kb` directories to identify project roots.
For example, `Projects/a/b/c/.openclaw_kb/file.md` sets the `projectId` to `a` (the project dirname). Skills are routed identically (e.g. `skills/repo-rag/.openclaw_kb/...` routes to `skill-repo-rag`).

## Default Behavior

OpenClaw defaults to **background sync** for memory ingestion:

| Setting                | Default | Reason                                                                         |
| ---------------------- | ------- | ------------------------------------------------------------------------------ |
| `sync.onSearch`        | `false` | Prevents interactive lane starvation; blocking embeddings freeze request lanes |
| `sync.watch`           | `true`  | Real-time file system updates without blocking                                 |
| `sync.intervalMinutes` | `120`   | Background reconciliation every 2 hours                                        |

**Why `onSearch` is disabled by default:** Synchronous indexing blocks request lanes while computing embeddings. At scale (10K+ files), batch embedding failures or API rate limits cause sequential retries that paralyze interactive sessions.

## Example Configs

### Small/Legacy (single DB)

```yaml
agents:
  defaults:
    memorySearch:
      store:
        path: "~/.openclaw/memory/main.sqlite"
      sync:
        onSearch: false # Default
        watch: true # Default
        intervalMinutes: 120 # Default
```

### Large Corpora (split-DB)

```yaml
agents:
  defaults:
    memorySearch:
      store:
        path: "~/.openclaw/memory/fallback.sqlite"
        corePath: "~/.openclaw/memory/core.sqlite"
        projectPathTemplate: "~/.openclaw/memory/projects/{projectId}.sqlite"
      sync:
        onSearch: false # Never enable on large corpora
        watch: true # Real-time updates
        intervalMinutes: 60 # More frequent background reconciliation
```

## Anti-pattern: "sync-on-search death spiral"

- **Symptom:** Interactive requests hang indefinitely during early project phases.
- **Root cause:** `sync.onSearch: true` on a large corpus. Batch embeddings fail, leading to sequential fallback that paralyzes lanes.
- **Why it happens:** Default was `onSearch: true` in older versions. This is now disabled by default.
- **Fix:** Keep `sync.onSearch: false` (default). Use background `intervalMinutes` and `watch: true` for ingestion.

## Migration and recovery

- **How to move from monolithic to split DB:** Set `corePath` and `projectPathTemplate` in your config. Let the background sync run—it will automatically re-route updates to the new project DBs as they are touched.
- **How to detect misrouting:** If your main `corePath` (or legacy `path`) DB continues growing enormously while project DBs exist, check your `.openclaw_kb` placement.
- **What NOT to delete blindly:** Keep all `.sqlite` files (especially the old monolithic DB) until you have fully verified the migration is complete and project routing is stable.
