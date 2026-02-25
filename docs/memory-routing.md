# Memory Split-DB Routing

## How file routing works

When memory sync discovers `.openclaw_kb/` directories, it routes their contents to
project-specific SQLite databases instead of the main `main.sqlite`. The routing is
determined by the file's relative path from the workspace root.

## Routing rules

### Projects

Pattern: `Projects/<projectName>/.../.openclaw_kb/<file>`

The **first path segment** after `Projects/` is extracted as the `projectId`.
The `.openclaw_kb/` directory can appear at any depth under the project.

| Path                                                                  | projectId         |
| --------------------------------------------------------------------- | ----------------- |
| `Projects/myapp/.openclaw_kb/notes.md`                                | `myapp`           |
| `Projects/myapp/src/deep/.openclaw_kb/ref.md`                         | `myapp`           |
| `Projects/openclaw-builds/openclaw-source-main-fix/.openclaw_kb/x.md` | `openclaw-builds` |

### Skills

Pattern: `skills/<skillName>/.openclaw_kb/<file>`

Routed to `skill-<skillName>` as the projectId, preventing skill KB content from
polluting the core database.

| Path                                     | projectId           |
| ---------------------------------------- | ------------------- |
| `skills/truenas-api/.openclaw_kb/ref.md` | `skill-truenas-api` |

### Everything else

Files not matching either pattern route to the core database (`main.sqlite`).

## Diagnosing misroutes

If you see an unusually large number of files in the core DB while project routes are
active, check:

1. **Nested project structures** — The first segment after `Projects/` becomes the
   projectId. `Projects/a/b/.openclaw_kb/` routes to project `a`, not `b`.
2. **Missing `.openclaw_kb/`** — Only directories literally named `.openclaw_kb` trigger
   routing. Variants like `_kb` or `knowledge_base` do not.
3. **corePath configuration** — If `store.corePath` is not explicitly set, core and
   project queries share the same DB, reducing the benefit of split routing.

## Configuration

Set `store.corePath` to a dedicated path (different from `store.path`) to fully
separate core memory from project memory:

```json
{
  "store": {
    "path": "~/.openclaw/memory/main.sqlite",
    "corePath": "~/.openclaw/memory/core.sqlite",
    "projectPathTemplate": "~/.openclaw/memory/projects/{projectId}.sqlite"
  }
}
```
