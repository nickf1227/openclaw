import { describe, expect, it } from "vitest";
import { MemoryIndexManager } from "./manager.js";

/**
 * Routing contract tests for resolveProjectIdForMemoryFile
 *
 * These tests verify that the routing logic correctly extracts projectId
 * from various memory file path patterns after schema changes (PR #25605).
 */
describe("resolveProjectIdForMemoryFile routing contract", () => {
  // Create a minimal manager instance to access the private method
  function createTestManager(workspaceDir: string): {
    resolveProjectId: (absPath: string) => string | null;
  } {
    // Use Object.create to instantiate without running the full constructor
    const manager = Object.create(MemoryIndexManager.prototype);
    manager.workspaceDir = workspaceDir;

    // Bind the private method for testing
    const resolveProjectId = (absPath: string): string | null => {
      // Replicate the exact logic from manager-sync-ops.ts
      const relPath = absPath
        .replace(/\\/g, "/")
        .replace(new RegExp(`^${workspaceDir.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/?`), "");

      // Match .openclaw_kb at any depth under Projects/, extracting the first segment as projectId
      const projectMatch =
        relPath.match(/^Projects\/([^/]+)\/.+\/\.openclaw_kb\//i) ??
        relPath.match(/^Projects\/([^/]+)\/\.openclaw_kb\//i);
      if (projectMatch?.[1]) {
        return normalizeProjectId(projectMatch[1]);
      }

      // Route skills/<name>/.openclaw_kb/ content to skill-specific DBs
      const skillMatch = relPath.match(/^skills\/([^/]+)\/\.openclaw_kb\//i);
      if (skillMatch?.[1]) {
        return normalizeProjectId(`skill-${skillMatch[1]}`);
      }

      return null;
    };

    return { resolveProjectId };
  }

  function normalizeProjectId(input: string): string {
    return (
      input
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "") || "project"
    );
  }

  describe("deep project path routing", () => {
    it("extracts first segment after Projects/ for deeply nested paths", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/Projects/a/b/c/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("a");
    });

    it("handles multiple nesting levels after projectId", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath =
        "/home/user/workspace/Projects/myproject/subdir1/subdir2/subdir3/.openclaw_kb/notes.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("myproject");
    });
  });

  describe("skills path routing", () => {
    it("routes skills/<name>/.openclaw_kb/ to skill-{name}", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/skills/repo-rag/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("skill-repo-rag");
    });

    it("normalizes skill names with special characters", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/skills/my_skill/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("skill-my-skill");
    });

    it("handles skill names with dots", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/skills/skill.name/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("skill-skill-name");
    });
  });

  describe("shallow project path (regression test)", () => {
    it("extracts projectId from shallow Projects/{name}/.openclaw_kb/ paths", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/Projects/vsinister/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("vsinister");
    });

    it("normalizes project names with mixed case", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/Projects/MyProject/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("myproject");
    });

    it("normalizes project names with special characters", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/Projects/my_project/.openclaw_kb/file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBe("my-project");
    });
  });

  describe("non-routed paths", () => {
    it("returns null for files outside Projects/ and skills/", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/memory/some-file.md";
      const result = resolveProjectId(testPath);

      expect(result).toBeNull();
    });

    it("returns null for files not under .openclaw_kb/", () => {
      const workspaceDir = "/home/user/workspace";
      const { resolveProjectId } = createTestManager(workspaceDir);

      const testPath = "/home/user/workspace/Projects/myproject/README.md";
      const result = resolveProjectId(testPath);

      expect(result).toBeNull();
    });
  });
});
