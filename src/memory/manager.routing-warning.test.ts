import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Validation tests for manager routing warnings
 *
 * These tests verify that the warning message for large corpus sync-on-search
 * is present in the manager implementation and includes actionable guidance.
 */
describe("manager routing/skip warnings", () => {
  it("includes actionable warning for large corpus onSearch starvation", () => {
    // Read the manager source to verify the warning message is present
    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const managerPath = path.join(__dirname, "manager.ts");
    const source = readFileSync(managerPath, "utf8");

    const warningText =
      "Using synchronous onSearch indexing with large codebases can lead to request timeouts";
    expect(source).toContain(warningText);

    // Verify it includes the recommendation
    expect(source).toContain("set sync.onSearch=false");
    expect(source).toContain("intervalMinutes (60-120)");
    expect(source).toContain("watch: true");
  });
});
