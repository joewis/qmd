/**
 * Tests for the remote (cloud) model backends.
 *
 * These pin the fallback contract from the README: a remote failure must surface as
 * `RemoteUnavailableError` (so the caller SKIPS the layer) and must never look like a
 * valid result - a partially-filled or reordered embedding set would silently corrupt
 * the index, which is worse than no embeddings at all.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  RemoteUnavailableError,
  hasRemoteEmbed,
  hasRemoteRerank,
  normalizeRemoteConfig,
  remoteEmbed,
  remoteRerank,
} from "../src/remote.js";

const EMBED_CFG = {
  url: "https://example.invalid/v1/embeddings",
  model: "test-embed",
  keyEnv: "TEST_EMBED_KEY",
  timeoutS: 5,
  batchSize: 2,
};

const RERANK_CFG = {
  url: "https://example.invalid/v1/rerank",
  model: "test-rerank",
  keyEnv: "TEST_RERANK_KEY",
  timeoutS: 5,
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const originalFetch = global.fetch;

beforeEach(() => {
  process.env.TEST_EMBED_KEY = "secret-embed";
  process.env.TEST_RERANK_KEY = "secret-rerank";
});

afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("capability checks", () => {
  it("reports a role as remote only when both url and model are set", () => {
    expect(hasRemoteEmbed(undefined)).toBe(false);
    expect(hasRemoteEmbed({ embed: { url: "u", model: "" } })).toBe(false);
    expect(hasRemoteEmbed({ embed: { url: "u", model: "m" } })).toBe(true);
    expect(hasRemoteRerank({ rerank: { url: "u", model: "m" } })).toBe(true);
  });
});

describe("normalizeRemoteConfig", () => {
  it("reads the snake_case spelling documented in the README", () => {
    // This is the bug the live test caught: YAML says key_env, the type says keyEnv, and
    // a dropped key_env produced an unauthenticated request rather than an error.
    const cfg = normalizeRemoteConfig({
      embed: { url: "u", model: "m", key_env: "K", timeout_s: 12, batch_size: 8 },
    });
    expect(cfg?.embed).toEqual({ url: "u", model: "m", keyEnv: "K", timeoutS: 12, batchSize: 8 });
  });

  it("accepts the camelCase spelling too", () => {
    const cfg = normalizeRemoteConfig({ embed: { url: "u", model: "m", keyEnv: "K" } });
    expect(cfg?.embed?.keyEnv).toBe("K");
  });

  it("reads rerank's top_n as topN", () => {
    const cfg = normalizeRemoteConfig({ rerank: { url: "u", model: "m", top_n: 25 } });
    expect(cfg?.rerank?.topN).toBe(25);
  });

  it("warns and drops an incomplete block rather than going silently unauthenticated", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(normalizeRemoteConfig({ embed: { model: "m" } })).toBeUndefined();
    expect(warn).toHaveBeenCalled();
  });

  it("keeps one role when only that role is configured", () => {
    const cfg = normalizeRemoteConfig({ rerank: { url: "u", model: "m" } });
    expect(cfg?.rerank?.url).toBe("u");
    expect(cfg?.embed).toBeUndefined();
  });

  it("returns undefined when no block is present", () => {
    expect(normalizeRemoteConfig(undefined)).toBeUndefined();
    expect(normalizeRemoteConfig(null)).toBeUndefined();
    expect(normalizeRemoteConfig("nonsense")).toBeUndefined();
  });
});

describe("remoteEmbed", () => {
  it("returns one vector per input, in input order", async () => {
    global.fetch = vi.fn(async () =>
      jsonResponse({ data: [{ embedding: [1, 2], index: 0 }, { embedding: [3, 4], index: 1 }] }),
    ) as unknown as typeof fetch;

    const vectors = await remoteEmbed(["a", "b"], EMBED_CFG);
    expect(vectors).toEqual([[1, 2], [3, 4]]);
  });

  it("reorders by index when the endpoint returns results out of order", async () => {
    // Providers are not required to preserve order; trusting arrival order would
    // attach the wrong vector to the wrong chunk.
    global.fetch = vi.fn(async () =>
      jsonResponse({ data: [{ embedding: [3, 4], index: 1 }, { embedding: [1, 2], index: 0 }] }),
    ) as unknown as typeof fetch;

    expect(await remoteEmbed(["a", "b"], EMBED_CFG)).toEqual([[1, 2], [3, 4]]);
  });

  it("fails rather than returning a short result set", async () => {
    global.fetch = vi.fn(async () => jsonResponse({ data: [{ embedding: [1, 2], index: 0 }] })) as unknown as typeof fetch;
    await expect(remoteEmbed(["a", "b"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
  });

  it("fails on inconsistent dimensions within one result set", async () => {
    global.fetch = vi.fn(async () =>
      jsonResponse({ data: [{ embedding: [1, 2], index: 0 }, { embedding: [1, 2, 3], index: 1 }] }),
    ) as unknown as typeof fetch;
    await expect(remoteEmbed(["a", "b"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
  });

  it("fails on an empty embedding", async () => {
    global.fetch = vi.fn(async () =>
      jsonResponse({ data: [{ embedding: [], index: 0 }, { embedding: [1], index: 1 }] }),
    ) as unknown as typeof fetch;
    await expect(remoteEmbed(["a", "b"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
  });

  it("does not call the network for an empty input", async () => {
    const spy = vi.fn();
    global.fetch = spy as unknown as typeof fetch;
    expect(await remoteEmbed([], EMBED_CFG)).toEqual([]);
    expect(spy).not.toHaveBeenCalled();
  });

  it("fails when the key env var is missing, without calling the network", async () => {
    delete process.env.TEST_EMBED_KEY;
    const spy = vi.fn();
    global.fetch = spy as unknown as typeof fetch;
    await expect(remoteEmbed(["a"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
    expect(spy).not.toHaveBeenCalled();
  });

  it("batches requests according to batchSize", async () => {
    const spy = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { input: string[] };
      return jsonResponse({ data: body.input.map((_, i) => ({ embedding: [i], index: i })) });
    });
    global.fetch = spy as unknown as typeof fetch;
    await remoteEmbed(["a", "b", "c", "d", "e"], EMBED_CFG); // batchSize 2 -> 3 calls
    expect(spy).toHaveBeenCalledTimes(3);
  });
});

describe("failure classification", () => {
  it("retries a 429 and succeeds", async () => {
    let calls = 0;
    global.fetch = vi.fn(async () => {
      calls += 1;
      if (calls < 3) return jsonResponse({ error: "rate limited" }, 429);
      return jsonResponse({ data: [{ embedding: [1], index: 0 }] });
    }) as unknown as typeof fetch;

    expect(await remoteEmbed(["a"], EMBED_CFG)).toEqual([[1]]);
    expect(calls).toBe(3);
  });

  it("does NOT retry a 404 (a retired model cannot come back by being asked again)", async () => {
    const spy = vi.fn(async () => jsonResponse({ error: "model not found" }, 404));
    global.fetch = spy as unknown as typeof fetch;

    await expect(remoteEmbed(["a"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry a 401 (rotated key)", async () => {
    const spy = vi.fn(async () => jsonResponse({ error: "unauthorized" }, 401));
    global.fetch = spy as unknown as typeof fetch;

    await expect(remoteEmbed(["a"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("treats a network error as unavailable", async () => {
    global.fetch = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    await expect(remoteEmbed(["a"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
  });

  it("treats a non-JSON body as unavailable", async () => {
    global.fetch = vi.fn(async () => new Response("<html>blocked by proxy</html>", { status: 200 })) as unknown as typeof fetch;
    await expect(remoteEmbed(["a"], EMBED_CFG)).rejects.toBeInstanceOf(RemoteUnavailableError);
  });
});

describe("model identity follows the remote backend", () => {
  const LOCAL = {
    embedModel: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
    rerankModel: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf",
  };

  it("reports the local URIs when no remote backend is configured", async () => {
    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp(LOCAL);
    expect(llm.embedModelName).toBe(LOCAL.embedModel);
    expect(llm.rerankModelName).toBe(LOCAL.rerankModel);
  });

  it("reports the REMOTE model names when the cloud backends are active", async () => {
    // This identity is what qmd fingerprints vectors with (content_vectors.model) and what
    // keys the rerank cache. If it kept reporting the local URI, a switch to the cloud model
    // would leave the previous model's vectors in place - silently, because the fingerprint
    // would still "match" - and a 2048-d query would be compared against 768-d rows.
    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({
      ...LOCAL,
      remote: {
        embed: { url: "https://x", model: "nvidia/nemotron-3-embed-1b", key_env: "K" },
        rerank: { url: "https://x", model: "nvidia/llama-nemotron-rerank-vl-1b-v2:free", key_env: "K" },
      },
    });
    expect(llm.embedModelName).toBe("nvidia/nemotron-3-embed-1b");
    expect(llm.rerankModelName).toBe("nvidia/llama-nemotron-rerank-vl-1b-v2:free");
  });
});

describe("remoteRerank", () => {
  it("maps scores back to files and sorts best-first", async () => {
    global.fetch = vi.fn(async () =>
      jsonResponse({
        results: [
          { index: 0, relevance_score: 0.2 },
          { index: 1, relevance_score: 0.9 },
        ],
      }),
    ) as unknown as typeof fetch;

    const docs = [{ file: "a.md", text: "a" }, { file: "b.md", text: "b" }];
    expect(await remoteRerank("q", docs, RERANK_CFG)).toEqual([
      { file: "b.md", score: 0.9, index: 1 },
      { file: "a.md", score: 0.2, index: 0 },
    ]);
  });

  it("accepts the `score` field as well as `relevance_score`", async () => {
    global.fetch = vi.fn(async () => jsonResponse({ results: [{ index: 0, score: 0.7 }] })) as unknown as typeof fetch;
    expect(await remoteRerank("q", [{ file: "a.md", text: "a" }], RERANK_CFG)).toEqual([
      { file: "a.md", score: 0.7, index: 0 },
    ]);
  });

  it("fails when no row carries a usable index and score", async () => {
    global.fetch = vi.fn(async () => jsonResponse({ results: [{ nonsense: true }] })) as unknown as typeof fetch;
    await expect(remoteRerank("q", [{ file: "a.md", text: "a" }], RERANK_CFG)).rejects.toBeInstanceOf(
      RemoteUnavailableError,
    );
  });

  it("fails on an empty result set rather than reporting 'nothing relevant'", async () => {
    // Empty must be an error: the caller has to be able to tell "provider is down"
    // from "provider scored everything irrelevant".
    global.fetch = vi.fn(async () => jsonResponse({ results: [] })) as unknown as typeof fetch;
    await expect(remoteRerank("q", [{ file: "a.md", text: "a" }], RERANK_CFG)).rejects.toBeInstanceOf(
      RemoteUnavailableError,
    );
  });

  it("sends the key as a bearer token from the named env var", async () => {
    const spy = vi.fn(async (_url: string, init: RequestInit) => {
      const headers = init.headers as Record<string, string>;
      expect(headers.authorization).toBe("Bearer secret-rerank");
      return jsonResponse({ results: [{ index: 0, relevance_score: 1 }] });
    });
    global.fetch = spy as unknown as typeof fetch;
    await remoteRerank("q", [{ file: "a.md", text: "a" }], RERANK_CFG);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
