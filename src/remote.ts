/**
 * Remote (cloud) backends for the two neural steps that dominate query latency.
 *
 * Why this exists
 * ---------------
 * On a CPU-only box the local neural steps are not slow, they are unusable in an
 * interactive path. Measured on the reference machine (16 cores, no GPU):
 *
 *   query expansion (1.7B, local)   p50 3.3 s, tail 25 s
 *   query embedding (300M, local)   sub-second
 *   re-ranking (0.6B, local)        52-57 s for 30 candidates (~1.8 s/doc)
 *
 * A full local pipeline is 60 s+ per search, which a caller experiences as a stall, not
 * as a slow search. Embeddings and re-ranking can go to an HTTP endpoint; expansion
 * stays local because its output format is enforced by a GBNF grammar that hosted
 * APIs do not accept.
 *
 * The fallback contract (deliberate - see README "Cloud Models with Local Fallback")
 * --------------------------------------------------------------------------------
 * A remote failure SKIPS the layer. It never substitutes the local model. Running the
 * local cross-encoder as a "fallback" would turn a 0.15 s search into a 55 s one - a
 * fallback worse than the outage it covers. So every function here either returns a
 * result or throws `RemoteUnavailableError`, and callers translate that into "skip this
 * layer and carry on with BM25 + RRF".
 *
 * Free-tier endpoints are expected to disappear
 * ---------------------------------------------
 * Default models here are free tiers, which get rate-limited and retired without notice
 * (cf. the retirement of ollama's `deepseek-v4-flash:cloud`, which silenced every agent
 * that depended on it). Hence: bounded retries with backoff for transient failures
 * (429/5xx/network), no retry for permanent ones (401/403/404), and a hard timeout on
 * every call so a hung socket cannot become a hung search.
 */

/** Cloud embedding endpoint (OpenAI-compatible `/v1/embeddings`). */
export type RemoteEmbedConfig = {
  url: string;
  model: string;
  /** NAME of the env var holding the key. The key itself never lives in index.yml. */
  keyEnv?: string;
  timeoutS?: number;
  /** Texts per request. Batching keeps a large `qmd embed` from serialising. */
  batchSize?: number;
};

/** Cloud re-ranking endpoint (Cohere-shape `/v1/rerank`). */
export type RemoteRerankConfig = {
  url: string;
  model: string;
  keyEnv?: string;
  timeoutS?: number;
  /** Upper bound on returned candidates; the endpoint may return fewer. */
  topN?: number;
};

export type RemoteConfig = {
  embed?: RemoteEmbedConfig;
  rerank?: RemoteRerankConfig;
};

/**
 * Raised for every remote failure the caller should treat as "skip this layer".
 * Distinguishing it from a bug matters: a bug should surface, an unavailable provider
 * should degrade quietly.
 */
export class RemoteUnavailableError extends Error {
  constructor(
    public readonly role: string,
    message: string,
    /** HTTP status when the failure was an HTTP response, else undefined. */
    public readonly status?: number,
  ) {
    super(`${role}: ${message}`);
    this.name = "RemoteUnavailableError";
  }
}

const DEFAULT_TIMEOUT_S = 20;
const MAX_ATTEMPTS = 3;
const BASE_BACKOFF_MS = 500;

/** Status codes worth retrying: the provider is up but momentarily unwilling. */
function isTransient(status: number): boolean {
  return status === 429 || status === 408 || status >= 500;
}

function resolveKey(cfg: { keyEnv?: string }): string | undefined {
  const name = cfg.keyEnv?.trim();
  if (!name) return undefined; // local/self-hosted endpoints may not need one
  const value = process.env[name];
  if (!value) {
    throw new RemoteUnavailableError(
      "remote",
      `environment variable ${name} is not set (it holds the API key)`,
    );
  }
  return value;
}

async function postJson(
  role: string,
  url: string,
  body: unknown,
  headers: Record<string, string>,
  timeoutS: number,
): Promise<unknown> {
  let response: Response;
  let text: string;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutS * 1000),
    });
  } catch (error) {
    // Network failure, DNS, TLS or timeout - all transient by nature.
    throw new RemoteUnavailableError(
      role,
      `request failed (${error instanceof Error ? error.message : String(error)})`,
    );
  }

  try {
    text = await response.text();
  } catch {
    text = "";
  }

  if (!response.ok) {
    const detail = text.replace(/\s+/g, " ").slice(0, 200);
    // 401/403 = key rotated or revoked; 404 = model retired. Retrying cannot help, and
    // retrying a retired model just delays the fallback.
    const permanent = response.status === 401 || response.status === 403 || response.status === 404;
    throw new RemoteUnavailableError(
      role,
      `HTTP ${response.status}${permanent ? " (permanent)" : ""}: ${detail}`,
      permanent || !isTransient(response.status) ? response.status : undefined,
    );
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new RemoteUnavailableError(role, `response was not JSON: ${text.slice(0, 120)}`);
  }
}

/** Retry only what is worth retrying: transient HTTP states and network errors. */
async function withRetry<T>(role: string, attempt: () => Promise<T>): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i < MAX_ATTEMPTS; i++) {
    try {
      return await attempt();
    } catch (error) {
      lastError = error;
      const status = error instanceof RemoteUnavailableError ? error.status : undefined;
      const retryable = !(error instanceof RemoteUnavailableError) || status === undefined || isTransient(status);
      if (!retryable || i === MAX_ATTEMPTS - 1) break;
      await new Promise((resolve) => setTimeout(resolve, BASE_BACKOFF_MS * 2 ** i));
    }
  }
  throw lastError instanceof RemoteUnavailableError
    ? lastError
    : new RemoteUnavailableError(role, lastError instanceof Error ? lastError.message : String(lastError));
}

/** True when this role is configured to use a remote endpoint at all. */
export function hasRemoteEmbed(cfg?: RemoteConfig): boolean {
  return Boolean(cfg?.embed?.url && cfg?.embed?.model);
}

export function hasRemoteRerank(cfg?: RemoteConfig): boolean {
  return Boolean(cfg?.rerank?.url && cfg?.rerank?.model);
}

type RawRemote = Record<string, unknown>;

function pickString(source: RawRemote, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

function pickPositiveNumber(source: RawRemote, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return undefined;
}

/**
 * Normalise the `models.remote:` block coming out of YAML.
 *
 * The YAML spelling is snake_case (`key_env`, `timeout_s`, `batch_size`, `top_n`) while
 * the SDK spelling is camelCase; both are accepted, and normalisation happens exactly
 * here so nothing downstream has to care.
 *
 * Two failure modes are deliberately loud rather than silent, because a misconfigured
 * remote block looks exactly like a healthy one until the first request goes out
 * unauthenticated: an incomplete block warns, and a block that names no usable endpoint
 * is dropped so the caller falls back to fully-local behaviour.
 */
export function normalizeRemoteConfig(raw: unknown): RemoteConfig | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const source = raw as { embed?: RawRemote; rerank?: RawRemote };

  const embed = source.embed && typeof source.embed === "object"
    ? normalizeEmbed(source.embed)
    : undefined;
  const rerank = source.rerank && typeof source.rerank === "object"
    ? normalizeRerank(source.rerank)
    : undefined;

  for (const [role, given, parsed] of [
    ["embed", source.embed, embed],
    ["rerank", source.rerank, rerank],
  ] as const) {
    if (given && !parsed) {
      console.warn(
        `⚠ models.remote.${role} is configured but has no usable url+model — ` +
        `ignoring it and staying local for that role`,
      );
    }
  }

  if (!embed && !rerank) return undefined;
  return { embed, rerank };
}

function normalizeEmbed(source: RawRemote): RemoteEmbedConfig | undefined {
  const url = pickString(source, "url");
  const model = pickString(source, "model");
  if (!url || !model) return undefined;
  return {
    url,
    model,
    keyEnv: pickString(source, "key_env", "keyEnv"),
    timeoutS: pickPositiveNumber(source, "timeout_s", "timeoutS"),
    batchSize: pickPositiveNumber(source, "batch_size", "batchSize"),
  };
}

function normalizeRerank(source: RawRemote): RemoteRerankConfig | undefined {
  const url = pickString(source, "url");
  const model = pickString(source, "model");
  if (!url || !model) return undefined;
  return {
    url,
    model,
    keyEnv: pickString(source, "key_env", "keyEnv"),
    timeoutS: pickPositiveNumber(source, "timeout_s", "timeoutS"),
    topN: pickPositiveNumber(source, "top_n", "topN"),
  };
}

/**
 * Embed texts via the remote endpoint. Returns one vector per input, in input order.
 *
 * Throws `RemoteUnavailableError` if the endpoint cannot deliver a complete, ordered,
 * internally-consistent result - a partial embedding set would silently corrupt the
 * index, so it is treated as failure rather than used.
 */
export async function remoteEmbed(texts: string[], cfg: RemoteEmbedConfig): Promise<number[][]> {
  if (texts.length === 0) return [];
  const role = "remote embed";
  const key = resolveKey(cfg);
  const timeoutS = cfg.timeoutS ?? DEFAULT_TIMEOUT_S;
  const batchSize = Math.max(1, cfg.batchSize ?? 32);
  const out: number[][] = [];

  for (let start = 0; start < texts.length; start += batchSize) {
    const batch = texts.slice(start, start + batchSize);
    const payload = await withRetry(role, () =>
      postJson(
        role,
        cfg.url,
        { model: cfg.model, input: batch },
        key ? { authorization: `Bearer ${key}` } : {},
        timeoutS,
      ),
    );

    const rows = (payload as { data?: { embedding?: number[]; index?: number }[] })?.data;
    if (!Array.isArray(rows) || rows.length === 0) {
      throw new RemoteUnavailableError(role, "response contained no embeddings");
    }
    // The API may reorder; `index` is authoritative when present.
    const ordered = rows
      .map((row, fallbackIndex) => ({ vector: row?.embedding, index: row?.index ?? fallbackIndex }))
      .sort((a, b) => a.index - b.index);
    if (ordered.length !== batch.length) {
      throw new RemoteUnavailableError(
        role,
        `expected ${batch.length} embeddings, received ${ordered.length}`,
      );
    }
    for (const row of ordered) {
      if (!Array.isArray(row.vector) || row.vector.length === 0) {
        throw new RemoteUnavailableError(role, "an embedding was empty");
      }
      out.push(row.vector);
    }
  }

  // Mixed dimensions within one response set mean a broken endpoint or a model swap
  // mid-flight; catching it here beats discovering it as a sqlite-vec error later.
  const dims = new Set(out.map((v) => v.length));
  if (dims.size > 1) {
    throw new RemoteUnavailableError(role, `inconsistent dimensions in one result set: ${[...dims].join(", ")}`);
  }
  return out;
}

/**
 * Re-rank candidates via a Cohere-shape `/v1/rerank` endpoint.
 *
 * This shape is spoken by both OpenRouter's rerank route and NVIDIA's own rerank
 * endpoint, which is why it is the first adapter: one implementation covers the
 * endpoints already proven by apinav. (The `typesafe` System One reranker is a
 * different API - stateful judgments rather than a scores response - and belongs in a
 * second adapter.)
 *
 * Returns `{ file, score }` aligned to the input candidates, best first. Throws
 * `RemoteUnavailableError` when the endpoint cannot answer, so the caller can skip
 * re-ranking entirely rather than fall back to the local cross-encoder.
 */
export async function remoteRerank(
  query: string,
  documents: { file: string; text: string }[],
  cfg: RemoteRerankConfig,
): Promise<{ file: string; score: number; index: number }[]> {
  if (documents.length === 0) return [];
  const role = "remote rerank";
  const key = resolveKey(cfg);
  const timeoutS = cfg.timeoutS ?? 30;

  const payload = await withRetry(role, () =>
    postJson(
      role,
      cfg.url,
      {
        model: cfg.model,
        query,
        documents: documents.map((d) => d.text),
        ...(cfg.topN ? { top_n: Math.min(cfg.topN, documents.length) } : {}),
      },
      key ? { authorization: `Bearer ${key}` } : {},
      timeoutS,
    ),
  );

  const results = (payload as { results?: { index?: number; relevance_score?: number; score?: number }[] })?.results;
  if (!Array.isArray(results) || results.length === 0) {
    throw new RemoteUnavailableError(role, "response contained no results");
  }

  const scored = results
    .map((row) => {
      const index = row?.index;
      const score = row?.relevance_score ?? row?.score;
      if (typeof index !== "number" || typeof score !== "number") return null;
      const doc = documents[index];
      if (!doc) return null;
      return { file: doc.file, score, index };
    })
    .filter((r): r is { file: string; score: number; index: number } => r !== null);

  if (scored.length === 0) {
    throw new RemoteUnavailableError(role, "no result carried a usable index and score");
  }
  return scored.sort((a, b) => b.score - a.score);
}
