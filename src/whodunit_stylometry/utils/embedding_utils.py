"""Embedding utilities for generating and processing embeddings."""

import hashlib
import json
import time
from typing import Any

import networkx as nx
import numpy as np
import openai
import pandas as pd
import tiktoken
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize


def get_encoding(model: str) -> tiktoken.Encoding:
    """Return the token encoding for a model.

    Looks up the tokenizer encoding associated with `model`. If the model is
    not known by `tiktoken`, falls back to the `cl100k_base` encoding.

    Args:
        model: Name of the model whose encoding should be retrieved.

    Returns:
        The `tiktoken` encoding associated with `model`, or the fallback
        `cl100k_base` encoding if the model is unknown.

    Raises:
        ValueError: If the fallback encoding name is invalid or unavailable.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text_by_tokens(
    text: str, chunk_tokens: int, overlap: int, min_tokens: int, encoding: tiktoken.Encoding
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks based on token count.

    Encodes `text` using the global `encoding` object and splits the resulting
    tokens into chunks of up to `chunk_tokens` tokens. Consecutive chunks overlap
    by `overlap` tokens, unless the computed step would be less than 1. Chunks
    with fewer than `min_tokens` tokens are discarded.

    Args:
        text: Text to split into token-based chunks.
        chunk_tokens: Maximum number of tokens per chunk.
        overlap: Number of tokens shared between consecutive chunks.
        min_tokens: Minimum number of tokens required for a chunk to be kept.
        encoding: The `tiktoken.Encoding` object used to encode and decode text.

    Returns:
        A list of dictionaries, one per chunk. Each dictionary contains:
        `chunk_index`, `text`, and `token_count`.

    Raises:
        AttributeError: If the global `encoding` object does not provide
            compatible `encode` or `decode` methods.
    """
    tokens = encoding.encode(text)
    chunks: list[dict[str, Any]] = []

    if len(tokens) <= chunk_tokens:
        return [{"chunk_index": 0, "text": text, "token_count": len(tokens)}] if len(tokens) >= min_tokens else []

    step = max(1, chunk_tokens - overlap)
    idx = 0
    chunk_index = 0

    while idx < len(tokens):
        window = tokens[idx : idx + chunk_tokens]
        if len(window) < min_tokens:
            break

        chunk_text = encoding.decode(window)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_text,
                "token_count": len(window),
            }
        )
        idx += step
        chunk_index += 1

    return chunks


def text_hash(text: str, model: str, dimensions: int | None) -> str:
    """Return a SHA-256 hash for text embedding parameters.

    Builds a JSON payload from the provided text, model name, and embedding
    dimensions, then returns the SHA-256 hexadecimal digest of that payload.

    Args:
        text: Text to include in the hash payload.
        model: Model name to include in the hash payload.
        dimensions: Embedding dimensionality to include in the hash payload, or
            `None` if no explicit dimensionality is used.

    Returns:
        The SHA-256 hash of the JSON-serialized payload as a hexadecimal string.
    """
    payload = json.dumps(
        {"model": model, "dimensions": dimensions, "text": text},
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def call_embeddings_api(
    texts: list[str], model: str, embedding_dimensions: int, max_retries: int, client: openai.OpenAI
) -> list[list[float]]:
    """Create embeddings for a list of texts using the provided API client.

    Sends `texts` to the embeddings API with the given `model`. If
    `embedding_dimensions` is not `None`, it is included in the request as the
    `dimensions` parameter.

    Failed requests are retried with exponential backoff, capped at 60 seconds
    between attempts. If all retry attempts fail, a `RuntimeError` is raised.

    Args:
        texts: Text inputs to embed.
        model: Name of the embedding model to use.
        embedding_dimensions: Optional embedding dimensionality to request. Use
            `None` to omit the `dimensions` parameter.
        max_retries: Maximum number of API call attempts.
        client: API client object exposing `client.embeddings.create(...)`.

    Returns:
        A list of embedding vectors, one per input text. Each embedding vector
        is represented as a list of floats.
    """
    kwargs: dict[str, Any] = {"model": model, "input": texts}

    if embedding_dimensions is not None:
        kwargs["dimensions"] = embedding_dimensions

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(**kwargs)
            return [item.embedding for item in response.data]
        except Exception as e:
            wait = min(60, 2**attempt)
            print(f"Error en embeddings: {e}. Reintentando en {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Se agotaron los reintentos contra la API de embeddings.")


def weighted_average_embeddings(
    df: pd.DataFrame,
    matrix: np.ndarray,
    group_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate embeddings by weighted average over grouped rows.

    Groups `df` by `group_cols`, uses each row's `token_count` as the weight,
    averages the corresponding rows from `matrix`, and normalizes each resulting
    group vector. The function returns both group-level metadata and the stacked
    normalized vectors.

    Args:
        df: DataFrame containing the grouping columns and a `token_count` column.
            Its row order must correspond to the row order of `matrix`.
        matrix: Embedding matrix where each row corresponds to the row at the
            same position in `df`.
        group_cols: Column names used to group rows before averaging embeddings.

    Returns:
        A tuple containing:
            - A DataFrame with one row per group, including the group column
              values, `n_chunks`, and summed `token_count`.
            - A NumPy array of normalized weighted-average vectors with dtype
              `np.float32`.
    """
    meta_rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []

    tmp = df.reset_index(drop=True).copy()
    tmp["row_idx"] = np.arange(len(tmp))

    for keys, group in tmp.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        idx = group["row_idx"].to_numpy()
        weights = group["token_count"].to_numpy(dtype=np.float32)
        vec = np.average(matrix[idx], axis=0, weights=weights)
        vec = normalize(vec.reshape(1, -1))[0]

        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_chunks"] = len(group)
        row["token_count"] = int(weights.sum())

        meta_rows.append(row)
        vectors.append(vec)

    return pd.DataFrame(meta_rows), np.vstack(vectors).astype(np.float32)


def simple_average_embeddings(
    meta_df: pd.DataFrame,
    embeddings: np.ndarray,
    group_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate embeddings by simple average over grouped metadata rows.

    Groups `meta_df` by `group_cols`, selects the corresponding rows from
    `embeddings`, computes the unweighted mean embedding for each group, and
    normalizes each averaged vector by its L2 norm.

    Args:
        meta_df: DataFrame containing the columns used to group embeddings.
            Its row positions must correspond to the row positions in
            `embeddings`.
        embeddings: Embedding matrix where each row corresponds to the row at
            the same position in `meta_df`.
        group_cols: Column names used to group rows before averaging embeddings.

    Returns:
        A tuple containing:
            - A DataFrame with one row per group, including the group column
              values and `n_items`.
            - A NumPy array containing one normalized average embedding per
              group.
    """
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []

    for group_values, idx in meta_df.groupby(group_cols).groups.items():
        idx = list(idx)

        group_embeddings = embeddings[idx]
        avg_embedding = group_embeddings.mean(axis=0)

        # Normalization, recommended for cosine similarity.
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = dict(zip(group_cols, group_values))
        row["n_items"] = len(idx)
        rows.append(row)
        vectors.append(avg_embedding)

    return pd.DataFrame(rows), np.vstack(vectors)


def nearest_neighbors_table(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Build a nearest-neighbor table from cosine similarities.

    Computes pairwise cosine similarity between embedding vectors and returns
    the top `top_k` nearest neighbors for each row in `meta`, excluding the row
    itself. The returned table includes the source work, source author, neighbor
    rank, neighbor work, neighbor author, whether both rows share the same
    author, and the cosine similarity score.

    Args:
        meta: DataFrame containing at least `work` and `author` columns. Its row
            positions are expected to correspond to the rows in `embeddings`.
        embeddings: Embedding matrix used to compute pairwise cosine
            similarities.
        top_k: Maximum number of neighbors to return for each row, excluding
            the row itself.

    Returns:
        A DataFrame with one row per source-neighbor pair.
    """
    sim = cosine_similarity(embeddings)
    rows: list[dict[str, object]] = []

    for i in range(len(meta)):
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i][:top_k]

        for rank, j in enumerate(order, 1):
            rows.append(
                {
                    "work": meta.loc[i, "work"],
                    "author": meta.loc[i, "author"],
                    "neighbor_rank": rank,
                    "neighbor_work": meta.loc[j, "work"],
                    "neighbor_author": meta.loc[j, "author"],
                    "same_author": meta.loc[i, "author"] == meta.loc[j, "author"],
                    "cosine_similarity": sim[i, j],
                }
            )

    return pd.DataFrame(rows)


def pairwise_author_metrics(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise work distances and author-level separation metrics.

    Computes cosine distances between all embedding vectors and builds two
    summary tables. The first table contains pairwise work-level comparisons.
    The second table contains author-level metrics based on intra-author and
    inter-author cosine distances.

    Args:
        meta: DataFrame containing at least `author` and `work` columns. Its row
            positions are expected to correspond to the rows in `embeddings`.
        embeddings: Embedding matrix used to compute pairwise cosine distances.

    Returns:
        A tuple containing:
            - A DataFrame with one row per unique pair of works, including
              author names, work names, whether both works have the same author,
              cosine distance, and cosine similarity.
            - A DataFrame with one row per author with at least two works,
              including `n_works`, mean intra-author distance, mean inter-author
              distance, and `separation_margin`, sorted by descending
              `separation_margin`.
    """
    dist = cosine_distances(embeddings)
    rows: list[dict[str, object]] = []

    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            rows.append(
                {
                    "author_i": meta.loc[i, "author"],
                    "work_i": meta.loc[i, "work"],
                    "author_j": meta.loc[j, "author"],
                    "work_j": meta.loc[j, "work"],
                    "same_author": meta.loc[i, "author"] == meta.loc[j, "author"],
                    "cosine_distance": dist[i, j],
                    "cosine_similarity": 1 - dist[i, j],
                }
            )

    pairs = pd.DataFrame(rows)
    by_author: list[dict[str, object]] = []

    for author, idxs in meta.groupby("author").groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue

        intra = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                intra.append(dist[idxs[a], idxs[b]])

        other_idxs = [i for i in range(len(meta)) if i not in idxs]
        inter = dist[np.ix_(idxs, other_idxs)].ravel() if other_idxs else np.array([])

        by_author.append(
            {
                "author": author,
                "n_works": len(idxs),
                "mean_intra_author_distance": (float(np.mean(intra)) if intra else np.nan),
                "mean_inter_author_distance": (float(np.mean(inter)) if len(inter) else np.nan),
                "separation_margin": (float(np.mean(inter) - np.mean(intra)) if intra and len(inter) else np.nan),
            }
        )

    return pairs, pd.DataFrame(by_author).sort_values(
        "separation_margin",
        ascending=False,
    )


def build_similarity_network(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 3,
) -> nx.Graph:
    """Build a similarity network from embedding nearest neighbors.

    Computes pairwise cosine similarities between embedding vectors and creates
    an undirected NetworkX graph. Each row in `meta` becomes a node, and each
    node is connected to its top `top_k` most similar neighbors, excluding
    itself.

    Node attributes include `label`, `author`, and `work`. Edge attributes
    include the cosine similarity as `weight` and whether the connected works
    have the same author as `same_author`.

    Args:
        meta: DataFrame containing at least `author` and `work` columns. Its row
            positions are expected to correspond to the rows in `embeddings`.
        embeddings: Embedding matrix used to compute pairwise cosine
            similarities.
        top_k: Maximum number of nearest neighbors to connect for each node.

    Returns:
        An undirected NetworkX graph representing nearest-neighbor similarities
        between works.
    """
    sim = cosine_similarity(embeddings)
    graph = nx.Graph()

    for i, row in meta.iterrows():
        graph.add_node(
            i,
            label=f"{row['author']} — {row['work']}",
            author=row["author"],
            work=row["work"],
        )

    for i in range(len(meta)):
        neighbors = [j for j in np.argsort(-sim[i]) if j != i][:top_k]

        for j in neighbors:
            graph.add_edge(
                i,
                j,
                weight=float(sim[i, j]),
                same_author=meta.loc[i, "author"] == meta.loc[j, "author"],
            )

    return graph
