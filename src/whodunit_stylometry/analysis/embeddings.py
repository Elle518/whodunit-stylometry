"""OpenAI embeddings workflows for interactive authorship exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from whodunit_stylometry.utils.embedding_utils import (
    build_similarity_network,
    call_embeddings_api,
    chunk_text_by_tokens,
    get_encoding,
    nearest_neighbors_table,
    pairwise_author_metrics,
    simple_average_embeddings,
    text_hash,
    weighted_average_embeddings,
)

EMBEDDING_MODELS = {
    "text-embedding-3-large": "text-embedding-3-large",
    "text-embedding-3-small": "text-embedding-3-small",
}


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Configuration for OpenAI embeddings experiments."""

    model: str = "text-embedding-3-large"
    dimensions: int | None = None
    chunk_tokens: int = 1200
    chunk_overlap: int = 150
    min_chunk_tokens: int = 120
    batch_size: int = 64
    max_retries: int = 6
    random_state: int = 42
    umap_neighbors: int = 10
    umap_min_dist: float = 0.08
    network_top_k: int = 3
    nearest_neighbors_k: int = 6


def estimate_embedding_chunks(df: pd.DataFrame, config: EmbeddingsConfig) -> pd.DataFrame:
    """Split corpus works into token chunks without creating embeddings.

    Delegates chunk construction to `_build_chunks()` and returns the resulting
    chunk table. This can be used to estimate how many chunks would be sent to an
    embeddings API without making any API calls.

    Args:
        df: Input corpus data in the format expected by `_build_chunks()`.
        config: Embeddings configuration used by `_build_chunks()` to determine
            chunking behavior.

    Returns:
        A DataFrame containing the chunks produced by `_build_chunks()`.
    """

    return _build_chunks(df, config)


def run_openai_embeddings_experiment(
    df: pd.DataFrame,
    config: EmbeddingsConfig,
    api_key: str,
) -> dict[str, Any]:
    """Generate OpenAI embeddings and derived analysis tables.

    Splits the corpus into chunks, generates embeddings for those chunks with an
    OpenAI client, aggregates chunk embeddings to work-level and author-level
    embeddings, and computes projection, similarity, nearest-neighbor, cohesion,
    and network outputs.

    Args:
        df: Input corpus data in the format expected by `_build_chunks()`.
        config: Embeddings configuration. Uses chunking, projection,
            nearest-neighbor, and network settings, and is also passed to
            `_embed_texts()`.
        api_key: OpenAI API key used to create the embeddings client.

    Returns:
        A dictionary containing generated embeddings, metadata tables, projection
        tables, similarity tables, nearest-neighbor summaries, pairwise author
        metrics, silhouette score, network figure, PCA variance, embedding
        dimension, API input count, and total chunk-token count.
    """

    chunks_df = _build_chunks(df, config)
    if chunks_df.empty:
        raise ValueError("No se han generado fragmentos. Revisa CHUNK_TOKENS y MIN_CHUNK_TOKENS.")

    client = OpenAI(api_key=api_key)
    texts = chunks_df["chunk_text"].tolist()
    chunk_embeddings = _embed_texts(texts, config, client)

    work_meta, work_embeddings = weighted_average_embeddings(
        chunks_df,
        chunk_embeddings,
        group_cols=["work_id", "author", "work", "filename"],
    )
    author_meta, author_embeddings = simple_average_embeddings(
        work_meta,
        work_embeddings,
        group_cols=["author"],
    )

    work_projection_df, pca_variance = _build_work_projections(work_meta, work_embeddings, config)
    chunks_projection_df = _build_chunk_projection(chunks_df, chunk_embeddings, config)
    work_similarity_df = _work_similarity_matrix(work_meta, work_embeddings)
    author_similarity_df = _author_similarity_matrix(work_meta, work_embeddings)
    nearest_neighbors_df = nearest_neighbors_table(
        work_meta.reset_index(drop=True),
        work_embeddings,
        top_k=config.nearest_neighbors_k,
    )
    neighbor_summary_df = _neighbor_summary(nearest_neighbors_df)
    pairs_df, author_metrics_df = pairwise_author_metrics(work_meta.reset_index(drop=True), work_embeddings)
    silhouette = _author_silhouette(work_meta, work_embeddings)
    network_fig = build_similarity_network_figure(
        work_meta.reset_index(drop=True), work_embeddings, config.network_top_k
    )

    chunk_display_df = chunks_df.drop(columns=["chunk_text"])
    return {
        "chunks_df": chunk_display_df,
        "work_meta": work_meta,
        "author_meta": author_meta,
        "chunk_embeddings": chunk_embeddings,
        "work_embeddings": work_embeddings,
        "author_embeddings": author_embeddings,
        "work_projection_df": work_projection_df,
        "chunks_projection_df": chunks_projection_df,
        "work_similarity_df": work_similarity_df,
        "author_similarity_df": author_similarity_df,
        "nearest_neighbors_df": nearest_neighbors_df,
        "neighbor_summary_df": neighbor_summary_df,
        "pairs_df": pairs_df,
        "author_metrics_df": author_metrics_df,
        "silhouette": silhouette,
        "network_fig": network_fig,
        "pca_variance": pca_variance,
        "embedding_dimension": int(work_embeddings.shape[1]),
        "n_api_inputs": int(len(chunks_df)),
        "total_chunk_tokens": int(chunks_df["token_count"].sum()),
    }


def build_similarity_network_figure(meta: pd.DataFrame, embeddings: np.ndarray, top_k: int) -> go.Figure:
    """Build a Plotly similarity network from nearest-neighbor embedding links.

    Builds a similarity graph from work metadata and embeddings, lays it out with
    NetworkX's spring layout, and renders it as a Plotly figure. Edges represent
    nearest-neighbor similarity links, and nodes are grouped into Plotly traces by
    author.

    Args:
        meta: Work-level metadata in the format expected by
            `build_similarity_network()`. Graph nodes are expected to expose
            `label`, `author`, and `work` attributes.
        embeddings: Embedding matrix aligned row-by-row with `meta`.
        top_k: Number of nearest-neighbor links to use per work when building
            the similarity network.

    Returns:
        A Plotly figure containing the similarity network.
    """

    graph = build_similarity_network(meta, embeddings, top_k=top_k)
    pos = nx.spring_layout(graph, seed=42, weight="weight")

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 0.8, "color": "#9ca3af"},
        hoverinfo="none",
        mode="lines",
        name="Similitud",
        showlegend=False,
    )

    node_rows = []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_rows.append({"x": x, "y": y, "label": data["label"], "author": data["author"], "work": data["work"]})
    node_df = pd.DataFrame(node_rows)

    fig = go.Figure(data=[edge_trace])
    for author, sub in node_df.groupby("author", sort=True):
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                marker={"size": 11},
                text=sub["label"],
                hovertemplate="%{text}<extra></extra>",
                name=str(author),
            )
        )

    fig.update_layout(
        title=f"Red de similitud entre obras, top-{top_k} vecinos por obra",
        height=760,
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False},
        legend={"title": "Autor"},
    )
    return fig


def _build_chunks(df: pd.DataFrame, config: EmbeddingsConfig) -> pd.DataFrame:
    """Split corpus works into token-based chunks.

    Encodes each work's text with the configured model encoding, splits it into
    token chunks, and returns one row per generated chunk with work metadata and
    chunk identifiers.

    Args:
        df: Input corpus data containing `text`, `author`, `work`, and
            `filename` columns.
        config: Embeddings configuration. Uses `model`, `chunk_tokens`,
            `chunk_overlap`, and `min_chunk_tokens` to build chunks.

    Returns:
        A DataFrame with columns `work_id`, `author`, `work`, `filename`,
        `chunk_index`, `chunk_text`, `token_count`, and `chunk_id`. The returned
        DataFrame has these columns even when no chunks are generated.
    """

    encoding = get_encoding(config.model)
    rows: list[dict[str, Any]] = []
    for work_id, row in df.reset_index(drop=True).iterrows():
        chunks = chunk_text_by_tokens(
            row["text"],
            config.chunk_tokens,
            config.chunk_overlap,
            config.min_chunk_tokens,
            encoding,
        )
        for chunk in chunks:
            rows.append(
                {
                    "work_id": work_id,
                    "author": row["author"],
                    "work": row["work"],
                    "filename": row["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "chunk_id": f"{work_id:05d}_{chunk['chunk_index']:05d}",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "work_id",
            "author",
            "work",
            "filename",
            "chunk_index",
            "chunk_text",
            "token_count",
            "chunk_id",
        ],
    )


def _embed_texts(texts: list[str], config: EmbeddingsConfig, client: OpenAI) -> np.ndarray:
    """Embed texts with an OpenAI client and return a normalized matrix.

    Computes cache keys for each text, reuses duplicate embeddings within the
    current call, sends uncached texts to the embeddings API in batches, stacks
    the resulting vectors into a matrix, and applies L2 normalization.

    Args:
        texts: Text inputs to embed.
        config: Embeddings configuration. Uses `model`, `dimensions`,
            `batch_size`, and `max_retries`.
        client: OpenAI client used by `call_embeddings_api()`.

    Returns:
        A two-dimensional NumPy array of L2-normalized embeddings with dtype
        `np.float32`. Rows correspond to the input `texts` order.
    """

    hashes = [text_hash(text, config.model, config.dimensions) for text in texts]
    in_memory_cache: dict[str, np.ndarray] = {}
    vectors: list[np.ndarray | None] = [None] * len(texts)

    missing_indices = []
    for i, hash_value in enumerate(hashes):
        if hash_value in in_memory_cache:
            vectors[i] = in_memory_cache[hash_value]
        else:
            missing_indices.append(i)

    for start in range(0, len(missing_indices), config.batch_size):
        batch_indices = missing_indices[start : start + config.batch_size]
        batch_texts = [texts[i] for i in batch_indices]
        batch_vectors = call_embeddings_api(
            batch_texts,
            config.model,
            config.dimensions,
            config.max_retries,
            client,
        )
        for i, vector in zip(batch_indices, batch_vectors, strict=False):
            array = np.array(vector, dtype=np.float32)
            in_memory_cache[hashes[i]] = array
            vectors[i] = array

    matrix = np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32)
    return normalize(matrix, norm="l2")


def _build_work_projections(
    work_meta: pd.DataFrame,
    work_embeddings: np.ndarray,
    config: EmbeddingsConfig,
) -> tuple[pd.DataFrame, list[float]]:
    """Build PCA and UMAP projection columns for work-level embeddings.

    Copies the work metadata, projects the work embeddings into up to three
    dimensions with PCA and UMAP, and appends the resulting coordinates as new
    columns.

    Args:
        work_meta: Work-level metadata aligned row-by-row with
            `work_embeddings`.
        work_embeddings: Two-dimensional embedding matrix for works.
        config: Embeddings configuration. Uses `random_state` for PCA and is
            passed to `_fit_umap()`.

    Returns:
        A tuple containing:
            - A copy of `work_meta` with added `pca_1`, `pca_2`, `pca_3`,
              `umap_1`, `umap_2`, and `umap_3` columns when three components
              are available. Fewer component columns are added when there are
              fewer works or embedding dimensions.
            - A list of PCA explained-variance ratios, one per PCA component.
    """

    n_components = min(3, len(work_embeddings), work_embeddings.shape[1])
    out = work_meta.copy()

    pca = PCA(n_components=n_components, random_state=config.random_state)
    pca_coords = pca.fit_transform(work_embeddings)
    for i in range(n_components):
        out[f"pca_{i + 1}"] = pca_coords[:, i]

    umap_coords = _fit_umap(work_embeddings, n_components=n_components, config=config)
    for i in range(n_components):
        out[f"umap_{i + 1}"] = umap_coords[:, i]

    return out, [float(value) for value in pca.explained_variance_ratio_]


def _build_chunk_projection(
    chunks_df: pd.DataFrame,
    chunk_embeddings: np.ndarray,
    config: EmbeddingsConfig,
) -> pd.DataFrame:
    """Build UMAP projection columns for chunk-level embeddings.

    Projects chunk embeddings into up to three UMAP dimensions and appends the
    resulting coordinates to a copy of the chunk metadata. The raw `chunk_text`
    column is removed from the returned DataFrame.

    Args:
        chunks_df: Chunk metadata aligned row-by-row with `chunk_embeddings`.
            Expected to contain a `chunk_text` column.
        chunk_embeddings: Two-dimensional embedding matrix for chunks.
        config: Embeddings configuration passed to `_fit_umap()`.

    Returns:
        A copy of `chunks_df` without `chunk_text` and with added `umap_1`,
        `umap_2`, and `umap_3` columns when three components are available.
        Fewer UMAP columns are added when there are fewer chunks or embedding
        dimensions.
    """

    n_components = min(3, len(chunk_embeddings), chunk_embeddings.shape[1])
    coords = _fit_umap(chunk_embeddings, n_components=n_components, config=config)
    out = chunks_df.drop(columns=["chunk_text"]).copy()
    for i in range(n_components):
        out[f"umap_{i + 1}"] = coords[:, i]
    return out


def _fit_umap(embeddings: np.ndarray, n_components: int, config: EmbeddingsConfig) -> np.ndarray:
    """Project embeddings with UMAP, falling back to PCA for small inputs.

    Uses PCA instead of UMAP when the number of embeddings is too small for a
    stable neighborhood-based projection. For larger inputs, fits a UMAP model
    with cosine distance and configuration-driven neighbor, minimum-distance,
    and random-state settings.

    Args:
        embeddings: Two-dimensional embedding matrix to project.
        n_components: Number of projection dimensions to compute.
        config: Embeddings configuration. Uses `random_state`,
            `umap_neighbors`, and `umap_min_dist`.

    Returns:
        A NumPy array of projected coordinates with one row per embedding and
        `n_components` columns.
    """

    if len(embeddings) <= max(5, n_components + 2):
        coords = PCA(n_components=n_components, random_state=config.random_state).fit_transform(embeddings)
        return coords

    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(config.umap_neighbors, max(2, len(embeddings) - 1)),
        min_dist=config.umap_min_dist,
        metric="cosine",
        random_state=config.random_state,
    )
    return model.fit_transform(embeddings)


def _work_similarity_matrix(work_meta: pd.DataFrame, work_embeddings: np.ndarray) -> pd.DataFrame:
    """Build a cosine-similarity matrix for work-level embeddings.

    Sorts works by author and title, reorders the embedding matrix to match that
    sorted metadata order, and returns a labeled square similarity matrix.

    Args:
        work_meta: Work-level metadata aligned row-by-row with `work_embeddings`.
            Expected to contain `author` and `work` columns.
        work_embeddings: Two-dimensional work embedding matrix.

    Returns:
        A square DataFrame whose rows and columns are labeled as
        `"{author} | {work}"` and whose values are pairwise cosine similarities
        between work embeddings.
    """

    ordered_meta = work_meta.sort_values(["author", "work"]).reset_index()
    ordered_embeddings = work_embeddings[ordered_meta["index"].to_numpy()]
    labels = [f"{row.author} | {row.work}" for row in ordered_meta.itertuples(index=False)]
    return pd.DataFrame(cosine_similarity(ordered_embeddings), index=labels, columns=labels)


def _author_similarity_matrix(work_meta: pd.DataFrame, work_embeddings: np.ndarray) -> pd.DataFrame:
    """Build an author-level cosine-similarity matrix from work embeddings.

    Computes pairwise cosine similarities between works, then aggregates those
    similarities by author pair. For same-author comparisons with more than one
    work, self-similarities on the diagonal are excluded.

    Args:
        work_meta: Work-level metadata aligned row-by-row with `work_embeddings`.
            Expected to contain an `author` column.
        work_embeddings: Two-dimensional work embedding matrix.

    Returns:
        A square DataFrame indexed and columned by sorted author names. Each
        value is the mean cosine similarity between works by the corresponding
        author pair.
    """

    sim = cosine_similarity(work_embeddings)
    authors = work_meta["author"].values
    author_names = sorted(work_meta["author"].unique())
    author_sim = pd.DataFrame(index=author_names, columns=author_names, dtype=float)

    for author_i in author_names:
        for author_j in author_names:
            idx_i = np.where(authors == author_i)[0]
            idx_j = np.where(authors == author_j)[0]
            values = sim[np.ix_(idx_i, idx_j)]
            if author_i == author_j and len(idx_i) > 1:
                values = values[~np.eye(len(idx_i), dtype=bool)]
            author_sim.loc[author_i, author_j] = float(values.mean())

    return author_sim


def _neighbor_summary(nearest_neighbors_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-author nearest-neighbor rates by neighbor rank.

    Groups nearest-neighbor rows by rank and computes the mean of the
    `same_author` indicator. The resulting mean is also expressed as a
    percentage.

    Args:
        nearest_neighbors_df: Nearest-neighbor table containing `neighbor_rank`
            and `same_author` columns. `same_author` is expected to be boolean
            or numeric.

    Returns:
        A DataFrame with columns `neighbor_rank`, `same_author`, and
        `pct_same_author`, where `pct_same_author` is `same_author * 100`.
    """

    summary = nearest_neighbors_df.groupby("neighbor_rank")["same_author"].mean().reset_index()
    summary["pct_same_author"] = 100 * summary["same_author"]
    return summary


def _author_silhouette(work_meta: pd.DataFrame, work_embeddings: np.ndarray) -> float:
    """Compute the author-label silhouette score for work embeddings.

    Encodes authors as categorical labels and computes a cosine-distance
    silhouette score for the work embeddings. Returns `nan` when the score is not
    defined because there are fewer than two author labels or because every work
    has a unique author label.

    Args:
        work_meta: Work-level metadata aligned row-by-row with `work_embeddings`.
            Expected to contain an `author` column.
        work_embeddings: Two-dimensional work embedding matrix.

    Returns:
        The cosine-distance silhouette score as a float, or `nan` when the score
        is not defined for the available author labels.
    """

    labels = work_meta["author"].astype("category").cat.codes.to_numpy()
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return float("nan")
    return float(silhouette_score(work_embeddings, labels, metric="cosine"))
