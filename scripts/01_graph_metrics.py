import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PATCH_PATH = "data/USG_Patch_08.shp"
GRID_PATH = "data/Grid.shp"
REGION_PATH = "data/Region.shp"
GRID_ID_COL = "网格ID"
D_LIST = [200, 500, 1000]
OUT_DIR = Path("outputs/graph")


def ensure_projected_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS.")
    if gdf.crs.is_projected:
        return gdf

    centroid = gdf.unary_union.centroid
    lon, lat = float(centroid.x), float(centroid.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    print(f"[INFO] Geographic CRS detected, reprojecting to EPSG:{epsg}.")
    return gdf.to_crs(epsg=epsg)


def prepare_data():
    patch = gpd.read_file(PATCH_PATH)
    grid = gpd.read_file(GRID_PATH)
    region = gpd.read_file(REGION_PATH)

    if GRID_ID_COL not in grid.columns:
        raise ValueError(f"{GRID_ID_COL} not found in grid fields: {grid.columns.tolist()}")

    region = ensure_projected_meters(region)
    patch = ensure_projected_meters(patch).to_crs(region.crs)
    grid = ensure_projected_meters(grid).to_crs(region.crs)

    patch = gpd.clip(patch, region)
    grid = gpd.clip(grid, region)

    patch = patch[~patch.geometry.is_empty & patch.geometry.notna()].copy()
    grid = grid[~grid.geometry.is_empty & grid.geometry.notna()].copy()
    patch["geometry"] = patch.geometry.buffer(0)
    grid["geometry"] = grid.geometry.buffer(0)

    print(f"[INFO] Prepared data: patches={len(patch)}, grids={len(grid)}")
    return patch, grid


def build_graph_metrics(patch: gpd.GeoDataFrame, d: float):
    centroids = patch.geometry.centroid
    coords = np.column_stack((centroids.x.values, centroids.y.values))
    n_nodes = len(coords)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=d)
    for i, j in pairs:
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        graph.add_edge(i, j, weight=dist)

    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    closeness = nx.closeness_centrality(graph)

    metrics = patch.copy()
    metrics["degree"] = [degree.get(i, 0) for i in range(n_nodes)]
    metrics["betweenness"] = [betweenness.get(i, 0.0) for i in range(n_nodes)]
    metrics["w_close"] = [closeness.get(i, 0.0) for i in range(n_nodes)]

    return metrics, graph.number_of_nodes(), graph.number_of_edges()


def summarize_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, part in df.groupby("D"):
        row = {"D": d, "nodes": int(part["nodes"].max()), "edges": int(part["edges"].max())}
        for col in ["degree", "betweenness", "w_close"]:
            mean_v = float(part[col].mean())
            std_v = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
            cv_v = std_v / mean_v if mean_v != 0 else np.nan
            row[f"{col}_mean"] = mean_v
            row[f"{col}_std"] = std_v
            row[f"{col}_cv"] = cv_v
        rows.append(row)
    return pd.DataFrame(rows).sort_values("D")


def make_plots(df_all: pd.DataFrame, summary: pd.DataFrame):
    plt.figure(figsize=(8, 5), dpi=150)
    for col in ["degree_mean", "betweenness_mean", "w_close_mean"]:
        plt.plot(summary["D"], summary[col], marker="o", label=col)
    plt.xlabel("D (m)")
    plt.ylabel("Mean value across grids")
    plt.title("Graph metric sensitivity")
    plt.legend()
    plt.tight_layout()
    p = OUT_DIR / "sensitivity_mean_vs_D.png"
    plt.savefig(p)
    plt.close()
    print(f"[OUT] {p}")

    for col in ["degree", "betweenness", "w_close"]:
        plt.figure(figsize=(8, 5), dpi=150)
        data = [df_all.loc[df_all["D"] == d, col].dropna().values for d in D_LIST]
        plt.boxplot(data, tick_labels=[str(d) for d in D_LIST], showfliers=False)
        plt.xlabel("D (m)")
        plt.ylabel(col)
        plt.title(f"{col} by D")
        plt.tight_layout()
        p = OUT_DIR / f"sensitivity_box_{col}.png"
        plt.savefig(p)
        plt.close()
        print(f"[OUT] {p}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch, grid = prepare_data()

    long_rows = []
    for d in D_LIST:
        print(f"[INFO] Running graph metrics for D={d}...")
        patch_metrics, n_nodes, n_edges = build_graph_metrics(patch, d)

        joined = gpd.sjoin(
            patch_metrics[["degree", "betweenness", "w_close", "geometry"]],
            grid[[GRID_ID_COL, "geometry"]],
            predicate="intersects",
            how="inner",
        )

        agg = joined.groupby(GRID_ID_COL)[["degree", "betweenness", "w_close"]].mean().reset_index()
        agg["D"] = d
        agg["nodes"] = n_nodes
        agg["edges"] = n_edges

        out_d = OUT_DIR / f"graph_grid_metrics_D{d}.csv"
        agg.to_csv(out_d, index=False, encoding="utf-8-sig")
        print(f"[OUT] {out_d}")

        long_rows.append(agg)

    df_all = pd.concat(long_rows, ignore_index=True)
    all_path = OUT_DIR / "graph_metrics_ALL_D.csv"
    df_all.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] {all_path}")

    summary = summarize_sensitivity(df_all)
    summary_path = OUT_DIR / "sensitivity_summary_table.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OUT] {summary_path}")

    make_plots(df_all, summary)
    print("[DONE] Graph metrics finished.")


if __name__ == "__main__":
    main()
