from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PATCH_PATH = "data/USG_Patch_08.shp"
GRID_PATH = "data/Grid.shp"
REGION_PATH = "data/Region.shp"
GRID_ID_COL = "网格ID"
D_LIST = [200, 500, 1000]
RESOLUTION = 100
OUT_DIR = Path("outputs/circuit")


def ensure_projected_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS")
    if gdf.crs.is_projected:
        return gdf
    c = gdf.unary_union.centroid
    zone = int((c.x + 180) // 6) + 1
    epsg = 32600 + zone if c.y >= 0 else 32700 + zone
    print(f"[INFO] Geographic CRS detected, reprojecting EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)


def build_resistance_raster(region, patch):
    minx, miny, maxx, maxy = region.total_bounds
    width = int(np.ceil((maxx - minx) / RESOLUTION))
    height = int(np.ceil((maxy - miny) / RESOLUTION))
    transform = from_origin(minx, maxy, RESOLUTION, RESOLUTION)

    arr = np.full((height, width), 100.0, dtype=np.float32)
    patch_shapes = [(geom, 1.0) for geom in patch.geometry if geom is not None and not geom.is_empty]
    patch_raster = rasterize(patch_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    arr[patch_raster == 1] = 1.0

    region_mask = geometry_mask(region.geometry, transform=transform, invert=True, out_shape=(height, width))
    arr[~region_mask] = np.nan
    return arr, transform


def cell_index(r, c, cols):
    return r * cols + c


def least_cost_proxy(resistance: np.ndarray, node_rc):
    """Fallback proxy when Circuitscape is unavailable.

    We compute a raster 'current-like' intensity as the sum of inverse least-cost distance
    from each source node to all pixels using 4-neighbor graph Dijkstra.
    """
    h, w = resistance.shape
    valid = np.isfinite(resistance)
    n = h * w

    rows, cols, data = [], [], []
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            u = cell_index(r, c, w)
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and valid[rr, cc]:
                    v = cell_index(rr, cc, w)
                    wgt = (float(resistance[r, c]) + float(resistance[rr, cc])) / 2.0
                    rows.append(u)
                    cols.append(v)
                    data.append(wgt)

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    current = np.zeros((h, w), dtype=np.float64)

    sources = [cell_index(r, c, w) for r, c in node_rc if 0 <= r < h and 0 <= c < w and valid[r, c]]
    for src in sources:
        dist = dijkstra(graph, directed=False, indices=src)
        dist = dist.reshape(h, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = 1.0 / (dist + 1.0)
        contrib[~np.isfinite(contrib)] = 0.0
        current += contrib

    return current


def summarize_over_grid(current, transform, grid_geom):
    mask = geometry_mask([grid_geom], transform=transform, invert=True, out_shape=current.shape)
    vals = current[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.sum(vals)), float(np.max(vals))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch = gpd.read_file(PATCH_PATH)
    grid = gpd.read_file(GRID_PATH)
    region = gpd.read_file(REGION_PATH)

    if GRID_ID_COL not in grid.columns:
        raise ValueError(f"Missing field {GRID_ID_COL} in grid")

    region = ensure_projected_meters(region)
    patch = ensure_projected_meters(patch).to_crs(region.crs)
    grid = ensure_projected_meters(grid).to_crs(region.crs)

    patch = gpd.clip(patch, region)
    grid = gpd.clip(grid, region)

    resistance, transform = build_resistance_raster(region, patch)
    centroids = patch.geometry.centroid

    print("[INFO] Circuitscape package check...")
    try:
        import circuitscape  # noqa: F401
        print("[INFO] Circuitscape detected, but this script uses documented least-cost proxy for runtime control.")
    except Exception:
        print("[INFO] Circuitscape not available, using least-cost proxy fallback.")

    all_rows = []
    for d in D_LIST:
        print(f"[INFO] Running circuit proxy for D={d}")
        d_rows = []
        for _, grow in grid.iterrows():
            gid = grow[GRID_ID_COL]
            aoi = grow.geometry.buffer(d).intersection(region.unary_union)
            nodes = centroids[centroids.within(aoi)]

            if len(nodes) < 2:
                d_rows.append({"网格ID": gid, "D": d, "mean_current": 0.0, "sum_current": 0.0, "max_current": 0.0})
                continue

            inv = ~transform
            node_rc = []
            for pt in nodes:
                c, r = inv * (pt.x, pt.y)
                node_rc.append((int(np.floor(r)), int(np.floor(c))))

            current = least_cost_proxy(resistance, node_rc)
            mean_c, sum_c, max_c = summarize_over_grid(current, transform, grow.geometry)
            d_rows.append({"网格ID": gid, "D": d, "mean_current": mean_c, "sum_current": sum_c, "max_current": max_c})

        df_d = pd.DataFrame(d_rows)
        out_d = OUT_DIR / f"circuit_grid_metrics_D{d}.csv"
        df_d.to_csv(out_d, index=False, encoding="utf-8-sig")
        print(f"[OUT] {out_d}")
        all_rows.append(df_d)

    df_all = pd.concat(all_rows, ignore_index=True)
    out_all = OUT_DIR / "circuit_metrics_ALL_D.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[OUT] {out_all}")

    summary_rows = []
    for d, part in df_all.groupby("D"):
        row = {"D": d}
        for col in ["mean_current", "sum_current", "max_current"]:
            m = float(part[col].mean())
            s = float(part[col].std(ddof=1)) if len(part) > 1 else 0.0
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
            row[f"{col}_cv"] = s / m if m != 0 else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("D")
    out_summary = OUT_DIR / "sensitivity_summary_table.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"[OUT] {out_summary}")

    plt.figure(figsize=(8, 5), dpi=150)
    for col in ["mean_current_mean", "sum_current_mean", "max_current_mean"]:
        plt.plot(summary["D"], summary[col], marker="o", label=col)
    plt.xlabel("D (m)")
    plt.ylabel("Mean across grids")
    plt.title("Circuit proxy sensitivity")
    plt.legend()
    plt.tight_layout()
    p = OUT_DIR / "sensitivity_mean_vs_D.png"
    plt.savefig(p)
    plt.close()
    print(f"[OUT] {p}")

    for col in ["mean_current", "sum_current", "max_current"]:
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

    print("[DONE] Circuit metrics finished.")


if __name__ == "__main__":
    main()
