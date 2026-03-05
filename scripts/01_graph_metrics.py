import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# ==============================
# 输入
# ==============================
PATCH_PATH = "data/USG_Patch_08.shp"
GRID_PATH  = "data/Grid.shp"
REGION_PATH = "data/Region.shp"  # 可选裁剪（你有就用）

GRID_ID_COL = "网格ID"

# ==============================
# 输出
# ==============================
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# 敏感性检验的 d（米）
# ==============================
D_LIST = [200, 500, 1000]

# ------------------------------
# 工具：确保米制坐标（投影）
# ------------------------------
def ensure_projected(gdf):
    if gdf.crs is None:
        raise ValueError("❌ 数据缺少 CRS（.prj）。请确认 shp 的 prj 文件已上传且有效。")

    if gdf.crs.is_geographic:
        # 自动UTM
        centroid = gdf.geometry.unary_union.centroid
        lon, lat = centroid.x, centroid.y
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return gdf.to_crs(epsg=epsg)

    return gdf

# ------------------------------
# 读取数据
# ------------------------------
patch = gpd.read_file(PATCH_PATH)
grid  = gpd.read_file(GRID_PATH)
region = gpd.read_file(REGION_PATH)

# 裁剪（保险）
patch = gpd.overlay(patch, region, how="intersection")
grid  = gpd.overlay(grid,  region, how="intersection")

# 投影到米制
patch = ensure_projected(patch)
grid  = ensure_projected(grid.to_crs(patch.crs))

# 修复几何
patch["geometry"] = patch["geometry"].buffer(0)
grid["geometry"] = grid["geometry"].buffer(0)

# 检查网格ID
if GRID_ID_COL not in grid.columns:
    raise ValueError(f"❌ Grid.shp 缺少字段：{GRID_ID_COL}。请确认字段名完全一致（含大小写/中文）。")

# patch centroid
cent = patch.geometry.centroid
coords = np.column_stack([cent.x.values, cent.y.values])
n_nodes = len(coords)

# ------------------------------
# 跑一个 d 的函数
# ------------------------------
def run_one_d(D):
    # 构建图
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # O(n^2) 距离判断（斑块很多会慢；但先保证你能跑通）
    for i in range(n_nodes):
        xi, yi = coords[i]
        for j in range(i + 1, n_nodes):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dist = float(np.sqrt(dx*dx + dy*dy))
            if dist <= D:
                G.add_edge(i, j, weight=dist)

    # 指标（节点级）
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)  # 无权版更稳定
    closeness = nx.closeness_centrality(G)                       # 无权版更稳定

    patch_tmp = patch.copy()
    patch_tmp["degree"] = patch_tmp.index.map(lambda idx: degree.get(idx, 0))
    patch_tmp["betweenness"] = patch_tmp.index.map(lambda idx: betweenness.get(idx, 0.0))
    patch_tmp["w_close"] = patch_tmp.index.map(lambda idx: closeness.get(idx, 0.0))

    # 空间连接到网格，做均值汇总
    join = gpd.sjoin(patch_tmp[["degree","betweenness","w_close","geometry"]],
                     grid[[GRID_ID_COL,"geometry"]],
                     how="left", predicate="intersects")

    grid_metrics = (
        join.groupby(GRID_ID_COL)[["degree","betweenness","w_close"]]
        .mean()
        .reset_index()
    )
    grid_metrics["D"] = D
    grid_metrics["edges"] = G.number_of_edges()
    grid_metrics["nodes"] = n_nodes

    # 输出单个 d 的表
    out_csv = os.path.join(OUT_DIR, f"graph_grid_metrics_D{D}.csv")
    grid_metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(" saved:", out_csv, "| edges:", G.number_of_edges())

    return grid_metrics

# ------------------------------
# 1) 跑全部 d，输出每个 d 的表
# ------------------------------
all_results = []
for D in D_LIST:
    all_results.append(run_one_d(D))

df_all = pd.concat(all_results, ignore_index=True)

# 输出 long-format 总表
out_all = os.path.join(OUT_DIR, "graph_grid_metrics_ALL_D.csv")
df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
print(" saved:", out_all)

# ------------------------------
# 2) 敏感性检验表（每个 d：均值、标准差、变异系数）
# ------------------------------
def summarize_metric(df, col):
    g = df.groupby("D")[col]
    out = pd.DataFrame({
        "D": g.mean().index.values,
        f"{col}_mean": g.mean().values,
        f"{col}_std": g.std(ddof=1).values
    })
    out[f"{col}_cv"] = out[f"{col}_std"] / out[f"{col}_mean"].replace(0, np.nan)
    return out

sum_degree = summarize_metric(df_all, "degree")
sum_betw  = summarize_metric(df_all, "betweenness")
sum_close = summarize_metric(df_all, "w_close")

summary = sum_degree.merge(sum_betw, on="D").merge(sum_close, on="D")

# 加上图结构信息（nodes/edges）
edges_by_d = df_all.groupby("D")[["edges","nodes"]].max().reset_index()
summary = summary.merge(edges_by_d, on="D", how="left")

out_summary = os.path.join(OUT_DIR, "sensitivity_summary_table.csv")
summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
print(" saved:", out_summary)

# ------------------------------
# 3) 出图：均值随 D 变化（折线）
# ------------------------------
plt.figure(figsize=(8.5, 5.2), dpi=160)
plt.plot(summary["D"], summary["degree_mean"], marker="o", label="degree_mean")
plt.plot(summary["D"], summary["betweenness_mean"], marker="o", label="betweenness_mean")
plt.plot(summary["D"], summary["w_close_mean"], marker="o", label="w_close_mean")
plt.xlabel("Distance threshold D (m)")
plt.ylabel("Mean metric (grid-level)")
plt.title("Sensitivity: Mean Graph Metrics vs D")
plt.legend()
plt.tight_layout()
out_fig1 = os.path.join(OUT_DIR, "sensitivity_mean_vs_D.png")
plt.savefig(out_fig1, bbox_inches="tight")
plt.close()
print(" saved:", out_fig1)

# ------------------------------
# 4) 出图：箱线图（每个 D 的分布）
# ------------------------------
def boxplot_metric(col, title, outname):
    plt.figure(figsize=(8.5, 5.2), dpi=160)
    data = [df_all[df_all["D"]==D][col].dropna().values for D in D_LIST]
    plt.boxplot(data, labels=[str(D) for D in D_LIST], showfliers=False)
    plt.xlabel("Distance threshold D (m)")
    plt.ylabel(col)
    plt.title(title)
    plt.tight_layout()
    outp = os.path.join(OUT_DIR, outname)
    plt.savefig(outp, bbox_inches="tight")
    plt.close()
    print("saved:", outp)

boxplot_metric("degree", "Sensitivity: degree distribution by D", "sensitivity_box_degree.png")
boxplot_metric("betweenness", "Sensitivity: betweenness distribution by D", "sensitivity_box_betweenness.png")
boxplot_metric("w_close", "Sensitivity: w_close distribution by D", "sensitivity_box_w_close.png")

print("DONE: d =", D_LIST)
