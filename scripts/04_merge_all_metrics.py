from pathlib import Path

import pandas as pd

D_LIST = [200, 500, 1000]
GRID_ID_COL = "网格ID"

GRAPH_DIR = Path("outputs/graph")
MAK_DIR = Path("outputs/makurhini")
CIR_DIR = Path("outputs/circuit")
OUT_DIR = Path("outputs/final")


def read_one(path: Path, d: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["D"] = d
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_parts = []
    wide_parts = []

    for d in D_LIST:
        graph = read_one(GRAPH_DIR / f"graph_grid_metrics_D{d}.csv", d)
        mak = read_one(MAK_DIR / f"makurhini_grid_metrics_D{d}.csv", d)
        cir = read_one(CIR_DIR / f"circuit_grid_metrics_D{d}.csv", d)

        merged = graph.merge(mak, on=[GRID_ID_COL, "D"], how="outer").merge(cir, on=[GRID_ID_COL, "D"], how="outer")
        long_parts.append(merged)

        renamed = merged.drop(columns=["D"]).rename(
            columns={c: f"{c}_D{d}" for c in merged.columns if c not in [GRID_ID_COL, "D"]}
        )
        wide_parts.append(renamed)

    long_df = pd.concat(long_parts, ignore_index=True)
    wide_df = wide_parts[0]
    for part in wide_parts[1:]:
        wide_df = wide_df.merge(part, on=GRID_ID_COL, how="outer")

    long_out = OUT_DIR / "grid_network_metrics_1493_long.csv"
    wide_out = OUT_DIR / "grid_network_metrics_1493_wide.csv"
    long_df.to_csv(long_out, index=False, encoding="utf-8-sig")
    wide_df.to_csv(wide_out, index=False, encoding="utf-8-sig")

    print(f"[OUT] {wide_out}")
    print(f"[OUT] {long_out}")


if __name__ == "__main__":
    main()
