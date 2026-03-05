suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(Makurhini)
})

patch_path <- "data/USG_Patch_08.shp"
grid_path <- "data/Grid.shp"
region_path <- "data/Region.shp"
grid_id_col <- "网格ID"
d_list <- c(200, 500, 1000)
out_dir <- "outputs/makurhini"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ensure_projected_meters <- function(gdf) {
  crs_obj <- st_crs(gdf)
  if (is.na(crs_obj)) stop("Input has no CRS.")

  if (sf::st_is_longlat(gdf)) {
    ctd <- st_centroid(st_union(st_geometry(gdf)))
    lonlat <- st_coordinates(st_transform(ctd, 4326))
    lon <- lonlat[1]
    lat <- lonlat[2]
    zone <- floor((lon + 180) / 6) + 1
    epsg <- ifelse(lat >= 0, 32600 + zone, 32700 + zone)
    message(sprintf("[INFO] Geographic CRS detected, reproject to EPSG:%s", epsg))
    return(st_transform(gdf, epsg))
  }
  gdf
}

patch <- st_read(patch_path, quiet = TRUE)
grid <- st_read(grid_path, quiet = TRUE)
region <- st_read(region_path, quiet = TRUE)

if (!(grid_id_col %in% names(grid))) {
  stop(sprintf("Field %s not found in grid", grid_id_col))
}

region <- ensure_projected_meters(region)
patch <- st_transform(ensure_projected_meters(patch), st_crs(region))
grid <- st_transform(ensure_projected_meters(grid), st_crs(region))

patch <- st_intersection(patch, region)
grid <- st_intersection(grid, region)

calc_metrics_one_grid <- function(grid_geom, patch_sf, d) {
  # Makurhini API note:
  # MK_dPCIIC(nodes, metric = c("PC", "IIC"), distance_thresholds = d, probability = 0.5)
  # expects node polygons and computes indices from Euclidean links under threshold.
  p_sel <- patch_sf[st_intersects(patch_sf, grid_geom, sparse = FALSE)[,1], ]

  if (nrow(p_sel) < 2) {
    return(data.frame(PC = 0, IIC = 0, ECA = 0))
  }

  mk_res <- tryCatch({
    MK_dPCIIC(
      nodes = p_sel,
      metric = c("PC", "IIC"),
      distance = d,
      probability = 0.5,
      LA = "area"
    )
  }, error = function(e) NULL)

  if (is.null(mk_res)) {
    return(data.frame(PC = 0, IIC = 0, ECA = 0))
  }

  # common MK_dPCIIC outputs include PC and IIC in summary tables.
  pc <- suppressWarnings(as.numeric(mk_res$PC[1]))
  iic <- suppressWarnings(as.numeric(mk_res$IIC[1]))

  if (is.na(pc)) pc <- 0
  if (is.na(iic)) iic <- 0

  # ECA derived from PC: ECA = sqrt(PC * landscape_area)
  a_total <- as.numeric(sum(st_area(p_sel)))
  eca <- sqrt(max(pc, 0) * max(a_total, 0))

  data.frame(PC = pc, IIC = iic, ECA = eca)
}

all_rows <- list()

for (d in d_list) {
  message(sprintf("[INFO] Running Makurhini metrics for D=%s", d))
  rows <- list()

  for (i in seq_len(nrow(grid))) {
    gid <- as.character(grid[[grid_id_col]][i])
    metr <- calc_metrics_one_grid(grid[i, ], patch, d)
    rows[[i]] <- data.frame(网格ID = gid, D = d, PC = metr$PC, IIC = metr$IIC, ECA = metr$ECA)
  }

  out_d <- bind_rows(rows)
  out_csv <- file.path(out_dir, sprintf("makurhini_grid_metrics_D%s.csv", d))
  write.csv(out_d, out_csv, row.names = FALSE, fileEncoding = "UTF-8")
  message(sprintf("[OUT] %s", out_csv))

  all_rows[[as.character(d)]] <- out_d
}

all_df <- bind_rows(all_rows)
all_csv <- file.path(out_dir, "makurhini_metrics_ALL_D.csv")
write.csv(all_df, all_csv, row.names = FALSE, fileEncoding = "UTF-8")
message(sprintf("[OUT] %s", all_csv))

summary_df <- all_df %>%
  group_by(D) %>%
  summarise(
    PC_mean = mean(PC, na.rm = TRUE), PC_std = sd(PC, na.rm = TRUE), PC_cv = ifelse(PC_mean == 0, NA, PC_std / PC_mean),
    IIC_mean = mean(IIC, na.rm = TRUE), IIC_std = sd(IIC, na.rm = TRUE), IIC_cv = ifelse(IIC_mean == 0, NA, IIC_std / IIC_mean),
    ECA_mean = mean(ECA, na.rm = TRUE), ECA_std = sd(ECA, na.rm = TRUE), ECA_cv = ifelse(ECA_mean == 0, NA, ECA_std / ECA_mean),
    .groups = "drop"
  )

summary_csv <- file.path(out_dir, "sensitivity_summary_table.csv")
write.csv(summary_df, summary_csv, row.names = FALSE, fileEncoding = "UTF-8")
message(sprintf("[OUT] %s", summary_csv))

png(file.path(out_dir, "sensitivity_mean_vs_D.png"), width = 1200, height = 700)
matplot(
  summary_df$D,
  as.matrix(summary_df[, c("PC_mean", "IIC_mean", "ECA_mean")]),
  type = "b", pch = 19, lty = 1,
  xlab = "D (m)", ylab = "Mean metric", main = "Makurhini sensitivity"
)
legend("topleft", legend = c("PC", "IIC", "ECA"), col = 1:3, lty = 1, pch = 19)
dev.off()

png(file.path(out_dir, "sensitivity_box_PC.png"), width = 1200, height = 700)
boxplot(PC ~ D, data = all_df, outline = FALSE, xlab = "D", ylab = "PC", main = "PC by D")
dev.off()

png(file.path(out_dir, "sensitivity_box_IIC.png"), width = 1200, height = 700)
boxplot(IIC ~ D, data = all_df, outline = FALSE, xlab = "D", ylab = "IIC", main = "IIC by D")
dev.off()

png(file.path(out_dir, "sensitivity_box_ECA.png"), width = 1200, height = 700)
boxplot(ECA ~ D, data = all_df, outline = FALSE, xlab = "D", ylab = "ECA", main = "ECA by D")
dev.off()

message("[DONE] Makurhini metrics finished.")
