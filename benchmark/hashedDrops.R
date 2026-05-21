#!/usr/bin/env Rscript
# hashedDrops.R

suppressPackageStartupMessages({
  library(Matrix)
  library(DropletUtils)
})

input_dir <- "./data/hashedDrops"
output_dir <- "./results/hashedDrops"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

hd_params <- list(
  min.prop = 0.05,
  pseudo.count = 5,
  doublet.nmads = 3,
  doublet.min = 2,
  doublet.mixture = FALSE,
  confident.nmads = 3,
  confident.min = 2
)

# =========================
# Read input
# =========================
matrix <- as.matrix(readMM(file.path(input_dir, "matrix.mtx")))
barcodes <- readLines(file.path(input_dir, "barcodes.tsv"))

features <- read.table(
  file.path(input_dir, "features.tsv"),
  sep = "\t",
  header = FALSE,
  stringsAsFactors = FALSE
)

colnames(matrix) <- barcodes

# 如果 features.tsv 是：
# HTO_sim_01    HTO_sim_01    Antibody Capture
# 用第 1 列或第 2 列都可以。这里用第 1 列最稳。
rownames(matrix) <- make.unique(as.character(features[, 1]))

cat("Input matrix dimension:", nrow(matrix), "HTOs x", ncol(matrix), "droplets\n")

# =========================
# Run hashedDrops with runtime
# =========================
start_time <- Sys.time()

hd_res <- do.call(
  DropletUtils::hashedDrops,
  c(list(x = matrix), hd_params)
)

end_time <- Sys.time()
runtime_sec <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("hashedDrops runtime:", runtime_sec, "seconds\n")

runtime_df <- data.frame(
  method = "hashedDrops",
  input_dir = input_dir,
  n_hto = nrow(matrix),
  n_droplets = ncol(matrix),
  start_time = as.character(start_time),
  end_time = as.character(end_time),
  runtime_sec = runtime_sec,
  stringsAsFactors = FALSE
)

write.csv(
  runtime_df,
  file = file.path(output_dir, "hashedDrops_runtime.csv"),
  row.names = FALSE
)

# =========================
# Save classification
# =========================
hd_df <- as.data.frame(hd_res)
hd_df$droplet_id <- rownames(hd_df)

write.csv(
  hd_df,
  file = file.path(output_dir, "hashedDrops_classification.csv"),
  row.names = FALSE
)

# =========================
# Save summary
# =========================
norm_mat <- t(apply(matrix, 2, function(x) {
  if(sum(x) > 0) {
    x / sum(x)
  } else {
    x
  }
}))

top_indices <- apply(norm_mat, 1, which.max)

summary_df <- data.frame(
  droplet_id = hd_df$droplet_id,
  nHTO_total = colSums(matrix),
  HTO_maxID = rownames(matrix)[top_indices],
  HTO_margin = hd_df$LogFC,
  stringsAsFactors = FALSE
)

write.csv(
  summary_df,
  file = file.path(output_dir, "hashedDrops_summary.csv"),
  row.names = FALSE
)

saveRDS(
  hd_res,
  file = file.path(output_dir, "hashedDrops_result.rds")
)

cat("Finished hashedDrops.\n")