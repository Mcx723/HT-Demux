#!/usr/bin/env Rscript
# BFF.R (cellhashR)

suppressPackageStartupMessages({
  library(cellhashR)
  library(Matrix)
})

input_dir <- "./data/BFF"
output_dir <- "./results/BFF"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

script_start_time <- Sys.time()

# =========================
# Read input
# =========================
matrix_mtx <- readMM(file.path(input_dir, "matrix.mtx"))
barcodes <- readLines(file.path(input_dir, "barcodes.tsv"))

features_raw <- read.table(
  file.path(input_dir, "features.tsv"),
  sep = "\t",
  header = FALSE,
  stringsAsFactors = FALSE
)

matrix <- as.matrix(matrix_mtx)

# matrix: HTO x droplets
colnames(matrix) <- barcodes

# 如果 features.tsv 是：
# HTO_sim_01    HTO_sim_01    Antibody Capture
# 用第 1 列最稳，避免读到重复或整行字符串
rownames(matrix) <- make.unique(as.character(features_raw[, 1]))

cat("Input matrix dimension:", nrow(matrix), "HTOs x", ncol(matrix), "droplets\n")

# =========================
# Run BFF / cellhashR with runtime
# =========================
method_start_time <- Sys.time()

res <- GenerateCellHashingCalls(
  barcodeMatrix = matrix,
  methods = c("bff_raw", "bff_cluster"),
  minCountPerCell = 0,
  doTSNE = FALSE,
  doHeatmap = FALSE,
  verbose = TRUE
)

method_end_time <- Sys.time()
method_runtime_sec <- as.numeric(difftime(method_end_time, method_start_time, units = "secs"))

cat("BFF method runtime:", method_runtime_sec, "seconds\n")

# =========================
# Save classification
# =========================
write.csv(
  res,
  file = file.path(output_dir, "BFF_classification.csv"),
  row.names = FALSE
)

# =========================
# Save summary
# =========================
norm_mat <- t(apply(matrix, 2, function(x) {
  if (sum(x) > 0) {
    x / sum(x)
  } else {
    x
  }
}))

top_indices <- apply(norm_mat, 1, which.max)

final_assign <- if ("consensuscall" %in% colnames(res)) {
  res$consensuscall
} else {
  rep(NA, ncol(matrix))
}

bff_raw <- if ("bff_raw" %in% colnames(res)) {
  res$bff_raw
} else {
  rep(NA, ncol(matrix))
}

bff_cluster <- if ("bff_cluster" %in% colnames(res)) {
  res$bff_cluster
} else {
  rep(NA, ncol(matrix))
}

summary_df <- data.frame(
  droplet_id = colnames(matrix),
  nHTO_total = colSums(matrix),
  HTO_maxID = rownames(matrix)[top_indices],
  final_assign = final_assign,
  bff_raw = bff_raw,
  bff_cluster = bff_cluster,
  stringsAsFactors = FALSE
)

write.csv(
  summary_df,
  file = file.path(output_dir, "BFF_summary.csv"),
  row.names = FALSE
)

# =========================
# Save runtime
# =========================
script_end_time <- Sys.time()
total_runtime_sec <- as.numeric(difftime(script_end_time, script_start_time, units = "secs"))

runtime_df <- data.frame(
  method = "BFF_cellhashR",
  input_dir = input_dir,
  n_hto = nrow(matrix),
  n_droplets = ncol(matrix),
  methods = "bff_raw,bff_cluster",
  minCountPerCell = 0,
  doTSNE = FALSE,
  doHeatmap = FALSE,
  method_start_time = as.character(method_start_time),
  method_end_time = as.character(method_end_time),
  script_start_time = as.character(script_start_time),
  script_end_time = as.character(script_end_time),
  method_runtime_sec = method_runtime_sec,
  total_runtime_sec = total_runtime_sec,
  stringsAsFactors = FALSE
)

write.csv(
  runtime_df,
  file = file.path(output_dir, "BFF_runtime.csv"),
  row.names = FALSE
)

# =========================
# Save complete result
# =========================
saveRDS(
  res,
  file = file.path(output_dir, "BFF_complete_results.rds")
)

cat("BFF total runtime:", total_runtime_sec, "seconds\n")
cat("Finished BFF / cellhashR.\n")