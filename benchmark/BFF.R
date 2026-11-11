#!/usr/bin/env Rscript
# BFF.R (cellhashR)

suppressPackageStartupMessages({
  library(cellhashR)
  library(Matrix)
})

input_dir <- Sys.getenv("BENCH_INPUT_DIR", "./data/BFF")
output_dir <- Sys.getenv("BENCH_OUTPUT_DIR", "./results/BFF")
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

strict_defaults <- Sys.getenv("BENCH_DEFAULT_PARAMETERS", "0") == "1"
min_hto_total_for_fit <- if (strict_defaults) 0 else as.numeric(Sys.getenv("BFF_MIN_HTO_TOTAL_FOR_FIT", "50"))
fit_cells <- colSums(matrix) >= min_hto_total_for_fit
cat(
  "BFF fitting cells:", sum(fit_cells),
  "of", ncol(matrix),
  "with nHTO_total >=", min_hto_total_for_fit, "\n"
)

call_args <- list(
  barcodeMatrix = matrix[, fit_cells, drop = FALSE],
  methods = c("bff_raw", "bff_cluster"),
  methodsForConsensus = c("bff_raw", "bff_cluster"),
  cellbarcodeWhitelist = colnames(matrix),
  doTSNE = FALSE,
  doHeatmap = FALSE,
  verbose = TRUE
)
if (!strict_defaults) call_args <- c(call_args, list(
  bff_raw.min_average_reads = 0, bff_cluster.min_average_reads = 0,
  bff_cluster.doublet_thresh = 0.05, bff_cluster.neg_thresh = 0.2,
  bff_cluster.dist_frac = 0.1, maxAllowableDoubletRate = 1,
  minAllowableSingletRate = 0, majorityConsensusThreshold = 0.5
))
writeLines(capture.output(formals(GenerateCellHashingCalls)), file.path(output_dir, "BFF_function_defaults.txt"))
saveRDS(call_args[names(call_args) != "barcodeMatrix"], file.path(output_dir, "BFF_call_parameters.rds"))
res <- do.call(GenerateCellHashingCalls, call_args)

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
  final_assign = final_assign[match(colnames(matrix), res$cellbarcode)],
  bff_raw = bff_raw[match(colnames(matrix), res$cellbarcode)],
  bff_cluster = bff_cluster[match(colnames(matrix), res$cellbarcode)],
  used_for_fit = fit_cells,
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
  n_droplets_used_for_fit = sum(fit_cells),
  methods = "bff_raw,bff_cluster",
  statistical_defaults = strict_defaults,
  minHTOTotalForFit = min_hto_total_for_fit,
  bff_cluster_doublet_thresh = if (strict_defaults) NA_real_ else 0.05,
  bff_cluster_neg_thresh = if (strict_defaults) NA_real_ else 0.2,
  bff_cluster_dist_frac = if (strict_defaults) NA_real_ else 0.1,
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
