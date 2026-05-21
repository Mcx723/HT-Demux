#!/usr/bin/env Rscript
# hashDemux.R

suppressPackageStartupMessages({
  library(Seurat)
  library(hashDemux)
  library(Matrix)
})

input_dir <- "./data/hashDemux"
output_dir <- "./results/hashDemux"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

script_start_time <- Sys.time()

# =========================
# Read input
# =========================
matrix <- readMM(file.path(input_dir, "matrix.mtx"))
barcodes <- readLines(file.path(input_dir, "barcodes.tsv"))

features <- read.table(
  file.path(input_dir, "features.tsv"),
  sep = "\t",
  header = FALSE,
  stringsAsFactors = FALSE
)

matrix <- as.matrix(matrix)
colnames(matrix) <- barcodes
rownames(matrix) <- make.unique(as.character(features[, 1]))

cat("Input matrix dimension:", nrow(matrix), "HTOs x", ncol(matrix), "droplets\n")

# =========================
# Create Seurat object
# =========================
seurat_obj <- CreateSeuratObject(
  counts = matrix,
  assay = "HTO"
)

# =========================
# Run hashDemux with runtime
# =========================
method_start_time <- Sys.time()

seurat_obj <- NormalizeData(
  seurat_obj,
  assay = "HTO",
  normalization.method = "CLR",
  margin = 2,
  verbose = FALSE
)

seurat_obj <- clustering_based_demux(
  seurat_object = seurat_obj,
  assay = "HTO",
  expected_doublet_rate = 0.05,
  knns = seq(5, 30, 5),
  resolutions = c(1, 2, 3, 4),
  nCores = NULL
)

method_end_time <- Sys.time()
method_runtime_sec <- as.numeric(difftime(method_end_time, method_start_time, units = "secs"))

cat("hashDemux method runtime:", method_runtime_sec, "seconds\n")

# =========================
# Save classification
# =========================
write.csv(
  seurat_obj@meta.data,
  file = file.path(output_dir, "hashDemux_classification.csv"),
  row.names = TRUE
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

meta <- seurat_obj@meta.data

final_assign <- if ("classification" %in% colnames(meta)) {
  meta$classification
} else if ("hash.ID" %in% colnames(meta)) {
  meta$hash.ID
} else if ("HTO_classification" %in% colnames(meta)) {
  meta$HTO_classification
} else {
  rep(NA, nrow(meta))
}

confidence <- if ("confidence_score" %in% colnames(meta)) {
  meta$confidence_score
} else if ("confidence" %in% colnames(meta)) {
  meta$confidence
} else {
  rep(NA, nrow(meta))
}

summary_df <- data.frame(
  droplet_id = rownames(meta),
  nHTO_total = colSums(matrix),
  HTO_maxID = rownames(matrix)[top_indices],
  final_assign = final_assign,
  confidence = confidence,
  stringsAsFactors = FALSE
)

write.csv(
  summary_df,
  file = file.path(output_dir, "hashDemux_summary.csv"),
  row.names = FALSE
)

# =========================
# Save runtime
# =========================
script_end_time <- Sys.time()
total_runtime_sec <- as.numeric(difftime(script_end_time, script_start_time, units = "secs"))

runtime_df <- data.frame(
  method = "hashDemux",
  input_dir = input_dir,
  n_hto = nrow(matrix),
  n_droplets = ncol(matrix),
  expected_doublet_rate = 0.05,
  knns = paste(seq(5, 30, 5), collapse = ","),
  resolutions = paste(c(1, 2, 3, 4), collapse = ","),
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
  file = file.path(output_dir, "hashDemux_runtime.csv"),
  row.names = FALSE
)

saveRDS(
  seurat_obj,
  file = file.path(output_dir, "hashDemux_result.rds")
)

# =========================
# Diagnostics
# =========================
cat("Metadata columns:\n")
print(colnames(meta))

cat("Final assignment summary:\n")
print(table(final_assign, useNA = "ifany"))

cat("hashDemux total runtime:", total_runtime_sec, "seconds\n")
cat("Finished hashDemux.\n")