#!/usr/bin/env Rscript
# HTODemux.R

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

input_dir <- Sys.getenv("BENCH_INPUT_DIR", "./data/HTODemux")
output_dir <- Sys.getenv("BENCH_OUTPUT_DIR", "./results/HTODemux")
init_setting <- Sys.getenv("BENCH_HTODEMUX_INIT", "default")
htodemux_init <- if (init_setting == "default") NULL else as.integer(init_setting)
positive_quantile <- as.numeric(Sys.getenv("BENCH_HTODEMUX_POSITIVE_QUANTILE", "0.99"))
recovery_mode <- Sys.getenv("BENCH_HTODEMUX_RECOVERY", "none")
if (!recovery_mode %in% c("none", "merge_small_clusters")) stop("Unknown HTODemux recovery mode")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

script_start_time <- Sys.time()

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

# matrix: HTO x droplets
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
# Run HTODemux with runtime
# =========================
method_start_time <- Sys.time()

seurat_obj <- NormalizeData(
  seurat_obj,
  assay = "HTO",
  normalization.method = "CLR"
)

demux_function <- Seurat::HTODemux
if (recovery_mode == "merge_small_clusters") {
  # Insert before background averaging; leave the installed Seurat namespace unchanged.
  statements <- as.list(body(demux_function))
  insert_at <- which(vapply(statements, function(x) {
    is.call(x) && identical(x[[1]], as.name("<-")) &&
      identical(x[[2]], as.name("average.expression"))
  }, logical(1)))
  if (length(insert_at) != 1L) stop("Unsupported Seurat HTODemux body; recovery was not applied")
  repair <- quote({
    cluster_ids <- as.character(Idents(object))
    cluster_sizes <- table(cluster_ids)
    large <- names(cluster_sizes)[cluster_sizes >= recovery_min_cluster_size]
    if (!length(large)) large <- names(cluster_sizes)[which.max(cluster_sizes)]
    small <- setdiff(names(cluster_sizes), large)
    centers <- vapply(large, function(id) {
      rowMeans(as.matrix(data[, cluster_ids == id, drop = FALSE]))
    }, numeric(nrow(data)))
    merge_rows <- list()
    for (id in small) {
      center <- rowMeans(as.matrix(data[, cluster_ids == id, drop = FALSE]))
      distance <- colSums((centers - center)^2)
      destination <- large[which.min(distance)]
      merge_rows[[length(merge_rows) + 1L]] <- data.frame(
        source_cluster = id, n_cells = as.integer(cluster_sizes[id]),
        destination_cluster = destination
      )
      cluster_ids[cluster_ids == id] <- destination
    }
    Idents(object) <- factor(cluster_ids)
    merge_table <- if (length(merge_rows)) do.call(rbind, merge_rows) else
      data.frame(source_cluster = character(), n_cells = integer(), destination_cluster = character())
    utils::write.csv(merge_table, file.path(recovery_output_dir, "cluster_merges.csv"), row.names = FALSE)
  })
  body(demux_function) <- as.call(append(statements, list(repair), after = insert_at - 1L))
  environment(demux_function) <- list2env(
    list(recovery_min_cluster_size = 20L, recovery_output_dir = output_dir),
    parent = environment(Seurat::HTODemux)
  )
}

seurat_obj <- demux_function(
  seurat_obj,
  assay = "HTO",
  positive.quantile = positive_quantile,
  init = htodemux_init
)

method_end_time <- Sys.time()
method_runtime_sec <- as.numeric(difftime(method_end_time, method_start_time, units = "secs"))

cat("HTODemux method runtime:", method_runtime_sec, "seconds\n")

# =========================
# Save classification
# =========================
write.csv(
  seurat_obj@meta.data,
  file = file.path(output_dir, "HTODemux_classification.csv"),
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

final_assign <- as.character(seurat_obj$HTO_classification)
global_type <- as.character(seurat_obj$HTO_classification.global)
final_assign[global_type == "Doublet"] <- "Multiplet"
final_assign[global_type == "Negative"] <- "Unassigned"
summary_df <- data.frame(
  droplet_id = rownames(seurat_obj@meta.data),
  nHTO_total = colSums(matrix),
  HTO_maxID = rownames(matrix)[top_indices],
  final_assign = final_assign,
  global_type = global_type,
  stringsAsFactors = FALSE
)

write.csv(
  summary_df,
  file = file.path(output_dir, "HTODemux_summary.csv"),
  row.names = FALSE
)

# =========================
# Save runtime
# =========================
script_end_time <- Sys.time()
total_runtime_sec <- as.numeric(difftime(script_end_time, script_start_time, units = "secs"))

runtime_df <- data.frame(
  method = "HTODemux",
  input_dir = input_dir,
  n_hto = nrow(matrix),
  n_droplets = ncol(matrix),
  positive_quantile = positive_quantile,
  init = if (is.null(htodemux_init)) nrow(matrix) + 1L else htodemux_init,
  variant = recovery_mode,
  recovery_min_cluster_size = if (recovery_mode == "none") NA_integer_ else 20L,
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
  file = file.path(output_dir, "HTODemux_runtime.csv"),
  row.names = FALSE
)

# =========================
# Save object and optional heatmap
# =========================
saveRDS(
  seurat_obj,
  file = file.path(output_dir, "HTODemux_result.rds")
)

pdf(file.path(output_dir, "HTODemux_heatmap.pdf"))
HTOHeatmap(seurat_obj, assay = "HTO")
dev.off()

cat("HTODemux total runtime:", total_runtime_sec, "seconds\n")
cat("Finished HTODemux.\n")
