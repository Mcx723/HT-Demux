#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(deMULTIplex2)
  library(Matrix)
})

input_dir <- Sys.getenv("BENCH_INPUT_DIR", "./data/deMULTIplex2")
output_dir <- Sys.getenv("BENCH_OUTPUT_DIR", "./results/deMULTIplex2")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

script_start_time <- Sys.time()
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
warning_messages <- character()
init_cos_setting <- Sys.getenv("BENCH_DEMULTIPLEX2_INIT_COS_CUT", "default")

matrix <- as.matrix(readMM(file.path(input_dir, "matrix.mtx")))
barcodes <- readLines(file.path(input_dir, "barcodes.tsv"))
features <- read.table(
  file.path(input_dir, "features.tsv"),
  sep = "\t",
  header = FALSE,
  stringsAsFactors = FALSE
)

colnames(matrix) <- barcodes
rownames(matrix) <- make.unique(as.character(features[, 1]))
tag_mtx <- t(matrix)

method_start_time <- Sys.time()
status <- "completed"
error_message <- ""
res <- withCallingHandlers(tryCatch({
  call_args <- list(
    tag_mtx = tag_mtx,
    plot.path = output_dir,
    plot.name = "deMULTIplex2",
    plot.umap = "none",
    plot.diagnostics = FALSE
  )
  if (init_cos_setting != "default") call_args$init.cos.cut <- as.numeric(init_cos_setting)
  do.call(deMULTIplex2::demultiplexTags, call_args)
}, error = function(e) {
  status <<- "failed"
  error_message <<- conditionMessage(e)
  NULL
}), warning = function(w) {
  warning_messages <<- c(warning_messages, conditionMessage(w))
  invokeRestart("muffleWarning")
})
write.csv(data.frame(warning = unique(warning_messages)),
          file.path(output_dir, "deMULTIplex2_warnings.csv"), row.names = FALSE)
method_end_time <- Sys.time()

method_runtime_sec <- as.numeric(difftime(method_end_time, method_start_time, units = "secs"))
script_end_time <- Sys.time()
total_runtime_sec <- as.numeric(difftime(script_end_time, script_start_time, units = "secs"))

runtime_df <- data.frame(
  method = "deMULTIplex2",
  input_dir = input_dir,
  n_hto = nrow(matrix),
  n_droplets = ncol(matrix),
  status = status,
  init_cos_cut = if (init_cos_setting == "default") 0.5 else as.numeric(init_cos_setting),
  statistical_defaults = init_cos_setting == "default",
  error_message = error_message,
  method_runtime_sec = method_runtime_sec,
  total_runtime_sec = total_runtime_sec,
  stringsAsFactors = FALSE
)
write.csv(runtime_df, file.path(output_dir, "deMULTIplex2_runtime.csv"), row.names = FALSE)

if (is.null(res)) {
  write.csv(
    data.frame(method = "deMULTIplex2", status = status, error_message = error_message),
    file.path(output_dir, "deMULTIplex2_failed.csv"),
    row.names = FALSE
  )
  quit(status = 1)
}

saveRDS(res, file.path(output_dir, "deMULTIplex2_result.rds"))
if (is.list(res) && !is.null(res$prob_mtx) && !is.null(res$coefs)) {
  diagnostics <- data.frame(
    tag = colnames(res$prob_mtx),
    fit_valid = vapply(colnames(res$prob_mtx), function(tag) {
      all(is.finite(res$coefs[[tag]]))
    }, logical(1)),
    n_positive = colSums(res$prob_mtx > 0.5, na.rm = TRUE)
  )
  write.csv(diagnostics, file.path(output_dir, "deMULTIplex2_fit_diagnostics.csv"), row.names = FALSE)
  runtime_df$n_failed_tag_fits <- sum(!diagnostics$fit_valid)
  if (all(!diagnostics$fit_valid)) runtime_df$status <- "fit_failed"
  else if (any(!diagnostics$fit_valid)) runtime_df$status <- "partial_fit_failure"
  runtime_df$n_warnings <- length(warning_messages)
  write.csv(runtime_df, file.path(output_dir, "deMULTIplex2_runtime.csv"), row.names = FALSE)
}
if (is.list(res) && is.data.frame(res$assign_table)) {
  out_df <- res$assign_table
  out_df$droplet_id <- rownames(out_df)
  expected_ids <- barcodes[colSums(matrix) > 0]
  if (anyDuplicated(out_df$droplet_id) || !setequal(out_df$droplet_id, expected_ids)) {
    stop("deMULTIplex2 output IDs do not match nonzero input cells")
  }
  out_df <- out_df[match(expected_ids, out_df$droplet_id), , drop = FALSE]
} else if (is.data.frame(res)) {
  out_df <- res
} else if (is.list(res)) {
  out_df <- as.data.frame(res)
} else {
  out_df <- data.frame(result = as.character(res), stringsAsFactors = FALSE)
}

if (nrow(out_df) == ncol(matrix)) {
  if (is.null(out_df$droplet_id)) out_df$droplet_id <- colnames(matrix)
} else if (is.null(out_df$droplet_id)) {
  out_df$droplet_id <- seq_len(nrow(out_df))
}

write.csv(out_df, file.path(output_dir, "deMULTIplex2_classification.csv"), row.names = FALSE)

final_assign <- if ("final_assign" %in% colnames(out_df)) {
  out_df$final_assign
} else if ("assign_table.final_assign" %in% colnames(out_df)) {
  out_df$assign_table.final_assign
} else {
  candidate_cols <- colnames(out_df)[grepl("class|assign|hash|hto|tag|sample|call", colnames(out_df), ignore.case = TRUE)]
  if (length(candidate_cols) > 0) out_df[[candidate_cols[1]]] else NA
}

if (length(final_assign) != nrow(out_df)) stop("deMULTIplex2 assignment length does not match output IDs")
summary_df <- data.frame(
  droplet_id = out_df$droplet_id,
  nHTO_total = colSums(matrix)[match(out_df$droplet_id, barcodes)],
  HTO_maxID = rownames(matrix)[apply(matrix, 2, which.max)[match(out_df$droplet_id, barcodes)]],
  final_assign = final_assign,
  stringsAsFactors = FALSE
)
write.csv(summary_df, file.path(output_dir, "deMULTIplex2_summary.csv"), row.names = FALSE)
runtime_df$n_output_droplets <- nrow(out_df)
runtime_df$n_zero_count_excluded <- sum(colSums(matrix) == 0)
write.csv(runtime_df, file.path(output_dir, "deMULTIplex2_runtime.csv"), row.names = FALSE)

saveRDS(res, file.path(output_dir, "deMULTIplex2_result.rds"))
cat("Finished deMULTIplex2 with status:", runtime_df$status, "\n")
