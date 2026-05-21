#!/usr/bin/env Rscript
# demuxmix.R

suppressPackageStartupMessages({
  library(demuxmix)
  library(Matrix)
})

input_dir <- "./input/demuxmix"
output_dir <- "./results/demuxmix"
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

colnames(matrix) <- barcodes
rownames(matrix) <- make.unique(as.character(features[, 1]))

cat("Input matrix dimension:", nrow(matrix), "HTOs x", ncol(matrix), "droplets\n")

# =========================
# Basic diagnostics
# =========================
diag_df <- data.frame(
  HTO = rownames(matrix),
  min = apply(matrix, 1, function(x) min(x, na.rm = TRUE)),
  q25 = apply(matrix, 1, function(x) as.numeric(quantile(x, 0.25, na.rm = TRUE))),
  median = apply(matrix, 1, function(x) median(x, na.rm = TRUE)),
  mean = apply(matrix, 1, function(x) mean(x, na.rm = TRUE)),
  q75 = apply(matrix, 1, function(x) as.numeric(quantile(x, 0.75, na.rm = TRUE))),
  max = apply(matrix, 1, function(x) max(x, na.rm = TRUE)),
  zero_rate = apply(matrix, 1, function(x) mean(x == 0, na.rm = TRUE)),
  variance = apply(matrix, 1, function(x) var(x, na.rm = TRUE)),
  nonfinite = apply(matrix, 1, function(x) sum(!is.finite(x))),
  stringsAsFactors = FALSE
)

write.csv(
  diag_df,
  file = file.path(output_dir, "demuxmix_input_diagnostics.csv"),
  row.names = FALSE
)

# =========================
# Run demuxmix safely
# =========================
method_start_time <- Sys.time()

status <- "completed"
error_message <- ""
warning_messages <- character()

dmm <- withCallingHandlers(
  tryCatch({
    demuxmix(
      hto = matrix,
      model = "naive"
    )
  }, error = function(e) {
    status <<- "failed"
    error_message <<- conditionMessage(e)
    NULL
  }),
  warning = function(w) {
    warning_messages <<- c(warning_messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)

classes_df <- NULL

if (!is.null(dmm)) {
  classes_df <- withCallingHandlers(
    tryCatch({
      dmmClassify(dmm)
    }, error = function(e) {
      status <<- "failed_at_classification"
      error_message <<- conditionMessage(e)
      NULL
    }),
    warning = function(w) {
      warning_messages <<- c(warning_messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
}

method_end_time <- Sys.time()
method_runtime_sec <- as.numeric(difftime(method_end_time, method_start_time, units = "secs"))

script_end_time <- Sys.time()
total_runtime_sec <- as.numeric(difftime(script_end_time, script_start_time, units = "secs"))

cat("demuxmix status:", status, "\n")
cat("demuxmix method runtime:", method_runtime_sec, "seconds\n")

if (status != "completed") {
  cat("demuxmix failed but script will continue.\n")
  cat("Error message:", error_message, "\n")
}

# =========================
# Save runtime no matter success or failure
# =========================
runtime_df <- data.frame(
  method = "demuxmix",
  input_dir = input_dir,
  n_hto = nrow(matrix),
  n_droplets = ncol(matrix),
  model = "naive",
  status = status,
  error_message = error_message,
  n_warnings = length(warning_messages),
  warnings = paste(unique(warning_messages), collapse = " | "),
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
  file = file.path(output_dir, "demuxmix_runtime.csv"),
  row.names = FALSE
)

# =========================
# If failed, save failure marker and exit normally
# =========================
if (is.null(dmm) || is.null(classes_df) || status != "completed") {
  failure_df <- data.frame(
    method = "demuxmix",
    input_dir = input_dir,
    status = status,
    error_message = error_message,
    stringsAsFactors = FALSE
  )

  write.csv(
    failure_df,
    file = file.path(output_dir, "demuxmix_failed.csv"),
    row.names = FALSE
  )

  cat("Saved failure information to demuxmix_failed.csv\n")
  cat("Finished demuxmix with failure status, exiting with status 0.\n")
  quit(status = 0)
}

# =========================
# Save classification
# =========================
write.csv(
  classes_df,
  file = file.path(output_dir, "demuxmix_classification.csv"),
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

hto_col <- rep(NA, ncol(matrix))

if ("HTO" %in% colnames(classes_df)) {
  if (nrow(classes_df) == ncol(matrix)) {
    hto_col <- classes_df$HTO
  } else if (!is.null(rownames(classes_df))) {
    matched_idx <- match(colnames(matrix), rownames(classes_df))
    hto_col <- classes_df$HTO[matched_idx]
  }
}

summary_df <- data.frame(
  droplet_id = colnames(matrix),
  nHTO_total = colSums(matrix),
  HTO_maxID = rownames(matrix)[top_indices],
  final_assign = hto_col,
  stringsAsFactors = FALSE
)

write.csv(
  summary_df,
  file = file.path(output_dir, "demuxmix_summary.csv"),
  row.names = FALSE
)

# =========================
# Save model
# =========================
saveRDS(
  dmm,
  file = file.path(output_dir, "demuxmix_result.rds")
)

# =========================
# Save plot safely
# =========================
plot_status <- "completed"
plot_error <- ""

tryCatch({
  pdf(file.path(output_dir, "demuxmix_histograms.pdf"))
  plotDmmHistogram(dmm)
  dev.off()
}, error = function(e) {
  plot_status <<- "failed"
  plot_error <<- conditionMessage(e)
  try(dev.off(), silent = TRUE)
})

plot_df <- data.frame(
  plot = "demuxmix_histograms.pdf",
  status = plot_status,
  error_message = plot_error,
  stringsAsFactors = FALSE
)

write.csv(
  plot_df,
  file = file.path(output_dir, "demuxmix_plot_status.csv"),
  row.names = FALSE
)

cat("Finished demuxmix successfully.\n")