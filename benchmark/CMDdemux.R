#!/usr/bin/env Rscript
# HTO-only CMDdemux benchmark; see CMDdemux_README.md for invocation/options.
script_start <- Sys.time()
input_dir <- Sys.getenv("BENCH_INPUT_DIR", "./data/CMDdemux")
output_dir <- Sys.getenv("BENCH_OUTPUT_DIR", "./results/CMDdemux")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
# Avoid combining a failed rerun with classifications from a previous run.
artifacts <- file.path(output_dir, paste0("CMDdemux_", c("summary.csv", "classification.csv", "result.rds", "failed.csv", "runtime.csv", "config.rds")))
if (any(file.exists(artifacts))) stop("Output already contains CMDdemux results; choose a new BENCH_OUTPUT_DIR")
warnings <- character()
method_start <- NULL
method_runtime <- NA_real_
n_hto <- n_droplets <- NA_integer_
version <- sha <- NA_character_
config <- list()
number <- function(key, default, lower = -Inf, upper = Inf, integer = FALSE) {
  x <- suppressWarnings(as.numeric(Sys.getenv(key, as.character(default))))
  if (length(x) != 1L || !is.finite(x) || x < lower || x > upper || (integer && x != floor(x))) stop("Invalid ", key)
  if (integer) as.integer(x) else x
}
boolean <- function(key, default = FALSE) {
  x <- tolower(Sys.getenv(key, as.character(default)))
  if (!x %in% c("true", "false", "1", "0")) stop("Invalid ", key)
  x %in% c("true", "1")
}
pick <- function(name) {
  paths <- file.path(input_dir, c(name, paste0(name, ".gz")))
  found <- paths[file.exists(paths)]
  if (!length(found)) stop("Missing input: ", name)
  found[1]
}
read_input <- function(name, reader) {
  path <- pick(name)
  con <- if (endsWith(path, ".gz")) gzfile(path, "rt") else file(path, "rt")
  on.exit(close(con))
  reader(con)
}
status <- "completed"
error_message <- ""
tryCatch(withCallingHandlers({
  suppressPackageStartupMessages({library(CMDdemux); library(Matrix)})
  # Current CMDdemux calls iqr() internally but does not define/import it.
  # Give that one internal function an enclosing shim with stats::IQR.
  cmd_ns <- asNamespace("CMDdemux")
  label_fun <- get("LabelClusterHTO", envir = cmd_ns)
  if (!exists("iqr", envir = environment(label_fun), inherits = TRUE)) {
    shim <- new.env(parent = cmd_ns)
    shim$iqr <- function(x, ...) stats::IQR(x, ...)
    environment(label_fun) <- shim
    assignInNamespace("LabelClusterHTO", label_fun, ns = "CMDdemux")
  }
  version <- as.character(packageVersion("CMDdemux"))
  sha <- packageDescription("CMDdemux")$RemoteSha
  if (is.null(sha)) sha <- NA_character_
  writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
  config <- list(
    seed = number("BENCH_RANDOM_SEED", 42, 0, .Machine$integer.max, TRUE),
    fast_mode = boolean("BENCH_CMDDEMUX_FAST_MODE"),
    prop = number("BENCH_CMDDEMUX_PROP", 0.1, .Machine$double.eps, 1),
    eu_cut_q = number("BENCH_CMDDEMUX_EU_CUT_Q", 0.9, .Machine$double.eps, 1 - .Machine$double.eps),
    md_cut_q = number("BENCH_CMDDEMUX_MD_CUT_Q", 0.95, .Machine$double.eps, 1 - .Machine$double.eps),
    num_modes = number("BENCH_CMDDEMUX_NUM_MODES", 3, 1, Inf, TRUE),
    cut_no = number("BENCH_CMDDEMUX_CUT_NO", 2, 1, Inf, TRUE),
    label_method = Sys.getenv("BENCH_CMDDEMUX_LABEL_METHOD", "medoids"),
    use_gex_data = FALSE
  )
  if (!config$label_method %in% c("medoids", "expression")) stop("Invalid label method")
  if (config$cut_no > 2 * config$num_modes - 1) stop("cut_no exceeds number of modes/antimodes")
  if (nzchar(Sys.getenv("BENCH_CMDDEMUX_LIB_CUT"))) config$lib_cut <- number("BENCH_CMDDEMUX_LIB_CUT", 0)
  saveRDS(config, file.path(output_dir, "CMDdemux_config.rds"))
  writeLines(capture.output(dput(config)), file.path(output_dir, "CMDdemux_config.txt"))
  set.seed(config$seed)
  counts <- read_input("matrix.mtx", Matrix::readMM)
  barcodes <- read_input("barcodes.tsv", function(con) readLines(con, warn = FALSE))
  features <- read_input("features.tsv", function(con) read.delim(con, header = FALSE, colClasses = "character", check.names = FALSE, quote = "", comment.char = ""))
  n_hto <- nrow(counts); n_droplets <- ncol(counts)
  if (n_hto < 2 || n_droplets <= n_hto) stop("Need >=2 tags and more droplets than tags")
  if (length(barcodes) != n_droplets || nrow(features) != n_hto) stop("Input dimensions do not match features/barcodes")
  tags <- features[[1]]
  if (anyNA(tags) || anyNA(barcodes) || any(!nzchar(tags)) || any(!nzchar(barcodes)) || anyDuplicated(tags) || anyDuplicated(barcodes)) stop("Feature IDs and barcodes must be nonempty and unique")
  if (any(!is.finite(counts@x)) || any(counts@x < 0) || any(counts@x != floor(counts@x))) stop("Input must contain finite nonnegative integer counts")
  rownames(counts) <- tags; colnames(counts) <- barcodes
  # No zero-cell filtering: upstream LocalCLRNorm has its own pseudocount.
  totals <- Matrix::colSums(counts)
  method_start <- Sys.time()
  clr <- CMDdemux::LocalCLRNorm(counts)
  if (any(!is.finite(clr))) stop("Nonfinite local CLR values")
  clusters <- CMDdemux::KmedCluster(clr, fast_mode = config$fast_mode, prop = config$prop)
  distances <- CMDdemux::EuclideanClusterDist(clr, clusters)
  core <- CMDdemux::DefineNonCore(distances, clusters, eu_cut_q = config$eu_cut_q)
  labels <- CMDdemux::LabelClusterHTO(clr, clusters, core, label_method = config$label_method)
  if (length(labels) != n_hto || anyNA(labels) || anyDuplicated(labels)) stop("CMDdemux could not map each tag to a distinct cluster")
  md <- CMDdemux::CalculateMD(clr, core, clusters, labels)
  if (!identical(colnames(md), barcodes) || any(!is.finite(md))) stop("Invalid Mahalanobis matrix")
  outliers <- CMDdemux::AssignOutlierDrop(md, md_cut_q = config$md_cut_q)
  args <- list(md.mat = md, hash.count = counts, drop.outlier.assign = outliers,
               use_gex_data = FALSE, num_modes = config$num_modes, cut_no = config$cut_no)
  if (!is.null(config$lib_cut)) args$lib_cut <- config$lib_cut
  calls <- do.call(CMDdemux::CMDdemuxClass, args)
  method_runtime <- as.numeric(difftime(Sys.time(), method_start, units = "secs"))
  if (anyDuplicated(rownames(calls)) || !setequal(rownames(calls), barcodes)) stop("Output barcodes do not match input")
  calls <- calls[match(barcodes, rownames(calls)), , drop = FALSE]
  if (!all(c("demux_id", "global_class", "demux_global_class", "doublet_class") %in% names(calls))) stop("Missing native classification fields")
  types <- as.character(calls$global_class)
  if (anyNA(types) || any(!types %in% c("Singlet", "Doublet", "Negative"))) stop("Unresolved or unknown native classification; inspect CMDdemux cutoffs")
  ids <- as.character(calls$demux_id)
  if (anyNA(ids) || any(!ids %in% tags)) stop("Invalid native sample IDs")
  final <- ifelse(types == "Singlet", ids, ifelse(types == "Doublet", "Multiplet", "Unassigned"))
  max_idx <- max.col(t(as.matrix(counts)), ties.method = "first")
  max_tag <- tags[max_idx]; max_tag[totals == 0] <- NA_character_
  calls$droplet_id <- barcodes
  summary <- data.frame(droplet_id = barcodes, nHTO_total = totals, HTO_maxID = max_tag,
                        final_assign = final, classification_type = types, stringsAsFactors = FALSE)
  saveRDS(list(classification = calls, clusters = clusters, core = core, cluster_assign = labels,
               md = md, outliers = outliers, config = config), file.path(output_dir, "CMDdemux_result.rds"))
  write.csv(calls, file.path(output_dir, "CMDdemux_classification.csv"), row.names = FALSE)
  write.csv(summary, file.path(output_dir, "CMDdemux_summary.csv"), row.names = FALSE)
  print(table(summary$classification_type))
}, warning = function(w) {
  warnings <<- c(warnings, conditionMessage(w))
  invokeRestart("muffleWarning")
}), error = function(e) {
  status <<- "failed"
  error_message <<- conditionMessage(e)
})
if (is.na(method_runtime) && !is.null(method_start)) method_runtime <- as.numeric(difftime(Sys.time(), method_start, units = "secs"))
write.csv(data.frame(warning = unique(warnings)), file.path(output_dir, "CMDdemux_warnings.csv"), row.names = FALSE)
write.csv(data.frame(method = "CMDdemux", input_dir = input_dir, n_hto = n_hto, n_droplets = n_droplets,
                     package_version = version, source_sha = sha, input_scope = "HTO_only_no_RNA_covariate",
                     status = status, error_message = error_message, n_warnings = length(warnings),
                     method_runtime_sec = method_runtime,
                     total_runtime_sec = as.numeric(difftime(Sys.time(), script_start, units = "secs"))),
          file.path(output_dir, "CMDdemux_runtime.csv"), row.names = FALSE)
if (status != "completed") {
  write.csv(data.frame(method = "CMDdemux", status = status, error_message = error_message), file.path(output_dir, "CMDdemux_failed.csv"), row.names = FALSE)
  message("CMDdemux failed: ", error_message)
  quit(status = 1)
}
cat("Finished CMDdemux.\n")
