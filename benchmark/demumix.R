#!/usr/bin/env Rscript
# Backward-compatible entry; the implementation is now demuxmix.R.
source_files <- vapply(sys.frames(), function(frame) {
  if (is.null(frame$ofile)) "" else as.character(frame$ofile)[1]
}, character(1))
source_files <- source_files[nzchar(source_files)]
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
entry <- if (length(source_files)) tail(source_files, 1) else if (length(file_arg)) {
  sub("^--file=", "", file_arg[1])
} else file.path("benchmark", "demumix.R")
source(file.path(dirname(normalizePath(entry, mustWork = TRUE)), "demuxmix.R"), chdir = FALSE)
