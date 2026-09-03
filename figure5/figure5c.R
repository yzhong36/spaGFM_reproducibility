#!/usr/bin/env Rscript

# Reproduce panel c: kidney UMAP coloured by glomerulus grade with one
# Slingshot lineage. Despite the historical filename, this analysis uses
# Slingshot rather than monocle3.
#
# Scientific role: visualize the progression from grade 0-enriched tissue
# toward the grade 3-enriched terminal state in UMAP space.
# Figure type: single validation panel (UMAP + inferred trajectory).
#
#
# Optional environment variables:
#   PROJECT_DIR=/path/to/project
#   KIDNEY_H5AD=/path/to/input.h5ad
#   OUTPUT_DIR=/path/to/output
#   LINEAGE_INDEX=4
#   FLIP_UMAP1=true
#   PANEL_LABEL=c          # use an empty value to omit the panel label
#   SHOW_LEGEND=false

suppressPackageStartupMessages({
  required_packages <- c(
    "zellkonverter",
    "SingleCellExperiment",
    "slingshot",
    "ggplot2",
    "svglite",
    "ragg"
  )

  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(missing_packages) > 0L) {
    stop(
      paste0(
        "Missing R package(s): ", paste(missing_packages, collapse = ", "), ".\n",
        "Install CRAN packages with install.packages(c('ggplot2', 'svglite', 'ragg'))\n",
        "and Bioconductor packages with:\n",
        "  BiocManager::install(c('zellkonverter', 'SingleCellExperiment', 'slingshot'))"
      ),
      call. = FALSE
    )
  }

  library(SingleCellExperiment)
  library(ggplot2)
  library(slingshot)
  library(zellkonverter)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

parse_bool <- function(x, name) {
  value <- tolower(trimws(x))
  if (value %in% c("true", "t", "1", "yes", "y")) return(TRUE)
  if (value %in% c("false", "f", "0", "no", "n")) return(FALSE)
  stop(name, " must be true or false; received: ", x, call. = FALSE)
}

find_project_dir <- function() {
  override <- Sys.getenv("PROJECT_DIR", unset = "")
  if (nzchar(override)) {
    return(normalizePath(override, mustWork = TRUE))
  }

  # This branch supports `Rscript kidney_code/monocle3.R`.
  full_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", full_args, value = TRUE)
  if (length(file_arg) == 1L) {
    script_path <- normalizePath(sub("^--file=", "", file_arg), mustWork = TRUE)
    return(normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE))
  }

  # This branch supports interactive use from the project root.
  current_dir <- normalizePath(getwd(), mustWork = TRUE)
  if (file.exists(file.path(current_dir, "kidney", "IU04_embedding_pca_36new.h5ad"))) {
    return(current_dir)
  }

  stop(
    "Cannot infer the project directory. Set PROJECT_DIR explicitly.",
    call. = FALSE
  )
}

project_dir <- find_project_dir()
input_h5ad <- Sys.getenv(
  "KIDNEY_H5AD",
  unset = file.path(project_dir, "kidney", "IU04_embedding_pca.h5ad")
)
output_dir <- Sys.getenv(
  "OUTPUT_DIR",
  unset = file.path(project_dir, "kidney_img", "reproducibility")
)
lineage_index <- suppressWarnings(as.integer(Sys.getenv("LINEAGE_INDEX", "4")))
flip_umap1 <- parse_bool(Sys.getenv("FLIP_UMAP1", "true"), "FLIP_UMAP1")
show_legend <- parse_bool(Sys.getenv("SHOW_LEGEND", "false"), "SHOW_LEGEND")
panel_label <- Sys.getenv("PANEL_LABEL", "c")
target_grade <- "3"
random_seed <- 20260902L

if (!file.exists(input_h5ad)) {
  stop("Input file does not exist: ", input_h5ad, call. = FALSE)
}
if (is.na(lineage_index) || lineage_index < 1L) {
  stop("LINEAGE_INDEX must be a positive integer.", call. = FALSE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Read and validate the AnnData object
# -----------------------------------------------------------------------------

set.seed(random_seed)
sce <- zellkonverter::readH5AD(input_h5ad)

required_columns <- c("leiden", "glomerulus_grade")
missing_columns <- setdiff(required_columns, colnames(SummarizedExperiment::colData(sce)))
if (length(missing_columns) > 0L) {
  stop(
    "Missing required observation column(s): ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

if (!"X_umap" %in% SingleCellExperiment::reducedDimNames(sce)) {
  stop("The input AnnData object does not contain obsm['X_umap'].", call. = FALSE)
}

umap_original <- SingleCellExperiment::reducedDim(sce, "X_umap")
if (ncol(umap_original) != 2L) {
  stop("X_umap must contain exactly two dimensions.", call. = FALSE)
}
if (any(!is.finite(umap_original))) {
  stop("X_umap contains non-finite coordinates.", call. = FALSE)
}

SingleCellExperiment::reducedDim(sce, "UMAP") <- umap_original
sce$cluster <- factor(as.character(sce$leiden))
grade_character <- as.character(sce$glomerulus_grade)

allowed_grades <- c("0", "1", "2", "3")
unexpected_grades <- setdiff(unique(stats::na.omit(grade_character)), allowed_grades)
if (length(unexpected_grades) > 0L) {
  stop(
    "Unexpected glomerulus_grade value(s): ",
    paste(unexpected_grades, collapse = ", "),
    call. = FALSE
  )
}
if (!target_grade %in% grade_character) {
  stop("No grade ", target_grade, " observations are present.", call. = FALSE)
}

# Match the original analysis: choose the Leiden cluster containing the largest
# number of grade 0 observations as the root cluster.
grade0_clusters <- table(sce$cluster[grade_character == "0"])
if (length(grade0_clusters) == 0L) {
  stop("No grade 0 observations are available to define the root cluster.", call. = FALSE)
}
start_cluster <- names(sort(grade0_clusters, decreasing = TRUE))[1L]

# -----------------------------------------------------------------------------
# Infer Slingshot lineages
# -----------------------------------------------------------------------------

sce <- slingshot::slingshot(
  sce,
  clusterLabels = "cluster",
  reducedDim = "UMAP",
  start.clus = start_cluster
)

curves <- slingshot::slingCurves(sce)
if (lineage_index > length(curves)) {
  stop(
    "LINEAGE_INDEX=", lineage_index,
    " is unavailable; Slingshot returned ", length(curves), " lineage(s).",
    call. = FALSE
  )
}

# -----------------------------------------------------------------------------
# Prepare reproducible plotting data
# -----------------------------------------------------------------------------

plot_umap <- umap_original
if (flip_umap1) plot_umap[, 1L] <- -plot_umap[, 1L]

cell_ids <- colnames(sce)
if (is.null(cell_ids)) cell_ids <- sprintf("cell_%05d", seq_len(ncol(sce)))

plot_df <- data.frame(
  cell_id = cell_ids,
  cluster = as.character(sce$cluster),
  grade = factor(grade_character, levels = allowed_grades),
  UMAP1_original = umap_original[, 1L],
  UMAP2_original = umap_original[, 2L],
  UMAP1 = plot_umap[, 1L],
  UMAP2 = plot_umap[, 2L],
  stringsAsFactors = FALSE
)

selected_curve <- curves[[lineage_index]]
curve_order <- selected_curve$ord
if (is.null(curve_order)) curve_order <- seq_len(nrow(selected_curve$s))

curve_matrix <- selected_curve$s[curve_order, , drop = FALSE]
curve_matrix <- curve_matrix[stats::complete.cases(curve_matrix), , drop = FALSE]
if (nrow(curve_matrix) < 2L) {
  stop("The selected lineage has fewer than two finite trajectory points.", call. = FALSE)
}
if (flip_umap1) curve_matrix[, 1L] <- -curve_matrix[, 1L]

# Ensure the arrow points toward the grade 3-enriched endpoint regardless of
# Slingshot's internal storage direction.
target_centroid <- colMeans(
  plot_umap[grade_character == target_grade, , drop = FALSE],
  na.rm = TRUE
)
distance_to_target <- function(point) sqrt(sum((point - target_centroid)^2))
if (distance_to_target(curve_matrix[1L, ]) <
    distance_to_target(curve_matrix[nrow(curve_matrix), ])) {
  curve_matrix <- curve_matrix[nrow(curve_matrix):1L, , drop = FALSE]
}

curve_df <- data.frame(
  point_order = seq_len(nrow(curve_matrix)),
  lineage = lineage_index,
  UMAP1 = curve_matrix[, 1L],
  UMAP2 = curve_matrix[, 2L]
)

utils::write.csv(
  plot_df,
  file.path(output_dir, "panel_c_umap_source_data.csv"),
  row.names = FALSE
)
utils::write.csv(
  curve_df,
  file.path(output_dir, "panel_c_trajectory_source_data.csv"),
  row.names = FALSE
)

# -----------------------------------------------------------------------------
# Build the figure
# -----------------------------------------------------------------------------

grade_colors <- c(
  "0" = "#31B44B",
  "1" = "#FF8C00",
  "2" = "#FF4A3D",
  "3" = "#A526B5"
)

x_range <- range(plot_df$UMAP1, na.rm = TRUE)
y_range <- range(plot_df$UMAP2, na.rm = TRUE)
x_span <- diff(x_range)
y_span <- diff(y_range)

# Extra lower-left space is reserved for the compact UMAP direction axes.
x0 <- x_range[1L] - 0.055 * x_span
y0 <- y_range[1L] - 0.125 * y_span
x_limits <- c(x_range[1L] - 0.10 * x_span, x_range[2L] + 0.035 * x_span)
y_limits <- c(y_range[1L] - 0.20 * y_span, y_range[2L] + 0.055 * y_span)

p <- ggplot(plot_df, aes(x = UMAP1, y = UMAP2)) +
  geom_point(
    aes(colour = grade),
    size = 0.26,
    alpha = 0.92,
    stroke = 0,
    na.rm = TRUE
  ) +
  geom_path(
    data = curve_df,
    aes(x = UMAP1, y = UMAP2),
    inherit.aes = FALSE,
    colour = "black",
    linewidth = 0.42,
    lineend = "round",
    arrow = grid::arrow(length = grid::unit(1.45, "mm"), type = "closed")
  ) +
  annotate(
    "segment",
    x = x0, xend = x0 + 0.12 * x_span,
    y = y0, yend = y0,
    linewidth = 0.34,
    arrow = grid::arrow(length = grid::unit(1.25, "mm"), type = "closed")
  ) +
  annotate(
    "segment",
    x = x0, xend = x0,
    y = y0, yend = y0 + 0.18 * y_span,
    linewidth = 0.34,
    arrow = grid::arrow(length = grid::unit(1.25, "mm"), type = "closed")
  ) +
  annotate(
    "text",
    x = x0 + 0.065 * x_span,
    y = y0 - 0.045 * y_span,
    label = "UMAP1",
    size = 2.55,
    family = "sans"
  ) +
  annotate(
    "text",
    x = x0 - 0.026 * x_span,
    y = y0 + 0.09 * y_span,
    label = "UMAP2",
    angle = 90,
    size = 2.55,
    family = "sans"
  ) +
  scale_colour_manual(
    values = grade_colors,
    limits = allowed_grades,
    drop = FALSE,
    name = "Grade"
  ) +
  coord_fixed(
    xlim = x_limits,
    ylim = y_limits,
    ratio = 1,
    expand = FALSE,
    clip = "off"
  ) +
  theme_void(base_family = "sans", base_size = 6.5) +
  theme(
    legend.position = if (show_legend) "right" else "none",
    legend.title = element_text(size = 6.2),
    legend.text = element_text(size = 5.8),
    legend.key.height = grid::unit(3.3, "mm"),
    plot.margin = margin(1.5, 1.5, 1.5, 1.5, unit = "mm")
  )

if (nzchar(panel_label)) {
  p <- p + annotate(
    "text",
    x = x_limits[1L] + 0.005 * x_span,
    y = y_limits[2L] - 0.005 * y_span,
    label = panel_label,
    hjust = 0,
    vjust = 1,
    fontface = "bold",
    family = "sans",
    size = 3.1
  )
}

# -----------------------------------------------------------------------------
# Export: raster preview + editable vector files
# -----------------------------------------------------------------------------

output_stem <- file.path(output_dir, "panel_c_slingshot_reproducibility")
width_mm <- if (show_legend) 112 else 89
height_mm <- 72

ggsave(
  paste0(output_stem, ".png"),
  plot = p,
  device = ragg::agg_png,
  width = width_mm,
  height = height_mm,
  units = "mm",
  dpi = 600,
  background = "white"
)
ggsave(
  paste0(output_stem, ".svg"),
  plot = p,
  device = svglite::svglite,
  width = width_mm,
  height = height_mm,
  units = "mm",
  background = "white"
)
ggsave(
  paste0(output_stem, ".pdf"),
  plot = p,
  device = grDevices::cairo_pdf,
  width = width_mm,
  height = height_mm,
  units = "mm",
  bg = "white"
)

metadata <- c(
  paste0("input_h5ad: ", normalizePath(input_h5ad, mustWork = TRUE)),
  paste0("random_seed: ", random_seed),
  paste0("start_cluster: ", start_cluster),
  paste0("lineage_index: ", lineage_index),
  paste0("target_grade: ", target_grade),
  paste0("flip_umap1: ", flip_umap1),
  paste0("show_legend: ", show_legend),
  paste0("panel_label: ", panel_label),
  "",
  "Session information:",
  capture.output(utils::sessionInfo())
)
writeLines(metadata, file.path(output_dir, "panel_c_session_info.txt"))

message("Finished. Outputs written to: ", normalizePath(output_dir, mustWork = TRUE))
