# spaGFM reproducibility repository

This repository contains the code, workflows, and plotting scripts used to reproduce spaGFM analyses and related evaluations. The project is organized by figure and analysis module, with each directory containing the scripts, notebooks, and shell commands needed to reproduce a specific result or study component.

The structure is designed for transparency and extensibility: major figures are grouped into separate folders, and subfolders further organize experiments by setting or task.

## Repository layout

The project is organized in a simple figure-based structure:

```text
spaGFM_reproducibility/
├── figure1/
├── figure2/
│   ├── figure2a/
│   ├── figure2b/
│   ├── figure2c/
│   └── figure2d/
├── figure3/
└── ...
```

In this layout:

- Each top-level folder such as `figure1/`, `figure2/`, or `figure3/` corresponds to a major figure or analysis block.
- Subfolders like `figure2a/`, `figure2b/`, and `figure2d/` represent different parts or views within that figure.
