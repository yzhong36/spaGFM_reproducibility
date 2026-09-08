import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


INPUT_H5AD = "your_file.h5ad"
SAMPLE_ORDER = [
    "sample_1",
    "sample_2",
    "sample_3",
    "sample_4",
    "sample_5",
]


adata = sc.read_h5ad(INPUT_H5AD)

required_columns = {"sample", "cell_type"}
missing_columns = required_columns.difference(adata.obs.columns)
if missing_columns:
    raise ValueError(f"Missing columns in adata.obs: {sorted(missing_columns)}")

if len(SAMPLE_ORDER) != 5:
    raise ValueError("SAMPLE_ORDER must contain exactly 5 sample names.")

comp = pd.crosstab(adata.obs["sample"], adata.obs["cell_type"]).reindex(
    SAMPLE_ORDER,
    fill_value=0,
)

ax = comp.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    width=0.8,
)

ax.set_ylabel("Number of Cells")
ax.set_xlabel("Sample")
ax.set_title("Cell Type Composition by Sample")
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
