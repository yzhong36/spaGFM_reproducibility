# Panel: figure3d
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_e_benchmark_box_24pair.py
# Original filename: fig1_e_benchmark_box_24pair.py
# Last modified: 2026-08-28 11:07:18  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/derived/{procB_spagfm_70pair.csv, procB_genept_70pair.csv, gears_recompute_long.csv, g2g_null_long.csv}
# Output: figure_bundle_20260817/data/work/fig1_e/box_four_arm_v22_procB_meanline.{pdf,png,svg} + box_v22_procB_meanline_pvalues.csv
# Match evidence: Run log make_figure_v22_box_meanline.log prints spaGFM 0.6005, GenePT* 0.5576, GEARS 0.5552, Celcomen 0.5008 -> the annotated 0.601/0.558/0.555/0.501. Arm colours pink/pale-green/blue/green, ylim (0.30,0.85), n=24 per arm. It is the ONLY 4-method balanced-accuracy BOXPLOT in the whole tree (verified by scanning every script mentioning 'Celcomen' for box-drawing code).
# Other candidates considered: fig1_e_benchmark_violin_24pair.py -- identical numbers but VIOLIN glyph. fig1_e_benchmark_violin_70pair.py -- violin AND different values (0.624/0.581), that is figure3_supp_f.
# ---- copied verbatim below; NOT modified ----
"""
BOX-PLOT VARIANT of fig1_e_benchmark_violin_24pair.py, requested 2026-08-28.

The ONLY change is the distribution glyph: the KDE violin body is replaced by a Tukey box
(IQR body, whiskers to the furthest point within 1.5*IQR, median line). Data, arms, arm
order, colours, pairing, statistics, brackets, axis limits, tick sets, font sizes, jitter
(same rng seed and same draw order, so the points land in identical positions), the black
mean bar with its bold 3-decimal label, the n= labels, the suptitle, figsize (9.4 x 6.6 in)
and the tight_layout/bbox_inches save path are all carried over unchanged, so the canvas
size and aspect ratio match the violin figure.

Fliers are deliberately not drawn: every point is already present in the jitter layer, so a
separate outlier marker would draw the same observation twice.

Two horizontal lines sit inside each box. The THIN one is the median (box furniture); the
THICK one at lw 2.6 is the MEAN, and the bold number beside it is that mean -- the same
statistic the violin version labelled.

Outputs are suffixed `box_` and nothing the violin script wrote is touched.

--- inherited notes from the violin version ---------------------------------------------

v22 -- v21 with the spaGFM and GenePT* arms refitted under pipeline B.

Copy of benchmark_v19_g2g_null/make_figure_v21.py; the original is untouched. Layout,
ordering, colour scheme, axis limits, annotation style and the statistical test are v21's.

Arm sources:
  spaGFM   REFITTED under B -- panel_e_triangle_procB_joint/violin_arm_procB_joint_long.csv
  GenePT*  REFITTED under B -- this directory, genept_arm_procB_long.csv
  GEARS    UNCHANGED from v21 -- benchmark_v10_celcomen_ridge/gears_recompute_long.csv
  Celcomen UNCHANGED from v21 -- benchmark_v19_g2g_null/g2g_null_long.csv (variant "real")

GEARS uses the external `gears` package and Celcomen a simcomen relaxation; neither shares
a training interface with TwoBranchMLP, so refitting them under B is not defined.

MEAN-MARKER VARIANT of make_figure_v22.py, requested 2026-08-17. Data, arms, pairing,
statistics and layout are byte-for-byte v22's; the ONLY change is that the black horizontal
mean line and the bold numeric mean label are drawn again, restored verbatim from
benchmark_v19_g2g_null/make_figure_v20.py:112-115 (bar x +/-0.22, lw 2.6, zorder 4; label at
x +0.26, fontsize 9.5, bold, 3 decimals).

CAVEAT, unchanged from v22: the 24 pairs are ~10 distinct perturbations with split-level
repeats, so a mean marker over them overstates the effective sample size. v22 dropped the
marker for that reason. This variant exists because the marker was asked for explicitly; it
does not supersede v22. All outputs are suffixed `_meanline` and nothing v22 wrote is touched.

Figures (per-violin mean bar + label + n, y in [0.30, 0.98]):
  box_four_arm_v22_procB_meanline.{pdf,png,svg}  4 arms, soft edges, p-value brackets

Arms are plain filled violins: the v18 zero-shot hatching was removed by request, so the
`zero_shot` field is retained as metadata but no longer affects style.

STATISTICS. Two-sided Wilcoxon signed-rank, PAIRED on the 24 (split, perturbation) pairs
that every arm shares. The family is the four comparisons listed in COMPARISONS, corrected
with Holm-Bonferroni WITHIN each metric panel; adjusted p is what the stars encode, matching
the convention in benchmark_v10_celcomen_ridge/figure_v12_pvalues.csv. Raw and adjusted p
are both written to box_v22_procB_meanline_pvalues.csv. Only the three spaGFM-vs-X
comparisons are drawn as brackets; GenePT* vs GEARS is in the family and in the CSV but not
drawn, to keep the panel legible.

Stars: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise.

Reads only. Nothing in benchmark_v10_celcomen_ridge/, benchmark_v16_control_seeded/ or
benchmark_v18_clean_g2g/ is written. Hard assertions at the end raise on failure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, wilcoxon  # noqa: F401  (kde unused in the box variant)

BOX_HW = 0.25     # box half-width; the violin body reached 0.34 at its widest
CAP_HW = 0.10     # whisker cap half-width

HERE = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/fig1_e")
FINAL = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived")
V10 = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived")
V19 = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived")
PROCB_SPAGFM = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived/procB_spagfm_70pair.csv")
PROCB_GENEPT = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived/procB_genept_70pair.csv")

BA = "balanced_accuracy_global"
F1 = "f1_macro_global"
METRICS = [(BA, "Balanced accuracy", "ba"), (F1, "F1-macro", "f1")]
NULL_COLORS = {"genelabel": "#C7C7C7", "entryshuffle": "#BFD3E6",
               "gaussian": "#E8C9C9", "zeros": "#D9D2E9"}

# family of comparisons; only the spaGFM-vs-X ones are drawn
COMPARISONS = [("spaGFM", "GenePT*", True), ("spaGFM", "GEARS", True),
               ("spaGFM", "Celcomen", True), ("GenePT*", "GEARS", False)]

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(str(s))


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def load_method(csv: Path, method: str) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d = d[d.method == method]
    if len(d) == 0:
        raise SystemExit(f"ERROR: no rows for {method!r} in {csv}")
    return d.sort_values(["split_idx", "perturbation"]).reset_index(drop=True)


def load_plain(csv: Path) -> pd.DataFrame:
    """Refitted-arm CSVs have no `method` column; they are already single-arm."""
    d = pd.read_csv(csv)
    return d.sort_values(["split_idx", "perturbation"]).reset_index(drop=True)


def restrict(d: pd.DataFrame, pairs: set) -> pd.DataFrame:
    k = d.apply(lambda r: (int(r.split_idx), str(r.perturbation)) in pairs, axis=1)
    return d[k].sort_values(["split_idx", "perturbation"]).reset_index(drop=True)


def draw(ax, arms, metric, ylabel, fs=11, outline=False, brackets=None):
    rng = np.random.default_rng(42)
    for xi, a in enumerate(arms):
        v = a["data"][metric].to_numpy(float)
        if len(v) >= 2 and np.ptp(v) > 0:
            # Tukey box: IQR body, whiskers to the most extreme point within 1.5*IQR.
            # Outliers are NOT drawn as separate markers -- every point is already in
            # the jitter layer below, so a flier symbol would double-draw it.
            q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
            iqr = q3 - q1
            inl = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
            lo, hi = (float(inl.min()), float(inl.max())) if len(inl) \
                else (float(v.min()), float(v.max()))
            ec = "black" if outline else a["color"]
            lw = 1.0 if outline else 0.9
            ax.add_patch(Rectangle((xi - BOX_HW, q1), 2 * BOX_HW, q3 - q1,
                                   facecolor=to_rgba(a["color"], 0.60),
                                   edgecolor=ec, linewidth=lw, zorder=2))
            ax.plot([xi, xi], [q3, hi], color=ec, lw=lw, zorder=2)
            ax.plot([xi, xi], [lo, q1], color=ec, lw=lw, zorder=2)
            for yv in (lo, hi):
                ax.plot([xi - CAP_HW, xi + CAP_HW], [yv, yv], color=ec, lw=lw, zorder=2)
            ax.plot([xi - BOX_HW, xi + BOX_HW], [med, med], color=ec, lw=lw, zorder=2)
        ax.scatter(xi + rng.uniform(-0.10, 0.10, len(v)), v, color=a["color"],
                   s=16 if len(v) <= 30 else 9, alpha=0.85 if len(v) <= 30 else 0.6,
                   zorder=3, edgecolors="white", linewidths=0.3)
        mu = float(v.mean())
        ax.plot([xi - 0.22, xi + 0.22], [mu, mu], color="black", lw=2.6, zorder=4)
        ax.text(xi + 0.26, mu, f"{mu:.3f}", va="center", ha="left",
                fontsize=9.5, fontweight="bold")
        ax.text(xi, 0.315, f"n={len(v)}", ha="center", va="bottom", fontsize=8.0,
                color="#444444")

    top = 0.85
    if brackets:
        for lvl, (i, j, lab) in enumerate(brackets):
            y = 0.795 + lvl * 0.055
            ax.plot([i, i, j, j], [y, y + 0.014, y + 0.014, y],
                    color="black", lw=1.1, clip_on=False)
            ax.text((i + j) / 2, y + 0.019, lab, ha="center", va="bottom",
                    fontsize=10.5 if lab != "ns" else 8.5)
            top = max(top, y + 0.055)

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a["label"] for a in arms], fontsize=fs, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0.30, max(0.85, top))
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlim(-0.7, len(arms) - 0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render(arms, stem, subtitle, figw, fs=11, outline=False, brackets=None, figh=5.9):
    fig, axes = plt.subplots(1, 2, figsize=(figw, figh))
    fig.patch.set_facecolor("white")
    for ax, (metric, ylabel, tag) in zip(axes, METRICS):
        draw(ax, arms, metric, ylabel, fs=fs, outline=outline,
             brackets=(brackets or {}).get(tag))
    fig.suptitle(subtitle, fontsize=9, y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    say(f"  saved {stem}.{{pdf,png}}  ({len(arms)} arms)")


def main() -> None:
    say("=" * 78)
    say("v22 BOX MEANLINE -- box-plot rendering of the v22 meanline violin panel")
    say("=" * 78)

    gears = load_method(V10 / "gears_recompute_long.csv", "GEARS")
    pair24 = set(zip(gears.split_idx.astype(int), gears.perturbation.astype(str)))
    assert len(pair24) == 24, f"expected 24 pairs, got {len(pair24)}"
    say(f"  reference pair set (GEARS): n={len(pair24)}")

    long = pd.read_csv(V19 / "g2g_null_long.csv")
    cel = {v: g for v, g in long.groupby("variant")}

    base = [
        {"label": "spaGFM", "color": "#F2A1A7", "zero_shot": False,
         "data": restrict(load_plain(PROCB_SPAGFM), pair24)},
        {"label": "GenePT*", "color": "#D5EAD9", "zero_shot": False,
         "data": restrict(load_plain(PROCB_GENEPT), pair24)},
        {"label": "GEARS", "color": "#9BD7F3", "zero_shot": False,
         "data": restrict(gears, pair24)},
        {"label": "Celcomen", "color": "#7DC69B", "zero_shot": True,
         "data": cel["real"].reset_index(drop=True)},
    ]
    idx = {a["label"]: i for i, a in enumerate(base)}
    for a in base:
        got = set(zip(a["data"].split_idx.astype(int), a["data"].perturbation.astype(str)))
        assert got == pair24, f"{a['label']}: {len(got)}/24 pairs"
        say(f"  {a['label']:10s} n={len(a['data']):3d}  "
            f"BA={a['data'][BA].mean():.4f}  F1={a['data'][F1].mean():.4f}")

    # ---- paired Wilcoxon + Holm, within each metric ---------------------------------
    say("")
    say("=" * 78)
    say("PAIRED WILCOXON SIGNED-RANK (two-sided, 24 pairs) + HOLM-BONFERRONI")
    say("=" * 78)
    rows, brackets = [], {}
    for metric, _yl, tag in METRICS:
        raw, meta = [], []
        for a, b, drawn in COMPARISONS:
            xa = base[idx[a]]["data"].sort_values(["split_idx", "perturbation"])
            xb = base[idx[b]]["data"].sort_values(["split_idx", "perturbation"])
            assert (xa.perturbation.values == xb.perturbation.values).all() and \
                   (xa.split_idx.values == xb.split_idx.values).all(), \
                   f"pairing misaligned for {a} vs {b}"
            va, vb = xa[metric].to_numpy(float), xb[metric].to_numpy(float)
            stat, p = wilcoxon(va, vb, alternative="two-sided")
            raw.append(float(p))
            meta.append((a, b, drawn, float(stat), float(va.mean() - vb.mean())))
        adj = holm(raw)
        for (a, b, drawn, stat, dmean), p, pa in zip(meta, raw, adj):
            rows.append({"metric": tag, "comparison": f"{a} vs {b}", "n_pairs": 24,
                         "test": "wilcoxon signed-rank (two-sided, paired)",
                         "statistic": stat, "mean_diff": dmean,
                         "p_raw": p, "p_holm": pa, "stars_holm": stars(pa),
                         "drawn_on_figure": drawn})
            say(f"  [{tag}] {a:8s} vs {b:9s}  dmean={dmean:+.4f}  W={stat:7.1f}  "
                f"p_raw={p:.3e}  p_holm={pa:.3e}  {stars(pa)}"
                f"{'' if drawn else '   (not drawn)'}")
        brackets[tag] = [(idx[a], idx[b], stars(pa))
                         for (a, b, drawn, _s, _d), pa in zip(meta, adj) if drawn]

    pdf = pd.DataFrame(rows)
    pdf.to_csv(HERE / "box_v22_procB_meanline_pvalues.csv", index=False)
    say("  saved box_v22_procB_meanline_pvalues.csv")

    # ---- figures ---------------------------------------------------------------------
    say("")
    sub4 = ("Same 24 (split, perturbation) pairs; spaGFM and GenePT* refitted under "
            "pipeline B; GEARS and Celcomen unchanged from v21\n"
            "brackets: paired Wilcoxon vs spaGFM, Holm-adjusted "
            "(*** p<0.001, ** p<0.01, * p<0.05, ns otherwise)")
    render(base, "box_four_arm_v22_procB_meanline", sub4, 9.4, brackets=brackets, figh=6.6)

    nulls = []

    # ---- hard assertions -----------------------------------------------------------
    say("")
    say("FIGURE SELF-VERIFICATION (raises on failure)")
    checks = []
    for a in base + nulls:
        lbl = a["label"].replace("\n", " ")
        checks.append((f"{lbl}: no NaN in metric columns",
                       int(a["data"][[BA, F1]].isna().sum().sum()) == 0))
    checks.append(("4 base arms each have exactly 24 rows",
                   all(len(a["data"]) == 24 for a in base)))
    checks.append(("Celcomen arm carried over unchanged from v21",
                   abs(base[3]["data"][BA].mean() - 0.500773) < 1e-6))
    checks.append((f"p-value table has {2 * len(COMPARISONS)} rows",
                   len(pdf) == 2 * len(COMPARISONS)))
    checks.append(("no NaN p-values", int(pdf[["p_raw", "p_holm"]].isna().sum().sum()) == 0))
    checks.append(("p_holm >= p_raw for every comparison",
                   bool((pdf.p_holm >= pdf.p_raw - 1e-15).all())))
    checks.append(("3 brackets drawn per metric",
                   all(len(brackets[t]) == 3 for _m, _y, t in METRICS)))
    for stem in ("box_four_arm_v22_procB_meanline",):
        for ext in ("pdf", "png", "svg"):
            f = HERE / f"{stem}.{ext}"
            checks.append((f"{f.name} exists, nonzero", f.exists() and f.stat().st_size > 0))
    for name, ok in checks:
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    bad = [n for n, ok in checks if not ok]
    (HERE / "make_figure_v22_box_meanline.log").write_text("\n".join(lines) + "\n")
    if bad:
        raise SystemExit(f"FIGURE VERIFICATION FAILED: {bad}")
    print("\nSaved make_figure_v22_box_meanline.log")


if __name__ == "__main__":
    main()
