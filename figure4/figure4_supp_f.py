# Panel: figure3_supp_f
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_e_benchmark_violin_70pair.py
# Original filename: fig1_e_benchmark_violin_70pair.py
# Last modified: 2026-08-18 10:29:09  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/derived/{procB_spagfm_70pair.csv, procB_genept_70pair.csv, gears_recompute_long.csv, g2g_null_long.csv}
# Output: data/work/fig1_e/violin_four_arm_v23_procB_70pair.{pdf,png,svg}
# Match evidence: Run log make_figure_v23_70pair.log prints BA 0.6238 / 0.5812 / 0.5552 / 0.5008 -> the annotated 0.624/0.581/0.555/0.501, and F1 0.6219 / 0.5779 / 0.5527 / 0.4784 -> 0.622/0.578/0.553/0.478. Sample sizes n=70, 70, 24, 24 match ('GEARS and Celcomen shadowed at n=24'). Brackets ***, *, ***.
# Other candidates considered: fig1_e_benchmark_violin_24pair.py / fig1_e_benchmark_box_24pair.py -- all four arms at n=24 and different values (0.601/0.558); the 24-pair box version is figure3d.
# ---- copied verbatim below; NOT modified ----
"""
v23 -- the pipeline-B (v22) results drawn over the FULL 70 (split, perturbation) pairs,
with GEARS and Celcomen SHADOWED because they only ever covered 24 of those pairs.
Requested 2026-08-17. Derived from make_figure_v22_meanline.py; v22 is untouched.

WHAT CHANGED vs v22/v22-meanline
  v22 restricted every arm to the 24 pairs GEARS covers, so all four violins were
  like-for-like. v23 instead shows spaGFM and GenePT* over all 70 pairs -- the full
  population those two arms were actually scored on -- and keeps GEARS and Celcomen at
  their native 24. THE PANEL IS THEREFORE MIXED-n BY CONSTRUCTION. The shadow styling
  exists to make that unmissable; it is not decoration.

  Same underlying numbers as v22: no arm is refitted or rescored here. The only
  difference is which rows are plotted.

SHADOWING (arms with `shadow: True`)
  fill blended 55% toward #BFBFBF and alpha 0.60 -> 0.34; points alpha 0.85 -> 0.45;
  mean bar and its label in #7A7A7A instead of black; x-tick label and n= label greyed.
  The mean VALUE is still printed -- shadowed means what "do not compare this directly
  against the 70-pair arms", not "unreliable in itself".

STATISTICS -- read this before quoting any star on this figure.
  A paired Wilcoxon signed-rank test requires the same pairs on both sides, so the
  comparisons on this panel do NOT all share a pairing:
    spaGFM vs GenePT*   paired on all 70 pairs   (both arms cover 70)  -> BLACK bracket
    spaGFM vs GEARS     paired on the 24 subset  (GEARS covers 24)     -> GREY bracket
    spaGFM vs Celcomen  paired on the 24 subset                        -> GREY bracket
    GenePT* vs GEARS    paired on the 24 subset  (in CSV, not drawn)
  Grey brackets are therefore tests on a SUBSET of the plotted spaGFM violin, not on the
  70 pairs the violin displays. The `n_pairs` column of the CSV records which. Holm-
  Bonferroni is applied within each metric panel across the four comparisons, as in v22;
  because the family mixes pairings the correction is a convention here, not a clean
  single-population adjustment.

  CONSEQUENCE, stated plainly: the two grey brackets are the v22 numbers unchanged. Only
  the spaGFM vs GenePT* bracket is a genuinely new 70-pair test.

CAVEAT carried from v22 RUN_NOTES.md: the 24-pair population is ~10 distinct
perturbations with split-level repeats, so tests involving GEARS/Celcomen remain
optimistic about their effective sample size. The 70-pair population is 30 distinct
perturbations over 10 splits.

Arm sources (all read-only; identical to v22):
  spaGFM   pipeline B -- panel_e_triangle_procB_joint/violin_arm_procB_joint_long.csv
  GenePT*  pipeline B -- this directory, genept_arm_procB_long.csv
  GEARS    from v21   -- benchmark_v10_celcomen_ridge/gears_recompute_long.csv
  Celcomen from v21   -- benchmark_v19_g2g_null/g2g_null_long.csv (variant "real")

Verified before plotting: the spaGFM and GenePT* 70-pair sets are identical, and the
GEARS/Celcomen 24-pair sets are identical to each other and a strict subset of the 70.

Figures (per-violin mean bar + label + n, y in [0.30, 0.98]):
  violin_four_arm_v23_procB_70pair.{pdf,png,svg}

Stars: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise.

Reads only. Nothing v22 wrote, and nothing in benchmark_v10_celcomen_ridge/,
benchmark_v16_control_seeded/, benchmark_v18_clean_g2g/ or benchmark_v19_g2g_null/, is
written. Hard assertions at the end raise on failure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wilcoxon

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


SHADOW_GREY = "#BFBFBF"     # colour the shadowed fills are blended toward
SHADOW_INK = "#7A7A7A"      # mean bar / label / tick colour for shadowed arms


def shade(hexcol: str, frac: float = 0.55) -> tuple:
    """Blend a fill colour toward SHADOW_GREY. frac=0 keeps it, frac=1 makes it grey."""
    c = np.asarray(mcolors.to_rgb(hexcol), dtype=float)
    g = np.asarray(mcolors.to_rgb(SHADOW_GREY), dtype=float)
    return tuple((1.0 - frac) * c + frac * g)


def draw(ax, arms, metric, ylabel, fs=11, outline=False, brackets=None):
    rng = np.random.default_rng(42)
    for xi, a in enumerate(arms):
        v = a["data"][metric].to_numpy(float)
        sh = bool(a.get("shadow", False))
        fill = shade(a["color"]) if sh else a["color"]
        f_alpha = 0.34 if sh else 0.60
        p_alpha = 0.45 if sh else (0.85 if len(v) <= 30 else 0.6)
        ink = SHADOW_INK if sh else "black"
        if len(v) >= 2 and np.ptp(v) > 0:
            kde = gaussian_kde(v, bw_method="scott")
            yg = np.linspace(v.min() - 0.02, v.max() + 0.02, 300)
            dens = kde(yg)
            dens = dens / dens.max() * 0.34
            ax.fill_betweenx(yg, xi - dens, xi + dens, color=fill, alpha=f_alpha,
                             linewidth=0.0)
            ec = "black" if outline else fill
            lw = 1.0 if outline else 0.7
            # shadowed arms get a dashed rim so the distinction survives greyscale print
            ls = (0, (4, 2)) if sh else "-"
            ax.plot(xi - dens, yg, color=ec, lw=lw, ls=ls,
                    alpha=0.9 if outline else (0.6 if sh else 0.8))
            ax.plot(xi + dens, yg, color=ec, lw=lw, ls=ls,
                    alpha=0.9 if outline else (0.6 if sh else 0.8))
            if outline:
                for k in (0, -1):
                    ax.plot([xi - dens[k], xi + dens[k]], [yg[k], yg[k]],
                            color=ec, lw=lw, alpha=0.9)
        ax.scatter(xi + rng.uniform(-0.10, 0.10, len(v)), v, color=fill,
                   s=16 if len(v) <= 30 else 9, alpha=p_alpha,
                   zorder=3, edgecolors="white", linewidths=0.3)
        mu = float(v.mean())
        ax.plot([xi - 0.22, xi + 0.22], [mu, mu], color=ink, lw=2.6, zorder=4)
        ax.text(xi + 0.26, mu, f"{mu:.3f}", va="center", ha="left",
                fontsize=9.5, fontweight="bold", color=ink)
        ax.text(xi, 0.315, f"n={len(v)}", ha="center", va="bottom", fontsize=8.0,
                color=SHADOW_INK if sh else "#444444",
                fontstyle="italic" if sh else "normal")

    top = 0.85
    if brackets:
        for lvl, (i, j, lab, b_sh) in enumerate(brackets):
            y = 0.795 + lvl * 0.055
            bc = SHADOW_INK if b_sh else "black"
            ax.plot([i, i, j, j], [y, y + 0.014, y + 0.014, y],
                    color=bc, lw=1.1, ls=(0, (4, 2)) if b_sh else "-", clip_on=False)
            ax.text((i + j) / 2, y + 0.019, lab, ha="center", va="bottom",
                    color=bc, fontsize=10.5 if lab != "ns" else 8.5)
            top = max(top, y + 0.055)

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a["label"] for a in arms], fontsize=fs, rotation=45, ha="right")
    for tick, a in zip(ax.get_xticklabels(), arms):
        if a.get("shadow", False):
            tick.set_color(SHADOW_INK)
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
    say("v23 -- pipeline-B results over all 70 pairs; GEARS + Celcomen shadowed at n=24")
    say("=" * 78)

    gears = load_method(V10 / "gears_recompute_long.csv", "GEARS")
    pair24 = set(zip(gears.split_idx.astype(int), gears.perturbation.astype(str)))
    assert len(pair24) == 24, f"expected 24 pairs, got {len(pair24)}"

    long = pd.read_csv(V19 / "g2g_null_long.csv")
    cel = {v: g for v, g in long.groupby("variant")}

    spagfm70 = load_plain(PROCB_SPAGFM)
    genept70 = load_plain(PROCB_GENEPT)
    pair70 = set(zip(spagfm70.split_idx.astype(int), spagfm70.perturbation.astype(str)))
    assert len(pair70) == 70, f"expected 70 pairs, got {len(pair70)}"
    assert set(zip(genept70.split_idx.astype(int),
                   genept70.perturbation.astype(str))) == pair70, \
        "spaGFM and GenePT* do not cover the same 70 pairs"
    assert pair24 < pair70, "the 24-pair set is not a strict subset of the 70-pair set"
    say(f"  full pair set (pipeline B arms): n={len(pair70)} "
        f"({spagfm70.perturbation.nunique()} perturbations x "
        f"{spagfm70.split_idx.nunique()} splits)")
    say(f"  shadowed pair set (GEARS/Celcomen): n={len(pair24)}  -- a strict subset")

    # Arms are plotted at their NATIVE coverage: no restriction to the 24-pair intersection.
    base = [
        {"label": "spaGFM", "color": "#F2A1A7", "zero_shot": False, "shadow": False,
         "n_expect": 70, "data": spagfm70},
        {"label": "GenePT*", "color": "#D5EAD9", "zero_shot": False, "shadow": False,
         "n_expect": 70, "data": genept70},
        {"label": "GEARS", "color": "#9BD7F3", "zero_shot": False, "shadow": True,
         "n_expect": 24, "data": restrict(gears, pair24)},
        {"label": "Celcomen", "color": "#7DC69B", "zero_shot": True, "shadow": True,
         "n_expect": 24, "data": cel["real"].sort_values(
             ["split_idx", "perturbation"]).reset_index(drop=True)},
    ]
    idx = {a["label"]: i for i, a in enumerate(base)}
    for a in base:
        got = set(zip(a["data"].split_idx.astype(int), a["data"].perturbation.astype(str)))
        want = pair70 if a["n_expect"] == 70 else pair24
        assert got == want, f"{a['label']}: {len(got)} pairs, expected {len(want)}"
        say(f"  {a['label']:10s} n={len(a['data']):3d}"
            f"{'  [SHADOWED]' if a['shadow'] else '            '}  "
            f"BA={a['data'][BA].mean():.4f}  F1={a['data'][F1].mean():.4f}")

    # ---- paired Wilcoxon + Holm, within each metric ---------------------------------
    say("")
    say("=" * 78)
    say("PAIRED WILCOXON SIGNED-RANK (two-sided) + HOLM-BONFERRONI")
    say("pairing = intersection of the two arms' coverage; n is per-comparison")
    say("=" * 78)
    rows, brackets = [], {}
    for metric, _yl, tag in METRICS:
        raw, meta = [], []
        for a, b, drawn in COMPARISONS:
            # Pair on the INTERSECTION of the two arms' coverage. For spaGFM vs GenePT*
            # that is all 70; every comparison touching GEARS/Celcomen collapses to 24.
            da, db = base[idx[a]]["data"], base[idx[b]]["data"]
            common = (set(zip(da.split_idx.astype(int), da.perturbation.astype(str)))
                      & set(zip(db.split_idx.astype(int), db.perturbation.astype(str))))
            xa, xb = restrict(da, common), restrict(db, common)
            assert (xa.perturbation.values == xb.perturbation.values).all() and \
                   (xa.split_idx.values == xb.split_idx.values).all(), \
                   f"pairing misaligned for {a} vs {b}"
            assert len(xa) == len(common), f"{a} vs {b}: duplicate pairs after restrict"
            va, vb = xa[metric].to_numpy(float), xb[metric].to_numpy(float)
            stat, p = wilcoxon(va, vb, alternative="two-sided")
            raw.append(float(p))
            # a bracket is shadowed when its test used fewer pairs than the panel shows
            meta.append((a, b, drawn, float(stat), float(va.mean() - vb.mean()),
                         len(common), len(common) < len(pair70)))
        adj = holm(raw)
        for (a, b, drawn, stat, dmean, npair, sh), p, pa in zip(meta, raw, adj):
            rows.append({"metric": tag, "comparison": f"{a} vs {b}", "n_pairs": npair,
                         "test": "wilcoxon signed-rank (two-sided, paired)",
                         "pairing": "70-pair (full)" if not sh else "24-pair subset",
                         "statistic": stat, "mean_diff": dmean,
                         "p_raw": p, "p_holm": pa, "stars_holm": stars(pa),
                         "drawn_on_figure": drawn, "bracket_shadowed": sh})
            say(f"  [{tag}] {a:8s} vs {b:9s}  n={npair:3d}  dmean={dmean:+.4f}  "
                f"W={stat:7.1f}  p_raw={p:.3e}  p_holm={pa:.3e}  {stars(pa)}"
                f"{'  [grey bracket: 24-pair subset]' if sh else ''}"
                f"{'' if drawn else '   (not drawn)'}")
        brackets[tag] = [(idx[a], idx[b], stars(pa), sh)
                         for (a, b, drawn, _s, _d, _n, sh), pa in zip(meta, adj) if drawn]

    pdf = pd.DataFrame(rows)
    pdf.to_csv(HERE / "violin_v23_procB_70pair_pvalues.csv", index=False)
    say("  saved violin_v23_procB_70pair_pvalues.csv")

    # ---- figures ---------------------------------------------------------------------
    say("")
    sub4 = ("spaGFM and GenePT* (pipeline B) over all 70 (split, perturbation) pairs\n"
            "GEARS and Celcomen shaded: they cover only 24 of those pairs, so the panel "
            "is mixed-n and the shaded arms\nare not directly comparable to the other two. "
            "Brackets = paired Wilcoxon vs spaGFM, Holm-adjusted:\n"
            "black solid = all 70 pairs, grey dashed = 24-pair subset "
            "(*** p<0.001, ** p<0.01, * p<0.05, ns otherwise)")
    render(base, "violin_four_arm_v23_procB_70pair", sub4, 9.4, brackets=brackets, figh=6.9)

    nulls = []

    # ---- hard assertions -----------------------------------------------------------
    say("")
    say("FIGURE SELF-VERIFICATION (raises on failure)")
    checks = []
    for a in base + nulls:
        lbl = a["label"].replace("\n", " ")
        checks.append((f"{lbl}: no NaN in metric columns",
                       int(a["data"][[BA, F1]].isna().sum().sum()) == 0))
    checks.append(("each arm has exactly its native row count (70/70/24/24)",
                   [len(a["data"]) for a in base] == [70, 70, 24, 24]))
    checks.append(("exactly 2 arms are shadowed, and they are GEARS + Celcomen",
                   [a["label"] for a in base if a["shadow"]] == ["GEARS", "Celcomen"]))
    checks.append(("Celcomen arm carried over unchanged from v21",
                   abs(base[3]["data"][BA].mean() - 0.500773) < 1e-6))
    checks.append(("GEARS arm carried over unchanged from v21",
                   abs(base[2]["data"][BA].mean() - 0.555216) < 1e-4))
    # the 70-pair arms must still reproduce the v22 24-pair means when restricted back
    for a, want in ((base[0], 0.6005), (base[1], 0.5576)):
        got = float(restrict(a["data"], pair24)[BA].mean())
        checks.append((f"{a['label']} restricted to the 24 pairs reproduces v22 "
                       f"({want:.4f}, got {got:.4f})", abs(got - want) < 5e-5))
    checks.append((f"p-value table has {2 * len(COMPARISONS)} rows",
                   len(pdf) == 2 * len(COMPARISONS)))
    checks.append(("no NaN p-values", int(pdf[["p_raw", "p_holm"]].isna().sum().sum()) == 0))
    checks.append(("p_holm >= p_raw for every comparison",
                   bool((pdf.p_holm >= pdf.p_raw - 1e-15).all())))
    checks.append(("3 brackets drawn per metric",
                   all(len(brackets[t]) == 3 for _m, _y, t in METRICS)))
    checks.append(("exactly 1 black (70-pair) bracket per metric, 2 grey",
                   all(sum(1 for b in brackets[t] if not b[3]) == 1
                       for _m, _y, t in METRICS)))
    checks.append(("only spaGFM vs GenePT* used all 70 pairs",
                   set(pdf.loc[pdf.n_pairs == 70, "comparison"]) == {"spaGFM vs GenePT*"}))
    for stem in ("violin_four_arm_v23_procB_70pair",):
        for ext in ("pdf", "png", "svg"):
            f = HERE / f"{stem}.{ext}"
            checks.append((f"{f.name} exists, nonzero", f.exists() and f.stat().st_size > 0))
    for name, ok in checks:
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    bad = [n for n, ok in checks if not ok]
    (HERE / "make_figure_v23_70pair.log").write_text("\n".join(lines) + "\n")
    if bad:
        raise SystemExit(f"FIGURE VERIFICATION FAILED: {bad}")
    print("\nSaved make_figure_v23_70pair.log")


if __name__ == "__main__":
    main()
