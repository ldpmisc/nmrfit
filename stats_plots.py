# stats_plots.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from stats_extract import classify_param_name

# -------------------------
# Helpers
# -------------------------

_CLASS_ORDER = {"amp": 0, "pos": 1, "lor": 2, "gauss": 3, "k": 4, "other": 5}


def _tokenize(name: str) -> List[str]:
    s = (name or "").lower().replace("-", "_")
    return [t for t in s.split("_") if t]


def _sort_key_for_name(name: str) -> Tuple[int, int, int, str]:
    info = classify_param_name(name)
    s = info.get("slice")
    p = info.get("peak")
    c = info.get("pclass", "other")
    s_i = int(s) if isinstance(s, int) else 10**9
    p_i = int(p) if isinstance(p, int) else 10**9
    c_i = _CLASS_ORDER.get(str(c), 99)
    return (s_i, p_i, c_i, str(name))


def _get_pairs_list(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    corr = (stats or {}).get("corr", {}) or {}
    # Prefer annotated list if present (more complete than top_n pairs)
    bundle_all = corr.get("bundle_all", {}) or {}
    ann = bundle_all.get("annotated_pairs", None)
    if isinstance(ann, list) and ann:
        return [p for p in ann if isinstance(p, dict)]
    pairs = corr.get("pairs_all", []) or []
    return [p for p in pairs if isinstance(p, dict)]


def _param_value_map(stats: Dict[str, Any]) -> Dict[str, float]:
    rows = (stats or {}).get("params", []) or []
    out: Dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "") or "")
        v = r.get("value", None)
        if isinstance(v, (int, float)):
            out[name] = float(v)
    return out


def _amp_width_points(stats: Dict[str, Any], *, width_kind: str = "both") -> Dict[str, List[Tuple[float, float]]]:
    """
    Build points (amp, width) from params by matching s#_p#_amp with s#_p#_lor / s#_p#_gauss.
    This does NOT depend on correlation pairs.
    """
    width_kind = (width_kind or "both").strip().lower()
    if width_kind not in ("lor", "gauss", "both"):
        width_kind = "both"

    rows = (stats or {}).get("params", []) or []
    # Map: (s, p) -> {class: value}
    by_sp: Dict[Tuple[Optional[int], Optional[int]], Dict[str, float]] = {}

    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "") or "")
        v = r.get("value", None)
        if not isinstance(v, (int, float)):
            continue
        info = classify_param_name(name)
        s = info.get("slice")
        p = info.get("peak")
        cls = str(info.get("pclass", "other"))
        if cls not in ("amp", "lor", "gauss"):
            continue
        key = (s if isinstance(s, int) else None, p if isinstance(p, int) else None)
        by_sp.setdefault(key, {})[cls] = float(v)

    pts = {"lor": [], "gauss": []}
    for _sp, d in by_sp.items():
        if "amp" not in d:
            continue
        a = d["amp"]
        if (width_kind in ("lor", "both")) and ("lor" in d):
            pts["lor"].append((a, d["lor"]))
        if (width_kind in ("gauss", "both")) and ("gauss" in d):
            pts["gauss"].append((a, d["gauss"]))
    return pts

# -------------------------
# Plotters
# -------------------------

def make_corr_heatmap_fig(
    stats: Dict[str, Any],
    *,
    title: str = "Correlation matrix (r)",
    sort_params: bool = True,
    cmap: str = "RdBu_r",
) -> Figure:
    """
    Build correlation heatmap figure.
    This is the core implementation used by both GUI display and file export.
    """
    corr = (stats or {}).get("corr", {}) or {}
    names = list(corr.get("names", []) or [])
    mat = corr.get("matrix", None)

    if mat is None or not names:
        raise ValueError("Correlation heatmap requires stats['corr']['matrix'] and ['names'].")

    M = np.array(mat, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != len(names):
        raise ValueError("Correlation matrix shape mismatch with names list.")

    if sort_params:
        order = sorted(range(len(names)), key=lambda i: _sort_key_for_name(names[i]))
        names = [names[i] for i in order]
        M = M[np.ix_(order, order)]

    fig = Figure()
    ax = fig.add_subplot(111)

    M_plot = M.copy()
    np.fill_diagonal(M_plot, np.nan)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    im = ax.imshow(
        M_plot,
        cmap=cmap_obj,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation (r)")
    cbar.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_title(title)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Parameters")

    n = len(names)
    if n <= 100:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def plot_corr_heatmap(
    stats: Dict[str, Any],
    out_path: str,
    *,
    title: str = "Correlation matrix (r)",
    sort_params: bool = True,
    dpi: int = 200,
) -> str:
    """
    Save correlation heatmap to disk.
    Thin wrapper around make_corr_heatmap_fig().
    """
    fig = make_corr_heatmap_fig(
        stats,
        title=title,
        sort_params=sort_params,
        cmap="RdBu_r",
    )

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi)
    plt.close(fig)
    return str(out)


def make_absr_distribution_fig(
    stats: Dict[str, Any],
    *,
    kind: str = "cdf",
    thresholds: Tuple[float, float] = (0.6, 0.8),
    title: str = "Distribution of |r|",
) -> Figure:
    """
    Build |r| distribution figure.
    This is the core implementation used by both GUI display and file export.
    """
    pairs = _get_pairs_list(stats)
    abs_r = np.array([float(p.get("abs_r", 0.0) or 0.0) for p in pairs], dtype=float)
    abs_r = abs_r[(abs_r >= 0.0) & (abs_r <= 1.0)]

    if abs_r.size == 0:
        raise ValueError("No correlation pairs with abs_r available.")

    fig = Figure()
    ax = fig.add_subplot(111)

    kind = (kind or "cdf").strip().lower()

    if kind == "hist":
        ax.hist(abs_r, bins=40)
        ax.set_ylabel("Count")
    else:
        x = np.sort(abs_r)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y)
        ax.set_ylabel("CDF")

    for t in thresholds:
        ax.axvline(float(t), linestyle="--")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("|r|")
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_absr_distribution(
    stats: Dict[str, Any],
    out_path: str,
    *,
    kind: str = "cdf",
    thresholds: Tuple[float, float] = (0.6, 0.8),
    title: str = "Distribution of |r|",
    dpi: int = 200,
) -> str:
    """
    Save |r| distribution figure to disk.
    Thin wrapper around make_absr_distribution_fig().
    """
    fig = make_absr_distribution_fig(
        stats,
        kind=kind,
        thresholds=thresholds,
        title=title,
    )

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi)
    plt.close(fig)
    return str(out)


def make_amp_width_scatter_fig(
    stats: Dict[str, Any],
    *,
    width_kind: str = "both",
    title: str = "Amplitude vs linewidth",
) -> Figure:
    """
    Build amplitude-vs-linewidth scatter figure.
    This is the core implementation used by both GUI display and file export.
    """
    pts = _amp_width_points(stats, width_kind=width_kind)

    fig = Figure()
    ax = fig.add_subplot(111)

    if pts["lor"]:
        x, y = zip(*pts["lor"])
        ax.scatter(x, y, label="lor")

    if pts["gauss"]:
        x, y = zip(*pts["gauss"])
        ax.scatter(x, y, label="gauss")

    if not pts["lor"] and not pts["gauss"]:
        raise ValueError("No amp-width points found in params (need amp + lor/gauss for same s,p).")

    ax.set_xlabel("amp")
    ax.set_ylabel("linewidth")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_amp_width_scatter(
    stats: Dict[str, Any],
    out_path: str,
    *,
    width_kind: str = "both",
    title: str = "Amplitude vs linewidth",
    dpi: int = 200,
) -> str:
    """
    Save amplitude-vs-linewidth scatter figure to disk.
    Thin wrapper around make_amp_width_scatter_fig().
    """
    fig = make_amp_width_scatter_fig(
        stats,
        width_kind=width_kind,
        title=title,
    )

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi)
    plt.close(fig)
    return str(out)


def write_stats_figures(
    out_dir: str,
    stats: Dict[str, Any],
    *,
    prefix: str = "fit_stats",
    include_pdf: bool = False,
) -> Dict[str, str]:
    """
    Write a small figure bundle to a directory.
    Returns dict of written paths.
    """
    outd = Path(out_dir).expanduser().resolve()
    outd.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    # 1) heatmap
    written["corr_heatmap_png"] = plot_corr_heatmap(
        stats,
        str(outd / f"{prefix}_corr_heatmap.png"),
        title="Correlation matrix (r)",
        sort_params=True,
    )

    # 2) |r| CDF
    written["absr_cdf_png"] = plot_absr_distribution(
        stats,
        str(outd / f"{prefix}_absr_cdf.png"),
        kind="cdf",
        title="Distribution of |r|",
    )

    # 3) amp vs width scatter
    written["amp_width_scatter_png"] = plot_amp_width_scatter(
        stats,
        str(outd / f"{prefix}_amp_width_scatter.png"),
        width_kind="both",
        title="Amplitude vs linewidth",
    )

    if include_pdf:
        # Re-render the same figures as PDF.
        written["corr_heatmap_pdf"] = plot_corr_heatmap(
            stats,
            str(outd / f"{prefix}_corr_heatmap.pdf"),
            title="Correlation matrix (r)",
            sort_params=True,
            dpi=300,
        )
        written["absr_cdf_pdf"] = plot_absr_distribution(
            stats,
            str(outd / f"{prefix}_absr_cdf.pdf"),
            kind="cdf",
            title="Distribution of |r|",
            dpi=300,
        )
        written["amp_width_scatter_pdf"] = plot_amp_width_scatter(
            stats,
            str(outd / f"{prefix}_amp_width_scatter.pdf"),
            width_kind="both",
            title="Amplitude vs linewidth",
            dpi=300,
        )

    return written
