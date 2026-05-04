# stats_extract.py
# Purpose: extract (1) a GUI-friendly parameter table and (2) correlation info
#          from an lmfit MinimizerResult (no string parsing).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import json
from pathlib import Path
from wsgiref import headers
import numpy as np
import re

def ppm_to_hz(x_ppm, ref: float):
    if ref is None or ref <= 0:
        raise ValueError("Invalid reference frequency. Must be > 0.")
    a = np.asarray(x_ppm, dtype=float)
    out = a * ref
    return float(out) if out.ndim == 0 else out

def hz_to_ppm(x_hz, ref: float):
    if ref is None or ref <= 0:
        raise ValueError("Invalid reference frequency. Must be > 0.")
    a = np.asarray(x_hz, dtype=float)
    out = a / ref
    return float(out) if out.ndim == 0 else out

@dataclass(frozen=True) #prevent modification
class FitResult:
    name: str
    value: float
    stderr: Optional[float]
    spercent: Optional[float]
    vary: bool
    min: Optional[float]
    max: Optional[float]
    expr: Optional[str]
    init_values: Optional[float]

@dataclass(frozen=True)
class CorrPair:
    i: int
    j: int
    name_i: str
    name_j: str
    r: float
    abs_r: float

@dataclass(frozen=True)
class CorrInfo:
    names: List[str]                 # ordering used in corr_mat
    corr_mat: Optional[np.ndarray]   # NxN, floats in [-1,1] (None if unavailable)
    pairs_all: List[CorrPair]        # all off-diagonal pairs
    pairs_filtered: List[CorrPair]   # threshold/top_n filtered pairs

    note: str = None                 # reserved

def extract_FitResult(
    result: Any, # lmfit.MinimizerResult object
    *,
    include_fixed: bool = True,
    include_expr: bool = True,
    sort_by: str = "name",  # "name" | "vary_first" | "none"
) -> List[FitResult]:
    """
    Build a flat parameter table from result.params.

    include_fixed: include vary=False parameters
    include_expr:  include expr-defined parameters (linked/deterministic)
    """
    params = getattr(result, "params", None) #dictionary-like of lmfit.Parameter objects aka Parameters object
    if params is None:
        return []

    rows: List[FitResult] = []
    for name, p in params.items(): #p: a lmfit.Parameter object
        vary = bool(getattr(p, "vary", False))
        expr = getattr(p, "expr", None)

        if (not include_fixed) and (not vary):
            continue
        if (not include_expr) and expr:
            continue
        value=float(getattr(p, "value", float("nan")))
        stderr=(None if getattr(p, "stderr", None) is None else float(p.stderr))
        if stderr is not None and np.isfinite(stderr) and np.isfinite(value) and abs(value) != 0:
            spercent = abs(stderr / value) * 100
        else:
            spercent = None
        # initial value of only varied parameters
        # init_values = result.init_values.get(name, None) if hasattr(result, "init_values") else None
        # initial value of all parameters
        init_values = p.init_value if hasattr(p, "init_value") else None

        rows.append(
            FitResult(
                name=name,
                value=value,
                stderr=stderr,
                spercent=spercent,
                vary=vary,
                min=(None if getattr(p, "min", None) is None else float(p.min)),
                max=(None if getattr(p, "max", None) is None else float(p.max)),
                expr=(None if not expr else str(expr)),
                init_values=init_values,
            )
        )

    if sort_by == "name":
        rows.sort(key=lambda r: r.name)
    elif sort_by == "vary_first":
        rows.sort(key=lambda r: (not r.vary, r.name))  # vary=True first
    elif sort_by == "none":
        pass
    else:
        raise ValueError(f"Unknown sort_by={sort_by!r}")

    return rows

def extract_correlation_info(
    result: Any,
    *,
    abs_threshold: float = 0.10,
    top_n: int = 60,
    include_matrix: bool = True,
) -> CorrInfo:
    """
    Extract full correlation matrix and both full + filtered correlation pairs.

    Notes:
    - corr_mat is the full NxN correlation matrix.
    - pairs_all contains all off-diagonal finite correlation pairs.
    - pairs_filtered contains only pairs with |r| >= abs_threshold,
      optionally limited to top_n.
    """
    covar = getattr(result, "covar", None)
    var_names = getattr(result, "var_names", []) or []

    if covar is None or len(var_names) == 0:
        note = "Covariance unavailable (singular fit, insufficient curvature, or lmfit did not estimate covar)."
        return CorrInfo(
            names=var_names,
            corr_mat=None,
            pairs_all=[],
            pairs_filtered=[],
            note=note,
        )

    cov = np.asarray(covar, dtype=float)
    n = cov.shape[0]

    if cov.shape != (n, n) or n != len(var_names):
        note = "Covariance shape mismatch vs var_names; cannot compute correlations reliably."
        return CorrInfo(
            names=var_names,
            corr_mat=None,
            pairs_all=[],
            pairs_filtered=[],
            note=note,
        )

    diag = np.diag(cov)

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(np.outer(diag, diag))
        corr = cov / denom

    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    pairs_all: List[CorrPair] = []

    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])

            if not math.isfinite(r):
                continue

            pairs_all.append(
                CorrPair(
                    i=i,
                    j=j,
                    name_i=var_names[i],
                    name_j=var_names[j],
                    r=r,
                    abs_r=abs(r),
                )
            )

    pairs_all.sort(key=lambda x: x.abs_r, reverse=True)

    pairs_filtered = [p for p in pairs_all if p.abs_r >= abs_threshold]

    if top_n is not None and top_n > 0:
        pairs_filtered = pairs_filtered[:top_n]

    note = (
        f"Correlations computed for all independent varied parameters "
        f"({len(pairs_all)} total off-diagonal pairs). "
        f"Filtered view shows {len(pairs_filtered)} pairs with |r| ≥ {abs_threshold:.2f}"
        + (f", top {top_n}." if top_n else ".")
    )

    return CorrInfo(
        names=var_names,
        corr_mat=(corr if include_matrix else None),
        pairs_all=pairs_all,
        pairs_filtered=pairs_filtered,
        note=note,
    )
# =========================
# Correlation classification
# =========================

_PARAM_CLASSES = ("amp", "pos", "lor", "gauss", "k", "other")

# Regexes that match common naming conventions in nmrFit (adjust if needed)
_RE_SLICE = re.compile(r"(?:^|[_\-])s(?P<s>\d+)(?:[_\-]|$)", re.IGNORECASE)
_RE_PEAK  = re.compile(r"(?:^|[_\-])p(?P<p>\d+)(?:[_\-]|$)", re.IGNORECASE)

# "pos/lor/gau/amp/k" token detection (string heuristics)
# Keep this intentionally conservative to avoid mislabeling.
_CLASS_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("amp", ("amp", "ampl", "amplitude")),
    ("pos", ("pos", "ppm", "shift", "cs")),
    ("lor", ("lor", "lorentz", "lorentzian")),
    ("gauss", ("gauss", "gaussian")),
    ("k",   ("k", "rate")),       # NOTE: avoids labeling "t" or "T" here on purpose
]


def _tokenize_param_name(name: str) -> List[str]:
    s = (name or "").lower().replace("-", "_")
    return [t for t in s.split("_") if t]


def classify_param_name(name: str) -> Dict[str, Any]:
    """
    Return a minimal classification for a parameter name.

    Output keys:
      - slice: Optional[int]
      - peak: Optional[int]
      - pclass: str in {"amp","pos","lor","gau","k","other"}
    """
    name = str(name or "")
    tokens = _tokenize_param_name(name)

    # slice index
    m = _RE_SLICE.search(name)
    s_idx = int(m.group("s")) if m else None

    # peak index
    m = _RE_PEAK.search(name)
    p_idx = int(m.group("p")) if m else None

    # parameter class
    pclass = "other"
    for cls, keys in _CLASS_RULES:
        if any(k in tokens for k in keys):
            pclass = cls
            break

    return {"slice": s_idx, "peak": p_idx, "pclass": pclass}


def annotate_corr_pairs(
    pairs: List[Dict[str, Any]],
    *,
    include_same_peak: bool = True,
) -> List[Dict[str, Any]]:
    """
    Expects dicts like those produced by CorrPair.__dict__ in extract_correlation_info().
    """
    out: List[Dict[str, Any]] = []
    for p in (pairs or []):
        d = dict(p)
        ni = str(d.get("name_i", ""))
        nj = str(d.get("name_j", ""))

        ci = classify_param_name(ni)
        cj = classify_param_name(nj)

        d["slice_i"] = ci["slice"]
        d["slice_j"] = cj["slice"]
        d["peak_i"] = ci["peak"]
        d["peak_j"] = cj["peak"]

        d["class_i"] = ci["pclass"]
        d["class_j"] = cj["pclass"]

        d["intra_class"] = (ci["pclass"] == cj["pclass"])
        d["pair_class"] = (
            ci["pclass"] if d["intra_class"]
            else f"{ci['pclass']}-{cj['pclass']}"
        )

        # same-slice / cross-slice (only if slice indices exist)
        si, sj = ci["slice"], cj["slice"]
        if (si is None) or (sj is None):
            d["slice_relation"] = "unknown"
            d["same_slice"] = None
        else:
            d["same_slice"] = (si == sj)
            d["slice_relation"] = ("same_slice" if si == sj else "cross_slice")

        # same-peak (optional, only meaningful if peak indices exist)
        if include_same_peak:
            pi, pj = ci["peak"], cj["peak"]
            if (pi is None) or (pj is None):
                d["same_peak"] = None
            else:
                d["same_peak"] = (pi == pj)

        out.append(d)
    return out


def filter_corr_pairs(
    pairs: List[Dict[str, Any]],
    *,
    abs_r_min: float = 0.60,
    slice_relation: Optional[str] = None,  # "same_slice" | "cross_slice" | "unknown" | None
    intra_class: Optional[bool] = None,
    class_in: Optional[Tuple[str, ...]] = None,   # e.g. ("amp","lor")
    top_n: Optional[int] = 40,
) -> List[Dict[str, Any]]:
    """
    Generic filter for annotated correlation pairs.
    """
    def ok(d: Dict[str, Any]) -> bool:
        ar = float(d.get("abs_r", 0.0) or 0.0)
        if ar < abs_r_min:
            return False
        if slice_relation is not None and d.get("slice_relation") != slice_relation:
            return False
        if intra_class is not None and bool(d.get("intra_class")) != bool(intra_class):
            return False
        if class_in is not None:
            if (d.get("class_i") not in class_in) and (d.get("class_j") not in class_in):
                return False
        return True

    filt = [d for d in (pairs or []) if ok(d)]
    filt.sort(key=lambda x: float(x.get("abs_r", 0.0) or 0.0), reverse=True)
    if top_n is not None:
        filt = filt[:max(0, int(top_n))]
    return filt


def corr_metrics(
    annotated_pairs: List[Dict[str, Any]],
    *,
    thresholds: Tuple[float, ...] = (0.6, 0.8),
) -> Dict[str, Any]:
    """
    Compute scalar metrics.
    Works only on PAIRS (not full matrix).
    """
    abs_rs = np.array([float(p.get("abs_r", 0.0) or 0.0) for p in (annotated_pairs or [])], dtype=float)
    if abs_rs.size == 0:
        return {
            "n_pairs": 0,
            "median_abs_r": None,
            "p90_abs_r": None,
            "max_abs_r": None,
            "counts_ge": {str(t): 0 for t in thresholds},
        }

    counts = {str(t): int(np.sum(abs_rs >= float(t))) for t in thresholds}
    return {
        "n_pairs": int(abs_rs.size),
        "median_abs_r": float(np.median(abs_rs)),
        "p90_abs_r": float(np.percentile(abs_rs, 90)),
        "max_abs_r": float(np.max(abs_rs)),
        "counts_ge": counts,
    }


def build_corr_bundle(
    corr_pairs: List[Dict[str, Any]],
    *,
    abs_r_min_for_lists: float = 0.60,
    top_n_each: int = 30,
) -> Dict[str, Any]:
    """
    Single entry point: annotate pairs and produce structured subsets + metrics.
    """
    ann = annotate_corr_pairs(corr_pairs)

    bundle = {
        "metrics_all": corr_metrics(ann),
        "metrics_same_slice": corr_metrics([p for p in ann if p.get("slice_relation") == "same_slice"]),
        "metrics_cross_slice": corr_metrics([p for p in ann if p.get("slice_relation") == "cross_slice"]),
        "top_all": filter_corr_pairs(ann, abs_r_min=abs_r_min_for_lists, top_n=top_n_each),
        "top_same_slice": filter_corr_pairs(ann, abs_r_min=abs_r_min_for_lists, slice_relation="same_slice", top_n=top_n_each),
        "top_cross_slice": filter_corr_pairs(ann, abs_r_min=abs_r_min_for_lists, slice_relation="cross_slice", top_n=top_n_each),
        "top_intra_class": filter_corr_pairs(ann, abs_r_min=abs_r_min_for_lists, intra_class=True, top_n=top_n_each),
        "top_inter_class": filter_corr_pairs(ann, abs_r_min=abs_r_min_for_lists, intra_class=False, top_n=top_n_each),
        # particularly important degeneracies in spectroscopy:
        "top_amp_width": filter_corr_pairs(ann, abs_r_min=abs_r_min_for_lists, class_in=("amp", "lor", "gau"), intra_class=False, top_n=top_n_each),
        "annotated_pairs": ann,  # keep full annotated list for downstream plotting if needed
    }
    return bundle

def stats_summary(result: Any, mode, slice_indices_list: List[int]) -> Dict: #result: lmfit.MinimizerResult object
    summary = {}
    chisqr = getattr(result, "chisqr", None)
    redchi = getattr(result,"redchi", None)
    aic = getattr(result,"aic", None)
    bic = getattr(result,"bic", None)
    nfev = getattr(result,"nfev", None)
    nvarys = getattr(result,"nvarys", None)
    nfree = getattr(result, "nfree", None)
    method = getattr(result, "method", None)
    data_points = getattr(result, 'ndata', None)
    
    summary["Mode"] = mode
    summary["Slices"] = slice_indices_list
    summary["Fitting method"] = method
    summary["Degree of freedoms"] = int(nfree)
    summary["Data points"] = int(data_points)
    summary["Number of free variables"] = int(nvarys)
    summary["Function evals"] = int(nfev)
    summary["chi-square"] = float(chisqr)
    summary["reduced chi-square"] = float(redchi)
    summary["Akaike info crit"] = float(aic)
    summary["Bayesian info crit"] = float(bic)
    
    return summary

def extract_FitResult_corr_and_sum(
    result: Any,
    mode: str,
    slice_indices_list: List[int],
    ref_MHz: Optional[float] = None,
    *,
    include_fixed: bool = True,
    include_expr: bool = True,
    param_sort_by: str = "name",
    abs_threshold: float = 0.10,
    top_n: int = 60,
    include_matrix: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper returning plain python structures (easy to json-ify if needed).
    """
    param_rows = extract_FitResult(
        result,
        include_fixed=include_fixed,
        include_expr=include_expr,
        sort_by=param_sort_by,
    )
    corr_info = extract_correlation_info(
        result,
        abs_threshold=abs_threshold,
        top_n=top_n,
        include_matrix=include_matrix,
    )
    corr_pairs_all = [p.__dict__ for p in corr_info.pairs_all]
    corr_pairs_filtered = [p.__dict__ for p in corr_info.pairs_filtered]
    corr_bundle_all = build_corr_bundle(corr_pairs_all, abs_r_min_for_lists=0.60, top_n_each=40)
    corr_bundle_filtered = build_corr_bundle(corr_pairs_filtered, abs_r_min_for_lists=0.60, top_n_each=40)

    matrix = None
    if corr_info.corr_mat is not None:
        # Convert ndarray -> nested python lists (JSON-serializable)
        matrix = corr_info.corr_mat.tolist()

    return {
        "summary": stats_summary(result, mode=mode, slice_indices_list=slice_indices_list),
        "params": [r.__dict__ for r in param_rows],
        "corr": {
            "names": list(corr_info.names),
            "note": corr_info.note,
            "pairs_all": corr_pairs_all,
            "pairs_filtered": corr_pairs_filtered,
            "bundle_all": corr_bundle_all,
            "bundle_filtered": corr_bundle_filtered,
            "matrix": matrix,
        },
        "meta": {
            "include_fixed": include_fixed,
            "include_expr": include_expr,
            "param_sort_by": param_sort_by,
            "abs_threshold": abs_threshold,
            "top_n": top_n,
            "include_matrix": include_matrix,
            "ref_MHz": ref_MHz,
        },        
    }


def _export_stats_payload(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and ensure JSON-serializable structure.
    Expected stats from extract_FitResult_corr_and_sum():
      {
        "params": [ { ... init_values ... }, ... ],
        "corr": { "pairs": [...], "names": [...], "note": str, "matrix": list|None },
        "summary": { "Mode": ..., "Slices": ..., ... }
      }
    """
    out = dict(stats or {})
    out.setdefault("summary", {})
    out.setdefault("params", [])
    out.setdefault("corr", {})

    if not isinstance(out["summary"], dict):
        out["summary"] = {}
    if not isinstance(out["corr"], dict):
        out["corr"] = {}

    out["corr"].setdefault("pairs_all", [])
    out["corr"].setdefault("pairs_filtered", [])
    out["corr"].setdefault("bundle_all", {})
    out["corr"].setdefault("bundle_filtered", {})
    out["corr"].setdefault("names", [])
    out["corr"].setdefault("note", "")
    out["corr"].setdefault("matrix", None)

    return out


def save_stats_json(path: str, stats: Dict[str, Any], *, indent: int = 2) -> str:
    payload = _export_stats_payload(stats)
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
    return str(p)


#def save_params_txt(path: str, stats: Dict[str, Any]) -> str:
#    payload = _export_stats_payload(stats)
#    rows = payload.get("params", []) or []
#
#    p = Path(path).expanduser().resolve()
#    p.parent.mkdir(parents=True, exist_ok=True)
#
#    headers = ["name", "value", "stderr", "spercent", "vary", "min", "max", "expr", "init_values"]
#
#    with p.open("w", encoding="utf-8") as f:
#        f.write("\t".join(headers) + "\n")
#        for r in rows:
#
#            r = r if isinstance(r, dict) else {}
#            line = []
#            for h in headers:
#                v = r.get(h, "")
#                line.append("" if v is None else str(v))
#            f.write("\t".join(line) + "\n")
#
#    return str(p)
#
#
#def save_corr_pairs_txt(path: str, stats: Dict[str, Any]) -> str:
#    payload = _export_stats_payload(stats)
#    corr = payload.get("corr", {}) or {}
#    pairs = corr.get("pairs", []) or []
#
#    p = Path(path).expanduser().resolve()
#    p.parent.mkdir(parents=True, exist_ok=True)
#
#    headers = ["name_i", "name_j", "r", "abs_r", "i", "j"]
#
#    with p.open("w", encoding="utf-8") as f:
#        f.write("\t".join(headers) + "\n")
#        for r in pairs:
#            r = r if isinstance(r, dict) else {}
#            line = []
#            for h in headers:
#                v = r.get(h, "")
#                line.append("" if v is None else str(v))
#            f.write("\t".join(line) + "\n")
#
#    return str(p)

def save_fit_report_txt(path: str, stats: Dict[str, Any]) -> str:
    payload = _export_stats_payload(stats)

    summary = payload.get("summary", {}) or {}
    rows = payload.get("params", []) or []
    corr = payload.get("corr", {}) or {}
    pairs = corr.get("pairs_all", []) or []
    note = corr.get("note", "") or ""

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Column order (match your data keys)
    param_headers = ["name", "value", "stderr", "spercent", "vary", "min", "max", "expr", "init_values"]
    corr_headers = ["name_i", "name_j", "r", "abs_r", "i", "j"]

    def _fmt(v):
        if v is None:
            return "—"
        return str(v)

    with p.open("w", encoding="utf-8") as f:
        # ========= Summary =========
        f.write("=== Fit Summary ===\n")

        preferred = [
            "Mode", "Slices",
            "chi-square", "reduced chi-square",
            "Data points", "Number of free variables", "Degree of freedoms",
            "Fitting method", "Function evals",
            "Akaike info crit", "Bayesian info crit",
        ]
        for k in preferred:
            if k in summary:
                f.write(f"{k}: {_fmt(summary.get(k))}\n")
        # Write any extra keys not in preferred
        for k, v in summary.items():
            if k not in preferred:
                f.write(f"{k}: {_fmt(v)}\n")

        f.write("\n")

        # ========= Parameters =========
        f.write("=== Parameters ===\n")
        f.write("\t".join(param_headers) + "\n")
        for r in rows:
            r = r if isinstance(r, dict) else {}
            line = []
            for h in param_headers:
                v = r.get(h, "")
                line.append("" if v is None else str(v))
            f.write("\t".join(line) + "\n")

        f.write("\n")

        # ========= Correlations =========
        f.write("=== Correlations (pairs) ===\n")
        if note:
            f.write(f"Note: {note}\n")
        f.write("\t".join(corr_headers) + "\n")
        for r in pairs:
            r = r if isinstance(r, dict) else {}
            line = []
            for h in corr_headers:
                v = r.get(h, "")
                line.append("" if v is None else str(v))
            f.write("\t".join(line) + "\n")

    return str(p)

def save_stats_bundle(base_path: str, stats: Dict[str, Any]) -> Dict[str, str]:
    """
    Write:
      <name>.json
      <name>_report.txt
    """
    base = Path(base_path).expanduser().resolve()

    # user may pick .json or .txt; normalize to a stem
    if base.suffix.lower() in (".json", ".txt"):
        stem = base.with_suffix("")
    else:
        stem = base

    written = {}
    written["json"] = save_stats_json(str(stem) + ".json", stats)
    written["report_txt"] = save_fit_report_txt(str(stem) + "_report.txt", stats)
    return written
