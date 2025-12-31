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
    pairs: List[CorrPair]            # sorted by abs_r desc (filtered)
    note: str = None                 # reserved


def extract_FitResult(
    result: Any,
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
    params = getattr(result, "params", None)
    if params is None:
        return []

    rows: List[FitResult] = []
    for name, p in params.items():
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
                init_values=(None if getattr(p, "init_values", None) is None else p.init_values[0]),
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
    abs_threshold: float = 0.30,  # only keep pairs with |r| >= threshold
    top_n: int = 60,              # keep strongest pairs after filtering
    include_matrix: bool = True,  # set False if you only want pairs
) -> CorrInfo:
    """
    Extract correlation matrix & strong pairs for *varied independent parameters* only.

    Notes:
    - lmfit covariance/correlation is defined for the independent varied parameters.
      That's typically result.var_names ordering.
    - If result.covar is None, correlations cannot be computed.
    """
    covar = getattr(result, "covar", None)
    var_names = getattr(result, "var_names", []) or []

    if covar is None or len(var_names) == 0:
        note = "Covariance unavailable (singular fit, insufficient curvature, or lmfit did not estimate covar)."
        return CorrInfo(names=var_names, corr_mat=None, pairs=[], note=note)

    cov = np.asarray(covar, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n) or n != len(var_names):
        note = "Covariance shape mismatch vs var_names; cannot compute correlations reliably."
        return CorrInfo(names=var_names, corr_mat=None, pairs=[], note=note)

    # Compute correlation matrix corr[i,j] = cov[i,j] / sqrt(cov[i,i]*cov[j,j])
    diag = np.diag(cov)
    # Guard against non-positive or zero variances
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(np.outer(diag, diag))
        corr = cov / denom

    # Clean up numerical junk
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    pairs: List[CorrPair] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            if not math.isfinite(r):
                continue
            ar = abs(r)
            if ar < abs_threshold:
                continue
            pairs.append(
                CorrPair(
                    i=i,
                    j=j,
                    name_i=var_names[i],
                    name_j=var_names[j],
                    r=r,
                    abs_r=ar,
                )
            )

    pairs.sort(key=lambda x: x.abs_r, reverse=True)
    if top_n is not None and top_n > 0:
        pairs = pairs[:top_n]

    note = (
        f"Correlations computed for {n} independent varied parameters (lmfit var_names). "
        f"Showing {len(pairs)} pairs with |r| ≥ {abs_threshold:.2f}."
    )

    return CorrInfo(
        names=var_names,
        corr_mat=(corr if include_matrix else None),
        pairs=pairs,
        note=note,
    )

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
    *,
    include_fixed: bool = True,
    include_expr: bool = True,
    param_sort_by: str = "vary_first",
    abs_threshold: float = 0.30,
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
            "pairs": [p.__dict__ for p in corr_info.pairs],
            "matrix": matrix,
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

    out["corr"].setdefault("pairs", [])
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
    pairs = corr.get("pairs", []) or []
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
        # If you want a stable order, define it; otherwise, summary dict insertion order is used.
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
