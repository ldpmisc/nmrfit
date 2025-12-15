# peak_table_io_v2.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import io

# ---------------------- Minimal types ----------------------
PeakRow     = Dict[str, object]
GlobalsMeta = Dict[str, float]   # {"offset":..., "multiplier":..., "phi0_deg":...}
excluded        = Dict[str, float]   # {"x_min": float, "x_max": float}
FitStats    = Dict[str, object]  # {"method": str, "success": bool, "nfev": int, "rmse": float, "chi2": float}

# ---------------------- Helpers ----------------------
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _fmt_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

def _fmt_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)

def _fmt_bool01(v: object) -> int:
    # Accept bools or "0/1" or truthy values
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return 1 if int(v) != 0 else 0
    except Exception:
        return 1 if bool(v) else 0

def _fmt_excluded(excluded: List[excluded]) -> str:
    if not excluded:
        return "[]"
    parts = []
    for m in excluded:
        try:
            parts.append(f'{{x_min={_fmt_float(m.get("x_min")):.3f}, x_max={_fmt_float(m.get("x_max")):.3f}}}')
        except Exception:
            continue
    return "[ " + ", ".join(parts) + " ]"

def _parse_excluded(s: str) -> List[excluded]:
    """
    Parse the simple textual 'excluded' format: [ {x_min=..., x_max=...}, {...} ].
    Robust to whitespace and missing pieces. Returns [] if parsing fails.
    """
    out: List[excluded] = []
    if not s or s.strip() in ("[]",):
        return out
    try:
        inner = s.strip()
        if inner.startswith("["):
            inner = inner[1:]
        if inner.endswith("]"):
            inner = inner[:-1]
        chunks = [c.strip() for c in inner.split("}") if c.strip()]
        for ch in chunks:
            if "x_min" not in ch and "x_max" not in ch:
                continue
            # remove any "{", commas → spaces, then split
            ch = ch.replace("{", "").replace("}", "").replace(",", " ")
            kv = {}
            for token in ch.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    kv[k.strip()] = v.strip()
            if "x_min" in kv and "x_max" in kv:
                out.append({"x_min": float(kv["x_min"]), "x_max": float(kv["x_max"])})
    except Exception:
        return []
    return out

# ---------------------- Writer ----------------------
def export_peak_table(path: str,
                      peaks: List[PeakRow],
                      ref: float,
                      globals_meta: GlobalsMeta,
                      globals_fix_flags: Dict[str, int],
                      excluded: List[excluded],
                      fit_stats: Optional[FitStats],
                      data_file: str,
                      program_version: str,
                      extra_meta: Optional[Dict[str, str]] = None) -> None:
    """
    Write a _fit.txt file. Backward-compatible with older exports.
    New: optional extra header lines via `extra_meta` (e.g., AxisMode, SliceIndex).
    """
    buf = io.StringIO()

    # --- Header ---
    print(f"# Program\t{program_version}", file=buf)
    print(f"# SavedUTC\t{_iso_utc_now()}", file=buf)
    print(f"# DataFile\t{data_file}", file=buf)

    # New optional meta header lines
    if extra_meta:
        for k, v in extra_meta.items():
            print(f"# {k}\t{v}", file=buf)
    # Reference
    print(f"# Ref_Hz\t{ref}", file=buf)

    # Globals (values) and fix flags
    g_off = _fmt_float(globals_meta.get("offset", 0.0))
    g_mul = _fmt_float(globals_meta.get("multiplier", 1.0))
    g_ph0 = _fmt_float(globals_meta.get("phi0_deg", 0.0))
    print(f"# Globals\toffset={g_off:.12g}; multiplier={g_mul:.3f}; phi0_deg={g_ph0:.3f}", file=buf)

    gf_off = _fmt_bool01(globals_fix_flags.get("offset", 0))
    gf_mul = _fmt_bool01(globals_fix_flags.get("multiplier", 0))
    gf_ph0 = _fmt_bool01(globals_fix_flags.get("phi0_deg", 0))
    print(f"# GlobalsFix\toffset={gf_off}; multiplier={gf_mul}; phi0_deg={gf_ph0}", file=buf)

    # Excluded
    print(f"# Excluded\t{_fmt_excluded(excluded)}", file=buf)

    # Fit stats (optional)
    if fit_stats:
        method  = str(fit_stats.get("method", ""))
        success = int(bool(fit_stats.get("success", False)))
        nfev    = _fmt_int(fit_stats.get("nfev", 0))
        rmse    = _fmt_float(fit_stats.get("rmse", 0.0))
        chi2    = _fmt_float(fit_stats.get("chi2", 0.0))
        print(f"# FitStats\tmethod={method}; success={success}; nfev={nfev}; rmse={rmse:.12g}; chi2={chi2:.12g}", file=buf)

    print("", file=buf)  # spacer

    # --- Table header ---
    header_cols = [
        "id", "pos", "pos_fix", "amp", "amp_fix",
        "lor_hz", "lor_fix", "gauss_disp", "gauss_fix", "notes"
    ]
    print("\t".join(header_cols), file=buf)

    # --- Rows ---
    for r in peaks:
        pid = str(r.get("id", ""))
        pos = _fmt_float(r.get("pos", 0.0))
        pfx = _fmt_bool01(r.get("pos_fix", 0))

        amp = _fmt_float(r.get("amp", 0.0))
        afx = _fmt_bool01(r.get("amp_fix", 0))

        lor = _fmt_float(r.get("lor_hz", 0.0))
        lfx = _fmt_bool01(r.get("lor_fix", 0))

        gau = _fmt_float(r.get("gauss_disp", 0.0))
        gfx = _fmt_bool01(r.get("gauss_fix", 0))

        print(f"{pid}\t{pos:.3f}\t{pfx}\t{amp:.3f}\t{afx}\t{lor:.3f}\t{lfx}\t{gau:.3f}\t{gfx}", file=buf)

    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(buf.getvalue())

# ---------------------- Reader ----------------------
def import_peak_table(path: str) -> Tuple[List[PeakRow], Dict[str, object]]:
    """
    Read a _fit.txt file and return (peaks, meta).
    - meta contains keys: program, saved_utc, data_file, globals, globals_fix,
                        excluded, fit_stats, axis_mode, slice_index
    """
    peaks: List[PeakRow] = []
    meta: Dict[str, object] = {}
    columns: List[str] = []
    excluded = []

    def _parse_kv_line(payload: str) -> Dict[str, str]:
        # "a=1; b=2; c=hi" -> {"a":"1","b":"2","c":"hi"}
        out: Dict[str, str] = {}
        for part in payload.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Header
            if line.startswith("#"):
                # Strip leading '#', split into key<TAB>value
                keyval = line[1:].strip().split("\t", 1)
                key = keyval[0].strip() if keyval else ""
                val = keyval[1].strip() if len(keyval) > 1 else ""

                if key == "Program":
                    meta["program"] = val
                elif key == "SavedUTC":
                    meta["saved_utc"] = val
                elif key.lower() == "data_file":
                    meta["data_file"] = val
                # NEW optional headers
                elif key.lower() == "axismode":
                    meta["axis_mode"] = val.lower()
                elif key.lower() == "sliceindex":
                    try:
                        meta["slice_index"] = int(val)
                    except Exception:
                        meta["slice_index"] = val
                elif key.startswith("GlobalsFix"):
                    kv = _parse_kv_line(val)
                    meta["globals_fix"] = {
                        "offset":     _fmt_bool01(kv.get("offset", 0)),
                        "multiplier": _fmt_bool01(kv.get("multiplier", 0)),
                        "phi0_deg":   _fmt_bool01(kv.get("phi0_deg", 0)),
                    }
                elif key.startswith("Globals"):
                    kv = _parse_kv_line(val)
                    meta["globals"] = {
                        "offset":     _fmt_float(kv.get("offset", 0.0)),
                        "multiplier": _fmt_float(kv.get("multiplier", 1.0)),
                        "phi0_deg":   _fmt_float(kv.get("phi0_deg", 0.0)),
                    }

                elif key.lower() == "excluded":
                    meta["excluded"] = _parse_excluded(val)
                elif key == "FitStats":
                    kv = _parse_kv_line(val)
                    meta["fit_stats"] = {
                        "method":  kv.get("method", ""),
                        "success": bool(int(kv.get("success", "0")) != 0),
                        "nfev":    _fmt_int(kv.get("nfev", 0)),
                        "rmse":    _fmt_float(kv.get("rmse", 0.0)),
                        "chi2":    _fmt_float(kv.get("chi2", 0.0)),
                    }
                continue  # next line

            # Table header
            if not columns:
                cols = [c.strip() for c in line.split("\t") if c.strip()]
                columns = cols
                continue

            # Data rows
            parts = line.split("\t")
            row: PeakRow = {}
            for c, v in zip(columns, parts):
                if c in ("pos", "amp", "lor_hz", "gauss_disp"):
                    row[c] = _fmt_float(v)
                elif c in ("pos_fix", "amp_fix", "lor_fix", "gauss_fix"):
                    row[c] = _fmt_bool01(v)
                else:
                    row[c] = v
            peaks.append(row)

    # Back-fill absent structures
    meta.setdefault("globals", {"offset": 0.0, "multiplier": 1.0, "phi0_deg": 0.0})
    meta.setdefault("globals_fix", {"offset": 0, "multiplier": 0, "phi0_deg": 0})
    meta.setdefault("excluded", [])
    # Provide axis_mode for older files (fallback to x_unit)
    if "axis_mode" not in meta and "x_unit" in meta:
        meta["axis_mode"] = str(meta["x_unit"]).lower()

    return peaks, meta
