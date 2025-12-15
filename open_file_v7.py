from __future__ import annotations

import json, os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def load_ssnake(path: str,
                echo_path: Optional[str] = None,
                t_f1_unit: str = "ms"
               ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Returns
    -------
    x_f2_ppm : (F,) float64
    x_f2_Hz  : (F,) float64
    t_f1     : (N,) float64 or None           # only for 2D
    y_traces : (N, F) float64                 # 1D -> (1, F)
    meta     : dict
        keys: dim, n_traces, n_f2, t_f1_unit, sw_Hz, ref_Hz, transmitter_freq, fname
    """
    with open(path, "r") as f:
        struct = json.load(f)

    # -------- (4) Axis & meta extraction (Hz, y, SW, ref, MHz) --------

    xaxArray = struct.get("xaxArray", None)  #list
    if xaxArray is None or not isinstance(xaxArray, list) or len(xaxArray) == 0:
        raise ValueError("xaxArray is missing or invalid")
    
    y_raw = np.asarray(struct.get("dataReal", None)) #array 
    if y_raw is None:
        raise ValueError("data field missing")
    
    ref_arr  = struct.get("ref", None) #list
    sw_arr   = struct.get("sw", None) #list
    freq_arr = struct.get("freq", None)  #list transmitter frequency in Hz (per axis)
    
    # reference (Hz) for ppm conversion; guard ref==0
    ref_f2 = None
    if isinstance(ref_arr, (list, tuple)) and len(ref_arr) > 0:
        ref_f2 = float(ref_arr[-1])
    elif isinstance(ref_arr, (int, float)):
        ref_f2 = float(ref_arr)
    if not ref_f2:
        # fallback to 0 → ppm disabled
        ref_f2 = 0.0

    # spectral width & transmitter freq (Hz) for F2
    sw_f2 = None
    if isinstance(sw_arr, (list, tuple)) and len(sw_arr) > 0:
        sw_f2 = float(sw_arr[-1])
    elif isinstance(sw_arr, (int, float)):
        sw_f2 = float(sw_arr)

    tx_f2 = None
    if isinstance(freq_arr, (list, tuple)) and len(freq_arr) > 0:
        tx_f2 = float(freq_arr[-1])
    elif isinstance(freq_arr, (int, float)):
        tx_f2 = float(freq_arr)

    # -------- Raw data --------   
    # F2 is always the last axis
    x_f2_Hz = np.asarray(xaxArray[-1], dtype=np.float64)
    x_f2_ppm = (x_f2_Hz / ref_f2 * 1e6) if ref_f2 != 0.0 else np.zeros_like(x_f2_Hz)   
    F = x_f2_Hz.size
    

    # -------- (2) Normalize shapes to (N, F); also accept (F, N) --------
    if y_raw.ndim - 1 == 1:
        # (F,) -> (1, F)
        if y_raw.size != F:
            raise ValueError(f"1D data length {y_raw.size} != F2 axis length {F}")
        y_traces = y_raw.reshape(1, y_raw.size)

    elif y_raw.ndim - 1 == 2:
        _, N0, F0 = y_raw.shape
        if F0 == F:
            # already (N, F)
            y_traces = y_raw[0, :, :]
        elif N0 == F and F0 != F:
            # it is (F, N) -> transpose to (N, F)
            y_traces = y_raw[0].T

    else:
        raise ValueError(f"Only 1D or 2D supported. Got ndim={y_raw.ndim}")
    
    N = y_traces.shape[0] #number of rows. should equals to number of relaxation times

    # -------- Meta --------
    meta = {
        "dim": 1 if N == 1 else 2,
        "number of traces": int(N),
        "n_f2": int(F),
        "t_f1_unit": str(t_f1_unit),
        "sw_Hz": float(sw_f2) if sw_f2 is not None else None,
        "ref_Hz": float(ref_f2) if ref_f2 is not None else None,
        "transmitter_freq": float(tx_f2) if tx_f2 is not None else None,
        "fname": os.path.basename(path),
    }

    return x_f2_ppm, x_f2_Hz, y_traces, meta

def load_time(path: str, N):        
    t_f1 = None
    try:
        t_f1 = np.loadtxt(path, skiprows=1, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Failed to read echo times from {path}: {e}")
    if t_f1.ndim != 1:
        raise ValueError(f"Relaxation time must be one column; got shape {t_f1.shape}")
    if t_f1.size != N:
        raise ValueError(f"Echo time count {t_f1.size} != number of traces {N}")
    # Optional: ensure units are consistent (t_f1_unit is informational here)
    t_f1 = np.asarray(t_f1, dtype=np.float64)       
    return t_f1

def get_trace(x_f2_ppm: np.ndarray, y_traces: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds-checked accessor for one spectrum."""
    if not (0 <= k < y_traces.shape[0]):
        raise IndexError(f"k={k} out of range [0, {y_traces.shape[0]-1}]")
    return x_f2_ppm, y_traces[k, :]


def quick_plot(
    x_f2_ppm: np.ndarray,
    y_traces: np.ndarray,
    k: int | None = None,
    max_traces: int = 6,
    title: str = "Quick NMR Data Check",
) -> None:
    """
    Quick overlay plot.
    - If k is given: plot that one trace.
    - Else: overlay up to `max_traces` traces spread across the set.
    """
    if k is not None:
        x, y = get_trace(x_f2_ppm, y_traces, k)
        plt.plot(x, y, lw=1, label=f"trace {k}")
    else:
        n = y_traces.shape[0]
        if n <= max_traces:
            ks = range(n)
        else:
            ks = np.linspace(0, n - 1, max_traces, dtype=int)
        for kk in ks:
            plt.plot(x_f2_ppm, y_traces[kk], lw=1, label=f"trace {kk}")

    plt.xlabel("Chemical shift (ppm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title)
    plt.legend(loc="best")
    plt.gca().invert_xaxis()  # common for NMR; remove if you prefer ascending ppm
    plt.tight_layout()
    plt.show()


# --- Test (keep commented) ---
#x_f2_ppm, x_f2_Hz, y_traces, meta = load_ssnake(r"D:\OneDrive\work\nmr-fit-sync\13.json")            # 2D
# x_f2_ppm, t_f1, y_traces, meta = load_ssnake(r"C:\path\to\13.json", echo_path=r"C:....")
#quick_plot(x_f2_ppm, y_traces, k=None)
#ref = meta.get("ref_Hz", None)
#print(ref)
# x_f2_ppm, t_f1, y_traces, meta = load_ssnake(r"C:\path\to\13-s15.json")        # 1D
# quick_plot(x_f2_ppm, y_traces, k=0)
