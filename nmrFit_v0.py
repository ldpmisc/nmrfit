"""
NMR-PeakDeconv-GUI (lmfit, PyQt5)

Purpose
-------
Interactive multi-peak fitting for a single 1D NMR spectrum using the
ssNake-style forward model: build a time-domain FID per peak, then FFT
(fft + fftshift) and compare to the measured frequency-domain data.

Stack
-----
- Python 3.10+
- PyQt5
- matplotlib
- lmfit
- numpy

Usage
-----
python mpFit.py

Click "Open" → select your spectrum. Currently only support .json input data from ssnake
Future versions will include Bruker and .txt file.

Choose axis units (ppm/Hz), set reference frequency if needed, then:
- Click on Add peak modes to enable add peak function
- Use the Peak Table to tweak initial guesses (Pos, Amp, Lorentz, Gauss).
- Press "Fit" to select a desired mode of fitting.
- Click Show Stat to see Statistic detail.
- Click Excluded to open excluded functions to add an excluded region for data. The excluded region will be masked and will not influence the residual
- Right click on a parameter cell in the Peak Table to set Link or Bounds. Link is to set mathematical relations between parameters. Bounds are to set limits on a parameter.
- Click File -> Export or Import to export or import peak table in .txt format.


Notes
-----
1) The auto-pick function is not working currently

2) The option to switch a display from ppm to Hz is not working currently. The software only works properly in ppm.

3) The fitting is taking input in ppm except for Lorentz width. The input is internally converted to Hz and optimization take place in Hz.


Design
------
The design centers around the SliceFitState and ParamRef objects.

When user make a peak, either by click on plot or add one in a table, or import from a table, a Peak object is created.
Peak object --> Peaks[List] aka a list of Peaks
Peak[List] --> SliceFitState object (Peak[List] is stored in SliceFitState object, one per slice (1D data has one slice)).
Peaks[List] <--> Peak --> ParamRef object <--> key --> lmfit.Parameter object -> lmfit.Parameters object <--> lmfit.minimize()

When creating a link between fitting parameters (e.g s0_p0_pos = s1_p0_pos), the link is a linkExpr object. The object and alike are stored in linkStore. 
Fetching linkStore gives linkExpr and ParamRef. Use ParamRef to call Parameter object from Parameters. Use linkExpr to find a link expression of a Parameter object.
Use Parameter.set() to set an expression and forward that to lmfit.minimize()
In short, User set a link --> LinkExpr <--> LinkStore --> ParamRef --> lmfit.Parameter --> lmfit.Parameters --> lmfit.minimize()

When setting a bound of a fitting parameter (e.g 90 < s0_p0_pos < 100 ppm). ....


Copyright
---------
This script is for research/educational use.
"""

from __future__ import annotations
import sys
import math
import io

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Iterable, Set
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from types import MethodType
from datetime import datetime, timezone


from open_file_v7 import load_ssnake, load_time
from peak_table_io_v3 import export_peak_table, import_peak_table

import numpy as np
from numpy.fft import fft, fftfreq, fftshift

from lmfit import Parameters, Parameter, minimize, fit_report, report_fit

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QHeaderView, QInputDialog, QAction, QMenu, QMenuBar, QToolButton
)
from PyQt5.QtGui import QDoubleValidator

from stats_extract import extract_FitResult_corr_and_sum
from stats_view import StatsView
from constraint_rules import (
    ConstraintRule,
    LinearConstraint,
    RelaxDecayConstraint,
    ConstraintRuleFactory,
    LinkExpr_to_ConstraintRule,
    ConstraintRule_to_LinkExpr,
    ConstrainedPeak,
    ConstraintStore,
)

import os
os.environ["QT_API"] = "pyqt5"
os.environ["MPLBACKEND"] = "Qt5Agg"
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector

# ---- logging setup (console + file) ----
import logging, sys
DEBUG_LOGGING = False
log = logging.getLogger("fit")
if not log.handlers:  # avoid double-adding if reloaded
    log.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler("fit_debug.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)

    log.addHandler(ch); log.addHandler(fh)

def _arr_summary(name, a, head=5):
    import numpy as _np
    a = _np.asarray(a)
    try:
        return (f"{name}: shape={a.shape}, "
                f"min={_np.min(a):.6g}, max={_np.max(a):.6g}, "
                f"head={_np.array2string(a[:head], precision=6, separator=', ')})")
    except Exception:
        return f"{name}: <unprintable>"


# ----------------------------- Data model ---------------------------------
@dataclass
class Peak:
    pos: float     # in display units (ppm or Hz)
    amp: float
    lor_hz: float  # Hz
    gauss_disp: float  # in display units (ppm or Hz)

@dataclass
class SliceFitState:
    
    peaks: List[Peak] = field(default_factory=list)                         # deep copy of fitted/display peaks
    constrained_peaks: List[Any] = field(default_factory=list)              # Phase 2: List[ConstrainedPeak] for constraint management
    constraint_store: Optional[Any] = None                                  # Phase 3: Optional[ConstraintStore] for per-slice constraint registry
    fix_flags: List[Tuple[bool,bool,bool,bool]] = field(default_factory=list)  # per-peak (pos, area, lor, gauss), True = fixed
    x_disp: np.ndarray = None                       # display x used for plotting
    y_data: np.ndarray = None                       # data trace used for plotting
    y_model: Optional[np.ndarray] = None      # cached model AFTER φ0 + offset
    y_diff: Optional[np.ndarray] = None       # cached (model - data), same grid
    y_peaks: Optional[List[np.ndarray]] = None
    has_fit: bool = False                     # was it from Fit/Sim?
    dirty_model: bool = False                 # set True if globals/peaks changed

    # minimal “globals snapshot” to check reusability
    axis_mode: str = "ppm"
    excluded: List[Tuple[float, float]] = field(default_factory=list)
    ref_MHz: Optional[float] = None
    sw_hz: Optional[float] = None
    N: int = 0
    multiplier: float = 1.0
    offset: float = 0.0
    phi0_deg: float = 0.0

    # optional: last plot view (restore zoom if needed)
    xlim: Optional[Tuple[float,float]] = None
    ylim: Optional[Tuple[float,float]] = None
    redchi: Optional[float] = None
    state_stats = None

class LinkType(Enum):
    LINEAR = auto()       # target = a * driver + b
    RELAX_EXP = auto()    # target(i) = driver * A * exp(-t_i / T) + C

@dataclass(frozen=True)
class ParamRef:
    slice_id: int
    peak_id: int
    name: str             # "pos" | "amp" | "lor" | "gauss"


@dataclass
class LinkExpr: #LinkExpr aka math expression of the link}
    type: LinkType
    target: ParamRef
    driver: Optional[ParamRef]   
    args: Dict[str, float]       # LINEAR: {a,b}; RELAX_EXP: {A,T,t_override,C}
    enabled: bool = True

class LinkStore:
    """
    One link per target ParamRef.
    Also maintains a reverse index for fast 'dependents_of(driver)'.
    """
    def __init__(self) -> None:
        self._by_target: Dict[ParamRef, LinkExpr] = {} #dict {target, LinkExpr aka math expression of the link}
        self._dependents: Dict[ParamRef, Set[ParamRef]] = {}  # dict{driver, set of targets}

    def set_link(self, expr: LinkExpr) -> None:
        # Remove old reverse dependency if replacing
        old = self._by_target.get(expr.target) #expr.target = ParamRef instance. old = LinkExpr instance
        if old and old.driver:
            self._dependents.get(old.driver, set()).discard(expr.target)

        self._by_target[expr.target] = expr
        if expr.driver is not None:
            self._dependents.setdefault(expr.driver, set()).add(expr.target)

    def remove_link(self, target: ParamRef) -> None:
        old = self._by_target.pop(target, None)
        if old and old.driver:
            self._dependents.get(old.driver, set()).discard(target)

    def get(self, target: ParamRef) -> Optional[LinkExpr]:
        return self._by_target.get(target)

    def is_linked(self, target: ParamRef) -> bool:
        s = self._by_target.get(target)
        return bool(s and s.enabled)

    def all_expr(self) -> Iterable[LinkExpr]:
        return self._by_target.values()

    def dependents_of(self, driver: ParamRef) -> List[ParamRef]:
        return list(self._dependents.get(driver, set()))
    
class LinkManagerDialog(QtWidgets.QDialog):
    """
    Viewer/editor for all links in a LinkStore.
    Editable Expr column with grammar:

    LINEAR (implicit target):
        s15_p1_amp
        2*s15_p1_amp
        s15_p1_amp + 1
        2*s15_p1_amp - 0.5
        a=0.95,b=0.02,driver=s15_p1_amp

    LINEAR (full):
        s14_p2_amp = 2*s15_p1_amp + 1

    EXP (implicit target):
        exp(t=0.1, T_value=0.035, A=1.2,,C=0)
        exp(t=0.1, T_name=T_0, A=1.2,C=0)
        s15_p2_amp * A * exp(-0.1/T_name + C)
    
    EXP (full):
        s14_p2_amp = s15_p2_amp * A * exp(-0.1/T_name + C)
    """

    COL_ENABLED = 0
    COL_TARGET  = 1
    COL_TYPE    = 2
    COL_DRIVER  = 3
    COL_EXPR    = 4
    def __init__(self, parent, link_store: LinkStore, *, slice_count: int = 1, peaks_per_slice: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Link Manager")
        self.resize(1000, 700)
        self._link_store = link_store
        self._slice_count = slice_count
        self._peaks_per_slice = peaks_per_slice
        self._row_targets: list[ParamRef] = []

        layout = QtWidgets.QVBoxLayout(self)

        # ---------------- table ----------------

        self.tbl = QtWidgets.QTableWidget(0, 5, self)
        self.tbl.setHorizontalHeaderLabels(["✓", "Target", "Type", "Driver", "Expr"])
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)

        # we still want editing for Expr
        self.tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked |
            QtWidgets.QAbstractItemView.EditKeyPressed |
            QtWidgets.QAbstractItemView.AnyKeyPressed
        )
        layout.addWidget(self.tbl, 1)

        # ---------------- hint panel ----------------
        hint_box = QtWidgets.QGroupBox("Hint for expr")
        hint_layout = QtWidgets.QGridLayout(hint_box)

        def make_label(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(hint_box)
            lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lbl.setText(text)
            return lbl

        linear_lbl = make_label(
            "Linear<br>"
            "s15_p1_amp<br>"
            "2*s15_p1_amp<br>"
            "s15_p1_amp + 1<br>"
            "2*s15_p1_amp - 0.5<br>"
            "a=0.95, b=0.02, driver=s15_p1_amp<br>"
            "s14_p2_amp = 2*s15_p1_amp + 1"
        )

        exp_lbl = make_label(
            "Exponential<br>"
            "exp(A=1.2, T=0.035, C=0)<br>"
            "exp(A=1.2, T_name=Tglobal, C=0)<br>"
            "s15_p1_amp*1*exp(-0.1/T_name + 2)<br>"
            "s14_p2_amp = s15_p1_amp*1*exp(-0.1/T_name + 2)<br>"
            "s14_p2_amp = exp(A=1.2, T_name=Tglobal, C=0)"
        )

        hint_layout.addWidget(linear_lbl, 0, 0)
        hint_layout.addWidget(exp_lbl,    0, 1)
        hint_layout.setColumnStretch(0, 1)
        hint_layout.setColumnStretch(1, 1)
        layout.addWidget(hint_box)

        # ---------------- buttons ----------------
        btns = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_gen  = QtWidgets.QPushButton("Generate…")
        self.btn_copy = QtWidgets.QPushButton("Copy")
        self.btn_edit = QtWidgets.QPushButton("Edit…")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        self.btn_export = QtWidgets.QPushButton("Export…")
        self.btn_import = QtWidgets.QPushButton("Import…")


        btns.addStretch(1)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_gen)
        btns.addWidget(self.btn_copy)
        btns.addWidget(self.btn_edit)
        btns.addWidget(self.btn_clear)
        btns.addWidget(self.btn_clear_all)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_import)
        layout.addLayout(btns)

        # close
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        layout.addWidget(self.btn_close, 0, QtCore.Qt.AlignRight)

        # wire
        self.btn_add.clicked.connect(self._on_add_row)
        self.btn_gen.clicked.connect(self._on_generate_targets)
        self.btn_copy.clicked.connect(self._on_copy_row)
        self.btn_edit.clicked.connect(self._on_edit)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_clear_all.clicked.connect(self._on_clear_all)
        self.btn_export.clicked.connect(self._on_export_links)
        self.btn_import.clicked.connect(self._on_import_links)
        self.tbl.itemChanged.connect(self._on_item_changed)

        self._reload()



    # --------------- UI helpers ---------------
    def _fmt_pref(self, pref: ParamRef) -> str:
        return f"s{pref.slice_id}_p{pref.peak_id}_{pref.name.lower()}"
    
    def _fmt_exp_args(self, args: dict, driver: ParamRef) -> str:
        drv_txt = self._fmt_pref(driver) if driver else "(None)"
        A = args.get("A", 1.0)
        k = args.get("t_override", 1.0)
        C = args.get("C", 0.0)
        denom = args.get("T_name") or args.get("T", "?")
        return f"{drv_txt}*{A:g}*exp(-{k:g}/{denom}+{C:g})"
    
    def _fmt_linear_args(self, args: dict, driver: ParamRef) -> str:
        drv_txt = self._fmt_pref(driver) if driver else "(None)"
        a = args.get("a", 1.0)
        b = args.get("b", 0.0)
        if b == 0:
            return f"{a:g}*{drv_txt}"
        return f"{a:g}*{drv_txt} + {b:g}"
    
    def _ensure_row_targets_len(self, upto_row: int) -> None:
    # make sure _row_targets has at least (upto_row + 1) elements
        while len(self._row_targets) <= upto_row:
            self._row_targets.append(None)

    def _driver_pref_from_cell(self, row: int) -> ParamRef | None:
        it = self.tbl.item(row, self.COL_DRIVER)
        if it is None:
            return None
        txt = (it.text() or "").strip()
        if not txt or txt.lower() == "(none)":
            return None
        try:
            return self._parse_pref(txt)
        except Exception:
            return None


    def _reload(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(0)
        self._row_targets.clear()   # keep “row → ParamRef” mapping
        for expr in self._link_store.all_expr():
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            # enabled (col 0)
            en_item = QtWidgets.QTableWidgetItem()
            en_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            en_item.setCheckState(QtCore.Qt.Checked if getattr(expr, "enabled", True) else QtCore.Qt.Unchecked)
            en_item.setText("") 
            self.tbl.setItem(row, 0, en_item)

            # target (col 1)
            tgt = expr.target
            tgt_txt = self._fmt_pref(tgt)
            # CHANGED: make item directly + make editable
            tgt_item = QtWidgets.QTableWidgetItem(tgt_txt)
            tgt_item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tbl.setItem(row, self.COL_TARGET, tgt_item)
            self._row_targets.append(tgt)  # remember which target belongs to this row

            # type (col 2)
            type_txt = "LINEAR" if expr.type == LinkType.LINEAR else "RELAX_EXP"
            type_item = QtWidgets.QTableWidgetItem(type_txt)
            self.tbl.setItem(row, 2, type_item)

            # driver
            if expr.driver is not None:
                drv_txt = self._fmt_pref(expr.driver)
            else:
                drv_txt = "(none)"
            drv_item = QtWidgets.QTableWidgetItem(drv_txt)

            self.tbl.setItem(row, 3, drv_item)

            # Expr (editable)
            if expr.type == LinkType.LINEAR:
                if expr.driver is not None:
                    expr_txt = self._fmt_linear_args(args=expr.args, driver=expr.driver)
                else:
                    a = expr.args.get("a", 1.0)
                    b = expr.args.get("b", 0.0)
                    expr_txt = f"a={a}, b={b}"
            
            elif expr.type == LinkType.RELAX_EXP:
                if expr.driver is not None:
                    expr_txt = self._fmt_exp_args(args=expr.args, driver=expr.driver)
                else:
                    A = expr.args.get("A", 1.0)
                    C = expr.args.get("C", 0.0)
                    Tn = expr.args.get("T_name")
                    T = expr.args.get("T")
                    k = expr.args.get("t_override", 1)
                    # fallback display for driverless exp (rare)
                    if Tn is not None:
                        expr_txt = f"exp(-{k:g}/{Tn}+{C:g})"
                    else:
                        expr_txt = f"exp(-{k:g}/{T:g}+{C:g})"
            expr_item = QtWidgets.QTableWidgetItem(expr_txt)
            # NEW: make expr editable (for drag-fill too)
            expr_item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tbl.setItem(row, self.COL_EXPR, expr_item)

            self.tbl.setRowHeight(row, 22)
        self.tbl.blockSignals(False)
    
    def _refresh_row_from_expr(self, row: int, expr: LinkExpr) -> None:
        """
        Avoids full _reload() inside itemChanged.
        Keep ALL cells editable.
        """
        self.tbl.blockSignals(True)
        # 0) enabled checkbox
        en_item = self.tbl.item(row, self.COL_ENABLED)

        if en_item is None:
            en_item = QtWidgets.QTableWidgetItem()
            en_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            self.tbl.setItem(row, 0, en_item)
        en_item.setCheckState(QtCore.Qt.Checked if getattr(expr, "enabled", True) else QtCore.Qt.Unchecked)
        en_item.setText("") 

        # 1) type
        type_txt = "LINEAR" if expr.type == LinkType.LINEAR else "RELAX_EXP"
        type_item = self.tbl.item(row, self.COL_TYPE)
        if type_item is None:
            type_item = QtWidgets.QTableWidgetItem()
            self.tbl.setItem(row, 2, type_item)
        type_item.setText(type_txt)

        # 2) driver
        drv_item = self.tbl.item(row, self.COL_DRIVER)
        if drv_item is None:
            drv_item = QtWidgets.QTableWidgetItem()
            self.tbl.setItem(row, 3, drv_item)
        if expr.driver is not None:
            drv_item.setText(self._fmt_pref(expr.driver))
        else:
            drv_item.setText("(none)")

        # 3) expr cell
        expr_item = self.tbl.item(row, self.COL_EXPR)
        if expr_item is None:
            expr_item = QtWidgets.QTableWidgetItem()
            self.tbl.setItem(row, 4, expr_item) 
        if expr.type == LinkType.LINEAR:
            if expr.driver is not None:
                expr_txt = self._fmt_linear_args(args=expr.args, driver=expr.driver)
            else:
                a = expr.args.get("a", 1.0)
                b = expr.args.get("b", 0.0)
                expr_txt = f"a={a:g}, b={b:g}"
        else:  # RELAX_EXP
            if expr.driver is not None:
                expr_txt = self._fmt_exp_args(args=expr.args, driver=expr.driver)
            else:
                A = expr.args.get("A", 1.0)
                C = expr.args.get("C", 0.0)
                Tn = expr.args.get("T_name")
                T = expr.args.get("T")
                k = expr.args.get("t_override", 1)
                if Tn is not None:
                    expr_txt = f"exp(-{k:g}/{Tn}+{C:g})"
                else:
                    expr_txt = f"exp(-{k:g}/{T:g}+{C:g})"

        expr_item.setText(expr_txt)

        self.tbl.blockSignals(False)
    
    def _mark_error(self, item: QtWidgets.QTableWidgetItem, msg: str):
        # IMPORTANT: don’t let these UI changes trigger itemChanged again
        self.tbl.blockSignals(True)
        item.setBackground(QtGui.QColor(255, 180, 180))
        item.setToolTip(msg)
        self.tbl.blockSignals(False)

    def _clear_error(self, item: QtWidgets.QTableWidgetItem):
        self.tbl.blockSignals(True)
        item.setBackground(QtGui.QColor(255, 255, 255))
        item.setToolTip("")
        self.tbl.blockSignals(False)
    
    def _make_item_with_ref(self, text, pref: ParamRef):
        item = QtWidgets.QTableWidgetItem(text)
        item.setData(QtCore.Qt.UserRole, pref)
        return item

    def _selected_paramref(self) -> Optional[ParamRef]:
        row = self.tbl.currentRow()
        if row < 0:
            # try selection as fallback
            idxs = self.tbl.selectedIndexes()
            if not idxs:
                return None
            row = idxs[0].row()

        self._ensure_row_targets_len(row)
        pref = self._row_targets[row]
        return pref


    # ---------------------------Slots---------------------------------------
    def _on_add_row(self):
        n, ok = QtWidgets.QInputDialog.getInt(
        self, "Add Rows", "How many rows to add?", value=1, min=1, max=50000
        )
        if not ok:
            return
        self.tbl.blockSignals(True)
        for _ in range(n):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            en_item = QtWidgets.QTableWidgetItem()
            en_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            en_item.setCheckState(QtCore.Qt.Checked)
            self.tbl.setItem(r, 0, en_item)

            tgt = QtWidgets.QTableWidgetItem("")
            self.tbl.setItem(r, 1, tgt)

            typ = QtWidgets.QTableWidgetItem("LINEAR")
            self.tbl.setItem(r, 2, typ)

            drv = QtWidgets.QTableWidgetItem("")
            self.tbl.setItem(r, 3, drv)

            ex = QtWidgets.QTableWidgetItem("")
            self.tbl.setItem(r, 4, ex)

            self.tbl.setCurrentCell(r, 0)
        self.tbl.blockSignals(False)

        # --------------------- generate targets ---------------------
    def _on_generate_targets(self):
        """
        Generate a block of targets from the currently selected target.
        Optionally: immediately apply an exponential template to JUST the generated rows.
        """
        cur_row = self.tbl.currentRow()
        if cur_row < 0:
            QtWidgets.QMessageBox.warning(self, "Generate targets", "Select a row with a valid target first.")
            return
    
        src_item = self.tbl.item(cur_row, self.COL_TARGET)
        src_txt = (src_item.text() if src_item else "").strip()
        if not src_txt:
            QtWidgets.QMessageBox.warning(self, "Generate targets", "The selected row has no target.")
            return
    
        try:
            src_pref = self._parse_pref(src_txt)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Generate targets", f"Invalid source target: {e}")
            return
    
        dlg = _GenerateTargetsDialog(
            parent=self,
            source_text=self._fmt_pref(src_pref),
            max_slices=int(self._slice_count),
            max_peaks=int(self._peaks_per_slice),
            start_row=cur_row + 1,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
    
        opts = dlg.result_options()
        if not opts:
            return
    
        mode        = opts["mode"]              # "peak" | "slice"
        start_row   = opts["start_row"]         # int
        overwrite   = opts["overwrite"]         # bool
        make_exp    = opts.get("make_exp", False)
    
        # --------- build list of ParamRef we will create ----------
        targets: list[ParamRef] = []
        if mode == "peak":
            for pid in range(opts["peak_from"], opts["peak_to"] + 1):
                tgt = ParamRef(slice_id=src_pref.slice_id, peak_id=pid, name=src_pref.name)
                targets.append(tgt)
        else:  # "slice"
            for sid in range(opts["slice_from"], opts["slice_to"] + 1):
                tgt = ParamRef(slice_id=sid, peak_id=src_pref.peak_id, name=src_pref.name)
                targets.append(tgt)
    
        if not targets:
            return
    
        self.tbl.blockSignals(True)
        row = start_row
        newly_created_rows: list[int] = []
    
        for pref in targets:
            if row >= self.tbl.rowCount():
                self.tbl.insertRow(row)
    
            tgt_item = self.tbl.item(row, self.COL_TARGET)
            tgt_txt  = (tgt_item.text() if tgt_item else "").strip() if tgt_item else ""
    
            if tgt_txt and not overwrite:
                QtWidgets.QMessageBox.information(
                    self,
                    "Generate targets",
                    f"Stopped at row {row+1} because it already has a target."
                )
                break
            
            # enabled
            en_item = self.tbl.item(row, self.COL_ENABLED)
            if en_item is None:
                en_item = QtWidgets.QTableWidgetItem()
                en_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                self.tbl.setItem(row, self.COL_ENABLED, en_item)
            en_item.setCheckState(QtCore.Qt.Checked)
            en_item.setText("")
    
            # target
            if tgt_item is None:
                tgt_item = QtWidgets.QTableWidgetItem()
                tgt_item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.tbl.setItem(row, self.COL_TARGET, tgt_item)
            tgt_item.setText(self._fmt_pref(pref))
    
            # remember
            self._ensure_row_targets_len(row)
            self._row_targets[row] = pref
    
            # type = LINEAR (we may overwrite to RELAX_EXP later)
            type_item = self.tbl.item(row, self.COL_TYPE)
            if type_item is None:
                type_item = QtWidgets.QTableWidgetItem()
                self.tbl.setItem(row, self.COL_TYPE, type_item)
            type_item.setText("LINEAR")
    
            # driver = ""
            drv_item = self.tbl.item(row, self.COL_DRIVER)
            if drv_item is None:
                drv_item = QtWidgets.QTableWidgetItem()
                self.tbl.setItem(row, self.COL_DRIVER, drv_item)
            drv_item.setText("")
    
            # expr = ""
            expr_item = self.tbl.item(row, self.COL_EXPR)
            if expr_item is None:
                expr_item = QtWidgets.QTableWidgetItem()
                expr_item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.tbl.setItem(row, self.COL_EXPR, expr_item)
            expr_item.setText("")
    
            self.tbl.setRowHeight(row, 22)
            newly_created_rows.append(row)
    
            row += 1
    
        self.tbl.blockSignals(False)

        # -------- optional exponential pass --------
        if make_exp and newly_created_rows:
            self._apply_exp_template_to_rows(newly_created_rows)

    def _apply_exp_template_to_rows(self, rows: list[int]) -> None:
        """
        Turn the given rows (which already have targets) into RELAX_EXP links
        using A/C/T from a small dialog and t from:
            - parent.t_f1    (if present)
            - parent.on_open_time() → parent.t_f1
            - or constant t
        """
        dlg = _ExpTemplateDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        tpl = dlg.result_template()
        if not tpl:
            return

        # get slice times if needed
        times = None
        if tpl["t_mode"] in ("parent", "load"):
            parent = self.parent()
            if parent is not None:
                # load if user asked "load"
                if tpl["t_mode"] == "load":
                    loader = getattr(parent, "on_open_time", None)
                    if callable(loader):
                        try:
                            loader()
                        except Exception as e:
                            QtWidgets.QMessageBox.warning(self, "Exp template", f"Failed to load times: {e}")
                    else:
                        QtWidgets.QMessageBox.warning(self, "Exp template", "Parent has no on_open_time().")
                # now read
                times = getattr(parent, "t_f1", None)
            if times is None:
                QtWidgets.QMessageBox.warning(self, "Exp template", "No slice times available from parent.")
                # we can bail or continue with const t; here we bail
                return

        driver_txt = (tpl["driver"] or "").strip()
        A = float(tpl["A"])
        C = float(tpl["C"])
        T_is_name = tpl["T_is_name"]
        T_val = tpl["T_val"]
        const_t = float(tpl["const_t"])

        self.tbl.blockSignals(True)
        for r in rows:
            self._ensure_row_targets_len(r)
            pref = self._row_targets[r]
            if pref is None:
                continue

            # determine t for this row
            if tpl["t_mode"] in ("parent", "load"):
                sid = int(pref.slice_id)
                if sid < 0 or sid >= len(times):
                    # skip or set dummy
                    continue
                t_val = float(times[sid])
            else:
                t_val = const_t

            # build expression text
            if driver_txt:
                # driver-based: driver*A*exp(-t/T + C)
                if T_is_name:
                    expr_txt = f"{driver_txt}*{A:g}*exp(-{t_val:g}/{T_val}+{C:g})"
                else:
                    expr_txt = f"{driver_txt}*{A:g}*exp(-{t_val:g}/{float(T_val):g}+{C:g})"
            else:
                # driverless KV form
                if T_is_name:
                    expr_txt = f"exp(A={A:g}, T_name={T_val}, C={C:g}, t={t_val:g})"
                else:
                    expr_txt = f"exp(A={A:g}, T={float(T_val):g}, C={C:g}, t={t_val:g})"

            # write into table
            # type
            type_item = self.tbl.item(r, self.COL_TYPE)
            if type_item is None:
                type_item = QtWidgets.QTableWidgetItem()
                self.tbl.setItem(r, self.COL_TYPE, type_item)
            type_item.setText("RELAX_EXP")

            # driver cell (if user gave driver)
            if driver_txt:
                drv_item = self.tbl.item(r, self.COL_DRIVER)
                if drv_item is None:
                    drv_item = QtWidgets.QTableWidgetItem()
                    self.tbl.setItem(r, self.COL_DRIVER, drv_item)
                drv_item.setText(driver_txt)

            # expr cell
            expr_item = self.tbl.item(r, self.COL_EXPR)
            if expr_item is None:
                expr_item = QtWidgets.QTableWidgetItem()
                expr_item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.tbl.setItem(r, self.COL_EXPR, expr_item)
            expr_item.setText(expr_txt)

            # now let existing pipeline parse and push to LinkStore
            self._apply_expr_from_cell(r, expr_txt)

        self.tbl.blockSignals(False)

    def _on_copy_row(self):
        cur = self.tbl.currentRow()
        if cur < 0:
            cur = self.tbl.rowCount() - 1
            if cur < 0:
                return

        src = cur
        dst = self.tbl.rowCount()

        self.tbl.blockSignals(True)
        self.tbl.insertRow(dst)

        for col in range(self.tbl.columnCount()):
            it_src = self.tbl.item(src, col)
            if it_src is not None:
                # clone everything: text, flags, checkstate
                it_new = QtWidgets.QTableWidgetItem(it_src)
            else:
                it_new = QtWidgets.QTableWidgetItem()
            self.tbl.setItem(dst, col, it_new)

        self.tbl.blockSignals(False)

        # keep ParamRef mapping in sync
        self._ensure_row_targets_len(src)
        self._ensure_row_targets_len(dst)
        self._row_targets[dst] = self._row_targets[src]

        self.tbl.setCurrentCell(dst, 0)




    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None:
            return
        row = item.row()
        col = item.column()
        self._ensure_row_targets_len(row)

        # figure out which change happened and update the source row first
        if col == self.COL_ENABLED:
            self._apply_enabled_from_cell(row)
            value_for_others = self.tbl.item(row, col).checkState()
            self._propagate_to_selected_rows(src_row=row, col=col, value=value_for_others)
            return

        if col == self.COL_TARGET:
            text = item.text().strip()
            # NEW: user just created an empty row → do nothing, but keep mapping
            if not text:
                if row < len(self._row_targets):
                    self._row_targets[row] = None
                else:
                    self._row_targets.append(None)
                return
            self._apply_target_from_cell(row, text)
            self._propagate_to_selected_rows(src_row=row, col=col, value=text)
            return


        if col == self.COL_TYPE:
            text = item.text().strip().upper()
            if text not in ("LINEAR", "RELAX_EXP"):
                text = "LINEAR"
            self._apply_type_from_cell(row, text)
            self._propagate_to_selected_rows(src_row=row, col=col, value=text)
            return

        if col == self.COL_DRIVER:
            text = item.text().strip()
            self._apply_driver_from_cell(row, text)
            self._propagate_to_selected_rows(src_row=row, col=col, value=text)
            return

        if col == self.COL_EXPR:
            text = item.text().strip()
            self._apply_expr_from_cell(row, text)
            self._propagate_to_selected_rows(src_row=row, col=col, value=text)
            return

    # -------------- per-column appliers (source row) --------------

    def _get_expr_for_row(self, row: int) -> LinkExpr | None:
        try:
            pref = self._row_targets[row]
        except Exception:
            return None
        # assuming LinkStore has something like get_expr(pref)
        try:
            return self._link_store.get_expr(pref)
        except Exception:
            # or: self._link_store._by_target.get(pref)
            return self._link_store._by_target.get(pref, None)

    def _apply_enabled_from_cell(self, row: int) -> None:
        expr = self._get_expr_for_row(row)
        if expr is None:
            return
        item = self.tbl.item(row, self.COL_ENABLED)
        expr.enabled = (item.checkState() == QtCore.Qt.Checked)

    def _apply_target_from_cell(self, row: int, text: str) -> None:
        self._ensure_row_targets_len(row)
        
        text = (text or "").strip()
        if not text:
            # user cleared target → keep None
            self._row_targets[row] = None
            return


        try:
            new_pref = self._parse_pref(text)
        except Exception as e:
            # optional: mark error, but do NOT crash
            it = self.tbl.item(row, self.COL_TARGET)
            if it is not None:
                self._mark_error(it, str(e))
            return

        # clear error if previously marked
        it = self.tbl.item(row, self.COL_TARGET)
        if it is not None:
            self._clear_error(it)

        # update link_store key; simplest is: remove old, add new
        old_pref = self._row_targets[row]

        # if there was no old link (new/blank row), we may not have anything in store yet
        if old_pref is not None:
            expr = self._link_store._by_target.pop(old_pref, None)
        else:
            expr = None

        if expr is None:
            self._row_targets[row] = new_pref
            return

        # existing expr: move it under new key
        expr.target = new_pref
        self._link_store._by_target[new_pref] = expr
        self._row_targets[row] = new_pref


    def _apply_type_from_cell(self, row: int, type_txt: str) -> None:
        expr = self._get_expr_for_row(row)
        if expr is None:
            return
        expr.type = LinkType.LINEAR if type_txt == "LINEAR" else LinkType.RELAX_EXP
        # re-render row to show correct expr text/driver formatting
        self._refresh_row_from_expr(row, expr)

    def _apply_driver_from_cell(self, row: int, text: str) -> None:
        expr = self._get_expr_for_row(row)
        if expr is None:
            return
        if text and text != "(none)":
            drv = self._parse_pref(text)
        else:
            drv = None
        expr.driver = drv
        self._refresh_row_from_expr(row, expr)

    def _apply_expr_from_cell(self, row: int, text: str) -> None:
        self._ensure_row_targets_len(row)

        tgt = self._row_targets[row]
        if tgt is None:
            return

        # read type + driver from the row
        type_item = self.tbl.item(row, self.COL_TYPE)
        type_txt = type_item.text().strip() if type_item is not None else "LINEAR"

        drv_pref = self._driver_pref_from_cell(row)

        try:
            new_expr = self._parse_row_to_expr(
                target=tgt,
                type_txt=type_txt,
                driver=drv_pref,
                expr_txt=text.strip(),
            )
        except Exception as e:
            # mark error but keep UI alive
            it = self.tbl.item(row, self.COL_EXPR)
            if it is not None:
                self._mark_error(it, str(e))
            return

        # clear error if any
        it = self.tbl.item(row, self.COL_EXPR)
        if it is not None:
            self._clear_error(it)

        # write back to store
        self._link_store.set_link(new_expr)

        # update row→target map (in case user typed a full lhs=... form later)
        self._row_targets[row] = new_expr.target

        # finally refresh row (this will reformat nicely)
        self._refresh_row_from_expr(row, new_expr)


    # -------------- propagation to other selected rows --------------

    def _propagate_to_selected_rows(self, src_row: int, col: int, value) -> None:
        sel = self.tbl.selectionModel()
        if sel is None:
            return

        target_rows: list[int] = []
        for idx in sel.selectedIndexes():
            if idx.column() == col:
                target_rows.append(idx.row())

        # fallback: maybe user selected whole rows
        if not target_rows:
            for idx in sel.selectedRows():
                target_rows.append(idx.row())

        if not target_rows:
            return

        for r in target_rows:
            if r == src_row:
                continue
            self._ensure_row_targets_len(r)

            # per-column apply
            if col == self.COL_ENABLED:
                self.tbl.blockSignals(True)
                it = self.tbl.item(r, col)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                    self.tbl.setItem(r, col, it)
                it.setCheckState(value)
                self.tbl.blockSignals(False)
                self._apply_enabled_from_cell(r)

            elif col in (self.COL_TARGET, self.COL_TYPE, self.COL_DRIVER, self.COL_EXPR):
                self.tbl.blockSignals(True)
                it = self.tbl.item(r, col)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    self.tbl.setItem(r, col, it)
                it.setText(str(value))
                self.tbl.blockSignals(False)

                # update backend too
                if col == self.COL_TARGET:
                    self._apply_target_from_cell(r, str(value))
                elif col == self.COL_TYPE:
                    self._apply_type_from_cell(r, str(value))
                elif col == self.COL_DRIVER:
                    self._apply_driver_from_cell(r, str(value))
                elif col == self.COL_EXPR:
                    self._apply_expr_from_cell(r, str(value))

    def _norm_type(self, txt: str) -> str:
        t = (txt or "").strip().lower()
        if t == "linear":
            return "LINEAR"
        if t in ("expo", "exp", "relax_exp", "relax-exp", "relaxexp", "relax"):
            return "RELAX_EXP"
        return txt.upper()


    # --------------- parsing ---------------

    def _parse_row_to_expr(self, *, target, type_txt: str, driver, expr_txt: str):
        """
        Single place to convert a 4-cell row into LinkExpr.
        We try in this order:
        1) if expr_txt starts with '=' → inline
        2) else if type is RELAX_EXP → parse KV style
        3) else → linear implicit
        """
        # CASE 1: user typed full inline in Expr cell
        if expr_txt.startswith("="):
            return self._parse_expr(target, expr_txt.lstrip("="))

        # CASE 2: exponential / relax form
        if type_txt.upper() == "RELAX_EXP":
            # you already have something like _parse_inline_or_kv(...)
            return self._parse_inline_or_kv(target, expr_txt)

        # CASE 3: plain linear: target = a*driver + b
        # if expr empty but driver present → assume a=1, b=0
        if driver is not None and not expr_txt:
            return LinkExpr(
                target=target,
                driver=driver,
                type=LinkType.LINEAR,
                enabled=True,
                args={"a": 1.0, "b": 0.0},
            )

        # last resort: let old expr parser handle it
        return self._parse_expr(target, expr_txt)
    def _parse_expr(self, target: ParamRef, text: str) -> LinkExpr:
        t = text.strip()

        # 1) full assignment: lhs = rhs
        #    now also allow: target = driver * ... * exp(...)
        if "=" in t and not t.startswith("=") and not t.lower().startswith("exp("): #detech full assignment
            lhs, rhs = t.split("=", 1)
            lhs_pref = self._parse_pref(lhs.strip())
            if not self._pref_equal(lhs_pref, target):
                raise ValueError(
                    f"LHS '{lhs.strip()}' does not match row target '{self._fmt_pref(target)}'"
                )
            rhs = rhs.strip()
            # NEW: RHS is driver*...*exp(...) → parse as exponential-with-driver or a*driver+b
            if self._detect_driver_exp(rhs):
                return self._parse_driver_exp(target, rhs)
            # otherwise parse as usual inline (=linear or kv)
            return self._parse_inline_or_kv(target, "=" + rhs)

        # 2) inline / kv / exp
        return self._parse_inline_or_kv(target, t)

    def _parse_inline_or_kv(self, target: ParamRef, t: str) -> LinkExpr:
        # CASE A: function-style exponential: exp(A=...,T=...,C=..., t=)
        if t.lower().startswith("exp(") and t.endswith(")"):
            args_raw = t[t.find("(")+1:-1]
            args = self._parse_kv(args_raw)
            norm = {}
            for k, v in args.items():                
                if k == "A":
                    norm["A"] = float(v)
                elif k == "T":
                    norm["T"] = float(v)
                elif k == "T_name":
                    norm["T_name"] = str(v)
                elif k == "C":
                    norm["C"] = float(v)
                elif k == "t":
                    norm["t_override"] = float(v)
                else:
                    norm[k] = v
            if "T_name" in norm and "T" in norm:
                norm.pop("T", None)
            return LinkExpr(
                target=target,
                driver=None,
                type=LinkType.RELAX_EXP,
                enabled=True,
                args=norm,
            )

        # NEW CASE: inline driver-based exponential (MATLAB)
        # e.g. s15_p1_amp*1*exp(-0.1/T_name+2)
        if self._detect_driver_exp(t):
            return self._parse_driver_exp(target, t)

        # CASE B: inline linear algebra
        if t.find("exp") == -1:
            return self._parse_inline_linear(target, t.strip())

        # CASE C: KV linear: a=...,b=...,driver=...
        if "driver=" in t or "a=" in t or "b=" in t:
            kv = self._parse_kv(t)
            if "driver" not in kv:
                raise ValueError("Linear KV form requires driver=...")
            driver_pref = self._parse_pref(str(kv.pop("driver")))
            a = float(kv.pop("a", 1.0))
            b = float(kv.pop("b", 0.0))
            return LinkExpr(
                target=target,
                driver=driver_pref,
                type=LinkType.LINEAR,
                enabled=True,
                args={"a": a, "b": b},
            )

        raise ValueError("Unrecognized expression")

    def _detect_driver_exp(self, txt: str) -> bool:
        # very simple: must contain 'exp(' and some '*' before it
        s = txt.strip()
        exp_idx = s.lower().find("exp")
        if exp_idx == -1:
            return False
        star_before = s.rfind("*", 0, exp_idx)
        return star_before != -1

    def _parse_driver_exp(self, target: ParamRef, expr: str) -> LinkExpr:
        """
        Parse:  s15_p1_amp * 1 * exp(-0.1/T_name + 2) or =s15_p1_amp * 1 * exp(-0.1/T_name + 2)
        or:     s15_p1_amp*exp(-0.1/T+2) or =s15_p1_amp*exp(-0.1/T+2)
        Left part → driver + optional scalar
        Right part (inside exp) → A,T/T_name,C
        Final: target = driver * A * exp(...)
        where A from left-mult is folded into args["A"]
        """
        txt = expr.strip()
        exp_pos = txt.lower().find("exp(")
        left = txt[:exp_pos].strip()
        right = txt[exp_pos:].strip()     # starts with exp(
        # 1) parse driver and leading factor(s)
        #    allow things like: driver*1*  or  2*driver*  or  driver
        #    split on '*', find first ParamRef, and multiply all numeric factors
        factors = [p.strip() for p in left.split("*") if p.strip()]
        driver_pref = None
        A_left = 1.0
        for f in factors:
            if self._is_float(f):
                A_left *= float(f)
            else:
                # assume first non-float is the driver paramref
                if driver_pref is None:
                    driver_pref = self._parse_pref(f)
                else:
                    # something like driver*otherParam*... → too weird
                    raise ValueError(f"Too many non-numeric factors in '{left}'")
        if driver_pref is None:
            raise ValueError(f"Cannot find driver in '{expr}'")

        # 2) parse inside exp(...)
        if not right.lower().startswith("exp(") or not right.endswith(")"):
            raise ValueError(f"Invalid exponential form '{expr}'")
        inner = right[right.find("(")+1:-1].strip()
        # inner is like:  -0.1/T_name + 2  or  -0.05/T + 0
        args = self._parse_exp_inside(inner)

        # 3) fold left A into args["A"]
        args["A"] = float(args.get("A", 1.0)) * A_left

        return LinkExpr(
            target=target,
            driver=driver_pref,
            type=LinkType.RELAX_EXP,
            enabled=True,
            args=args,
        )

    def _parse_exp_inside(self, inner: str) -> dict: #LinkExpr.args
        """
        inner examples:
            -0.1/T_name + 2
            -0.05/T + 0
            -0.1/Tslice
            -0.1/0.035 + 1

            exp(-k / T_name + C) → T_name=..., C=..., A=1, k=
            exp(-k / 0.035 + C)  → T=0.035, C=...

            k is t_time. use k internally in this function to avoid confusion. k can be t_f1 or t_overide
        """
        s = inner.replace(" ", "")
        # split on '+', but keep sign for second part
        C = 0.0
        if "+" in s[1:]:  # ignore first char which may be '-'
            main, c_part = s.split("+", 1)
            C = float(c_part)
        else:
            main = s

        # main should look like: -0.1/T_name   or  -0.1/0.035
        if not main.startswith("-"):
            # allow user mistakes but give warning here
            main_k = main
        else:
            main_k = main[1:]

        if "/" not in main_k:
            raise ValueError(f"Expected form like '-k/T_name' in '{inner}'")
        k_part, t_part = main_k.split("/", 1)
        k_val = float(k_part)

        out = {"A": 1.0, "C": C}
        # t_part can be numeric or name
        if self._is_float(t_part):
            # numeric T
            # original exp() used "T"
            out["T"] = float(t_part)
        else:
            out["T_name"] = t_part
        # keep k so evaluator knows rate = k / T
        out["t_override"] = k_val
        return out

    def _parse_inline_linear(self, target: ParamRef, rhs: str) -> LinkExpr:
        """
        Parse forms like:
            s15_p1_amp
            2*s15_p1_amp or s15_p1_amp*2
            s15_p1_amp + 1
            2*s15_p1_amp - 0.5
        """
        txt = rhs.strip()

        b = 0.0
        driver_part = txt

        plus_idx = txt.find("+")
        minus_idx = txt.find("-", 1)  # skip leading minus

        if plus_idx != -1:
            driver_part = txt[:plus_idx].strip()
            b_str = txt[plus_idx+1:].strip()
            b = float(b_str)
        elif minus_idx != -1:
            driver_part = txt[:minus_idx].strip()
            b_str = txt[minus_idx:].strip()
            b = float(b_str)

        if "*" in driver_part:
            left, right = [p.strip() for p in driver_part.split("*", 1)]
            if self._is_float(left) and not self._is_float(right):
                a = float(left)
                driver_pref = self._parse_pref(right)
            elif self._is_float(right) and not self._is_float(left):
                a = float(right)
                driver_pref = self._parse_pref(left)
            else:
                raise ValueError(f"Cannot interpret '{driver_part}' as a * driver")
        else:
            a = 1.0
            driver_pref = self._parse_pref(driver_part.strip())

        return LinkExpr(
            target=target,
            driver=driver_pref,
            type=LinkType.LINEAR,
            enabled=True,
            args={"a": a, "b": b},
        )

    def _parse_kv(self, s: str) -> dict:
        out = {}
        for chunk in s.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Expected key=value in '{chunk}'")
            k, v = chunk.split("=", 1)
            k = k.strip()
            v = v.strip()
            # don't force to float yet; some can be symbolic (T_name)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        return out

    def _parse_pref(self, txt: str) -> ParamRef:
        t = txt.strip()
        parts = t.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid ParamRef '{txt}' (expected sN_pM_name)")
        try:
            sid = int(parts[0].lstrip("sS"))
            pid = int(parts[1].lstrip("pP"))
        except Exception:
            raise ValueError(f"Invalid ParamRef numbers in '{txt}'")

        raw_name = "_".join(parts[2:]).strip().lower()

        if raw_name in ("pos", "position"):
            name = "pos"
        elif raw_name in ("amp", "area", "amplitude", "integral", "integrals"):
            name = "amp"
        elif raw_name in ("lor", "lorentz", "lor_hz"):
            name = "lor"
        elif raw_name in ("gau", "gauss", "gaussian", "gauss_disp"):
            name = "gauss"
        else:
            raise ValueError(f"Unknown param name '{raw_name}'")

        return ParamRef(slice_id=sid, peak_id=pid, name=name)


    def _pref_equal(self, a: ParamRef, b: ParamRef) -> bool:
        return int(a.slice_id) == int(b.slice_id) and int(a.peak_id) == int(b.peak_id) and str(a.name).lower() == str(b.name).lower()
    
    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False


    # --------------- buttons ---------------
    def _on_edit(self):
        pref = self._selected_paramref()
        if pref is None:
            return
        dlg = LinkEditorDialog(
            self,
            target=pref,
            slice_count=int(self._slice_count_provider()),
            peaks_per_slice=int(self._peaks_per_slice_provider()),
            link_store=self._link_store,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_expr = dlg.result_expr()
            if new_expr is not None:
                self._link_store.set_link(new_expr)
                self._reload()

            # If the expr is RELAX_EXP with T_name, ensure a registry row exists.
            try:
                if new_expr and new_expr.type == LinkType.RELAX_EXP:
                    Tn = new_expr.args.get("T_name", None)
                    if isinstance(Tn, str) and Tn.strip():
                        parent = self.parent()
                        if hasattr(parent, "_ensure_tseed_row"):
                            parent._ensure_tseed_row(str(Tn).strip())
            except Exception:
                pass

        

    def _on_clear(self):
        row = self.tbl.currentRow()
        if row < 0:
            return

        pref = self._selected_paramref()

        # case 1: no real link behind this row → just remove the row
        if pref is None:
            self._remove_table_row(row)
            return

        # how many rows point to this same ParamRef?
        same = sum(1 for p in self._row_targets if p == pref)

        if same > 1:
            # this row is just another view of the same link → drop this row only
            self._remove_table_row(row)
            return

        # case 3: this row is the ONLY one for that link → remove from store, then reload
        self._link_store.remove_link(pref)
        self._reload()



    def _clear_row_visual(self, row: int) -> None:
        self.tbl.blockSignals(True)
        for col in range(self.tbl.columnCount()):
            it = QtWidgets.QTableWidgetItem("")
            if col == 0:
                # plain cell, NO checkbox
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.tbl.setItem(row, col, it)
        self.tbl.blockSignals(False)

    def _remove_table_row(self, row: int) -> None:
        
        if 0 <= row < len(self._row_targets):
            self._row_targets.pop(row)

        # then drop from the UI
        self.tbl.blockSignals(True)
        self.tbl.removeRow(row)
        self.tbl.blockSignals(False)




    def _on_clear_all(self):
        if QtWidgets.QMessageBox.question(
            self,
            "Clear all links",
            "This will remove all links. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        ) == QtWidgets.QMessageBox.Yes:
            for expr in list(self._link_store.all_expr()):
                self._link_store.remove_link(expr.target)
            self._reload()

    def _on_export_links(self):
        """
        Export what is CURRENTLY shown in the Link Manager table.
        Don't go through LinkStore here on purpose: user might have just typed
        in the Expr cell and not pressed Enter/save yet.
        Format: similar headers to peak_table_io_v3, then a tab-separated table.
        """
        parent = self.parent()
        base_dir = getattr(parent, "default_save_dir", str(Path.home()))
        suggested = os.path.join(base_dir, "_links.txt")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Links",
            suggested,
            "Text files (*.txt);;All files (*)"
        )
        if not path:
            return

        # --- collect rows first
        rows = []
        enabled_count = 0
        rc = self.tbl.rowCount()
        for r in range(rc):
            en_item = self.tbl.item(r, self.COL_ENABLED)
            is_enabled = (
                en_item is not None
                and en_item.checkState() == QtCore.Qt.Checked
            )
            if is_enabled:
                enabled_count += 1

            tgt_item = self.tbl.item(r, self.COL_TARGET)
            typ_item = self.tbl.item(r, self.COL_TYPE)
            drv_item = self.tbl.item(r, self.COL_DRIVER)
            exp_item = self.tbl.item(r, self.COL_EXPR)

            rows.append({
                "enabled": "1" if is_enabled else "0",
                "target":  tgt_item.text().strip() if tgt_item else "",
                "type":    (typ_item.text().strip().upper() if typ_item else ""),
                "driver":  drv_item.text().strip() if drv_item else "",
                "expr":    exp_item.text().strip() if exp_item else "",
            })

        # gather some meta from parent if available
        program_version = getattr(parent, "program_version", "mpFit_links_v1")
        data_file = getattr(parent, "data_file", "")
        try:
            slice_count = int(self._slice_count_provider() or 0)
        except Exception:
            slice_count = 0
        try:
            peaks_per_slice = int(self._peaks_per_slice_provider() or 0)
        except Exception:
            peaks_per_slice = 0

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        buf = io.StringIO()
        # --- header ---
        print(f"# Program\t{program_version}", file=buf)
        print(f"# SavedUTC\t{utc_now}", file=buf)
        print(f"# DataFile\t{data_file}", file=buf)
        print(f"# Kind\tLinkTable", file=buf)
        print(f"# SliceCount\t{slice_count}", file=buf)
        print(f"# PeaksPerSlice\t{peaks_per_slice}", file=buf)        
        print(f"# NumberOfEnabledLinks\t{enabled_count}", file=buf)
        print(f"# NumberOfRows\t{rc}", file=buf)
        print("", file=buf)

        # --- table header ---
        print("enabled\ttarget\ttype\tdriver\texpr", file=buf)

        # --- table rows ---
        for row in rows:
            print(
                f"{row['enabled']}\t{row['target']}\t{row['type']}\t{row['driver']}\t{row['expr']}",
                file=buf,
            )

        # --- write to disk ---
        try:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(buf.getvalue())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Links", f"Failed to save:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Export Links", "Link table saved.")


    def _on_import_links(self):
        parent = self.parent()
        base_dir = getattr(parent, "default_save_dir", str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Links",
            os.path.join(base_dir, "_links.txt"),
            "Text files (*.txt);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                lines = f.readlines()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Links", f"Failed to read:\n{e}")
            return

        header_meta = {}
        colmap = {}
        parsed_rows = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                # parse "# Key<TAB>Value"
                try:
                    _, rest = line.split("#", 1)
                    rest = rest.strip()
                    if "\t" in rest:
                        k, v = rest.split("\t", 1)
                        header_meta[k.strip().lower()] = v.strip()
                except Exception:
                    pass
                continue

            parts = line.split("\t")
            # first non-# line is the header row
            if not colmap:
                for i, name in enumerate(parts):
                    colmap[name.strip().lower()] = i
                continue

            def _get(name, default=""):
                i = colmap.get(name, None)
                if i is None or i >= len(parts):
                    return default
                return parts[i].strip()

            parsed_rows.append({
                "enabled": _get("enabled", "1"),
                "target":  _get("target", ""),
                "type":    _get("type", "LINEAR").upper(),
                "driver":  _get("driver", ""),
                "expr":    _get("expr", ""),
            })

        # expected counts from header (may be missing)
        expected_links = None
        expected_rows = None
        try:
            expected_links = int(header_meta.get("numberoflinks", ""))
        except Exception:
            expected_links = None
        try:
            expected_rows = int(header_meta.get("numberofrows", ""))
        except Exception:
            expected_rows = None

        if not parsed_rows:
            QtWidgets.QMessageBox.information(self, "Import Links", "No rows found in file.")
            return

        # clear existing
        try:
            for expr in list(self._link_store.all_expr()):
                try:
                    self._link_store.remove_link(expr.target)
                except Exception:
                    pass
        except Exception:
            pass

        self.tbl.blockSignals(True)
        self.tbl.setRowCount(0)
        self._row_targets = []

        enabled_seen = 0

        for row in parsed_rows:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            # enabled
            en_item = QtWidgets.QTableWidgetItem()
            en_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            is_enabled = row["enabled"] in ("1", "true", "True")
            en_item.setCheckState(QtCore.Qt.Checked if is_enabled else QtCore.Qt.Unchecked)
            if is_enabled:
                enabled_seen += 1
            self.tbl.setItem(r, self.COL_ENABLED, en_item)

            # target
            tgt_item = QtWidgets.QTableWidgetItem(row["target"])
            self.tbl.setItem(r, self.COL_TARGET, tgt_item)

            # type
            typ_item = QtWidgets.QTableWidgetItem(row["type"])
            typ_item.setFlags(typ_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, self.COL_TYPE, typ_item)

            # driver
            drv_item = QtWidgets.QTableWidgetItem(row["driver"])
            drv_item.setFlags(drv_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, self.COL_DRIVER, drv_item)

            # expr
            expr_item = QtWidgets.QTableWidgetItem(row["expr"])
            self.tbl.setItem(r, self.COL_EXPR, expr_item)

            pref = self._parse_pref(row["target"])
            self._row_targets.append(pref)

            # rebuild LinkStore

            if pref is not None:
                type_txt  = self._norm_type(row["type"])            # "LINEAR" / "RELAX_EXP"
                expr_txt  = (row["expr"] or "").strip()
                driver_pref = self._parse_pref(row["driver"]) if row["driver"] else None
                try:
                    link_obj = self._parse_row_to_expr(
                        target=pref,
                        type_txt=type_txt,
                        driver=driver_pref,
                        expr_txt=expr_txt
                    )
                except Exception:
                    link_obj = None

                if link_obj is not None:
                    link_obj.enabled = is_enabled
                    try:
                        self._link_store.set_link(link_obj)
                    except Exception:
                        pass
                    # If the expr is RELAX_EXP with T_name, ensure a registry row exists.
                    try:
                        if link_obj and link_obj.type == LinkType.RELAX_EXP:
                            Tn = link_obj.args.get("T_name", None)
                            if isinstance(Tn, str) and Tn.strip():
                                parent = self.parent()
                                if hasattr(parent, "_ensure_tseed_row"):
                                    parent._ensure_tseed_row(str(Tn).strip())
                    except Exception:
                        pass


        self.tbl.blockSignals(False)
        try:
            self._reload()
        except Exception:
            pass

        # final small consistency hint
        msg = "Link table loaded."
        extra = []
        if expected_rows is not None and expected_rows != len(parsed_rows):
            extra.append(f"rows in file: {expected_rows}, parsed: {len(parsed_rows)}")
        if expected_links is not None and expected_links != enabled_seen:
            extra.append(f"enabled in file: {expected_links}, enabled parsed: {enabled_seen}")
        if extra:
            msg += "\n" + "\n".join(extra)

        QtWidgets.QMessageBox.information(self, "Import Links", msg)




class _GenerateTargetsDialog(QtWidgets.QDialog):
    """
    Tiny helper dialog for LinkManagerDialog._on_generate_targets.
    Lets user pick:
      - generate by: peaks / slices
      - range
      - start row
      - conflict: overwrite / stop
    """
    def __init__(self, parent, source_text: str, max_slices: int, max_peaks: int, start_row: int):
        super().__init__(parent)
        self.setWindowTitle("Generate targets")
        self._max_slices = max_slices
        self._max_peaks = max_peaks
        self._result = None

        lay = QtWidgets.QVBoxLayout(self)

        src_lbl = QtWidgets.QLabel(f"Source target: <b>{source_text}</b>")
        src_lbl.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(src_lbl)

        # mode
        mode_box = QtWidgets.QGroupBox("Generate by")
        mode_lay = QtWidgets.QHBoxLayout(mode_box)
        self.rad_peak = QtWidgets.QRadioButton("Peaks")
        self.rad_slice = QtWidgets.QRadioButton("Slices")
        self.rad_peak.setChecked(True)
        mode_lay.addWidget(self.rad_peak)
        mode_lay.addWidget(self.rad_slice)
        lay.addWidget(mode_box)

        frm = QtWidgets.QFormLayout()

        self.spn_peak_from = QtWidgets.QSpinBox()
        self.spn_peak_from.setRange(0, max_peaks - 1)
        self.spn_peak_from.setValue(1)
        self.spn_peak_to = QtWidgets.QSpinBox()
        self.spn_peak_to.setRange(0, max_peaks - 1)
        self.spn_peak_to.setValue(min(3, max_peaks - 1))

        self.spn_slice_from = QtWidgets.QSpinBox()
        self.spn_slice_from.setRange(0, max_slices - 1)
        self.spn_slice_from.setValue(1)
        self.spn_slice_to = QtWidgets.QSpinBox()
        self.spn_slice_to.setRange(0, max_slices - 1)
        self.spn_slice_to.setValue(min(3, max_slices - 1))

        frm.addRow("Peak from:", self.spn_peak_from)
        frm.addRow("Peak to:",   self.spn_peak_to)
        frm.addRow("Slice from:", self.spn_slice_from)
        frm.addRow("Slice to:",   self.spn_slice_to)

        lay.addLayout(frm)

        # start row
        self.spn_start_row = QtWidgets.QSpinBox()
        self.spn_start_row.setRange(0, 9999)
        self.spn_start_row.setValue(start_row)
        lay.addWidget(QtWidgets.QLabel("Start writing at row (0-based):"))
        lay.addWidget(self.spn_start_row)

        # conflict
        conf_box = QtWidgets.QGroupBox("If row already has a target")
        conf_lay = QtWidgets.QVBoxLayout(conf_box)
        self.rad_stop = QtWidgets.QRadioButton("Stop")
        self.rad_overwrite = QtWidgets.QRadioButton("Overwrite")
        self.rad_stop.setChecked(True)
        conf_lay.addWidget(self.rad_stop)
        conf_lay.addWidget(self.rad_overwrite)
        lay.addWidget(conf_box)
                
        # NEW: exp checkbox
        self.chk_make_exp = QtWidgets.QCheckBox("After generating, apply exponential template…")
        self.chk_make_exp.setChecked(False)
        lay.addWidget(self.chk_make_exp)
        
        # buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        lay.addWidget(btn_box)

        # toggle visibility
        self.rad_peak.toggled.connect(self._update_mode_visibility)
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        peak_mode = self.rad_peak.isChecked()
        self.spn_peak_from.setEnabled(peak_mode)
        self.spn_peak_to.setEnabled(peak_mode)
        self.spn_slice_from.setEnabled(not peak_mode)
        self.spn_slice_to.setEnabled(not peak_mode)

    def result_options(self):
        return self._result

    def accept(self):
        mode = "peak" if self.rad_peak.isChecked() else "slice"
        opts = {
            "mode": mode,
            "start_row": self.spn_start_row.value(),
            "overwrite": self.rad_overwrite.isChecked(),
        }
        if mode == "peak":
            a = self.spn_peak_from.value()
            b = self.spn_peak_to.value()
            if b < a:
                b = a
            opts["peak_from"] = a
            opts["peak_to"] = b
        else:
            a = self.spn_slice_from.value()
            b = self.spn_slice_to.value()
            if b < a:
                b = a
            opts["slice_from"] = a
            opts["slice_to"] = b
        opts["make_exp"] = self.chk_make_exp.isChecked()

        self._result = opts
        super().accept()

class _ExpTemplateDialog(QtWidgets.QDialog):
    """
    Ask for A, C, T(/T_name), optional driver, and where to take t from.
    t-mode:
        - parent  → just use parent.t_f1 as-is
        - load    → call parent.on_open_time()
        - const   → same t for all generated rows
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Exponential template")
        self._result = None

        lay = QtWidgets.QVBoxLayout(self)

        # driver
        self.ed_driver = QtWidgets.QLineEdit()
        self.ed_driver.setPlaceholderText("s0_p0_amp (optional)")
        lay.addWidget(QtWidgets.QLabel("Driver (optional):"))
        lay.addWidget(self.ed_driver)

        # A, C
        form = QtWidgets.QFormLayout()
        self.spn_A = QtWidgets.QDoubleSpinBox()
        self.spn_A.setRange(-1e9, 1e9)
        self.spn_A.setValue(1.0)
        self.spn_C = QtWidgets.QDoubleSpinBox()
        self.spn_C.setRange(-1e9, 1e9)
        self.spn_C.setValue(0.0)
        form.addRow("A:", self.spn_A)
        form.addRow("C:", self.spn_C)
        lay.addLayout(form)

        # T mode
        tgrp = QtWidgets.QGroupBox("T")
        tlay = QtWidgets.QHBoxLayout(tgrp)
        self.rad_T_num = QtWidgets.QRadioButton("Numeric T=")
        self.rad_T_name = QtWidgets.QRadioButton("T_name=")
        self.rad_T_num.setChecked(True)
        self.spn_T_num = QtWidgets.QDoubleSpinBox()
        self.spn_T_num.setRange(1e-9, 1e9)
        self.spn_T_num.setValue(0.035)
        self.ed_T_name = QtWidgets.QLineEdit()
        self.ed_T_name.setPlaceholderText("Tglobal")
        tlay.addWidget(self.rad_T_num)
        tlay.addWidget(self.spn_T_num)
        tlay.addWidget(self.rad_T_name)
        tlay.addWidget(self.ed_T_name)
        lay.addWidget(tgrp)

        # t source
        g = QtWidgets.QGroupBox("t source")
        gl = QtWidgets.QVBoxLayout(g)
        self.rad_t_parent = QtWidgets.QRadioButton("Use parent slice times (parent.t_f1)")
        self.rad_t_load   = QtWidgets.QRadioButton("Load times via parent.on_open_time()")
        self.rad_t_const  = QtWidgets.QRadioButton("Constant t =")
        self.rad_t_parent.setChecked(True)
        gl.addWidget(self.rad_t_parent)
        gl.addWidget(self.rad_t_load)
        hl_const = QtWidgets.QHBoxLayout()
        hl_const.addWidget(self.rad_t_const)
        self.spn_t_const = QtWidgets.QDoubleSpinBox()
        self.spn_t_const.setRange(0.0, 1e9)
        self.spn_t_const.setValue(0.01)
        hl_const.addWidget(self.spn_t_const)
        gl.addLayout(hl_const)
        lay.addWidget(g)

        # buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        # enable/disable T fields
        self.rad_T_num.toggled.connect(self._update_T_mode)
        self.rad_T_name.toggled.connect(self._update_T_mode)
        self._update_T_mode()

    def _update_T_mode(self):
        is_num = self.rad_T_num.isChecked()
        self.spn_T_num.setEnabled(is_num)
        self.ed_T_name.setEnabled(not is_num)

    def result_template(self):
        return self._result

    def accept(self):
        if self.rad_t_parent.isChecked():
            t_mode = "parent"
        elif self.rad_t_load.isChecked():
            t_mode = "load"
        else:
            t_mode = "const"

        T_is_name = self.rad_T_name.isChecked()
        if T_is_name:
            T_val = self.ed_T_name.text().strip()
            if not T_val:
                QtWidgets.QMessageBox.warning(self, "Exponential template", "T_name cannot be empty.")
                return
        else:
            T_val = self.spn_T_num.value()

        self._result = {
            "driver": self.ed_driver.text().strip(),
            "A": self.spn_A.value(),
            "C": self.spn_C.value(),
            "T_is_name": T_is_name,
            "T_val": T_val,
            "t_mode": t_mode,
            "const_t": self.spn_t_const.value(),
        }
        super().accept()




# --------------------------- Link editor dialog ---------------------------
class LinkEditorDialog(QtWidgets.QDialog):
    def __init__(self,
                 parent,
                 target: ParamRef,
                 slice_count: int,
                 peaks_per_slice: int,
                 link_store: LinkStore,
                 default_link_type: LinkType = LinkType.LINEAR):
        super().__init__(parent)
        self.setWindowTitle("Edit Link")
        self._target = target
        self._link_store = link_store

        # Try to get per-slice times from parent (t_f1 or slice_times)
        self._times = None
        try:
            self._times = getattr(parent, "t_f1", None)
            if self._times is None:
                self._times = getattr(parent, "slice_times", None)
        except Exception:
            self._times = None

        layout = QtWidgets.QFormLayout(self)

        # Target label
        layout.addRow("Target:", QtWidgets.QLabel(f"s{target.slice_id} p{target.peak_id} {target.name}"))

        # Link type
        self.cmb_type = QtWidgets.QComboBox()
        self.cmb_type.addItem("Linear (target = driver*a + b)", LinkType.LINEAR)
        self.cmb_type.addItem("Time decay (target = driver*A*exp(-t/T)+C)", LinkType.RELAX_EXP)
        self.cmb_type.setCurrentIndex(0 if default_link_type == LinkType.LINEAR else 1)
        layout.addRow("Type:", self.cmb_type)

        # ---------------- LINEAR (unchanged) ----------------
        self.cmb_drv_slice = QtWidgets.QSpinBox()
        self.cmb_drv_slice.setRange(0, max(0, slice_count-1))
        self.cmb_drv_peak  = QtWidgets.QSpinBox()
        self.cmb_drv_peak.setRange(0, max(0, peaks_per_slice-1))
        self.cmb_drv_param = QtWidgets.QComboBox()
        self.cmb_drv_param.addItems(COLUMN_NAMES)

        drv_grid = QtWidgets.QHBoxLayout()
        drv_grid.addWidget(QtWidgets.QLabel("slice"))
        drv_grid.addWidget(self.cmb_drv_slice)
        drv_grid.addWidget(QtWidgets.QLabel("peak"))
        drv_grid.addWidget(self.cmb_drv_peak)
        drv_grid.addWidget(QtWidgets.QLabel("param"))
        drv_grid.addWidget(self.cmb_drv_param)
        self._w_linear = QtWidgets.QWidget()
        self._w_linear.setLayout(drv_grid)
        layout.addRow("Driver:", self._w_linear)

        self.edt_a = QtWidgets.QDoubleSpinBox()
        self.edt_a.setDecimals(3); self.edt_a.setRange(-1e12, 1e12); self.edt_a.setValue(1.0)
        self.edt_b = QtWidgets.QDoubleSpinBox()
        self.edt_b.setDecimals(3); self.edt_b.setRange(-1e12, 1e12); self.edt_b.setValue(0.0)
        lin_grid = QtWidgets.QHBoxLayout()
        lin_grid.addWidget(QtWidgets.QLabel("a"))
        lin_grid.addWidget(self.edt_a)
        lin_grid.addWidget(QtWidgets.QLabel("b"))
        lin_grid.addWidget(self.edt_b)
        self._w_lin_args = QtWidgets.QWidget(); self._w_lin_args.setLayout(lin_grid)
        layout.addRow("Linear args:", self._w_lin_args)
        self._lbl_exp_args = QtWidgets.QLabel("Exponential args:")
        layout.addRow(self._lbl_exp_args)

        # ---------------- RELAX_EXP (new UI) ----------------
        # A spin
        self.spn_A = QtWidgets.QDoubleSpinBox()
        self.spn_A.setDecimals(3)
        self.spn_A.setRange(-1e12, 1e12)
        self.spn_A.setValue(1.0)
        layout.addRow("A (multiplier):", self.spn_A)

        # t (with units)
        t_row = QtWidgets.QHBoxLayout()
        self.spn_t = QtWidgets.QDoubleSpinBox()
        self.spn_t.setDecimals(6); self.spn_t.setRange(-1e6, 1e6); self.spn_t.setValue(0.0)
        self.cmb_t_unit = QtWidgets.QComboBox()
        self.cmb_t_unit.addItems(["s", "ms"])
        t_row.addWidget(QtWidgets.QLabel("t"))
        t_row.addWidget(self.spn_t)
        t_row.addWidget(self.cmb_t_unit)
        self.lbl_t_hint = QtWidgets.QLabel("Hint: leave as prefilled from time file or edit to override.")
        self.lbl_t_hint.setStyleSheet("color: gray;")
        t_col = QtWidgets.QVBoxLayout()
        t_col.addLayout(t_row); t_col.addWidget(self.lbl_t_hint)
        self._w_t = QtWidgets.QWidget(); self._w_t.setLayout(t_col)
        layout.addRow("Time:", self._w_t)

        # T (named or numeric)
        T_col = QtWidgets.QVBoxLayout()
        self.edt_T = QtWidgets.QLineEdit()
        self.edt_T.setPlaceholderText("e.g., T_1  (name → shared parameter)  or  0.0123  (fixed)")
        self.lbl_T_hint = QtWidgets.QLabel("Use a meaningful name like T_1. Numbers create a fixed T for this link.")
        self.lbl_T_hint.setStyleSheet("color: gray;")
        T_col.addWidget(self.edt_T); T_col.addWidget(self.lbl_T_hint)
        self._w_T = QtWidgets.QWidget(); self._w_T.setLayout(T_col)
        layout.addRow("Decay T:", self._w_T)

        # C
        self.spn_C = QtWidgets.QDoubleSpinBox()
        self.spn_C.setDecimals(3); self.spn_C.setRange(-1e12, 1e12); self.spn_C.setValue(0.0)
        layout.addRow("Offset C:", self.spn_C)

        # Preview (two lines)
        self.lbl_prev_human = QtWidgets.QLabel("")
        self.lbl_prev_engine = QtWidgets.QLabel("")
        self.lbl_prev_human.setStyleSheet("color: gray;")
        self.lbl_prev_engine.setStyleSheet("color: gray; font-family: Consolas, 'Courier New', monospace;")
        layout.addRow("Preview:", self.lbl_prev_human)
        layout.addRow("", self.lbl_prev_engine)

        # Enabled + buttons
        self.chk_enabled = QtWidgets.QCheckBox("Enabled")
        self.chk_enabled.setChecked(True)
        layout.addRow("", self.chk_enabled)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

        self._preload_existing()
        
        self._sync_visibility()
        self._prefill_time_from_file()
        self._update_preview()

        # Wire events
        self.cmb_type.currentIndexChanged.connect(self._sync_visibility)
        self.spn_A.valueChanged.connect(self._update_preview)

        self.spn_t.valueChanged.connect(self._update_preview)
        self.cmb_t_unit.currentIndexChanged.connect(self._update_preview)
        self.edt_T.textChanged.connect(self._update_preview)
        self.spn_C.valueChanged.connect(self._update_preview)
        self.cmb_drv_slice.valueChanged.connect(self._update_preview)
        self.cmb_drv_peak.valueChanged.connect(self._update_preview)
        self.cmb_drv_param.currentIndexChanged.connect(self._update_preview)


    # ---------- helpers ----------
    def _prefill_time_from_file(self):
        """Prefill t from loaded times if available; convert to chosen unit (default s)."""
        # don't swallow everything silently
        if self._times is None:
            # print("no times on parent")
            return
    
        sid = int(self._target.slice_id)
        # robust against "dialog opened for slice not in time file"
        if not hasattr(self._times, "__len__"):
            # print("times is not indexable:", type(self._times))
            return
    
        n = len(self._times)
        if sid < 0 or sid >= n:
            # print(f"time not available for slice {sid}; array has {n} entries")
            return
    
        t_sec = float(self._times[sid])
        if self.cmb_t_unit.currentText() == "ms":
            self.spn_t.setValue(t_sec * 1e3)
        else:
            self.spn_t.setValue(t_sec)

    def _sync_visibility(self):
        layout = self.layout()                     # QFormLayout
        link_type = self.cmb_type.currentData()
        is_linear = (link_type == LinkType.LINEAR)

        # driver: always
        self._w_linear.setVisible(True)
        lbl_drv = layout.labelForField(self._w_linear)
        if lbl_drv is not None:
            lbl_drv.setVisible(True)

        # linear-only
        self._w_lin_args.setVisible(is_linear)
        lbl_lin = layout.labelForField(self._w_lin_args)
        if lbl_lin is not None:
            lbl_lin.setVisible(is_linear)

        # exp-only
        self._lbl_exp_args.setVisible(not is_linear)
        for w in (self.spn_A, self._w_t, self._w_T,
                  self.spn_C, self.lbl_prev_human, self.lbl_prev_engine):
            if not w:
                continue
            w.setVisible(not is_linear)
            lbl = layout.labelForField(w)
            if lbl is not None:
                lbl.setVisible(not is_linear)

    @staticmethod
    def _normalize_T_name(text: str) -> str:
        s = (text or "").strip()
        # numeric? leave empty to signal "numeric"
        try:
            float(s)
            return ""  # numeric
        except Exception:
            pass
        # normalize to lowercase snake, leading letter
        import re
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
        s = s.strip("_")
        s = s.lower()
        if not s:
            return ""
        if not s[0].isalpha():
            s = "t_" + s
        return s

    def _current_t_seconds(self) -> float:
        val = float(self.spn_t.value())
        return val / 1e3 if self.cmb_t_unit.currentText() == "ms" else val



    def _T_as_term(self) -> tuple[str, bool]:
        """Returns (term, is_named). If numeric, term is the number string."""
        raw = (self.edt_T.text() or "").strip()
        # numeric?
        try:
            val = float(raw)
            return (f"{val:.6g}", False)
        except Exception:
            pass
        norm = self._normalize_T_name(raw)
        if not norm:
            return ("", True)  # invalid name; preview will show blank
        return (norm, True)

    def _update_preview(self):
        # Only for RELAX_EXP
        if self.cmb_type.currentData() != LinkType.RELAX_EXP:
            self.lbl_prev_human.setText("")
            self.lbl_prev_engine.setText("")
            return

        sid = int(self._target.slice_id)
        pid = int(self._target.peak_id)
        t_sec = self._current_t_seconds()
        T_term, is_named = self._T_as_term()
        C_val = float(self.spn_C.value())
        A_val = float(self.spn_A.value())

        sid = int(self._target.slice_id)
        pid = int(self._target.peak_id)
        tgt_txt = f"s{sid}_{self._target.name.lower()}_{pid}"

        drv_sid = self.cmb_drv_slice.value()
        drv_pid = self.cmb_drv_peak.value()
        drv_param = COLUMN_NAMES[self.cmb_drv_param.currentIndex()]
        drv_txt = f"s{drv_sid}_{drv_param.lower()}_{drv_pid}"

        # human line
        T_human = T_term if is_named and T_term else (self.edt_T.text().strip() or "T")
        self.lbl_prev_human.setText(
            f"{tgt_txt} = {drv_txt} * {A_val:.6g} * exp(- {t_sec:.6g} / {T_human}) + {C_val:.6g}"
        )

        # engine line (canonical)
        T_engine = T_term if (T_term and is_named) else (f"{float(self.edt_T.text()):.6g}" if self.edt_T.text().strip() else "T")
        if not T_engine:
            T_engine = "T"
        self.lbl_prev_engine.setText(
            f'expr → "{drv_txt}*({A_val:.6g})*exp(-({t_sec:.6g})/{T_engine})+{C_val:.6g}"'
        )

    def _preload_existing(self):
        expr = self._link_store.get(self._target)
        if not expr:
            # sensible defaults
            self.cmb_type.setCurrentIndex(0)  # linear by default
            return

        self.chk_enabled.setChecked(expr.enabled)
        if expr.type == LinkType.LINEAR:
            self.cmb_type.setCurrentIndex(0)
            if expr.driver:
                self.cmb_drv_slice.setValue(int(expr.driver.slice_id))
                self.cmb_drv_peak.setValue(int(expr.driver.peak_id))
                try:
                    name = expr.driver.name.lower()
                    idx = [n.lower() for n in COLUMN_NAMES].index(name)
                    self.cmb_drv_param.setCurrentIndex(idx)
                except Exception:
                    pass
            self.edt_a.setValue(float(expr.args.get('a', 1.0)))
            self.edt_b.setValue(float(expr.args.get('b', 0.0)))
        else:
            self.cmb_type.setCurrentIndex(1)
            
            A_val = float(expr.args.get("A", 1.0))
            self.spn_A.setValue(A_val)

            # 3) restore T
            if "T_name" in expr.args:
                self.edt_T.setText(str(expr.args["T_name"]))
            elif "T_value" in expr.args:
                self.edt_T.setText(f"{float(expr.args['T_value']):.6g}")

            # 4) restore C
            if "C" in expr.args:
                self.spn_C.setValue(float(expr.args["C"]))

            # 5) restore t override
            if "t_override" in expr.args:
                t_sec = float(expr.args["t_override"])
                if self.cmb_t_unit.currentText() == "ms":
                    self.spn_t.setValue(t_sec * 1e3)
                else:
                    self.spn_t.setValue(t_sec)
            elif self._times is not None:
                sid = int(self._target.slice_id)
                try:
                    t_sec = float(self._times[sid])
                    if self.cmb_t_unit.currentText() == "ms":
                        self.spn_t.setValue(t_sec * 1e3)
                    else:
                        self.spn_t.setValue(t_sec)
                except Exception:
                    pass
            if expr.driver:
                self.cmb_drv_slice.setValue(int(expr.driver.slice_id))
                self.cmb_drv_peak.setValue(int(expr.driver.peak_id))
                try:
                    self.cmb_drv_param.setCurrentIndex(COLUMN_NAMES.index(expr.driver.name))
                except Exception:
                    pass
                

        self._sync_visibility()
        self._update_preview()

    # ---------- result ----------
    def result_expr(self) -> Optional[LinkExpr]:
        drv = ParamRef(self.cmb_drv_slice.value(),
               self.cmb_drv_peak.value(),
               COLUMN_NAMES[self.cmb_drv_param.currentIndex()].lower())
        t = self.cmb_type.currentData()
        if t == LinkType.LINEAR:

            args = {'a': float(self.edt_a.value()), 'b': float(self.edt_b.value())}
            rev = self._link_store.get(drv)
            if rev and rev.driver == self._target and rev.type == LinkType.LINEAR:
                QtWidgets.QMessageBox.warning(self, "Invalid link",
                        "This would create a cycle: the driver is already linked back to the target.")
                return None
            return LinkExpr(type=LinkType.LINEAR, target=self._target, driver=drv, args=args,
                            enabled=self.chk_enabled.isChecked())

        # RELAX_EXP
        if t == LinkType.RELAX_EXP:           

            # Validate A
            args: dict = {}
            args["A"] = float(self.spn_A.value())

            # T: named or numeric
            T_text = (self.edt_T.text() or "").strip()
            T_name_norm = self._normalize_T_name(T_text)
            if T_name_norm:
                args["T_name"] = T_name_norm
                args["display_T"] = T_text
            else:
                # must be numeric
                try:
                    args["T_value"] = float(T_text)
                except Exception:
                    QtWidgets.QMessageBox.warning(self, "Invalid T", "Enter a T name (e.g., T_1) or a numeric value.")
                    return None

            # C
            args["C"] = float(self.spn_C.value())

            # t override (store only if user edited or no file)
            t_sec = self._current_t_seconds()
            # heuristic: store if no file known or nonzero edit
            if (self._times is None) or (t_sec > 0):
                args["t_override"] = float(t_sec)

        return LinkExpr(type=LinkType.RELAX_EXP, target=self._target, driver=drv, args=args,
                        enabled=self.chk_enabled.isChecked())

# --------------------------- View context menu wiring ---------------------------

def attach_link_context_menu(view: QtWidgets.QTableView,
                             model: PeakTableModel,
                             slice_count: int,
                             peaks_per_slice: int, parent=None, **kwargs):
    """
    Right-click → 'Edit Link…' for the clicked cell.
    """
    view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def _on_menu(pos: QtCore.QPoint):
        idx = view.indexAt(pos)
        if (not idx.isValid()) or (idx.column() not in (0, 2, 4, 6)):
            return  # only pos/amp/lor/gauss cells have links
        pref = model.index_to_paramref(idx)
        if pref is None:
            return

        menu = QtWidgets.QMenu(view)
        act_edit = menu.addAction("Edit Link…")
        act_clear = menu.addAction("Clear Link") if model._links.is_linked(pref) else None
        act_bounds = menu.addAction("Set Bounds…")
        act_clear_bounds = menu.addAction("Clear Bounds")

        chosen = menu.exec_(view.viewport().mapToGlobal(pos))
        if chosen is None:
            return

        if chosen == act_edit:
            dlg = LinkEditorDialog(
                parent or view,
                target=pref,
                slice_count=slice_count,
                peaks_per_slice=peaks_per_slice,
                link_store=model._links,
            )
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                expr = dlg.result_expr()
                if expr is not None:
                    model._links.set_link(expr)
                    # refresh flags + tooltip for this cell
                    model.dataChanged.emit(idx, idx, [
                        QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.ForegroundRole, QtCore.Qt.ToolTipRole
                    ])

                    mw = parent or view
                    try:
                        if expr.type == LinkType.RELAX_EXP:
                            tname = expr.args.get("T_name") or expr.args.get("t_name")
                            if isinstance(tname, str) and tname.strip():
                                tname = tname.strip()
                                if hasattr(mw, "_ensure_tseed_row"):
                                    mw._ensure_tseed_row(tname)
                    except Exception:
                        pass

        elif act_clear and chosen == act_clear:
            model._links.remove_link(pref)
            model.dataChanged.emit(idx, idx, [
                QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.ForegroundRole, QtCore.Qt.ToolTipRole
            ])

        elif chosen == act_bounds:
            dlg = BoundsDialog(parent or view, title=f"Set Bounds for {pref.name}")
            # preload current
            b = model.get_bounds_for(pref)
            dlg.preload(b.lo, b.hi)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                lo, hi = dlg.result()
                model.set_bounds_for(pref, lo, hi)
                model.dataChanged.emit(idx, idx, [QtCore.Qt.ToolTipRole, QtCore.Qt.DisplayRole])

        elif act_clear_bounds and chosen == act_clear_bounds:
            model.clear_bounds_for(pref)
            model.dataChanged.emit(idx, idx, [QtCore.Qt.ToolTipRole, QtCore.Qt.DisplayRole])

    view.customContextMenuRequested.connect(_on_menu)




# --------------------------- Link engine ---------------------------

class LinkEngine:
    @staticmethod
    def evaluate(registry: Dict[ParamRef, Dict[str, Any]],
                 slice_times: Optional[List[float]],
                 links: LinkStore) -> None:
        """
        Mutates registry[target]['value'] for all enabled links.
        registry[ParamRef] = {'value': float, 'fixed': bool, 'bounds': (lo, hi)}
        """
        # Build DAG (only LINEAR edges create dependencies).
        edges: Dict[ParamRef, List[ParamRef]] = {} # Dict[driver, list of targets]
        indeg: Dict[ParamRef, int] = {} # Dict[driver: 0, target: 1]
        # driver always has values of  0, means no other parameters control them
        # targets alsways have values of 1, means 1 driver control them

        targets = [ex.target for ex in links.all_expr() if ex.enabled]
        # links is LinkStore instance. links.all_expr() = LinkStore._by_target.values()
        # self._by_target = dict{ParamRef, LinkExpr}, an attribute of LinkStore
        # s = all values in self._by_target dict = LinkExpr instance
        # ex.target = LinkExpr.target. target is a ParamRef instance, which is an attribute of LinkExpr
        # target: list of ex.target (ParamRef instances)
        for t in targets:
            indeg[t] = 0
        for expr in links.all_expr():
            if not expr.enabled:
                continue
            if expr.driver is not None:
                d = expr.driver
                edges.setdefault(d, []).append(expr.target)
                indeg.setdefault(d, 0)
                indeg[expr.target] = indeg.get(expr.target, 0) + 1

        # Kahn topo-sort over LINEAR and Relaxation exponential dependencies
        q = [t for t, k in indeg.items() if k == 0] 
        # q = list of drivers, those are keys that have values of 0 in indeg
        ordered: List[ParamRef] = []
        while q:
            u = q.pop()
            ordered.append(u)
            for v in edges.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(ordered) != len(indeg):
            raise ValueError("Link cycle detected among links. Resolve before fitting.")

        # Compute order to fit. For each target that has either LINEAR or Exponential expr, compute from its driver
        exprs_by_target = {ex.target: ex for ex in links.all_expr() if ex.enabled}
        for tgt in ordered:
            expr = exprs_by_target.get(tgt)
            if not expr:
                continue
            if expr.type == LinkType.LINEAR:
                a = float(expr.args.get('a', 1.0))
                b = float(expr.args.get('b', 0.0))
                drv = expr.driver
                if drv is None:
                    raise ValueError("Link missing driver.")
                drv_entry = registry.get(drv)
                if not drv_entry or not math.isfinite(drv_entry.get('value', float('nan'))):
                    raise ValueError(f"Driver value missing/invalid for {drv}.")
                val = a * float(drv_entry['value']) + b
                registry.setdefault(tgt, {'value': val, 'fixed': False, 'bounds': (-math.inf, math.inf)})
                registry[tgt]['value'] = float(val)
            
            elif expr.type == LinkType.RELAX_EXP:
                if slice_times is None:
                    raise ValueError("RELAX_EXP link requires slice_times.")
                A = float(expr.args.get('A', 1.0))
                C = float(expr.args.get('C', 0.0))
                T = expr.args.get('T', 1.0)
                # sequential path: T must be numeric
                try:
                    T = float(T)
                except Exception:
                    raise ValueError("RELAX_EXP (sequential): T must be a numeric value.")
                if not (T > 0 and math.isfinite(T)):
                    raise ValueError("RELAX_EXP: T must be > 0 and finite.")
                drv = expr.driver
                if drv is None:
                    raise ValueError("RELAX_EXP link missing driver.")
                drv_entry = registry.get(drv)
                if not drv_entry or not math.isfinite(drv_entry.get('value', float('nan'))):
                    raise ValueError(f"Driver value missing/invalid for {drv}.")
                t_i = slice_times[expr.target.slice_id]
                val = float(drv_entry['value']) * A * math.exp(-t_i / T) + C
                registry.setdefault(expr.target, {'value': val, 'fixed': False, 'bounds': (-math.inf, math.inf)})
                registry[expr.target]['value'] = val

# --- Cross-slice helpers for sequential fitting ---

def _topo_sort_slices_for_links(links: "LinkStore", selected: list[int]) -> list[int]:
    """
    Topologically sort the selected slice IDs so that cross-slice LINEAR drivers
    come before their targets. If a cycle spans selected slices, raise ValueError.
    Slices with no deps keep their original relative order.
    """
    selected_set = set(int(s) for s in selected)
    # Build edges driver_slice -> target_slice for cross-slice LINEAR links
    indeg = {s: 0 for s in selected_set}
    edges = {s: [] for s in selected_set}

    for expr in links.all_expr():
        if not getattr(expr, "enabled", False):
            continue
        if getattr(expr, "type", None) != LinkType.LINEAR:
            continue
        d = getattr(expr, "driver", None)
        t = getattr(expr, "target", None)
        if d is None or t is None:
            continue
        ds, ts = int(d.slice_id), int(t.slice_id)
        if ds in selected_set and ts in selected_set and ds != ts:
            edges[ds].append(ts)
            indeg[ts] = indeg.get(ts, 0) + 1

    # Kahn's algorithm with stable layering (preserve user order within layers)
    order = []
    layer = [s for s in selected if indeg.get(s, 0) == 0]  # seed in user order
    seen = set(layer)
    while layer:
        nxt = []
        for u in layer:
            order.append(u)
            for v in edges.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0 and v not in seen:
                    nxt.append(v)
                    seen.add(v)
        layer = nxt

    if len(order) != len(selected_set):
        # Cycle among the selected slices – force joint fit or break links
        raise ValueError("Cross-slice link cycle detected among selected slices.")
    # Add any selected slices that had no nodes at all (paranoia)
    for s in selected:
        if s not in order:
            order.append(s)
    return order

def _links_for_target_slice(links: "LinkStore", s: int) -> "LinkStore":
    """
    Return a new LinkStore containing only links whose TARGET lives in slice `s`.
    (Keep both LINEAR and RELAX_EXP for those targets.)
    """
    sub = LinkStore()
    for expr in links.all_expr():
        if not getattr(expr, "enabled", False):
            continue
        tgt = getattr(expr, "target", None)
        if tgt is None or int(tgt.slice_id) != int(s):
            continue
        # Recreate an equivalent expr in the subset
        sub.set_link(LinkExpr(
            type=expr.type,
            target=expr.target,
            driver=expr.driver,
            args=dict(expr.args),
            enabled=True,
        ))
    return sub


def _seed_external_drivers_into_registry(
    *,
    registry: dict["ParamRef", dict],
    slice_states: dict[int, "SliceFitState"],
    current_slice_id: int,
    links: "LinkStore",
    strict: bool = True,
) -> None:
    """
    For each enabled link targeting current_slice_id whose driver lives in another slice,
    inject the driver's current value (read-only) into the registry so LinkEngine can use it.
    """
    missing = []  # collect (driver_ref, reason)
    # Collect targets in current slice
    targets_here = {ex.target for ex in links.all_expr()
                    if getattr(ex, "enabled", False)
                    and getattr(ex, "target", None) is not None
                    and int(ex.target.slice_id) == int(current_slice_id)} # set of targets

    def _norm(n: str) -> str: return (n or "").strip().lower()

    for expr in links.all_expr():
        if not getattr(expr, "enabled", False):
            continue
        if getattr(expr, "type", None) != LinkType.LINEAR:
            continue
        t = getattr(expr, "target", None)
        d = getattr(expr, "driver", None)
        if t is None or d is None or t not in targets_here:
            continue

        ds, ts = int(d.slice_id), int(t.slice_id)
        if ds == ts:
            continue  # same-slice driver is already in the per-slice registry

        other = slice_states.get(ds, None)
        if other is None:
            missing.append((d, f"slice_state[{ds}] not found"))
            continue

        # Check peak index exists
        try:
            pk = other.peaks[int(d.peak_id)]
        except Exception:
            missing.append((d, f"peak_id {d.peak_id} not present in slice {ds} "
                               f"(len={len(getattr(other, 'peaks', []))})"))
            continue

        # Map parameter name → value
        base = _norm(getattr(d, "name", ""))
        try:
            if base in ("amp", "area"):
                val = float(pk.amp)
            elif base in ("pos", "position", "freq", "hz", "ppm"):
                val = float(pk.pos)
            elif base in ("lor", "lorentz", "lor_hz"):
                val = float(pk.lor_hz)
            elif base in ("gauss", "gauss_disp", "gau"):
                val = float(pk.gauss_disp)
            else:
                missing.append((d, f"unknown driver name '{d.name}'"))
                continue
        except Exception as ex:
            missing.append((d, f"value read failed: {ex}"))
            continue

        # Seed read-only entry
        if d not in registry:
            registry[d] = {"value": val, "fixed": True, "bounds": (val, val)}

    if strict and missing:

        lines = []
        for d, why in missing:
            lines.append(f"- driver slice={d.slice_id} peak={d.peak_id} name={getattr(d, 'name', '?')}: {why}")
        raise RuntimeError("Cross-slice driver seeding failed:\n" + "\n".join(lines))
    

# -------------------------- Unit conversion. A fit uses Hz unit for pos, lor, gauss, and bound internally. ----------------------------
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

def Peaks_to_PeaksHz(self, peaks):
    if self.axis_mode.lower() == 'ppm':
        peaks_hz = []
        for pk in peaks:
            peaks_hz.append(Peak(
                pos=ppm_to_hz(pk.pos, self.ref),
                amp=pk.amp,
                lor_hz=pk.lor_hz,
                gauss_disp=ppm_to_hz(pk.gauss_disp, self.ref),
            ))
        return peaks_hz
    return peaks

def Bounds_to_BoundsHz(lo, hi, pref_name: str, axis_mode: str, ref_mhz: float):
    """Return (lo_int, hi_int) in internal Hz units, given display-unit bounds."""
    # pass-through if nothing set
    if lo is None and hi is None:
        return None, None

    def conv(v):
        if v is None:
            return None
        if axis_mode.lower() == "ppm" and pref_name in ("pos", "gauss"):
            # pos, gauss are shown in ppm when axis_mode == 'ppm'
            return float(ppm_to_hz(float(v), ref_mhz))
        # lor (Hz) and amp (arb.) are already in internal units
        return float(v)

    lo_i = conv(lo)
    hi_i = conv(hi)

    # if both set and inverted after conversion, swap them, just in case. 
    # It should never happen because BoundsDialog._on_ok() make sure that the input has hi > lo and ref_mhz is positive
    if (lo_i is not None) and (hi_i is not None) and (lo_i > hi_i):
        lo_i, hi_i = hi_i, lo_i
    return lo_i, hi_i


def calc_sw_hz(x_hz: np.ndarray) -> float:
    # Approximate SW from displayed window width.
    
    return float((np.max(x_hz) - np.min(x_hz)) * x_hz.size / (x_hz.size - 1))

def extract_sw_hz_from_meta(meta: dict | None) -> float | None:
    """Try to read SW (Hz) from meta; if missing/invalid, return None."""
    sw = None
    if isinstance(meta, dict):
        v = meta.get("sw_Hz", None)
        if v is not None:
            try:
                sw = float(v)
            except Exception:
                sw = None

    if sw is not None and sw > 0:
        return sw

    # nothing usable in meta → leave None so caller can auto-estimate
    return None

def extract_ref_from_meta(meta: dict | None) -> float | None:
    """Try to read reference frequency from meta, convert to MHz; if missing/invalid, return None."""
    ref = None
    if isinstance(meta, dict):
        v = meta.get("ref_Hz", None)
        if v is not None:
            try:
                ref = float(v) / 1e6
            except Exception:
                ref = None
    if ref is not None and ref > 0:
        return ref
    return None

def extract_freq_from_meta(meta: dict | None) -> float | None:
    """Try to read transmitter frequency from meta, convert to MHz; if missing/invalid, return None."""
    freq = None
    if isinstance(meta, dict):
        v = meta.get("transmitter_freq", None)
        if v is not None:
            try:
                freq = float(v) / 1e6
            except Exception:
                freq = None
    if freq is not None and freq > 0:
        return freq
    return None

def _assert_axis_ready(axis_mode: str, sw_hz, ref):
    """Backend-only guard for axis handling."""
    # SW must be positive
    if sw_hz is None or float(sw_hz) <= 0.0:
        raise ValueError("Spectral width (SW) must be > 0 Hz.")
    # If axis is ppm, MHz must be positive
    if str(axis_mode).lower() == "ppm":
        if ref is None or float(ref) <= 0.0:
            raise ValueError("Spectrometer frequency (MHz) required for ppm axis.")


# ---------- CORE KERNEL (physics only; canonical units = Hz) ----------
def peakSim_kernel(
    t: np.ndarray,                 # (N,)
    pos_hz: np.ndarray,            # (P,)
    amp: np.ndarray,               # (P,)
    lor_hz: np.ndarray,            # (P,)
    gauss_hz: np.ndarray,          # (P,)
    x: np.ndarray,                 # (N,)  -- MUST be in Hz; center at x[N//2]
    sw_eff_hz: float,              # scalar, used in amplitude prefactor
    multiplier: float              # scalar
) -> np.ndarray:
    """
    Vectorized single-peak kernel (FID), Lorentz+Gauss broadening in time.
    Returns complex FID of shape (N,). Two-sided t is assumed by caller.
    """
    # Guards / canonicalization
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    sw_eff_hz = float(sw_eff_hz) if (sw_eff_hz is not None and sw_eff_hz > 0) else 1.0

    # Broadcast to (P,N)
    N = x.size
    x_center_hz = float(x[N // 2])

    t_row   = t[None, :]                                   # (1, N)
    pos_col = (np.asarray(pos_hz,  dtype=float) - x_center_hz)[:, None]  # (P, 1)
    lor_col =  np.asarray(lor_hz,  dtype=float)[:, None]                  # (P, 1)
    gau_col =  np.asarray(gauss_hz, dtype=float)[:, None]                 # (P, 1)
    amp_col = (float(multiplier) * np.asarray(amp, dtype=float) / sw_eff_hz)[:, None]

    # Phase & apodization (note |t| for Lorentz; t^2 for Gauss)
    phase = 2j * np.pi * pos_col * t_row
    apod  = -np.pi * np.abs(lor_col) * np.abs(t_row) - (np.pi * np.abs(gau_col) * t_row)**2 / (4.0 * np.log(2.0))

    return (amp_col * np.exp(phase + apod)).sum(axis=0)    # (N,)



def build_fid_from_peaks(
    N: int,                        # Number of POINTS in the spectrum/FID
    sw_hz: float,                  # acquisition SW (Hz); dt = 1/sw_hz
    x: np.ndarray,                 # (N,) frequency axis in Hz used only for its center bin
    pos_hz: np.ndarray,            # (P,)
    amp: np.ndarray,               # (P,)
    lor_hz: np.ndarray,            # (P,)
    gauss_hz: np.ndarray,          # (P,)
    multiplier: float,
    *,
    return_fid: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Math-only path:
      - builds a two-sided time grid from (N, sw_hz)
      - sums the complex FID via kernel
      - FFT→fftshift to complex spectrum
    """
    # Guards
    if (N is None) or (N <= 0) or (sw_hz is None) or (sw_hz <= 0.0):
        empty_c = np.array([], dtype=complex)
        return (np.array([], dtype=float), empty_c, empty_c) if return_fid else empty_c

    x = np.asarray(x, dtype=float)
    if x.size != N:
        raise ValueError(f"'x' must have length N; got len(x)={x.size}, N={N}")
    # NOTE: x MUST be in Hz (caller responsibility)

    # Two-sided time grid
    t = np.fft.fftfreq(N, d=sw_hz / N)   # seconds

    # Empty-peak fast path
    if pos_hz is None or len(pos_hz) == 0:
        fid  = np.zeros(N, dtype=complex)
        spec = fid_to_spectrum(fid)
        return (t, fid, spec) if return_fid else spec

    # Build FID via kernel (use acquisition SW as effective SW)
    fid = peakSim_kernel(
        t=t,
        pos_hz=pos_hz,
        amp=amp,
        lor_hz=lor_hz,
        gauss_hz=gauss_hz,
        x=x,
        sw_eff_hz=float(sw_hz),
        multiplier=float(multiplier),
    ).astype(complex, copy=False)

    # FFT → complex spectrum (no phase/offset here)
    spec = fid_to_spectrum(fid)
    return (t, fid, spec) if return_fid else spec



# ---------- FFT UTILITY ----------
def fid_to_spectrum(fid: np.ndarray) -> np.ndarray:
    spec = fft(fid)
    spec = fftshift(spec)
    return spec

def apply_phase_and_offset(spec_cmplx: np.ndarray,
                           phi0_deg: float,
                           offset: float) -> np.ndarray:
    """
    Apply only zero-order phase (phi0) and offset. Return real spectrum
    """
    phi0 = np.deg2rad(float(phi0_deg))
    return np.real(spec_cmplx * np.exp(-1j * phi0)) + float(offset)



# ---------- ORCHESTRATOR (unit conversion & grid choice from acquisition) ----------
def model_spectrum(
    peaks: list,                 # Peak objects with .pos, .amp, .lor_hz, .gauss_disp
    *,
    axis_mode: str,              # 'ppm' or 'hz' (input interpretation only)
    ref: float,                  # MHz (Hz per ppm)
    sw_hz: float,                # acquisition SW (Hz)
    N: int,                      # number of points
    x: np.ndarray,               # (N,) axis in ppm or Hz (converted to Hz below)
    multiplier: float,
    return_fid: bool = False
):
    """
    Orchestrator/adapter that ensures the math core runs in Hz.
    - Fast path: axis_mode='hz' → no unit conversion is performed.
    - Backward Compatibility path: axis_mode='ppm' → convert x, pos, gauss_disp to Hz.

    Notes:
      * Core routine build_fid_from_peaks(...) expects Hz everywhere.
      * No plotting / phasing / offset here.
    """
    # --- guards ---
    if (N is None) or (N <= 0) or (sw_hz is None) or (sw_hz <= 0.0):
        empty_c = np.array([], dtype=complex)
        return (np.array([], dtype=float), empty_c, empty_c) if return_fid else empty_c

    x = np.asarray(x, dtype=float)
    if x.size != N:
        raise ValueError(f"'x' must have length N; got len(x)={x.size}, N={N}")

    mode = str(axis_mode).lower()
    is_hz  = (mode == 'hz')
    is_ppm = (mode == 'ppm')
    if not (is_hz or is_ppm):
        raise ValueError(f"axis_mode must be 'ppm' or 'hz', got {axis_mode!r}")

    # --- axis to Hz (fast path for Hz) ---
    x_hz = x if is_hz else ppm_to_hz(x, ref)

    P = len(peaks)
    if P == 0:
        # Empty spectrum: keep behavior consistent with previous code
        t = np.fft.fftfreq(N, d=sw_hz / N)  # (kept as-is for backward compatibility)
        fid = np.zeros(N, dtype=complex)
        spec = fid_to_spectrum(fid)
        return (t, fid, spec) if return_fid else spec

    # --- gather peak arrays in Hz (fast path avoids conversions) ---
    amp_arr      = np.empty(P, dtype=float)
    pos_hz_arr   = np.empty(P, dtype=float)
    lor_hz_arr   = np.empty(P, dtype=float)
    gauss_hz_arr = np.empty(P, dtype=float)

    if is_hz:
        # Fast path: all inputs are already Hz
        for i, pk in enumerate(peaks):
            amp_arr[i]      = float(pk.amp)
            pos_hz_arr[i]   = float(pk.pos)
            lor_hz_arr[i]   = float(getattr(pk, "lor_hz", 0.0))
            gauss_hz_arr[i] = float(getattr(pk, "gauss_disp", 0.0))
    elif is_ppm:
        # Compat path: convert ppm fields to Hz
        for i, pk in enumerate(peaks):
            amp_arr[i]      = float(pk.amp)
            pos_hz_arr[i]   = float(ppm_to_hz(pk.pos, ref))
            lor_hz_arr[i]   = float(getattr(pk, "lor_hz", 0.0))  # already Hz
            gdisp           = float(getattr(pk, "gauss_disp", 0.0))
            gauss_hz_arr[i] = float(ppm_to_hz(gdisp, ref))

    # --- core math in Hz ---
    return build_fid_from_peaks(
        N=int(N),
        sw_hz=float(sw_hz),
        x=x_hz,                           # Hz grid for center-bin logic
        pos_hz=pos_hz_arr,
        amp=amp_arr,
        lor_hz=lor_hz_arr,
        gauss_hz=gauss_hz_arr,
        multiplier=float(multiplier),
        return_fid=return_fid,
    )

def Peaks_to_ParamRefList(sid: int, peaks: List):
    ParamRefList = []
    peak_attr = ["pos", "amp", "lor", "gauss"]
    for pid in range(len(peaks)):
        for name in peak_attr:
            pref = ParamRef(slice_id=sid, peak_id=pid, name=name)
            ParamRefList.append(pref)
    return ParamRefList

def ParamRef_to_key(pref: ParamRef):
    key = f"s{int(pref.slice_id)}_p{int(pref.peak_id)}_{str(pref.name)}"
    return key

def global_ref(sid: int, name: str) -> "ParamRef":
    # globals still follow the same rule by using peak_id = 000
    return ParamRef(slice_id=int(sid), peak_id=000, name=str(name))

def canon_name(n: str) -> str:

    n = (n or "").strip().lower()
    # allow backwards-compat / synonyms
    return {
        "pos": "pos", "position": "pos", "freq": "pos",
        "amp": "amp", "area": "amp", "amplitude": "amp",
        "lor": "lor", "lorentz": "lor", "lor_hz": "lor",
        "gauss": "gauss", "gau": "gauss", "gaussian": "gauss",
        "gauss_disp": "gauss", "gauss_display": "gauss",
    }.get(n, n)


# ------------------------------ Fitting core -------------------------------
class FitContext:
    def __init__(self, x: np.ndarray, y: np.ndarray, axis_mode: str, ref: float, sw_hz: Optional[float]):
        self.x = x
        self.y = y
        self.axis_mode = axis_mode
        self.ref = ref
        self.sw_hz = sw_hz

    def build_params(self, peaks: List[Peak], sid: int, multiplier: float = 1.0, offset: float = 0.0, *, prefix: str = "") -> Parameters: 
        #Parameters always contains pos lor and gauss in Hz
        p = Parameters()

        p.add(f'{ParamRef_to_key(global_ref(sid, "mult"))}', value=float(multiplier), min=0.0)
        p.add(f'{ParamRef_to_key(global_ref(sid, "offset"))}', value=float(offset))
        p.add(f'{ParamRef_to_key(global_ref(sid, "phi0_deg"))}', value=0.0, min=-180.0, max=180.0)

        for pid in range(len(peaks)):
            pk = peaks[pid]
            pref_pos = ParamRef(slice_id=sid, peak_id=pid, name="pos")
            name_pos = ParamRef_to_key(pref_pos)
            pref_gauss = ParamRef(slice_id=sid, peak_id=pid, name="gauss")
            name_gauss = ParamRef_to_key(pref_gauss)
            if self.axis_mode.lower() == 'ppm':
                p.add(f'{name_pos}',   value=float(ppm_to_hz(pk.pos, self.ref)))
                p.add(f'{name_gauss}',   value=max(1e-9, float(ppm_to_hz(pk.gauss_disp, self.ref))), min=0.0)
            else:
                p.add(f'{name_pos}',   value=float(pk.pos))
                p.add(f'{name_gauss}', value=max(1e-6, float(pk.gauss_disp)), min=0.0)

            pref_lor = ParamRef(slice_id=sid, peak_id=pid, name="lor")
            name_lor = ParamRef_to_key(pref_lor)
            p.add(f'{name_lor}',   value=max(1.0, float(pk.lor_hz)), min=0.0)
            
            pref_amp = ParamRef(slice_id=sid, peak_id=pid, name="amp")
            name_amp = ParamRef_to_key(pref_amp)
            p.add(f'{name_amp}',   value=max(1e-9, float(pk.amp)), min=0.0)

        return p

    def result_to_peaks(self, peaks: List[Peak], params: Parameters, sid: int) -> None:

        for i in range(len(peaks)):
            peaks[i].pos        = hz_to_ppm(params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="pos"))}'].value, self.ref) if self.axis_mode.lower() == 'ppm' else params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="pos"))}'].value
            peaks[i].amp        = params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="amp"))}'].value
            peaks[i].lor_hz     = params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="lor"))}'].value
            peaks[i].gauss_disp = hz_to_ppm(params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'].value, self.ref) if self.axis_mode.lower() == 'ppm' else params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'].value



    def residual(self, params: Parameters, no_of_peaks, sid) -> np.ndarray:
        """
        peaks_shape is to get the number of peaks to build an updated peaks List during optimization
        no actual value of peak_shape is read.
        Residual with phi1 fully disabled.
        Applies only zero-order phase (phi0) and offset on the display grid.
        Keeps mask weighting and robust normalization.
        Supports optional self.param_prefix for joint fits.
        """
        # --- guards ---
        _assert_axis_ready(self.axis_mode, self.sw_hz, self.ref)
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length for residual computation.")



        # --- rebuild peaks from params (in Hz) ---
        peaks: List[Peak] = []
        for i in range(no_of_peaks):
            peaks.append(Peak(
                pos=params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="pos"))}'].value, #Hz
                amp=params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="amp"))}'].value,
                lor_hz=params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="lor"))}'].value, #Hz
                gauss_disp=params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'].value, #Hz
            ))

        x_disp = np.asarray(self.x, dtype=float)
        x_hz = ppm_to_hz(x_disp,float(self.ref)) if self.axis_mode == 'ppm' else x_disp

        # --- complex model spectrum (pure math; no phasing/offset here) ---
        spec_c = model_spectrum(
            peaks=peaks,
            axis_mode='hz',
            ref=self.ref,
            sw_hz=self.sw_hz,
            N=len(self.x),
            x=x_hz,                         # pass current display axis; orchestrator canonicalizes
            multiplier=params[f'{ParamRef_to_key(global_ref(sid, "mult"))}'].value,
            return_fid=False,
        )

        # --- apply ONLY zero-order phase and offset ---
        phi0   = params.get(f'{ParamRef_to_key(global_ref(sid, "phi0_deg"))}', 0.0).value
        offset = params.get(f'{ParamRef_to_key(global_ref(sid, "offset"))}',   0.0).value
        model = apply_phase_and_offset(spec_c, phi0, offset)

        # --- residual on display grid ---
        res = (self.y - model)

        # === apply GUI mask (1 outside, 0 inside excluded ranges) ===
        w = getattr(self, 'mask_w', None)
        if w is not None:
            res = res * w

        # === robust normalization (scale by MAD/STD of unmasked y) ===
        y_for_sigma = self.y if w is None else (self.y[w > 0] if np.any(w > 0) else self.y)

        if y_for_sigma.size >= 3:
            med = float(np.median(y_for_sigma))
            mad = float(np.median(np.abs(y_for_sigma - med)))
            sigma = 1.4826 * mad if mad > 0 else (float(np.std(y_for_sigma)) or 1.0)
        else:
            sigma = float(np.std(y_for_sigma)) or 1.0

        # ---- DEBUG LOGGING (1st, 10th, every 10th) ----
        if DEBUG_LOGGING:
            if not hasattr(self, "_dbg_call_count"):
                self._dbg_call_count = 0
            c = self._dbg_call_count

            x_disp = np.asarray(self.x, dtype=float)
            x_hz = x_disp * float(self.ref) if self.axis_mode == 'ppm' else x_disp
            sw_grid_hz = float((abs(x_hz[-1] - x_hz[0]))*len(self.x) / (len(self.x)-1)) if x_hz.size > 1 else float("nan")

            should_log = (c == 0) or (c == 9) or (c % 10 == 9)
            if should_log:
                log.info(
                    "RESIDUAL call #%d | pre='%s' | MHz=%.6f | sw_display=%.3f Hz | sw_hz(ctx)=%s | mult=%.3f | phi0=%.6g deg | %s | %s",
                    c, float(self.ref), sw_grid_hz,

                    _arr_summary("x_disp", x_disp),
                    _arr_summary("x_hz",   x_hz),
                )
            self._dbg_call_count = c + 1
            # ---- end DEBUG LOGGING ----

        return res / sigma
    
def apply_bounds_to_param(par: Parameter, model, slice_id: int, peak_idx: int,
                          pref_name: str, axis_mode: str, ref_mhz: float):
    """Look up and apply bounds for this parameter."""
    if not model:
        return
    try:
        pref = ParamRef(slice_id=slice_id, peak_id=peak_idx, name=pref_name)
        b = model.get_bounds_for(pref)
        if not b or (b.lo is None and b.hi is None):
            return

        lo_i, hi_i = Bounds_to_BoundsHz(b.lo, b.hi, pref_name, axis_mode, ref_mhz)
        if lo_i is not None:
            par.min = float(lo_i)
        if hi_i is not None:
            par.max = float(hi_i)
    except Exception:
        # keep optimizer robust if UI/model is unavailable
        pass

def _T_bounds_to_k_bounds(T_lo, T_hi):
    """Map T-bounds (seconds) → k-bounds (s^-1). Allows None on either side."""
    def inv(x): 
        try:
            return None if (x is None or float(x) <= 0.0) else 1.0/float(x)
        except Exception:
            return None
    # flip: k ∈ [1/T_hi, 1/T_lo]
    k_lo = inv(T_hi)
    k_hi = inv(T_lo)
    # keep order sane if both present and inverted
    if k_lo is not None and k_hi is not None and k_lo > k_hi:
        k_lo, k_hi = k_hi, k_lo
    return k_lo, k_hi

def apply_Tbounds_to_param(par, lo: float | None, hi: float | None) -> None:

    """Apply numeric bounds directly to an lmfit Parameter (no unit conversion)."""
    if lo is not None:
        par.min = float(lo)
    if hi is not None:
        par.max = float(hi)

def _get_seed(self, T_name: str):
    rec = None
    # prefer MainWindow-level registry
    reg1 = getattr(self, "_TSeedRegistry", None)
    if isinstance(reg1, dict):
        rec = reg1.get(T_name)
    if rec is None:
        # optional model-level fallback
        mdl = getattr(self, "table_model", None)
        reg2 = getattr(mdl, "_tseed_registry", None) if mdl is not None else None
        if isinstance(reg2, dict):
            rec = reg2.get(T_name)
    return rec  # expected keys: {'fixed': Bool, 'T_seed_s': float, 'T_result_s': None}
# --------------------------- Peak table model ---------------------------

COLUMN_NAMES = ["Pos", "Amp", "Lorentz", "Gauss"]

class PeakTableModel(QtCore.QAbstractTableModel):
    COLS = ["Pos", "Fix Pos", "Amp", "Fix Amp", "Lor", "Fix Lor", "Gau", "Fix Gau"]

    def __init__(self, peaks=None, fix_flags=None, redchi=None, parent=None):
        super().__init__(parent)
        self._peaks = peaks if peaks is not None else []                  # List[Peak]
        self._fix   = fix_flags if fix_flags is not None else []          # List[Tuple[bool,bool,bool,bool]]
        self._redchi = redchi if redchi is not None else float("nan")     # float
        self._links = None  # LinkStore
        self._slice_index_provider = None  # callable -> int
        # map visible columns to param names
        self._col_to_name = {0: "pos", 2: "amp", 4: "lor", 6: "gauss"}
        self._bounds: dict[ParamRef, ParamBounds] = {}   # per-target bounds (pos/amp/lor/gau)


    # ---- Public API ----
    def bind_state(self, state):  # state: SliceFitState object
        """Re-bind the model to a different slice state (no copy). Called when changing/updating slice states"""
        self.beginResetModel()
        self._peaks = state.peaks
        self._fix   = state.fix_flags
        self._redchi = state.redchi
        self.endResetModel()

    def replace_data(self, peaks, fix_flags, redchi):
        """Same semantics as bind_state but with explicit lists. Called when information comes from import, fit results, or peak picking"""
        self.beginResetModel()
        self._peaks = peaks
        self._fix   = fix_flags
        self._redchi = redchi
        self.endResetModel()

    def add_row(self, pk: Peak | None = None,
                 fix_flags: tuple[bool,bool,bool,bool] | None = None,
                 default_fix: dict[str,bool] | None = None):
        """
        Add a new peak row.

        If pk is None → create a blank Peak(0,0,...).
        If pk is given → append that existing Peak instance.
        """
        if pk is None:
            pk = Peak(pos=0.0, amp=0.0, lor_hz=0.2, gauss_disp=0.1)
        if fix_flags is None:
            if default_fix is None:
                default_fix = dict(pos=False, amp=False, lor=True, gauss=False)
            fix_flags = (default_fix["pos"], default_fix["amp"],
                         default_fix["lor"], default_fix["gauss"])

        r = len(self._peaks)
        self.beginInsertRows(QtCore.QModelIndex(), r, r)
        self._peaks.append(pk)
        self._fix.append(tuple(fix_flags))
        self.endInsertRows()

    def remove_rows(self, rows):
        if not rows: return
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self._peaks):
                self.beginRemoveRows(QtCore.QModelIndex(), r, r)
                del self._peaks[r]
                del self._fix[r]
                self.endRemoveRows()
    def return_model(self):
        """
        Adapter for controllers: return a shallow copy of peaks + fix flags
        so compute paths can read domain objects without touching the View.
        """
        return list(self._peaks), list(self._fix), self._redchi
    
    # Translate a cell into ParamRef
    def index_to_paramref(self, index):
        if (not index.isValid()) or (index.column() not in self._col_to_name):
            return None
        r, c = index.row(), index.column()
        if r < 0 or r >= len(self._peaks):
            return None
        # determine peak_id and slice_id
        peak_id = r
        slice_id = self._slice_index_provider() if self._slice_index_provider else 0
        name = self._col_to_name[c] # "pos" | "amp" | "lor" | "gauss"
        # ParamRef(slice, peak, name) is expected to exist in this module
        return ParamRef(slice_id, peak_id, name)

    # ---- Qt basics ----
    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._peaks)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.COLS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole: return None
        if orientation == QtCore.Qt.Horizontal: return self.COLS[section]
        return str(section)
    
        # ---- Data / Edit ----  
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r, c = index.row(), index.column()
        pk = self._peaks[r]

        # --- fix (checkbox) columns ---
        if self._is_fix_col(c):
            if role == QtCore.Qt.CheckStateRole:
                fi = self._fix_index_for_col(c)
                return QtCore.Qt.Checked if self._fix[r][fi] else QtCore.Qt.Unchecked
            if role == QtCore.Qt.DisplayRole:
                return ""
            if role == QtCore.Qt.TextAlignmentRole:
                return QtCore.Qt.AlignCenter
            return None

        # map columns → values
        if c == 0: val = float(pk.pos)
        elif c == 2: val = float(pk.amp)
        elif c == 4: val = float(pk.lor_hz)
        elif c == 6: val = float(pk.gauss_disp)
        else: val = None

        # --- numeric columns ---
        if role == QtCore.Qt.DisplayRole:
            return self._fmt(val)
        if role == QtCore.Qt.EditRole:
            return float(val)             # keep as float → default editor (spinbox)
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter

        # Show link hint tooltip
        if role == QtCore.Qt.ToolTipRole and self._links and (c in self._col_to_name):
            tips = []
            # link tip
            if self._links:
                pref = self.index_to_paramref(index)
                if pref and self._links.is_linked(pref):
                    expr = self._links.get(pref)
                    if expr and expr.type == LinkType.LINEAR and expr.driver:
                        tips.append(f"Linked: {pref.name} ← a*driver + b")
                    elif expr and expr.type == LinkType.RELAX_EXP:
                        tips.append("Linked: decay y=A*exp(-t/T)+C")
            # bounds tip
            pref = self.index_to_paramref(index)
            if pref:
                b = self.get_bounds_for(pref)
                if b.is_set():
                    lo = "" if b.lo is None else f"{b.lo:.6g}"
                    hi = "" if b.hi is None else f"{b.hi:.6g}"
                    tips.append(f"Bounds: [{lo}, {hi}]")
            return " | ".join(tips) if tips else None
         
        if role == QtCore.Qt.ForegroundRole and self._links and (c in self._col_to_name):
            pref = self.index_to_paramref(index)
            if pref and self._links.is_linked(pref):
                return QtGui.QBrush(QtGui.QColor("#827F7F"))

        return None   

    # Edit policy
    def flags(self, index):
        base = super().flags(index)
        if not index.isValid():
            return base
        c = index.column()
        if self._is_fix_col(c):
            return base | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
        if self._links and (c in self._col_to_name):
            pref = self.index_to_paramref(index)
            if pref and self._links.is_linked(pref):
                return (base | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable) & ~QtCore.Qt.ItemIsEditable
        return base | QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid(): return False
        r, c = index.row(), index.column()

        if self._is_fix_col(c):
            if role != QtCore.Qt.CheckStateRole: return False
            fi = self._fix_index_for_col(c)
            cur = self._fix[r]
            new = list(cur); new[fi] = (value == QtCore.Qt.Checked)
            if tuple(new) != cur:
                self._fix[r] = tuple(new)
                self.dataChanged.emit(index, index, [QtCore.Qt.CheckStateRole])
            return True

        if role == QtCore.Qt.EditRole:
            try: v = float(value)
            except Exception: return False
            pk = self._peaks[r]
            if c == 0: pk.pos = v
            elif c == 2: pk.amp = v
            elif c == 4: pk.lor_hz = v
            elif c == 6: pk.gauss_disp = v
            else: return False
            self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
            return True

        return False

    # ---- Column helpers ----
    @staticmethod
    def _is_fix_col(col): return col in (1, 3, 5, 7)
    @staticmethod
    def _fix_index_for_col(col):
        # map fix columns → index in (pos, amp, lor, gauss)
        return {1:0, 3:1, 5:2, 7:3}.get(col, None)

    # ---- Header checkbox helpers (for master fix in header) ----
    def column_fix_tristate(self, section: int) -> QtCore.Qt.CheckState:
        fi = self._fix_index_for_col(section)
        if fi is None or len(self._fix) == 0: return QtCore.Qt.Unchecked
        checked = sum(1 for t in self._fix if t[fi])
        if checked == 0: return QtCore.Qt.Unchecked
        if checked == len(self._fix): return QtCore.Qt.Checked
        return QtCore.Qt.PartiallyChecked

    def set_all_fix_in_column(self, section: int, checked: bool) -> None:
        fi = self._fix_index_for_col(section)
        if fi is None or len(self._fix) == 0: return
        changed = False
        newv = bool(checked)
        for i, t in enumerate(self._fix):
            if t[fi] != newv:
                lst = list(t); lst[fi] = newv; self._fix[i] = tuple(lst)
                changed = True
        if changed:
            tl = self.index(0, section)
            br = self.index(len(self._fix)-1, section)
            self.dataChanged.emit(tl, br, [QtCore.Qt.CheckStateRole])

    # ---- Format ----
    def _fmt(self, v: float) -> str:
        if v is None or not math.isfinite(v):
            return "—"
        v = float(v)
        # Switch to scientific for large or tiny values
        if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"   # 3 digits after decimal, scientific notation
        else:
            return f"{v:.3f}"   # 3 digits after decimal, fixed notation
   
    # Bind LinkStore and a provider for current slice index to link parameters
    def bind_links(self, store, *, slice_index_provider=None):
        self._links = store
        self._slice_index_provider = slice_index_provider
        self.layoutChanged.emit()
    # Set bound for parameters
    def get_bounds_for(self, pref: ParamRef) -> ParamBounds:
        return self._bounds.get(pref, ParamBounds())
    
    def set_bounds_for(self, pref: ParamRef, lo: float | None, hi: float | None):
        if lo is None and hi is None:
            self._bounds.pop(pref, None)
        else:
            self._bounds[pref] = ParamBounds(lo, hi)
    
    def clear_bounds_for(self, pref: ParamRef):
        self._bounds.pop(pref, None)
    

class ScientificDoubleDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None, *, decimals=12, bottom=None, top=None):
        super().__init__(parent)
        self.decimals = decimals
        self.bottom = bottom
        self.top = top

    def createEditor(self, parent, option, index):
        ed = QtWidgets.QLineEdit(parent)
        if self.bottom is None or self.top is None:
            val = QtGui.QDoubleValidator(parent)
        else:
            val = QtGui.QDoubleValidator(self.bottom, self.top, self.decimals, parent)
        val.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        val.setDecimals(self.decimals)
        ed.setValidator(val)
        ed.setAlignment(QtCore.Qt.AlignRight)
        return ed

    def setEditorData(self, editor, index):
        # pull model's EditRole (float), format compactly for editing
        v = index.model().data(index, QtCore.Qt.EditRole)
        try:
            editor.setText(f"{float(v):.12g}")
        except Exception:
            editor.setText("")

    def setModelData(self, editor, model, index):
        txt = editor.text().strip()
        try:
            model.setData(index, float(txt), QtCore.Qt.EditRole)
        except Exception:
            pass


class CheckableHeader(QtWidgets.QHeaderView):
    """
    Horizontal header with checkboxes embedded in exprific sections (columns).
    - Paints tri-state checkboxes aligned with header text.
    - Click on the checkbox toggles the entire column.
    """
    toggled = QtCore.pyqtSignal(int, bool)  # section, checked

    def __init__(self, orientation, parent=None, model=None, fix_sections=None):
        super().__init__(orientation, parent)
        self._model = model          # PeakTableModel8
        self._fix_sections = set(fix_sections or [])
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self._checkbox_margins = 6   # px padding inside header cell
        self.setStyleSheet("QHeaderView::section { padding-right: 22px; }")

    def setModelRef(self, model):
        self._model = model
        self.update()

    def setFixSections(self, sections):
        self._fix_sections = set(sections or [])
        self.update()

    
    
    
    def paintSection(self, painter, rect, logicalIndex):
        super().paintSection(painter, rect, logicalIndex)

        if logicalIndex not in self._fix_sections or self._model is None:
            return

        # Determine checkbox state from model
        state = self._model.column_fix_tristate(logicalIndex)

        # Compute checkbox rect robustly
        opt = QtWidgets.QStyleOptionButton()
        opt.state = QtWidgets.QStyle.State_Enabled
        if state == QtCore.Qt.Checked:
            opt.state |= QtWidgets.QStyle.State_On
        elif state == QtCore.Qt.PartiallyChecked:
            opt.state |= QtWidgets.QStyle.State_NoChange
        else:
            opt.state |= QtWidgets.QStyle.State_Off
        # size checkbox to ~60% of header height, square (better on HiDPI)
        base_rect = self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxIndicator, opt, None)
        s = int(min(rect.height() * 0.60, base_rect.width()))
        x = rect.right() - s - self._checkbox_margins
        y = rect.center().y() - s // 2
        opt.rect = QtCore.QRect(x, y, s, s)

        painter.save()
        self.style().drawControl(QtWidgets.QStyle.CE_CheckBox, opt, painter)
        painter.restore()

        setattr(self, f"_cb_rect_{logicalIndex}", opt.rect)


    def mousePressEvent(self, event):
        idx = self.logicalIndexAt(event.pos())
        if idx in self._fix_sections:
            cb_rect = getattr(self, f"_cb_rect_{idx}", None)
            if cb_rect and cb_rect.contains(event.pos()):
                # Toggle target state: Checked if not fully checked, else Unchecked
                current = self._model.column_fix_tristate(idx)
                new_checked = (current != QtCore.Qt.Checked)
                self.toggled.emit(idx, new_checked)
                return  # don’t pass to super to avoid sorting etc.
        super().mousePressEvent(event)

    # Ensure header repaints when model changes
    def sectionDataChanged(self, logicalFirst, logicalLast):
        super().sectionDataChanged(logicalFirst, logicalLast)
        self.update()

    def sectionResized(self, logicalIndex, oldSize, newSize):
        super().sectionResized(logicalIndex, oldSize, newSize)
        attr = f"_cb_rect_{logicalIndex}"
        if hasattr(self, attr):
            delattr(self, attr)
        self.updateSection(logicalIndex)

@dataclass
class ParamBounds:
    lo: float | None = None
    hi: float | None = None
    def is_set(self) -> bool:
        return (self.lo is not None) or (self.hi is not None)
    
class BoundsDialog(QtWidgets.QDialog):
    def __init__(self, parent, title="Set Bounds"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._lo = None; self._hi = None
        lay = QtWidgets.QFormLayout(self)

        self.ed_lo = QtWidgets.QLineEdit(self); self.ed_hi = QtWidgets.QLineEdit(self)
        v_lo = QtGui.QDoubleValidator(self); v_lo.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        v_hi = QtGui.QDoubleValidator(self); v_hi.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        self.ed_lo.setValidator(v_lo); self.ed_hi.setValidator(v_hi)
        self.ed_lo.setPlaceholderText("leave blank = no lower bound")
        self.ed_hi.setPlaceholderText("leave blank = no upper bound")

        lay.addRow("Lower bound:", self.ed_lo)
        lay.addRow("Upper bound:", self.ed_hi)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok); btns.rejected.connect(self.reject)
        lay.addRow(btns)

    def preload(self, lo: float | None, hi: float | None):
        if lo is not None: self.ed_lo.setText(f"{lo:.12g}")
        if hi is not None: self.ed_hi.setText(f"{hi:.12g}")

    def _on_ok(self):
        def parse(txt):
            t = txt.strip()
            return None if not t else float(t)
        try:
            self._lo = parse(self.ed_lo.text()); self._hi = parse(self.ed_hi.text())
            if (self._lo is not None) and (self._hi is not None) and (self._lo > self._hi):
                QtWidgets.QMessageBox.warning(self, "Bounds", "Lower bound > Upper bound.")
                return
            self.accept()
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Bounds", "Invalid number.")
    def result(self): return (self._lo, self._hi)

    
class TSeedTableModel(QtCore.QAbstractTableModel):
    COLS = ["T_name", "T_seed (s)", "Fix", "Lo (s)", "Hi (s)", "T_result (s)"]
    def __init__(self, registry: dict[str, dict], parent=None):
        super().__init__(parent)
        self._reg = registry
        self._rows = sorted(self._reg.keys())

    def refresh(self):
        self.beginResetModel()
        self._rows = sorted(self._reg.keys())
        self.endResetModel()

    def rowCount(self, parent=None): return len(self._rows)
    def columnCount(self, parent=None): return 6
    def headerData(self, section, orient, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orient == QtCore.Qt.Horizontal:
            return self.COLS[section]
        return None

    def flags(self, index):
        if not index.isValid(): return QtCore.Qt.NoItemFlags
        r, c = index.row(), index.column()
        base = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        # T_name, T_result are read-only; T_seed and Fix editable
        if c in (1, 3, 4):
            base |= QtCore.Qt.ItemIsEditable
        # make Fix a checkbox
        if c == 2:
            base |= QtCore.Qt.ItemIsUserCheckable
        return base

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        name = self._rows[index.row()]
        rec = self._reg.get(name, {})
        c = index.column()

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            if c == 0:  # T_name
                return name
            if c == 1:  # T_seed (s)
                v = rec.get("T_seed_s", None)
                return "" if v is None else (f"{float(v):.12g}" if role == QtCore.Qt.DisplayRole else float(v))
            
            if c == 3:  # Lo(s)
                v = rec.get("T_lo_s", None)
                return "" if v is None else (f"{float(v):.12g}" if role == QtCore.Qt.DisplayRole else float(v))
            if c == 4:  # Hi(s)
                v = rec.get("T_hi_s", None)
                return "" if v is None else (f"{float(v):.12g}" if role == QtCore.Qt.DisplayRole else float(v))
            
            if c == 5:  # T_result (s)
                v = rec.get("T_result_s", None)
                return "" if v is None else f"{float(v):.12g}"
        
        if c == 2 and role == QtCore.Qt.CheckStateRole:
            return QtCore.Qt.Checked if bool(rec.get("fixed", False)) else QtCore.Qt.Unchecked
        return None

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid(): return False
        name = self._rows[index.row()]
        rec = self._reg.setdefault(name, {"fixed": False, "T_seed_s": None, "T_result_s": None})
        c = index.column()

        if c in (1, 3, 4) and role == QtCore.Qt.EditRole:
            try:
                v = float(value)
                if v <= 0: return False
                if c == 1: rec["T_seed_s"] = float(v)
                elif c == 3: rec["T_lo_s"] = float(v)
                else: rec["T_hi_s"] = v
                self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
                return True
            except Exception:
                return False

        if c == 2 and role == QtCore.Qt.CheckStateRole:
            rec["fixed"] = (value == QtCore.Qt.Checked)
            self.dataChanged.emit(index, index, [QtCore.Qt.CheckStateRole, QtCore.Qt.DisplayRole])
            return True
        return False


class TSeedTableDialog(QtWidgets.QDialog):
    """Minimal T seeds table."""
    def __init__(self, parent, registry: dict[str, dict]):
        super().__init__(parent)
        self.setWindowTitle("T seeds")
        self.resize(520, 360)
        self._registry = registry
        lay = QtWidgets.QVBoxLayout(self)
        self.tbl = QtWidgets.QTableView(self)
        self.model = TSeedTableModel(self._registry, self)
        self.tbl.setModel(self.model)
        # ---- table styling to match PeakTableView ----
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.horizontalHeader().setSectionsClickable(False)
        self.tbl.setShowGrid(False)

        # scientific editor for "T_seed (s)" (col 1)
        self.tbl.setItemDelegateForColumn(
            1, ScientificDoubleDelegate(self.tbl, decimals=6)
        )

        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.tbl, 1)
        self.tbl.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject); btns.accepted.connect(self.accept)
        self.tbl.customContextMenuRequested.connect(self._on_tseed_menu)
        lay.addWidget(btns)

    def ensure_row(self, T_name: str):
        if T_name not in self._registry:
            self._registry[T_name] = {"fixed": False, "T_seed_s": None, "T_result_s": None}
            self.model.refresh()

    def _on_tseed_menu(self, pos: QtCore.QPoint):
        idx = self.tbl.indexAt(pos)
        if not idx.isValid(): return
        r = idx.row()
        name = self.model._rows[r]
        rec = self.model._reg.get(name, {})
        menu = QtWidgets.QMenu(self.tbl)
        act_bounds = menu.addAction("Set T bounds…")
        act_clear = menu.addAction("Clear T bounds")
        chosen = menu.exec_(self.tbl.viewport().mapToGlobal(pos))
        if chosen == act_bounds:
            dlg = BoundsDialog(self, title=f"Set Bounds for {name} (seconds)")
            dlg.preload(rec.get("T_lo_s"), rec.get("T_hi_s"))
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                lo, hi = dlg.result()
                if lo is not None: rec["T_lo_s"] = float(lo)
                else: rec.pop("T_lo_s", None)
                if hi is not None: rec["T_hi_s"] = float(hi)
                else: rec.pop("T_hi_s", None)
                self.model.dataChanged.emit(self.model.index(r, 3), self.model.index(r, 4),
                                            [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        elif chosen == act_clear:
            rec.pop("T_lo_s", None); rec.pop("T_hi_s", None)
            self.model.dataChanged.emit(self.model.index(r, 3), self.model.index(r, 4),
                                        [QtCore.Qt.DisplayRole])
    





# ------------------------------ Helper ------------------------------
def load_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal 2-column ASCII loader.

    """
    arr = np.loadtxt(path, dtype=float, skiprows=1, delimiter=',')
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError('Expected a 2-column text file: x y')
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    return x, y

def _replace_peaks_from_dicts(self, rows):
    """
    Replace self.peaks list from imported dicts.
    - importer gives gauss_hz in Hz → convert to display unit if ppm
    """
    self.peaks.clear()
    for r in rows:
        gauss_hz = float(r["gauss_hz"])
        if self.axis_mode == 'ppm':
            gauss_disp = hz_to_ppm(gauss_hz, self.ref)
        else:
            gauss_disp = gauss_hz
        self.peaks.append(Peak(
            pos=float(r["pos"]),
            amp=float(r["amp"]),
            lor_hz=float(r["lor_hz"]),
            gauss_disp=gauss_disp,
        ))
def add_peak_to_PeakTableModel(self,
             pk: Peak | None = None,
             fix_flags: tuple[bool, bool, bool, bool] | None = None,
             *, select_row: bool = True, refresh: bool = True) -> None:
    """
    Single source of truth for adding a peak.
    - pk=None creates a blank with sane widths from current grid.
    - fix_flags defaults to self._default_fix mapping.
    - Updates model, selection, UI state; marks current slice state dirty.
    """
    # Default peak guess if none given
    if pk is None:
        # guard for first call before x exists
        if self.x is not None and len(self.x) >= 2:
            step = abs(self.x[1] - self.x[0])
        else:
            step = 1.0
        lor0_hz  = max(0.32768 * self._grid_hz(), 0.0)
        ga0_disp = max(5 * step, 1e-6)
        pk = Peak(pos=0.0, amp=0.0, lor_hz=float(lor0_hz), gauss_disp=float(ga0_disp))

    # Default fix flags
    if fix_flags is None:
        d = getattr(self, "_default_fix", dict(pos=False, amp=False, lor=True, gauss=False))
        fix_flags = (d["pos"], d["amp"], d["lor"], d["gauss"])

    # Push to model
    self.peak_model.add_row(pk=pk, fix_flags=fix_flags, default_fix=self._default_fix)

    # Keep domain cache in sync for callers that read self.peaks/self.fix_flags()
    self.Model_to_Peaks()

    # Mark current slice state dirty (don’t rely on any missing _invalidate_slice_cache)
    st = self.slice_states.get(int(getattr(self, "slice_index", 0)))
    if st:
        st.dirty_model = True

    # UX niceties
    if select_row:
        try:
            self.tbl.selectRow(self.peak_model.rowCount() - 1)
        except Exception:
            pass

    self.update_ui_state()
    if refresh:
        self.refresh_plot(preserve_view=True)

# --------------------------- Matplotlib canvas -----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()


# ---------------------------- Main GUI window ------------------------------
class  MainWindow(QMainWindow):
    @property
    def peaks(self): 
        return self.slice_states[self.slice_index].peaks
    @peaks.setter
    def peaks(self, v): 
        self.slice_states[self.slice_index].peaks = v
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- initialize single-slice SSOT before GUI build ---

        self.slice_index = 0 # int
        self.slice_states = {} # : dict[int, SliceFitState()]
        self.slice_states[0] = SliceFitState()
        
        self.link_store = LinkStore()
        # ---- T-seed registry (authoritative) ----
        # schema: {T_name: {"fixed": bool, "T_seed_s": float|None, "T_result_s": float|None}
        self._TSeedRegistry: dict[str, dict] = {}

        
        self._buildMainWindow() 
        self._connect_signals()
        self._last_slice_selection = None
        self.update_ui_state()
        self._ui_busy = 0
        
        # A5: dataset-scope link store
        

    def _buildMainWindow(self):
        self.setWindowTitle('NMR Peak Deconvolution (FID→FFT) – lmfit')
        self.resize(1100, 720)
        # Data state
        self.x = None # Optional[np.ndarray] 
        self.y = None # Optional[np.ndarray] 
        self.axis_mode = 'ppm'  # str ppm or 'hz'
        self.ref = None #: Optional[float] 
        self.sw_hz = None #: Optional[float]
        self.multiplier = 1.0 #: float
        self.offset = 0.0 #: float 
        self.phi0_deg = 0.0 # float
        self.fix_mult = True
        self.fix_phi0 = True
        self.fix_offset = True
        self._in_bulk_import = False  # guard: avoid auto-normalizing while importing
        # Backing data for the CURRENT slice (list of dicts).

        # ---- 2D dataset state ----
        self.data2d = None      # shape (N, F) or None. Optional[np.ndarray]
        self.f2_ppm = None      # (F,). Optional[np.ndarray]
        self.f2_hz = None      # (F,). Optional[np.ndarray]
        self.t_f1 = None      # (N,) or None. Optional[np.ndarray]
        self.meta = None # Optional[dict]

        # Bound add_peak function to an object
        self.add_peak = MethodType(add_peak_to_PeakTableModel, self)

        # Place holder for fix flag
        self._fix_flags_cache = None

        # Two-step add-peak in click-on plot
        self._add_waiting_width = False
        self._pending_pos = None
        self._pending_height = None

        # Plotting
        self._peak_lines = []
        self._diff_line = None
        self._model_line = None
        self._data_line = None
        
        # defaults for new export/import flow
        self.dataset_label = ""                    # set when opening a file
        self.data_file = ""                        # : str
        self.default_save_dir = str(Path.home())   # where to start Save dialog
        self.default_open_dir = str(Path.home())   # where to start Open dialog
        self.program_version = "mpFit"
        
        self.fit_stats = None  # fill after fit
        self.lmfit_fit_report: str | None = None  # last lmfit.fit_report text
        
        # Widgets
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        self.status = self.statusBar()
        self.status.showMessage("Ready")
        layout = QtWidgets.QHBoxLayout(central)
        # Menu bar
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu("File")
        self.open_act = QAction("Open", self)
        self.open_act.setShortcut("Ctrl+O")  
        self.time_act = QAction("Load relaxation time", self)
        self.time_act.setShortcut("Ctrl+T")
        self.time_act.setEnabled(False)
        self.save_act = QAction("Save", self)
        self.save_act.setShortcut("Ctrl+S")
        self.export_act = QAction("Export peak table", self)
        self.export_act.setShortcut("Ctrl+E")
        self.import_act = QAction("Import peak table", self)
        self.import_act.setShortcut("Ctrl+I")
        self.file_menu.addAction(self.open_act)
        self.file_menu.addAction(self.time_act)
        self.file_menu.addAction(self.save_act)
        self.file_menu.addAction(self.export_act)
        self.file_menu.addAction(self.import_act)
        # Left: plot
        left = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas(self)
        left.addWidget(self.canvas)
        self.nav = NavigationToolbar(self.canvas, self)
        left.addWidget(self.nav)

        # Controls under plot
        ctrls_top = QtWidgets.QHBoxLayout()
        ctrls_mid = QtWidgets.QHBoxLayout()
        ctrls_bot = QtWidgets.QHBoxLayout()
        ctrls = QtWidgets.QVBoxLayout()
        
        # Controls under plot / Top row
        self.btn_auto = QtWidgets.QPushButton('Auto-Pick')
        self.btn_clear = QtWidgets.QPushButton('Clear Peaks')
        self.tgl_add = QtWidgets.QPushButton('Add Peaks Mode')
        self.tgl_add.setCheckable(True)
        self.tgl_add.setChecked(False)
        
        self.btn_fit = QToolButton()
        self.btn_fit.setText("Fit")
        self.btn_fit.setPopupMode(QToolButton.InstantPopup)        
        self.act_fit = QAction("Fit current slice", self, checkable=True)
        self.act_fit.setShortcut("Ctrl+F")
        self.act_fit.setStatusTip("Fit the current slice.")
        self.act_fit_selected = QAction("Fit selected", self, checkable=True)
        self.act_fit_selected.setShortcut("Shift+F")
        self.act_fit_selected.setStatusTip("Pick slices to fit.")        
        menu_fit = QMenu(self)
        menu_fit.addAction(self.act_fit)
        menu_fit.addAction(self.act_fit_selected)
        self.btn_fit.setMenu(menu_fit)
        
        self.btn_sim = QtWidgets.QPushButton('Simulate')
        self.btn_copy = QtWidgets.QPushButton('Copy Params')

        ctrls_top.addWidget(self.btn_auto)
        ctrls_top.addWidget(self.btn_clear)
        ctrls_top.addWidget(self.tgl_add)
        ctrls_top.addWidget(self.btn_copy)
        ctrls_top.addWidget(self.btn_sim)
        ctrls_top.addWidget(self.btn_fit)

        # Controls under plot / Mid row
        self.excluded = []  # list[tuple(lo, hi)] in display units (ppm or Hz)
        self.btn_excluded = QToolButton()
        self.btn_excluded.setText("Excluded")
        self.btn_excluded.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        self.act_drag = QAction("Drag on plot", self, checkable=True)
        self.act_add  = QAction("Add range…", self)
        self.act_rm   = QAction("Remove Selected", self)
        self.act_clr  = QAction("Clear all", self)
        menu.addAction(self.act_drag)
        menu.addSeparator()
        menu.addAction(self.act_add)
        menu.addAction(self.act_rm)
        menu.addAction(self.act_clr)
        self.btn_excluded.setMenu(menu)
        ctrls_mid.addWidget(self.btn_excluded)
        self.btn_link_manager = QtWidgets.QPushButton("Link Manager")
        ctrls_mid.addWidget(self.btn_link_manager)
        self.btn_tseed = QtWidgets.QPushButton("T seeds")
        self.btn_copy_bounds = QtWidgets.QPushButton("Copy bounds")
        ctrls_mid.addWidget(self.btn_tseed)
        ctrls_mid.addWidget(self.btn_copy_bounds)


        # ==== Additional buttons / bottom row ====  

        self.lbl_redchi = QtWidgets.QLabel("Reduced χ²: —")
        self.btn_stat = QtWidgets.QPushButton('Stat') 
        self.chk_2d_mode = QtWidgets.QCheckBox("2D mode")
        self.chk_2d_mode.setChecked(False)
        self.chk_2d_mode.toggled.connect(self.on_toggle_2d_mode)
        self.chk_relax = QtWidgets.QCheckBox("Relaxation mode")
        self.chk_relax.setChecked(False)
        self.chk_relax.toggled.connect(self.on_toggle_relax)
        self.lbl_slice = QtWidgets.QLabel("Trace: —")
        self.spn_slice = QtWidgets.QSpinBox()
        self.spn_slice.setMinimum(0); self.spn_slice.setMaximum(0); self.spn_slice.setValue(0)
        self.spn_slice.setEnabled(False)
        self.spn_slice.valueChanged.connect(self.on_slice_spin_changed)
        self.lbl_slice_caption = QtWidgets.QLabel("Slice:")
        ctrls_bot.addWidget(self.chk_2d_mode)
        ctrls_bot.addWidget(self.chk_relax)
        ctrls_bot.addWidget(self.lbl_slice_caption)
        ctrls_bot.addWidget(self.spn_slice)
        ctrls_bot.addWidget(self.lbl_slice)
        ctrls_bot.addWidget(self.lbl_redchi)
        ctrls_bot.addWidget(self.btn_stat)   
        ctrls.addLayout(ctrls_top)
        ctrls.addLayout(ctrls_mid)
        ctrls.addLayout(ctrls_bot)
        left.addLayout(ctrls)
        
        # Right: settings + table
        right = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        fixes = QtWidgets.QHBoxLayout()
        self.cmb_axis = QtWidgets.QComboBox()
        self.cmb_axis.addItems(['ppm', 'hz'])
        self.cmb_axis.setCurrentText('ppm')
        self.cmb_axis.currentTextChanged.connect(self.on_axis_mode_changed)
        self.edt_ref = QtWidgets.QLineEdit()
        self.edt_ref.setValidator(QDoubleValidator(0.0, 1500.0, 9, self.edt_ref))
        self.edt_ref.setText("0.0")  # MHz
        self.edt_ref.setPlaceholderText("MHz - to be loaded with correct reference frequency before running simulation/fitting")
        self.edt_sw = QtWidgets.QLineEdit()
        self.edt_sw.setValidator(QDoubleValidator(0.0, 5_000_000.0, 6, self.edt_sw))
        self.edt_sw.setText("0.0")  # 0 = auto
        self.edt_sw.setPlaceholderText("Hz (0 = auto)")
        self.edt_mult = QtWidgets.QLineEdit()
        self.edt_mult.setValidator(QDoubleValidator(0.0, 1e12, 3, self.edt_mult))
        self.edt_mult.setText("1.0")
        self.edt_offset = QtWidgets.QLineEdit()
        # allow negatives: set bottom < top then flip; PyQt’s QDoubleValidator needs bottom <= top
        v_off = QDoubleValidator(-1e12, 1e12, 9, self.edt_offset)
        v_off.setNotation(QDoubleValidator.StandardNotation)
        self.edt_offset.setValidator(v_off)
        self.edt_offset.setText("0.0")
        self.spn_phi0 = QtWidgets.QDoubleSpinBox()
        self.spn_phi0.setDecimals(2); self.spn_phi0.setRange(-180.0, 180.0); self.spn_phi0.setValue(0.0); self.spn_phi0.setSuffix(' °')
        self.chk_fix_mult   = QtWidgets.QCheckBox('Fix Multiplier');  self.chk_fix_mult.setChecked(True)
        self.chk_fix_offset = QtWidgets.QCheckBox('Fix Offset');      self.chk_fix_offset.setChecked(True)
        self.chk_fix_phi0   = QtWidgets.QCheckBox('Fix φ0');          self.chk_fix_phi0.setChecked(True)
        fixes.addWidget(self.chk_fix_mult)
        fixes.addWidget(self.chk_fix_offset)
        fixes.addWidget(self.chk_fix_phi0)
        right.addLayout(fixes)
        form.addRow('Axis unit:', self.cmb_axis)
        form.addRow('Reference frequency (MHz):', self.edt_ref)
        form.addRow('Spectral Width (Hz):', self.edt_sw)
        form.addRow('Multiplier:', self.edt_mult)
        form.addRow('Offset:', self.edt_offset)
        form.addRow('φ0 (deg):', self.spn_phi0)
        right.addLayout(form)
    
        # Peak table (view + model)
        self.peak_model = PeakTableModel(self.peaks)
        self.tbl = QtWidgets.QTableView()
        self.tbl.setModel(self.peak_model)
        delegate = ScientificDoubleDelegate(self.tbl, decimals=12)
        for c in (0, 2, 4, 6):
            self.tbl.setItemDelegateForColumn(c, delegate)

        # A6: bind links to model and provide current slice index
        self.peak_model.bind_links(self.link_store, slice_index_provider=lambda: getattr(self, "slice_index", 0))
        
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  # optional
        self.hdr = CheckableHeader(QtCore.Qt.Horizontal, self.tbl, model=self.peak_model,
                      fix_sections=[1, 3, 5, 7])
        self.tbl.setHorizontalHeader(self.hdr)
        # narrow for fix columns, stretch for value columns
        for c in (1, 3, 5, 7):  # fix columns
            self.hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        for c in (0, 2, 4, 6):  # value columns
            self.hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
        self.hdr.setStretchLastSection(False)
        self.hdr.setMinimumSectionSize(40)
        self.tbl.verticalHeader().setDefaultSectionSize(int(self.fontMetrics().height() * 1.6))
        self.tbl.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        right.addWidget(self.tbl, 1)
                # B1: mount link context menu
        # Capture slice and peak counts once at initialization (snapshot, not dynamic)
        _slice_count_snapshot = self._slice_count()
        _peaks_per_slice_snapshot = self._peaks_per_slice(0)  # Use slice 0 for initialization
        attach_link_context_menu(
            view=self.tbl,
            parent=self,
            model=self.peak_model,
            slice_count=_slice_count_snapshot,
            peaks_per_slice=_peaks_per_slice_snapshot,
            get_target_from_index=self.peak_model.index_to_paramref,
            link_store=self.link_store,
            enable_linear=True,
            enable_decay=True,   # placeholder
            enable_growth=False   # placeholder
        )
        
        def _on_header_toggled(section: int, checked: bool):
            self.peak_model.set_all_fix_in_column(section, checked)
            # force header to refresh tri-state if needed
            self.hdr.updateSection(section)

        self.hdr.toggled.connect(_on_header_toggled)
        # default per-peak fixed/varied flags. True = Fixed. False = Varied
        self._default_fix = dict(pos=False, amp=False, lor=True, gauss=False)

        self.peak_model.dataChanged.connect(lambda *_: self.hdr.viewport().update())
        self.peak_model.modelReset.connect(self.hdr.viewport().update)
        self.peak_model.rowsInserted.connect(lambda *_: self.hdr.viewport().update())
        self.peak_model.rowsRemoved.connect(lambda *_: self.hdr.viewport().update())

        # Buttons for table rows
        row_ctrls = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton('Add Peak')
        self.btn_remove = QtWidgets.QPushButton('Remove Selected')
        row_ctrls.addWidget(self.btn_add)
        row_ctrls.addWidget(self.btn_remove)
        right.addLayout(row_ctrls)
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)   
       
    def _slice_count(self) -> int:
        data2d = getattr(self, "data2d", None)
        if getattr(data2d, "ndim", 0) == 2:
            return int(data2d.shape[0])
        return 1

    def _peaks_per_slice(self, s: int) -> int:
        st = getattr(self, "slice_states", {}).get(s, None)
        if st and getattr(st, "peaks", None) is not None:
            return len(st.peaks)
        # fallback: current model length
        return len(getattr(self.peak_model, "_peaks", []))

    def get_links(self) -> "LinkStore":
        return self.link_store

    def clear_links(self) -> None:
        self.link_store.clear()
        self.peak_model.layoutChanged.emit()

    # C1: build registry for a slice
    def _build_registry_for_slice(self, s: int) -> dict:
        reg = {} #dict[ParamRef dict]
        st = getattr(self, "slice_states", {}).get(s, None)
        if not st:
            return reg
        for pid, pk in enumerate(st.peaks):
            reg[ParamRef(s, pid, "pos")] = {'value': float(pk.pos), 'fixed': False, 'bounds': (-math.inf, math.inf)}
            reg[ParamRef(s, pid, "amp")] = {'value': float(pk.amp), 'fixed': False, 'bounds': (0.0, math.inf)}
            reg[ParamRef(s, pid, "lor")] = {'value': float(pk.lor_hz), 'fixed': False, 'bounds': (0.0, math.inf)}
            reg[ParamRef(s, pid, "gauss")] = {'value': float(pk.gauss_disp), 'fixed': False, 'bounds': (0.0, math.inf)}
        return reg
    
    # C2: scatter evaluated target values back into states
    def _apply_links_to_slice_states(self, reg: dict) -> None:
        for pref, entry in reg.items():
            if not isinstance(pref, ParamRef):
                continue
            try:
                val = float(entry['value'])
            except Exception:
                continue
            st = getattr(self, "slice_states", {}).get(pref.slice_id, None)
            if not st:
                continue
            try:
                pk = st.peaks[pref.peak_id]
            except Exception:
                continue
            if pref.name == "pos":       pk.pos = val
            elif pref.name == "amp":     pk.amp = val
            elif pref.name == "lor":     pk.lor_hz = val
            elif pref.name == "gauss":     pk.gauss_disp = val
            st.dirty_model = True


    # C3: mark lmfit Parameters for linked targets in current slice
    def _apply_links_to_lmfit(self, params, s: int) -> None:
        """
        Resolve links for the current slice `s` and freeze linked targets in LMFit `params`.
        Key change: seed external (cross-slice) drivers into `reg` *before* LinkEngine.evaluate.
        """
        # 1) Build per-slice registry
        reg = self._build_registry_for_slice(int(s))
        # quick exit if nothing to do
        try:
            has_links = any(getattr(expr, "enabled", False) for expr in self.link_store.all_expr())
        except Exception:
            has_links = False
        if not reg or not has_links:
            return

        # 2) Inject external drivers so LinkEngine can resolve cross-slice targets
        _seed_external_drivers_into_registry(
            registry=reg,
            slice_states=self.slice_states,   # dict[int, SliceFitState]
            current_slice_id=int(s),
            links=self.link_store,
            strict=True,
        )

        # 3a) Evaluate links (mutates reg[target]['value'])
        links_subset = _links_for_target_slice(self.link_store, int(s))
        #links = LinkStore instance, links_subset = LinkStore_subset, a smaller LinkStore, that concerns only about a given slice s
          
        slice_times = getattr(self, "t_f1", None)
        # Thread time vector if present (needed for RELAX_EXP)
        try:
            _has_relax = any(ex.enabled and ex.type == LinkType.RELAX_EXP for ex in links_subset.all_expr())
        except Exception:
            _has_relax = False
        if _has_relax and slice_times is None:
            raise ValueError("RELAX_EXP link requires time data. Load time_echo.txt via File → Open Time.")

        LinkEngine.evaluate(registry=reg, slice_times=slice_times, links=links_subset)

        # 3b) Seed a shared T parameter for later joint-fitting (harmless in single-slice)
        # Don't use it here for numeric freeze, but adding it once avoids surprises later.
        if _has_relax and ("relax_T" not in params):
        # conservative, positive default; user can tweak later
            params.add("relax_T", value=1.0, min=1e-12)

        if _has_relax and ("relax_C" not in params):
            params.add("relax_C", value=0.0, vary=False)

        # 4) Freeze linked targets in this slice in LMFit params (set value + vary=False)
        #    Support a few common aliases for robustness.
        def _norm(n: str) -> str: return (n or "").strip().lower()
        name_map = {
            "pos": "pos", "position": "pos", "freq": "pos", "pos_hz": "pos", "pos_ppm": "pos",
            "amp": "amp", "area": "amp", "amplitude": "amp",
            "lor": "lor", "lorentz": "lor", "lor_hz": "lor", "lorentz_hz": "lor",
            "gauss": "gauss", "gauss_disp": "gauss",
        }

        for pref_tgt, entry in reg.items(): #pref_tgt is ParamRef instance
            if not isinstance(pref_tgt, ParamRef):
                continue
            if int(pref_tgt.slice_id) != int(s):
                continue
            # only freeze if target is actually linked
            if not self.link_store.is_linked(pref_tgt):
                continue
                       
            base_tgt = name_map.get(_norm(pref_tgt.name))
            if not base_tgt:
                continue
            key_tgt = f"{base_tgt}_{int(pref_tgt.peak_id)}"  # lmfit param names: pos_i, amp_i, lor_i, gauss_i
            p_tgt = params.get(key_tgt, None)
            if p_tgt is None:
                continue

            #fetch the link expression for this target            
            linkexpr = links_subset.get(pref_tgt)

            if getattr(linkexpr,"type", None) == LinkType.LINEAR and getattr(linkexpr,"driver", None) is not None:
                pref_drv = linkexpr.driver
                if int(pref_drv.slice_id) != int(s):
                # cross-slice driver not present; freeze numeric
                    try:
                        p_tgt.set(value=float(entry["value"]), vary=False)
                    except Exception:
                        pass
                    continue
            
                base_drv = name_map.get(_norm(pref_drv.name))
                if not base_drv:
                    continue
                key_drv = f"{base_drv}_{int(pref_drv.peak_id)}"
                if key_drv not in params:
                    try:
                        p_tgt.set(value=float(entry["value"]), vary=False)
                    except Exception: pass
                    continue

                a = float(linkexpr.args.get("a", 1))
                b = float(linkexpr.args.get("b", 0))
                p_tgt.set(value=float(entry["value"]), expr=f"{key_drv}*{a}+{b}")

            elif getattr(linkexpr, "type", None) == LinkType.RELAX_EXP:
                # numeric freeze (no algebraic tie)
                try:
                    p_tgt.set(value=float(entry["value"]), vary=False)
                except Exception:
                    pass
                

    def _get_external_driver_slice(self, s: int) -> set[int]:
        """
        Return slice IDs that drive (via enabled LINEAR links) targets in slice `s`,
        excluding `s` itself. Uses self.link_store.
        """
        out: set[int] = set()
        links = getattr(self, "link_store", None)
        if not links:
            return out
        for expr in links.all_expr():
            if not getattr(expr, "enabled", False):
                continue
            if getattr(expr, "type", None) != LinkType.LINEAR:
                continue
            t = getattr(expr, "target", None)
            d = getattr(expr, "driver", None)
            if not t or not d:
                continue
            if int(t.slice_id) == int(s) and int(d.slice_id) != int(s):
                out.add(int(d.slice_id))
        return out
    
    def _on_open_tseed_table(self):
        dlg = getattr(self, "_tseed_dlg", None)
        if dlg is None:
            dlg = TSeedTableDialog(self, self._TSeedRegistry)
            self._tseed_dlg = dlg
        # refresh each time it opens (new T_names might have been added)
        try:
            for ex in self.link_store.all_expr():
                if getattr(ex, "type", None) == LinkType.RELAX_EXP and getattr(ex, "enabled", True):
                    tname = ex.args.get("T_name") or ex.args.get("t_name")
                    if isinstance(tname, str) and tname.strip():
                        dlg.ensure_row(tname.strip())
            dlg.model.refresh()
        except Exception:
            pass

        dlg.exec_()

    def _ensure_tseed_row(self, T_name: str):
        if not T_name: return
        if T_name not in self._TSeedRegistry:
            self._TSeedRegistry[T_name] = {"fixed": False, "T_seed_s": None, "T_result_s": None, "updated_at": ""}
        # if dialog is open, reflect immediately
        dlg = getattr(self, "_tseed_dlg", None)
        if dlg: 
            try: dlg.ensure_row(T_name)
            except Exception: pass



    def _connect_signals(self):
        self.open_act.triggered.connect(self.on_open)
        self.save_act.triggered.connect(self.on_export_ascii)
        self.export_act.triggered.connect(self.on_export_peak_table)
        self.import_act.triggered.connect(self.on_import_peak_table)
        self.time_act.triggered.connect(self.on_open_time)
        self.act_fit.triggered.connect(self.on_fit_current)
        self.act_fit_selected.triggered.connect(
            lambda: self.show_slice_picker(
                N=int(self.data2d.shape[0]) if (hasattr(self, "data2d") and getattr(self.data2d, "ndim", 0) == 2) else 0,
                labels=self.make_slice_labels(int(self.data2d.shape[0])) if hasattr(self, "data2d") else [],
                prechecked=(getattr(self, "_last_slice_selection", None) or [int(getattr(self, "slice_index", 0))])
            )
        )
        self.btn_sim.clicked.connect(self.on_simulate)
        self.btn_copy.clicked.connect(self.on_copy_params)
        btn_copy_bounds = QtWidgets.QPushButton("Copy bounds")
        self.btn_copy_bounds.clicked.connect(self.on_copy_bounds)

        self.btn_clear.clicked.connect(self.on_clear)
        self.tgl_add.toggled.connect(self.on_toggle_add)
        self.btn_add.clicked.connect(self.on_add_peak)
        self.btn_remove.clicked.connect(self.on_remove_peak)
        self.btn_auto.clicked.connect(self.on_autopick)
        self.canvas.mpl_connect('button_press_event', self.on_click_plot)
        
        # --- Exclusion signals + span selector
        self.act_drag.toggled.connect(self.on_toggle_exclude_mode)
        self.act_add.triggered.connect(self.on_exclude_add_manual)
        self.act_rm.triggered.connect(self.on_exclude_remove_selected)
        self.act_clr.triggered.connect(self.on_exclude_clear)
        self._span_exc = SpanSelector(
            self.canvas.ax, self.on_span_exclude,
            direction='horizontal', useblit=True, interactive=False
        )
        self._span_exc.set_active(False)  # only active when toggle is ON
        
        self.btn_link_manager.clicked.connect(self.on_show_link_manager)
        self.btn_tseed.clicked.connect(self._on_open_tseed_table)
        self.btn_stat.clicked.connect(self.on_show_statistics)



        self.edt_mult.textChanged.connect(lambda _: self._mark_all_states_dirty())
        self.edt_offset.textChanged.connect(lambda _: self._mark_all_states_dirty())
        self.edt_ref.textChanged.connect(lambda _: self._mark_all_states_dirty())
        self.edt_sw.textChanged.connect(lambda _: self._mark_all_states_dirty())
        self.cmb_axis.currentTextChanged.connect(lambda _: self._mark_all_states_dirty())
        self.refresh_plot()
        self._wire_state_signals()

    # Call this after any data load / slice change that affects enablement
    def _on_data_state_changed(self) -> None:
        self.update_ui_state()

    def _bind_slice_direct(self, k: int) -> bool:
        st = getattr(self, "slice_states", {}).get(k, None)
        if st is None:
            return False
        self.slice_index = k
        # Rebind the table model to this slice
        if hasattr(self, "peak_model"):
            self.peak_model.bind_state(st)       # begin/endResetModel inside
        # Rebind x/y and refresh plot for this slice
        if hasattr(self, "load_slice_xy"):
            self.load_slice_xy(k)                # provide x,y for k if you have this helper
        if hasattr(self, "refresh_plot"):
            self.refresh_plot(preserve_view=False)
        return True
    
    def on_show_link_manager(self):
        # Capture counts once at dialog creation time (snapshot)
        _slice_count_snapshot = self._slice_count()
        _peaks_per_slice_snapshot = self._peaks_per_slice(getattr(self, "slice_index", 0))
        dlg = LinkManagerDialog(
            self,
            self.link_store,
            slice_count=_slice_count_snapshot,
            peaks_per_slice=_peaks_per_slice_snapshot,
        )
        dlg.exec_()
        # after user edits links, refresh current table so tooltips / gray text update
        self.peak_model.layoutChanged.emit()


    def show_slice_picker(self, N: int, labels: list[str] | None = None, prechecked: list[int] | None = None) -> list[int]:
        dlg = SlicePickerDialog(self, N=N, labels=labels or [], prechecked=prechecked or [])
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            return dlg.result_payload()
        return ([], None)

    def make_slice_labels(self, N: int) -> list[str]:
        """Optional pretty labels; default plain indices. If self.meta has echo times, use them."""
        meta = getattr(self, "meta", {}) or {}
        et = meta.get("echo_times") or meta.get("t_echo") or None
        if isinstance(et, (list, tuple)) and len(et) == N:
            return [f"t = {float(v):g}" for v in et]
        return [f"Slice {i}" for i in range(N)]
       
    def _assert_ready_for_model(self):
        """
        Backend guard for modeling/fit code. Raises ValueError with a clear message
        if required inputs are missing/invalid. Keep UI popups out of here.
        """
        # Data
        if self.x is None or self.y is None or len(self.x) < 2 or len(self.y) != len(self.x):
            raise ValueError("Load a spectrum first (x,y are missing or invalid).")
    
        # SW
        if self.sw_hz is None or float(self.sw_hz) <= 0.0:
            raise ValueError("Spectral width (SW) must be a positive number (Hz).")
    
        # Axis/MHz
        mode = str(getattr(self, "axis_mode", "hz")).lower()
        if mode == "ppm":
            if self.ref is None or float(self.ref) <= 0.0:
                raise ValueError("Spectrometer frequency (MHz) required for ppm axis.")
    
    # Hook signals so state stays fresh ---
    def _wire_state_signals(self):
        try:
            if hasattr(self, "edt_sw") and self.edt_sw is not None:
                self.edt_sw.textChanged.connect(self.update_ui_state)
            if hasattr(self, "edt_ref") and self.edt_ref is not None:
                self.edt_ref.textChanged.connect(self.update_ui_state)

            if hasattr(self, "cmb_axis") and self.cmb_axis is not None:
                self.cmb_axis.currentTextChanged.connect(self.update_ui_state)

            elif hasattr(self, "axis_mode_group") and self.axis_mode_group is not None:
                self.axis_mode_group.buttonToggled.connect(lambda *_: self.update_ui_state())

            # After peaks change, simulation readiness may change
            if hasattr(self, "peak_table_model") and self.peak_table_model is not None:
                try:
                    self.peak_table_model.changed.connect(self.update_ui_state)
                except Exception:
                    pass
        except Exception:
            pass

        # Initial compute
        self.update_ui_state()

    def current_settings(self):
        self.axis_mode = str(self.cmb_axis.currentText()).strip().lower()
        ref_val = self._read_line_float(self.edt_ref, None)
        if ref_val:
            self.ref = ref_val
        else:
            QMessageBox.warning(self, "Invalid reference frequency", "Please provide valid reference frequency")
        
        sw_val = self._read_line_float(self.edt_sw, None)
        if sw_val:
            self.sw_hz = sw_val
        else:
            QMessageBox.warning(self, "Invalid SW", "Spectral width must be > 0.")
        if getattr(self, "_in_bulk_import", False):
        # still allow other parts of current_settings to run if you need them,
        # but DO NOT re-parse/normalize globals here.
            pass
        else:
        # SAFE parsing: never reset to 1 or 0 just because text is momentarily empty
            self.offset      = self._float_from_lineedit(self.edt_offset,   self.offset)
            self.multiplier  = self._float_from_lineedit(self.edt_mult,     self.multiplier)
            self.phi0_deg    = float(self.spn_phi0.value())

        self.fix_mult   = bool(self.chk_fix_mult.isChecked())
        self.fix_phi0   = bool(self.chk_fix_phi0.isChecked())
        self.fix_offset = bool(self.chk_fix_offset.isChecked())

    
    # ------------------------- UI helpers ---------------------------------
    def on_axis_mode_changed(self, s: str):
        self.axis_mode = s.lower()
        # If we have 2D data loaded, re-emit current slice to swap x-axis
        if self.data2d is not None and self.data2d.shape[0] >= 1:
            self.display_slice(self.slice_index, preserve_view=True)
        else:
            self.refresh_plot(preserve_view=True)

    def display_slice(self, i: int, preserve_view: bool = False):
        """
        Extract one row from self.data2d and show it as a standard 1D spectrum.
        """
        if self.data2d is None:
            return
        n_traces, n_f2 = self.data2d.shape
        i = max(0, min(int(i), n_traces - 1))
        self.slice_index = i
        # keep spinbox synced without feedback loops
        try:
            self.spn_slice.blockSignals(True)
            self.spn_slice.setValue(i)
        finally:
            self.spn_slice.blockSignals(False)

        # choose x-axis based on ref availability / current mode
        x = self.f2_ppm if (self.axis_mode.lower() == 'ppm') and (self.ref is not None and self.ref > 0) else self.f2_hz
        if x is None:
            return  # should not happen if loader populated axes
        self.x = x.astype(float)
        self.y = self.data2d[i, :].astype(float)
        self._update_slice_label()
            # if we have a cache for this slice, restore table immediately
        st = self.slice_states.get(int(self.slice_index)) #slice_states = dicts

        if st is not None and not getattr(st, "dirty_model", False):
            # table _ plots from cache
            self._load_model_from_state(st)
            self.refresh_plot_from_state(st, preserve_view=preserve_view)
            return
        
        if st is not None:
            self._load_model_from_state(st)
            self.refresh_plot(show_model=False, preserve_view=preserve_view)
            return
        
        # no cache yet -> data only
        self._on_data_state_changed()
        self.refresh_plot(show_model=False, preserve_view=preserve_view)
   
    def _read_line_float(self, w: QtWidgets.QLineEdit, default: float | None = None) -> float | None:
        t = w.text().strip()
        if t == "":
            return default
        try:
            return float(t)
        except ValueError:
            return default

    def _float_from_lineedit(self, le, fallback):
        """
        Parse float safely from a QLineEdit.
        Keeps previous 'fallback' on empty/invalid text instead of forcing 1.0/0.0.
        """
        try:
            txt = le.text()
            if txt is None:
                return fallback
            txt = txt.strip()
            if txt == "":
                return fallback
            return float(txt)
        except Exception:
            return fallback
        
    def _set_slice_controls_enabled(self, enabled: bool, n_traces: int = 0):
        self.chk_2d_mode.setEnabled(enabled)
        self.spn_slice.setEnabled(enabled and self.chk_2d_mode.isChecked())
        if enabled and n_traces > 0:
            self.spn_slice.setMaximum(n_traces - 1)
        else:
            self.spn_slice.setMaximum(0)

    def _update_slice_label(self):
        if self.data2d is None:
            self.lbl_slice.setText("t = —")
            return
        i = int(self.slice_index)
        if self.t_f1 is not None and 0 <= i < len(self.t_f1):
            # try to show a t1 value 
            self.lbl_slice.setText(f"t = {self.t_f1[i]:.3g}")
        else:
            self.lbl_slice.setText(f"t = —")
          
    def _save_slice_state(self, k: int, *, has_fit: bool) -> None:
        """
        Capture current slice 'k' into the cache:
          - peaks + per-peak fix flags
          - x/y display arrays
          - cached model (after φ0 + offset) and diff
          - snapshot of globals and excluded ranges
          - last plot view (xlim/ylim)
        """
        if self.x is None or self.y is None:
            return

        # 1) deep-copy peaks and fix flags (read from the Model as SSOT)
        model_peaks, model_fix, model_redchi = self.peak_model.return_model()  # lists of Peak and tuples
        # sanitize lengths just in case
        n = min(len(model_peaks), len(model_fix)) if model_fix else len(model_peaks)
        peaks_copy: List[Peak] = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in model_peaks[:n]]
        fix_flags: List[Tuple[bool,bool,bool,bool]] = [tuple(ff) for ff in (model_fix[:n] if model_fix else [])]

        # 2) copy display arrays
        x_disp = np.asarray(self.x, dtype=float).copy()
        y_data = np.asarray(self.y, dtype=float).copy()

        # 3) build cached model/diff on the display grid (φ0 + offset applied)
        y_model = None
        y_diff  = None
        
        try:
            if len(peaks_copy) > 0 and self.sw_hz and self.sw_hz > 0:
                spec_c = model_spectrum(
                    peaks=peaks_copy,
                    axis_mode=self.axis_mode,
                    ref=self.ref,
                    sw_hz=self.sw_hz,
                    N=len(x_disp),
                    x=x_disp,
                    multiplier=self.multiplier,
                    return_fid=False,
                )
                y_model = apply_phase_and_offset(spec_c, self.phi0_deg, self.offset)
                y_diff  = (np.real(y_model) if np.iscomplexobj(y_model) else y_model) - y_data
        except Exception as _e:
            # If anything goes wrong, just skip caching model/diff; table will still be restored.
            y_model = None
            y_diff = None

        y_peaks = None
        try:
            if len(peaks_copy) > 0 and self.sw_hz and self.sw_hz > 0:
                y_peaks = []
                for pk in peaks_copy:
                    spec_i = model_spectrum(
                        peaks=[pk],
                        axis_mode=self.axis_mode,
                        ref=self.ref,
                        sw_hz=self.sw_hz,
                        N=len(x_disp),
                        x=x_disp,
                        multiplier=self.multiplier,
                        return_fid=False,
                    )
                    y_i = apply_phase_and_offset(spec_i, self.phi0_deg, self.offset)
                    y_peaks.append(np.real(y_i) if np.iscomplexobj(y_i) else y_i)
        except Exception:
            y_peaks = None


        # 4) remember last plot view
        try:
            ax = self.canvas.ax
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
        except Exception:
            xlim = ylim = None

        # 5) globals snapshot and excluded ranges
        excluded_copy = list(self.excluded) if getattr(self, "excluded", None) else []

        state = SliceFitState(
            peaks=peaks_copy,
            fix_flags=fix_flags,
            x_disp=x_disp,
            y_data=y_data,
            y_model=y_model,
            y_diff=y_diff,
            y_peaks=y_peaks,
            has_fit=bool(has_fit),
            dirty_model=False,
            axis_mode=str(self.axis_mode),
            excluded=excluded_copy,
            ref_MHz=(None if self.ref is None else float(self.ref)),
            sw_hz=(None if self.sw_hz is None else float(self.sw_hz)),
            N=int(len(x_disp)),
            multiplier=float(self.multiplier),
            offset=float(self.offset),
            phi0_deg=float(self.phi0_deg),
            redchi=model_redchi,
            xlim=xlim,
            ylim=ylim,
        )
        self.slice_states[int(k)] = state

    def _mark_all_states_dirty(self) -> None:
        for st in self.slice_states.values():
            st.dirty_model = True
    
    def _load_model_from_state(self, state: SliceFitState) -> None:
        """
        Rebind PeakTableModel to the cached state's peaks and fix flags.
        No UI globals are touched.
        """
        # 1) rebuild domain list (keep MainWindow.peaks in sync with model SSOT)
        peaks = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in state.peaks]

        # 2) sanitize / backfill fix flags to match length(peaks)
        if state.fix_flags and len(state.fix_flags) >= len(peaks):
            fix_flags = [tuple(ff) for ff in state.fix_flags[:len(peaks)]]
        else:
            d = getattr(self, "_default_fix", dict(pos=False, amp=False, lor=True, gauss=False))
            fix_flags = [(d["pos"], d["amp"], d["lor"], d["gauss"]) for _ in peaks]
            if state.fix_flags:
                for i, ff in enumerate(state.fix_flags):
                    if i < len(peaks):
                        fix_flags[i] = (bool(ff[0]), bool(ff[1]), bool(ff[2]), bool(ff[3]))
        
        rc = getattr(state, "redchi", None)
        redchi = rc if rc is not None else float("nan")

        # 3) commit to window state + model
        self.peaks = peaks
        self.peak_model.replace_data(self.peaks, fix_flags, redchi)

        # 4) nudge header tristate / view
        hdr = self.tbl.horizontalHeader()
        if isinstance(hdr, CheckableHeader):
            hdr.setModelRef(self.peak_model)
            hdr.update()
        self.tbl.resizeColumnsToContents()

    def Model_to_Peaks(self) -> None:
        """Sync peaks from the PeakTableModel (SSOT)."""
        peaks, fix_flags, _ = self.peak_model.return_model()
        self.peaks = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in peaks]
        self._fix_flags_cache = [tuple(ff) for ff in fix_flags]  # for fast access


    def fix_flags(self, i: int) -> tuple[bool,bool,bool,bool]:
        """Compat helper used by on_fit(); returns (pos, amp, lor, gauss) fixed flags."""
        try:
            return self._fix_flags_cache[i]
        except Exception:
            _, fix_flags, _ = self.peak_model.return_model()
            if 0 <= i < len(fix_flags):
                return tuple(fix_flags[i])
            # fallback default
            d = getattr(self, "_default_fix", dict(pos=False, amp=False, lor=True, gauss=False))
            return (d["pos"], d["amp"], d["lor"], d["gauss"])
        


    def _grid_hz(self) -> float:
        if self.x is None or len(self.x) < 2:
            return 10.0
        step = abs(self.x[1] - self.x[0])
        if self.axis_mode == 'ppm':
            return step * (self.ref)
        return step

    def refresh_plot(self, show_model: bool = True,
                     preserve_view: bool = False,
                     read_globals: bool = True):
        """
        Draw:
          - Data (single line)
          - Full model (single line, optional via show_model)
          - Difference curve = model − data (single line, when show_model)
          - Per-peak contributions (one line per peak, when show_model)
          - Peak markers and excluded regions
        """


        ax = self.canvas.ax

        # 1) capture current limits if we want to preserve
        if preserve_view:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # Clear axes and reset cached handles
        ax.clear()
        self._peak_lines = []
        self._diff_line = None
        self._model_line = None
        self._data_line = None

        ax.set_xlabel(self.axis_mode.upper())
        ax.set_ylabel('Intensity (a.u.)')

        # Helper: convert complex → real for plotting
        def _as_plot(y):
            return np.real(y) if np.iscomplexobj(y) else y

        if self.x is not None and self.y is not None:
            # ---- DATA ----
            self._data_line, = ax.plot(self.x, _as_plot(self.y), lw=1.5, label='Data',zorder=1)

            # === shade excluded regions ===
            if getattr(self, "excluded", None) and (self.x is not None):
                if preserve_view:
                    _ylim_saved = ax.get_ylim()
                for (lo, hi) in self.excluded:
                    ax.axvspan(lo, hi, color='grey', alpha=0.15, zorder=0)
                if preserve_view:
                    ax.set_ylim(_ylim_saved)
            self._draw_excluded(ax, getattr(self, "excluded", []))

            # === MODEL, DIFF, AND PER-PEAKS ===
            if show_model and getattr(self, "peaks", None):
                try:
                    if read_globals:
                        # Refresh globals from UI (multiplier, offset, phi0, etc.)
                        self.current_settings()

                    # --- Full complex model (no phase/offset inside math core) ---
                    spec_c = model_spectrum(
                        peaks=self.peaks,
                        axis_mode=self.axis_mode,
                        ref=self.ref,
                        sw_hz=self.sw_hz,
                        N=len(self.x),
                        x=self.x,                     # use current display grid
                        multiplier=self.multiplier,
                        return_fid=False,
                    )

                    # --- Apply ONLY zero-order phase and offset for display ---
                    y_model = apply_phase_and_offset(spec_c, self.phi0_deg, self.offset)

                    # plot full model
                    self._model_line, = ax.plot(self.x, _as_plot(y_model), lw=1.5, alpha=0.9, label='Model', zorder=4)

                    # plot difference curve: model − data
                    diff_raw = _as_plot(y_model) - _as_plot(self.y)
                    # Use current auto y-limits (after plotting data+model) to determine span
                    ymin_auto, ymax_auto = ax.get_ylim()
                    yspan = max(1e-12, ymax_auto - ymin_auto)

                    scale = 0.1  # scale to shift the diff curve. change if needed
                    
                    diff_disp = diff_raw - scale * yspan  # shift the diff curve by scale * yspan
                    self._diff_line, = ax.plot(
                        self.x, diff_disp, lw=1, alpha=0.75, color="0.35",
                        label="Difference"
                    )

                    # Ensure the negative residual band is visible
                    lower_needed = min(ymin_auto, float(np.min(diff_disp)))
                    ax.set_ylim(lower_needed, ymax_auto)

                    # plot per-peak contributions
                    for i, pk in enumerate(self.peaks):
                        spec_i = model_spectrum(
                            peaks=[pk],
                            axis_mode=self.axis_mode,
                            ref=self.ref,
                            sw_hz=self.sw_hz,
                            N=len(self.x),
                            x=self.x,
                            multiplier=self.multiplier,
                            return_fid=False,
                        )
                        y_i = apply_phase_and_offset(spec_i, self.phi0_deg, self.offset)
                        line_i, = ax.plot(self.x, _as_plot(y_i), lw=1.5, alpha=0.9, label=f"Peak {i+1}", zorder=2)
                        self._peak_lines.append(line_i)

                    ax.legend(loc='best')

                except ValueError as e:
                    QMessageBox.warning(self, "Inputs Required", str(e))
                    # fall through to axis inversion / view restore

        # 2) restore limits if preserving; else apply NMR ppm inversion once
        if preserve_view:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            if getattr(self, "axis_mode", "").lower() == 'ppm' or self.cmb_axis.setCurrentText() == 'ppm':
                ax.invert_xaxis()

        self.canvas.draw_idle()

    def refresh_plot_from_state(self, state, preserve_view: bool = True) -> None:
        """
        Fast redraw from cached arrays (no recomputation).
        Draws: data, model (if cached), diff (if cached), plus any UI styling you use.
        """
        ax = self.canvas.ax

        # 1) capture/restore zoom if asked
        if preserve_view:
            try:
                xlim_prev, ylim_prev = ax.get_xlim(), ax.get_ylim()
            except Exception:
                xlim_prev = ylim_prev = None
        ax.clear()

        # 2) axis labels
        ax.set_xlabel(self.axis_mode.upper())
        ax.set_ylabel("Intensity (a.u.)")

        # 3) data
        x = state.x_disp
        y = state.y_data
        if x is None or y is None or len(x) == 0:
            self.canvas.draw_idle()
            return
        ax.plot(x, y, lw=1.5, label="data")

        # 4) model (cached AFTER φ0 + offset)
        if state.y_model is not None:
            y_model = np.real(state.y_model) if np.iscomplexobj(state.y_model) else state.y_model
            ax.plot(x, y_model, lw=1.5, alpha=0.9, label="model")

        # 5a) diff (model - data), shown below x-axis if available
        if state.y_diff is not None:
            ymin_auto, ymax_auto = ax.get_ylim()
            yspan = max(1e-12, ymax_auto - ymin_auto)
            scale = 0.1  # scale to shift the diff curve. change if needed            
            diff_disp = np.asarray(state.y_diff, dtype=float) - scale * yspan  # shift the diff curve by scale * yspan
            ax.plot(x, diff_disp, lw=1, alpha=0.75, color="0.35", label="Difference")
            lower_needed = min(ymin_auto, float(np.min(diff_disp)))
            ax.set_ylim(lower_needed, ymax_auto)

        # 5b) per-peak cached curves
        if getattr(state, "y_peaks", None):
            for i, y_i in enumerate(state.y_peaks, start=1):
                ax.plot(x, y_i, lw=1.5, alpha=0.9, zorder=2, label=f"Peak {i}")

        # 6) legend (optional)
        try:
            ax.legend(loc="best", fontsize=8)
        except Exception:
            pass

        # shaded excluded regions (prefer state.excluded if provided)
        spans = getattr(state, "excluded", None)
        if spans is None:
            spans = getattr(self, "excluded", [])
        self._draw_excluded(ax, spans)

        # 7) restore view if requested
        if preserve_view and 'xlim_prev' in locals() and xlim_prev and ylim_prev:
            ax.set_xlim(xlim_prev)
            ax.set_ylim(ylim_prev)
        mode = (getattr(state, "axis_mode", None) or getattr(self, "axis_mode", "hz")).lower()
        lo, hi = ax.get_xlim()
        if mode == "ppm":
            if lo < hi:  # ensure decreasing for ppm
                ax.set_xlim(hi, lo)
        else:
            if lo > hi:  # ensure increasing for Hz
                ax.set_xlim(hi, lo)

        # 8) update redchi
        if state.redchi is not None and float("nan"):
            self.lbl_redchi.setText(f"Reduced χ²: {state.redchi:.4g}")
        else:
            self.lbl_redchi.setText(f"Reduced χ²: —")



        self.canvas.draw_idle()

    # ---------------------------- Slots ------------------------------------
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open spectrum', '',
            'Supported (*.json *.txt *.dat *.csv);;JSON (*.json);;Text (*.txt *.dat *.csv);;All files (*.*)'
        )

        if not path:
            return
        try:
            p = Path(path)
            self.dataset_label = p.stem
            self.data_file = p.name
            self.default_open_dir = str(p.parent)
            self.default_save_dir = self.default_open_dir
            # Rebuild SSOT from this file
            self.slice_states.clear()          
            
            if p.suffix.lower() == ".json":
                # Use ssNake-aware loader
                x_ppm, x_hz_raw, y_traces, meta = load_ssnake(p)
                
                # --- try to read SW from meta, else leave None (auto-estimate)
                sw_from_meta = extract_sw_hz_from_meta(meta)
                self.sw_hz = sw_from_meta if (sw_from_meta is not None and sw_from_meta > 0) else None              
                self.edt_sw.setText(f"{float(self.sw_hz):.1f}" if self.sw_hz else f"{calc_sw_hz((x_hz_raw)):.1f}")
                # --- try to read ref from meta, else leave None (set text)
                ref_from_meta = extract_ref_from_meta(meta)
                self.ref = ref_from_meta if (ref_from_meta is not None) else None
                if self.ref:
                    self.edt_ref.setText(f"{float(self.ref):.9f}")
                    self.edt_ref.setStyleSheet("")
                else:
                    self.edt_ref.clear()
                    self.edt_ref.setStyleSheet("background-color: #ffcccc;")
                    QMessageBox.warning(self, "Invalid reference frequency", "Please provide correct reference frequency")
                    raise ValueError("Invalid reference frequency")
                self.transmitter_freq = extract_freq_from_meta(meta)
                # Display axis: prefer ppm if ref is valid
                self.axis_mode = 'ppm'
                self.cmb_axis.setCurrentText('ppm')
                x_disp = x_ppm.astype(float) if str((self.axis_mode)).lower == 'ppm' else x_hz_raw.astype(float)
                self.meta = meta 

                if meta.get('dim') == 1:
                    self.slice_index = 0
                    self.slice_states[0] = SliceFitState(
                        peaks=[],
                        fix_flags=[],
                        x_disp=x_disp,
                        y_data=y_traces[0],
                        y_model=None,
                        y_diff=None,
                        y_peaks=None,
                        has_fit=False,
                        dirty_model=False,
                        axis_mode=str(self.axis_mode),
                    )
                    self.x = self.slice_states[0].x_disp
                    self.y = self.slice_states[0].y_data
                    n_traces = int(1)

                
                elif meta.get('dim') == 2:                
                    # ---- stash full 2D and axes; show first slice ----
                    self.data2d = y_traces.astype(float)          # (N, F)
                    self.f2_ppm = x_ppm.astype(float)             # (F,)
                    self.f2_hz  = x_hz_raw.astype(float)          # (F,)
                    n_traces = int(self.data2d.shape[0])

                    for k in range(n_traces):
                        self.slice_states[k] = SliceFitState(
                            peaks=[],
                            fix_flags=[],
                            x_disp=x_disp,
                            y_data=y_traces[k],
                            y_model=None,
                            y_diff=None,
                            y_peaks=None,
                            has_fit=False,
                            dirty_model=False,
                            axis_mode=str(self.axis_mode),  
                        )

                # Enable/disable 2D controls based on N
                self._set_slice_controls_enabled(n_traces >= 1, n_traces)

                # Show first slice
                self.slice_index = 0
                try:
                    self.peak_model.beginResetModel()
                    # PeakTableModel stores a ref list as _peaks in v6.1
                    self.peak_model._peaks = self.peaks
                    self.peak_model.endResetModel()
                except Exception:
                    # Fallback: recreate model if it lacks a reset API
                    self.tbl.setModel(None)
                    self.peak_model = PeakTableModel(self.peaks)
                    self.tbl.setModel(self.peak_model)   
                self.display_slice(0, preserve_view=False)

            else:
                # fallback: 2-column text loader
                x, y = load_spectrum(path)
                self.x, self.y = x.astype(float), y.astype(float)

            self._on_data_state_changed()
            self.refresh_plot(show_model=False)
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))

    def on_open_time(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open relaxation time', '',
            'Supported (*.txt *.dat *.csv);;Text (*.txt *.dat *.csv);;All files (*.*)'
        )
        if not path:
            return
        try:
            p = Path(path)
            self.dataset_label   = p.stem
            self.default_open_dir = str(p.parent)
            self.default_save_dir = self.default_open_dir

            if p.suffix.lower() in [".txt", ".csv", ".dat"]:
                N = self.meta.get("number of traces", None)
                self.t_f1 = load_time(p, N)
                self.status.showMessage('Relaxation time loaded', 1500)
                self._on_data_state_changed()

        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))

        # -------------------- 2D mode events --------------------
    def on_toggle_2d_mode(self, checked: bool):
        if self.data2d is None or self.data2d.shape[0] <= 1:
            # no usable 2D data
            self.chk_2d_mode.setChecked(False)
            self._set_slice_controls_enabled(False)
            self._on_data_state_changed()
            return
        # enable/disable slider+spin
        self._set_slice_controls_enabled(True, self.data2d.shape[0])
        if checked:
            # entering 2D mode → display current slice_index
            self.display_slice(self.slice_index, preserve_view=False)
        else:
            # leaving 2D mode → fall back to first (or current) trace
            self.display_slice(0, preserve_view=False)

    def on_slice_spin_changed(self, val: int):       
        if not self.chk_2d_mode.isChecked():
            return
        self._force_commit_table_edits()
        self.current_settings()
        self.Model_to_Peaks()
        self._save_slice_state(self.slice_index, has_fit=False)
        self.display_slice(int(val), preserve_view=True)
        self.update_ui_state()

    def _force_commit_table_edits(self) -> None:
        """
        Ensure any in-place editor (line edit, spinbox, checkbox) commits its value
        into the model before read/save from the table.
        """
        try:
            view: QtWidgets.QTableView = self.tbl

            # 1) If we're mid-edit, ask the view to end editing.
            if view.state() == QtWidgets.QAbstractItemView.EditingState:
                idx = view.currentIndex()
                if idx.isValid():
                    # close any transient editor for the current index
                    view.closeEditor(view.focusWidget(), QtWidgets.QAbstractItemDelegate.NoHint)

            # 2) Move focus off the editor to trigger delegate → commitData → model.setData
            target = getattr(self, "spn_slice", None) or self  # any non-table widget is fine
            if isinstance(target, QtWidgets.QWidget):
                target.setFocus(Qt.OtherFocusReason)

            # 3) Let Qt process the commit immediately
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)

        except Exception:
            pass


    def on_toggle_relax(self, checked:bool):
        if self.data2d is None or self.data2d.shape[0] <= 1:
            # no usable 2D data
            self.chk_relax.setEnabled(False)
            self.time_act.setEnabled(False)
        else:
            if checked:
                # enable time_act
                self.time_act.setEnabled(True)
            else:
                self.time_act.setEnabled(False)

    def on_clear(self):
        """Clear all peaks for the current slice and refresh."""
        # 1) clear domain + model
        self.peaks = []
        self._fix_flags_cache = []
        self.peak_model.replace_data([], [], float('nan'))

        # 2) invalidate this slice’s cache (if you keep per-slice states)
        try:
            st = self.slice_states.get(int(self.slice_index))
            if st:
                st.peaks.clear()
                st.fix_flags.clear()
                st.y_model = None
                st.y_diff = None
                st.y_peaks = None
                st.has_fit = False
                st.dirty_model = True
        except Exception:
            pass

        # 3) UI state + redraw
        self._on_data_state_changed()
        self.refresh_plot(show_model=False, preserve_view=True)


    def on_add_peak(self):
        if self.x is None or self.y is None:
            return

        # 1) choose the highest point as a quick seed
        idx = int(np.argmax(self.y))
        pos0 = float(self.x[idx])

        # 2) initial widths
        lor0_hz = max(0.32768 * self._grid_hz(), 0.0)
        ga0_disp = max(30 * (self.x[1] - self.x[0]), 1e-6)

        # 3) Gaussian width in Hz if axis is ppm
        ga0_hz = ga0_disp * self.ref if self.axis_mode == 'ppm' else ga0_disp

        # 4) height estimate (remove baseline offset)
        height = float(max(self.y[idx] - self.offset, 0.0))

        # 5) crude area guess (amp now = area)
        fwhm_hz = lor0_hz + ga0_hz
        amp0 = max(height * fwhm_hz / self.multiplier, 1e-9)

        # 6) add the peak
        self.add_peak(Peak(pos0, amp0, lor0_hz, ga0_disp), select_row=True, refresh=True)

    def on_remove_peak(self):
        """Remove selected peak rows (QTableView model path)."""
        sel = self.tbl.selectionModel()
        if sel is None:
            return

        rows = sorted({idx.row() for idx in sel.selectedIndexes()})
        if not rows:
            return

        # 1) remove in model
        self.peak_model.remove_rows(rows)

        # 2) sync domain cache and mark slice dirty
        self.Model_to_Peaks()
        st = self.slice_states.get(int(getattr(self, "slice_index", 0)))
        if st:
            st.peaks = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in self.peaks]
            st.fix_flags = list(getattr(self, "_fix_flags_cache", []))
            st.y_model = None
            st.y_diff = None
            st.y_peaks = None
            st.has_fit = False
            st.dirty_model = True

        # 3) keep UI coherent (header tristate, selection, state)
        hdr = self.tbl.horizontalHeader()
        if isinstance(hdr, CheckableHeader):
            hdr.update()

        rc = self.peak_model.rowCount()
        if rc > 0:
            self.tbl.selectRow(min(rows[0], rc - 1))

        self._on_data_state_changed()
        self.refresh_plot(preserve_view=True)
        self.update_ui_state()



    def on_click_plot(self, event):
        # 0) Ignore when toolbar is active
        mode = getattr(self.nav, 'mode', '')
        if mode:
            return

        # 1) Neutralize toolbar toggles
        if hasattr(self, 'nav') and hasattr(self.nav, 'mode'):
            self.nav.mode = ''
            for act in self.nav.actions():
                try:
                    t = act.text().lower()
                except Exception:
                    t = ''
                if 'zoom' in t or 'pan' in t:
                    act.setChecked(False)

        # 2) If SpanSelector is present and exclude-mode is OFF, disable span
        if hasattr(self, '_span_exc'):
            if not (hasattr(self, 'act_drag') and self.act_drag.isChecked()):
                self._span_exc.set_active(False)

        # 3) Preconditions for adding peaks
        if not getattr(self, 'tgl_add', None) or not self.tgl_add.isChecked():
            return
        if event.inaxes != self.canvas.ax or self.x is None or self.y is None:
            return
        if event.xdata is None:   # clicked outside data area
            return

        ax = self.canvas.ax

        # --- First LEFT click → choose peak position only ---
        if not self._add_waiting_width and event.button == 1:
            self.status.clearMessage()
            x0 = float(event.xdata)
            idx = int(np.argmin(np.abs(self.x - x0)))
            pos0 = float(self.x[idx])

            # store for second click
            self._pending_pos = pos0
            # use local signal height as a crude scale for later amp estimate
            self._pending_height = float(max(self.y[idx], 1.0))
            self.status.showMessage('Right click to choose peak width', 0)

            # flip state
            self._add_waiting_width = True
            return

        # --- Then RIGHT click → choose width, compute params, add peak ---
        if self._add_waiting_width and event.button == 3:
            # width from distance to the first click (display units)
            self.status.clearMessage()
            xw = float(event.xdata)
            pos0 = float(self._pending_pos)
            height = float(self._pending_height)

            ga0_disp = abs(pos0 - xw) * 2.0  # FWHM = HWHM * 2

            lor0_hz = max(0.32768 * self._grid_hz(), 0.0)
            ga0_hz  = ga0_disp * (self.ref if self.axis_mode == 'ppm' else 1.0)
            fwhm_hz = lor0_hz + ga0_hz
            amp0    = height * fwhm_hz / self.multiplier

            # create the peak now (constructor expects gauss in display units)
            self.add_peak(Peak(pos0, float(amp0), float(lor0_hz), float(ga0_disp)), select_row=True, refresh=True)

            # reset state and refresh
            self._add_waiting_width = False
            self._pending_pos = None
            self._pending_height = None
            self.status.showMessage('Peak added', 1500)

            self.refresh_plot(preserve_view=True)
            return

        # Optional: if user mis-clicks (e.g., left again), gently remind
        if self._add_waiting_width and event.button == 1:
            self.status.showMessage('Right click to choose width (distance from the chosen position)', 2000)
    
    def on_toggle_add(self, checked: bool):
        # When entering Add Peaks mode, force toolbar to neutral so clicks reach user
        if checked:
            if hasattr(self, 'nav') and hasattr(self.nav, 'mode'):
                self.nav.mode = ''
                for act in self.nav.actions():
                    try:
                        t = act.text().lower()
                    except Exception:
                        t = ''
                    if 'zoom' in t or 'pan' in t:
                        act.setChecked(False)
            # Ensure SpanSelector (select exclude region) never eats clicks while adding peaks
            if hasattr(self, '_span_exc'):
                self._span_exc.set_active(False)
            
            # Notify user
            if hasattr(self, "status"):
                self.status.showMessage('Left click to choose peak position', 0)

                self._add_waiting_width = False
                self._pending_pos = None
                self._pending_height = None
        else:
            # cancel any pending operation
            self._add_waiting_width = False
            self._pending_pos = None
            self._pending_height = None
            self.status.clearMessage()

    def on_autopick(self): #does not work. fix later
        if self.x is None or self.y is None:
            return
        N, ok = QInputDialog.getInt(self, 'Auto-Pick', 'How many peaks?', 3, 1, 50, 1)
        if not ok:
            return
        # Simple prominence-based detection
        y = self.y
        # Estimate noise via MAD
        sigma = 1.4826 * np.median(np.abs(y - np.median(y)))
        prom = max(sigma * 5, (np.max(y) - np.min(y)) * 0.02)
        # Peak finder
        idxs = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] >= y[i + 1] and (y[i] - min(y[i-1], y[i+1])) >= prom:
                idxs.append(i)
        # Sort by height
        idxs = sorted(idxs, key=lambda i: y[i], reverse=True)[:N]
        # Enforce minimum spacing
        idxs_sorted = []
        min_dist = max(3, len(y) // 200)
        for i in idxs:
            if all(abs(i - j) >= min_dist for j in idxs_sorted):
                idxs_sorted.append(i)
        # Add peaks
        
        for i in idxs_sorted:
            pos0 = float(self.x[i])
            
            height = float(y[i])
            lor0_hz = max(5 * self._grid_hz(), 1.0)
            ga0_disp = max(5 * abs(self.x[1] - self.x[0]), 1e-6)
            ga0_hz = ga0_disp * self.ref if self.axis_mode == 'ppm' else ga0_disp
            fwhm_hz = lor0_hz + ga0_hz
            amp0 = height * fwhm_hz

            lor0_hz = max(5 * self._grid_hz(), 1.0)
            # width guess using neighbor window (display units)
            w = max(5, len(y)//100)
            l = max(0, i - w); r = min(len(y) - 1, i + w)
            half = y[i] / 2.0
            li = l + np.argmin(np.abs(y[l:i+1] - half))
            ri = i + np.argmin(np.abs(y[i:r+1] - half))
            ga0_disp = float(abs(self.x[ri] - self.x[li])) if ri > li else float(abs(self.x[1]-self.x[0]) * 5)
            self.add_peak(Peak(pos0, amp0, lor0_hz, ga0_disp))
        self.refresh_plot(preserve_view=True)
        self.update_ui_state()

    def on_fit_current(self, *, allow_external_drivers: bool = False, status: bool = True):
        # 0) Guards (before busy)
        if self.x is None or self.y is None or self.peak_model.rowCount() == 0:
            QtWidgets.QMessageBox.warning(self, 'Fit', 'Load data and define at least one peak.')
            return

        self._begin_busy()
        try:
            # 1) Read peaks/fix from the Model (no table pokes)
            peaks, fix_flags, _ = self.peak_model.return_model()
            sid = int(getattr(self, "slice_index", 0))
            no_of_peaks = self._peaks_per_slice(int(getattr(self, "slice_index", 0)))

            # 2) Read other settings from the UI
            self.current_settings()
            mask_w = self._mask_from_excluded(self.x)

            # 3) Build context and LMFit params
            ctx = FitContext(self.x, self.y, self.axis_mode, self.ref, self.sw_hz)
            ctx.mask_w = mask_w
            params = ctx.build_params(peaks, sid, self.multiplier, self.offset) #params is a Paramters instance aka a dictionary

            # globals vary/freeze
            params[f'{ParamRef_to_key(global_ref(sid, "mult"))}'].set(vary=not self.fix_mult,    value=self.multiplier)
            params[f'{ParamRef_to_key(global_ref(sid, "phi0_deg"))}'].set(vary=not self.fix_phi0, value=self.phi0_deg)
            params[f'{ParamRef_to_key(global_ref(sid, "offset"))}'].set(vary=not self.fix_offset, value=self.offset)        

            for i in range(len(peaks)):
                apply_bounds_to_param(params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="pos"))}'],   self.peak_model, sid, i, "pos",   self.axis_mode, self.ref)
                apply_bounds_to_param(params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'], self.peak_model, sid, i, "gauss", self.axis_mode, self.ref)
                apply_bounds_to_param(params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="amp"))}'],   self.peak_model, sid, i, "amp",   self.axis_mode, self.ref)
                apply_bounds_to_param(params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="lor"))}'],   self.peak_model, sid, i, "lor",   self.axis_mode, self.ref)

            for i in range(len(peaks)):
                b_lor = self.peak_model.get_bounds_for(ParamRef(slice_id=sid, peak_id=i, name="lor")) if self.peak_model else None
                if not (b_lor and b_lor.is_set()):
                    params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="lor"))}'].min = 0.0
                    params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="lor"))}'].max = 10000.0

                b_g = self.peak_model.get_bounds_for(ParamRef(slice_id=sid, peak_id=i, name="gauss")) if self.peak_model else None
                if not (b_g and b_g.is_set()):
                    params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'].min = 0.0
                    params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'].max = 10000

                b_a = self.peak_model.get_bounds_for(ParamRef(slice_id=sid, peak_id=i, name="amp")) if self.peak_model else None
                if not (b_a and b_a.is_set()):
                    params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="amp"))}'].min = 0.0

            # 4) Apply per-peak vary for lmfit from GUI fix flags. lmfit set vary = 1. GUI set fix = 1 to be consistent with ssnake
            for i in range(len(peaks)):
                fpos, famp, flor, fgauss = fix_flags[i]
                params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="pos"))}'].vary   = not fpos
                params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="amp"))}'].vary   = not famp
                params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="lor"))}'].vary   = not flor
                params[f'{ParamRef_to_key(ParamRef(slice_id=sid, peak_id=i, name="gauss"))}'].vary = not fgauss

            # 5) Resolve links for THIS slice: seed external drivers + evaluate + freeze linked targets
            s = int(getattr(self, "slice_index", 0))
            ext_drv = self._get_external_driver_slice(s)
            if ext_drv and not allow_external_drivers:
                # Note: error messege
                msg = ("This slice has parameters driven by other slice(s): "
                       f"{sorted(ext_drv)}.\n"
                       "Please use ‘Fit Selected…’, include the driver slice(s), "
                       "and choose Sequential or Joint mode.")
                QtWidgets.QMessageBox.information(self, "Cross-slice driver detected", msg)
                return  # finally{} will _end_busy()

            try:
                self._apply_links_to_lmfit(params, s)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Link error", str(e))
                return  # finally{} will end_busy()

            # 6) Fit

            try:
                result = minimize(
                    lambda p: ctx.residual(p, no_of_peaks, sid),
                    params,
                    method='least_squares', max_nfev=5000,
                    ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale='jac', loss='soft_l1'
                )

                self.fit_stats = extract_FitResult_corr_and_sum(result, mode="Single Fit", slice_indices_list=[sid])
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Fit error', str(e))
                return  # finally{} will end_busy()

            # 7) Debug log (optional, to be removed later)
            if DEBUG_LOGGING:
                try:
                    x_disp = self.x.astype(float)
                    is_ppm = (self.axis_mode == 'ppm')
                    MHz    = float(self.ref)
                    x_hz   = ppm_to_hz(x_disp, MHz) if is_ppm else x_disp
                    sw_grid_hz = calc_sw_hz(x_hz) if x_hz.size > 1 else float("nan")
                    p = result.params
                    phi0_deg_final = p['phi0_deg'].value if 'phi0_deg' in p else 0.0
                    v_phi0_fin     = p['phi0_deg'].vary  if 'phi0_deg' in p else None
                    log.info(
                        "FINAL | MHz=%.6f | sw_display=%.3f Hz | sw_hz(ctx)=%s | mult=%.3f | "
                        "phi0=%.6g deg (vary=%s) | success=%s | nfev=%d | |res|=%.6g | %s | %s",
                        MHz, sw_grid_hz,
                        "None" if (self.sw_hz is None) else f"{float(self.sw_hz):.1f}",
                        float(p['mult'].value),
                        float(phi0_deg_final), v_phi0_fin,
                        result.success,
                        result.nfev,
                        np.linalg.norm(result.residual),
                        _arr_summary("x_disp", x_disp),
                        _arr_summary("x_hz",   x_hz),
                    )
                except Exception as _e:
                    log.warning("FINAL logging failed: %s", _e)

            # 8) Update peaks & globals from result
            ctx.result_to_peaks(peaks, result.params, sid)
            self.multiplier = result.params[f'{ParamRef_to_key(global_ref(sid, "mult"))}'].value
            self.offset     = result.params[f'{ParamRef_to_key(global_ref(sid, "offset"))}'].value
            self.phi0_deg   = result.params[f'{ParamRef_to_key(global_ref(sid, "phi0_deg"))}'].value

            # UI widgets reflect new globals (block signals while setting)
            for w in (self.edt_mult, self.edt_offset):
                try: w.blockSignals(True)
                except Exception: pass
            try: self.spn_phi0.blockSignals(True)
            except Exception: pass

            self.edt_mult.setText(f"{self.multiplier:.3f}")
            self.edt_offset.setText(f"{self.offset:.3f}")
            self.spn_phi0.setValue(self.phi0_deg)

            for w in (self.edt_mult, self.edt_offset):
                try: w.blockSignals(False)
                except Exception: pass
            try: self.spn_phi0.blockSignals(False)
            except Exception: pass

            # 9) Push fitted data back to the model
            fitted_peaks = peaks
            fitted_fix   = fix_flags
            redchi   = float(result.redchi)          
            try:
                self.peak_model.replace_data(fitted_peaks, fitted_fix, redchi)
            except Exception: pass

            # 10) Save ONLY this slice and refresh
            self._save_slice_state(s, has_fit=True)
            st = self.slice_states.get(s)
            if st is not None:
                self.refresh_plot_from_state(st, preserve_view=True)
                st.state_stats = self.fit_stats

            # 11) Quick GOF
            if status is True:
                try:
                    QtWidgets.QMessageBox.information(
                        self, 'Fit', f"Fit done. redchi={redchi:.4g}, nfev={int(result.nfev)}."
                    )
                except Exception:
                    QtWidgets.QMessageBox.information(self, 'Fit', "Fit done.")

        finally:
            # Always end busy (covers all early-returns above)
            try:
                self._end_busy()
            except Exception: pass
                

    def on_fit_pick_and_run(self) -> None:
        """Dialog → run sequential or joint."""
        data2d = getattr(self, "data2d", None)
        if not (hasattr(data2d, "shape") and data2d.ndim == 2 and data2d.shape[0] > 1):
            QtWidgets.QMessageBox.information(self, "Fit Selected…", "No 2D data with multiple slices.")
            return

        # Ensure 2D mode
        if not self.chk_2d_mode.isChecked():
            self.statusBar().showMessage("2D mode was OFF. Turning it ON to run multi-slice fit.", 4000)
            self.chk_2d_mode.setChecked(True)
            if hasattr(self, "on_toggle_2d_mode"):
                self.on_toggle_2d_mode(True)

        n_traces = int(data2d.shape[0])
        labels = self.make_slice_labels(n_traces)
        pre = self._last_slice_selection if self._last_slice_selection else [int(getattr(self, "slice_index", 0))]
        indices, mode = self.show_slice_picker(n_traces, labels=labels, prechecked=pre)
        if not indices or mode is None:
            return

        indices = sorted(set(indices))
        self._last_slice_selection = indices

        if mode == "sequential":
            self.on_fit_sequential(indices)
        elif mode == "joint":
            self.on_fit_jointed(indices)            
        
    def on_fit_sequential(self, indices: list[int]) -> None:
        """Sequential fit (driver-first) across 'indices'. Uses on_fit_current under the hood."""
        data2d = getattr(self, "data2d", None)
        if not (hasattr(data2d, "shape") and data2d.ndim == 2 and data2d.shape[0] > 1):
            QtWidgets.QMessageBox.information(self, "Sequential Fit", "No 2D data with multiple slices.")
            return
        if not indices:
            QtWidgets.QMessageBox.information(self, "Sequential Fit", "No slices selected.")
            return

        # Validate selection: require drivers present in selection
        selected = set(int(s) for s in indices)
        missing: set[int] = set()
        for t in selected:
            missing |= self._get_external_driver_slice(t)
        missing -= selected
        if missing:
            QtWidgets.QMessageBox.information(
                self, "Missing driver slices",
                "Your selection includes targets driven by other slice(s): "
                f"{sorted(missing)}.\nAdd those slice(s) or unlink the parameters."
            )
            return

        # 1) Order slices so cross-slice LINEAR drivers come first
        try:
            order = _topo_sort_slices_for_links(self.link_store, list(selected))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Sequential fit blocked", str(e))
            return

        # 2) Fit each slice in order; let on_fit_current handle link seeding/eval
        fitted = []
        try:
            for k in order:
                if getattr(self, "slice_index", None) != k:
                    self.slice_index = int(k)
                    st = self.slice_states.get(k, None)
                    if st is not None:
                        self.peak_model.bind_state(st)
                if not self._bind_slice_direct(k):
                    continue
                self.on_slice_spin_changed(k)
                self.on_fit_current(allow_external_drivers=True, status=False)
                fitted.append(k)
        finally:
            self.statusBar().showMessage(
                f"Fitted slices (sequential): {fitted}" if fitted else "No slices were fitted.", 4000
            )

    def on_fit_jointed(self, indices: list[int]) -> None:
        """
        Joint WLS fit over multiple slices using one shared optimizer.
        - Per-slice residuals are robust-scaled (MAD) and masked (excluded ranges).
        - Parameter names are prefixed by slice: s{sid}_pos_i, s{sid}_amp_i, ...
        - Cross-slice links are enforced; LINEAR ties are applied after assembly.
        - EXP uses k = 1 / T internally for optimization
        """
        data2d = getattr(self, "data2d", None)
        if not (hasattr(data2d, "shape") and data2d.ndim == 2 and data2d.shape[0] > 1):
            QtWidgets.QMessageBox.information(self, "Joint Fit", "No 2D data with multiple slices.")
            return
        if not indices:
            QtWidgets.QMessageBox.information(self, "Joint Fit", "No slices selected.")
            return
        
        current_sid = int(getattr(self, "slice_index", 0))

        # Read globals once
        self.current_settings()
        axis_mode = str(self.axis_mode).lower()
        ref_MHz   = float(self.ref)
        sw_hz     = float(self.sw_hz)
        # Display axis (shared across slices)
        x_disp = self.f2_ppm if (axis_mode == 'ppm') else self.f2_hz
        if x_disp is None:
            QtWidgets.QMessageBox.warning(self, "Joint Fit", "Display axis is not available.")
            return
        x_disp = x_disp.astype(float)

        load = self._build_joint_params(indices)
        if not load or load[0] is None:
            QtWidgets.QMessageBox.warning(self, "Joint Fit", "Failed to build joint parameters.")
            return

        params_all, slice_ctxs, report = load
        if report.get("errors"):
            QtWidgets.QMessageBox.warning(self, "Joint Fit", "\n".join(report["errors"]))
            return
        
        # Joint residual = concat of per-slice robust residuals
        def _joint_residual(params: Parameters) -> np.ndarray:
            chunks = []
            for sid, ctx, tmpl, _fix, _pre in slice_ctxs:
                r = ctx.residual(params, tmpl)
                chunks.append(np.asarray(r, dtype=float).ravel())
            return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        # Optimize
        self._begin_busy()
        try:
            result = minimize(
                _joint_residual, params_all,
                method='least_squares', max_nfev=5000,
                ftol=1e-8, xtol=1e-8, gtol=1e-8, loss='soft_l1', x_scale='jac'                
            )
            self.fit_stats = extract_FitResult_corr_and_sum(result, mode="Joint Fit", slice_indices_list=indices)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Joint Fit error', str(e))
            try: self._end_busy()
            except Exception: pass
            return

        # Scatter back per slice; refresh caches/UI
        for sid, ctx, tmpl, fix_flags, pre in slice_ctxs:
            ctx.result_to_peaks(tmpl, result.params, prefix=pre)
            mult_val  = result.params[f'{ParamRef_to_key(global_ref(sid, "mult"))}'].value
            off_val   = result.params[f'{ParamRef_to_key(global_ref(sid, "offset"))}'].value
            phi0_val  = result.params[f'{ParamRef_to_key(global_ref(sid, "phi0_deg"))}'].value

            st = self.slice_states.get(sid) or SliceFitState()
            self.slice_states[sid] = st
            st.peaks      = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in tmpl]
            st.fix_flags  = [tuple(ff) for ff in fix_flags]
            st.axis_mode  = str(axis_mode)
            st.multiplier = float(mult_val)
            st.offset     = float(off_val)
            st.phi0_deg   = float(phi0_val)
            st.ref_MHz    = float(ref_MHz)
            st.sw_hz      = float(sw_hz)
            st.x_disp     = x_disp.copy()
            st.y_data     = self.data2d[sid, :].astype(float).copy()
            st.has_fit    = True

            y_model = None
            y_diff  = None
            y_peaks = None
            try:
                if st.peaks and sw_hz > 0.0:
                    # full model
                    spec_c = model_spectrum(
                        peaks=st.peaks,
                        axis_mode=axis_mode,
                        ref=ref_MHz,
                        sw_hz=sw_hz,
                        N=len(x_disp),
                        x=x_disp,
                        multiplier=st.multiplier,
                        return_fid=False,
                    )
                    y_mod = apply_phase_and_offset(spec_c, st.phi0_deg, st.offset)
                    y_model = np.real(y_mod) if np.iscomplexobj(y_mod) else y_mod
                    y_diff  = y_model - st.y_data

                    # per-peak contributions
                    y_peaks = []
                    for pk in st.peaks:
                        spec_i = model_spectrum(
                            peaks=[pk],
                            axis_mode=axis_mode,
                            ref=ref_MHz,
                            sw_hz=sw_hz,
                            N=len(x_disp),
                            x=x_disp,
                            multiplier=st.multiplier,
                            return_fid=False,
                        )
                        y_i = apply_phase_and_offset(spec_i, st.phi0_deg, st.offset)
                        y_peaks.append(np.real(y_i) if np.iscomplexobj(y_i) else y_i)
            except Exception:
                y_model = None
                y_diff  = None
                y_peaks = None

            st.y_model = y_model
            st.y_diff  = y_diff
            st.y_peaks = y_peaks
            st.dirty_model = False

            try:
                r_slice = FitContext(x_disp, st.y_data, axis_mode, ref_MHz, sw_hz)
                r_slice.mask_w = getattr(ctx, "mask_w", None)
                r_slice.param_prefix = pre
                rs = r_slice.residual(result.params, st.peaks)
                st.redchi = float(np.mean(np.square(np.asarray(rs, dtype=float))))
            except Exception:
                st.redchi = float("nan")


            # --- only update GUI/view for the slice that was CURRENT before fit ---
            if int(sid) == current_sid:
                try:
                    # bind view
                    self.peak_model.bind_state(st)
                    self.refresh_plot_from_state(st, preserve_view=True)
                    # sync widgets
                    self.multiplier = st.multiplier
                    self.offset     = st.offset
                    self.phi0_deg   = st.phi0_deg
                    self.edt_mult.setText(f"{self.multiplier:.3f}")
                    self.edt_offset.setText(f"{self.offset:.3f}")
                    self.spn_phi0.setValue(self.phi0_deg)
                except Exception:
                    pass

        for T_name, members in report.get("T_results", {}).items():
            k_name = f"k__{T_name}"
            val = result.params[k_name].value if k_name in result.params else None
            if val is None:
                print(f"{T_name}: (not in result)  ← {len(members)} targets")
            else:
                # pretty print: s or ms
                T_val = float(1 / val)
                shown = f"{T_val:.4g} s" if T_val >= 1 else f"{T_val*1e3:.4g} ms"
                print(f"{T_name} = {shown} (k={val:.4g} s^-1)  ← {len(members)} targets")

            try:
                rec = self._TSeedRegistry.setdefault(T_name, {"fixed": False, "T_seed_s": None, "T_result_s": None})
                rec["T_result_s"] = float(T_val)
                # optional: if dialog is open, refresh it
                dlg = getattr(self, "_tseed_dlg", None)
                if dlg: dlg.model.refresh()
            except Exception:
                pass

        try:
            nfev = int(result.nfev)
            norm_res = float(np.linalg.norm(result.residual))
            msg = f"Joint fit done on slices {sorted(set(indices))}. nfev={nfev}, |res|={norm_res:.6g}."
            QtWidgets.QMessageBox.information(self, "Joint Fit", msg)
            self.statusBar().showMessage(msg, 4000)
        finally:
            try: self._end_busy()
            except Exception: pass



    def _build_joint_params(self, indices):
        """
        Build joint lmfit.Parameters and per-slice contexts for the selected slices.
        Supports:
          - linear links
          - exponential links with GUI schema:
            driver
            args = {
                "A" float               # optional, default 1.0
                "T_name": "t_1",          # shared fitted parameter
                "T_value": float,         # alternative: fixed T
                "t_override": float,      # seconds, optional
                "C": float,               # optional, default 0.0
            }
        1) obtain global parameters + relaxation time if any
        2) loop through all slices. build joint params of each slice for lmfit and put them in a temporary storage named pending_storage
            a) fetch peak parameters, fix flags
            b) fetch targets and convert to lmfit variables
            c) fetch drivers and convert to lmfit variables
            d) fetch a b in linear linking
            e) fetch A, T, C in exponential linking. Convert T to k with k = 1/T to pass to optimizer
        
        3) check all the slices and write lmfit expression, exp=..., that can be inputed to optimizer 

        Return  params_all, slice_ctxs, report for on_fitting_joint and debug
        """
        """
        New version: do ALL linking in ParamRef-space.
        lmfit names are produced ONLY by ParamRef_to_key(ParamRef).
        Naming rule: s{sid}_p{pid}_{name}
        Globals use pid=-1: s{sid}_p-1_mult, etc.
        """
        # 1) common display / context
        self.current_settings()
        axis_mode = str(self.axis_mode).lower()
        ref_MHz   = float(self.ref)
        sw_hz     = float(self.sw_hz)
        x_disp = self.f2_ppm if (axis_mode == "ppm") else self.f2_hz

        params_all = Parameters()
        slice_ctxs: list[tuple[int, "FitContext", list["Peak"], list[tuple[bool,bool,bool,bool]]]] = []
        report = {"errors": [], "applied_linear": [], "T_results": {}}

        if x_disp is None:
            report["errors"].append("Display axis is not available.")
            return params_all, slice_ctxs, report
        x_disp = x_disp.astype(float)

        slice_times = getattr(self, "t_f1", None)
        if slice_times is None:
            slice_times = getattr(self, "slice_times", None)

        # 2a) build all per-slice params (using ctx.build_params that emits ParamRef-based names)
        joint_slices = sorted(set(int(s) for s in indices))
        for sid in joint_slices:
            st = self.slice_states.get(sid)
            if st is None:
                if not self._bind_slice_direct(sid):
                    continue
                peaks, fix_flags, _ = self.peak_model.return_model()
                peaks = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in peaks]
            else:
                peaks = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in (st.peaks or [])]
                if st.fix_flags and len(st.fix_flags) >= len(peaks):
                    fix_flags = [tuple(ff) for ff in st.fix_flags[:len(peaks)]]
                else:
                    _, fix_flags2, _ = self.peak_model.return_model()
                    fix_flags = [tuple(ff) for ff in (fix_flags2[:len(peaks)] if fix_flags2 else [])]

            if not peaks:
                continue

            y_slice = self.data2d[sid, :].astype(float)
            ctx = FitContext(x_disp, y_slice, axis_mode, ref_MHz, sw_hz)
            ctx.mask_w = self._mask_from_excluded(x_disp)

            # IMPORTANT: build_params must now create:
            #   s{sid}_p-1_mult, s{sid}_p-1_offset, s{sid}_p-1_phi0_deg
            #   s{sid}_p{i}_pos/amp/lor/gauss
            p_s = ctx.build_params(
                peaks=peaks,
                sid=sid,
                multiplier=self.multiplier,
                offset=self.offset,
            )
            for name, par in p_s.items():
                params_all.add(par)

            # bounds (use ParamRef directly)
            for pid in range(len(peaks)):
                apply_bounds_to_param(params_all[ParamRef_to_key(ParamRef(sid, pid, "pos"))],   self.peak_model, sid, pid, "pos",   axis_mode, ref_MHz)
                apply_bounds_to_param(params_all[ParamRef_to_key(ParamRef(sid, pid, "gauss"))], self.peak_model, sid, pid, "gauss", axis_mode, ref_MHz)
                apply_bounds_to_param(params_all[ParamRef_to_key(ParamRef(sid, pid, "amp"))],   self.peak_model, sid, pid, "amp",   axis_mode, ref_MHz)
                apply_bounds_to_param(params_all[ParamRef_to_key(ParamRef(sid, pid, "lor"))],   self.peak_model, sid, pid, "lor",   axis_mode, ref_MHz)

            # fallback bounds if model does not provide
            for pid in range(len(peaks)):
                for nm, default_max in (("lor", 10000.0), ("gauss", 10000.0)):
                    pref = ParamRef(slice_id=sid, peak_id=pid, name=nm)
                    b = self.peak_model.get_bounds_for(pref) if self.peak_model else None
                    if not (b and b.is_set()):
                        k = ParamRef_to_key(pref)
                        params_all[k].min = 0.0
                        params_all[k].max = float(default_max)

                prefA = ParamRef(slice_id=sid, peak_id=pid, name="amp")
                bA = self.peak_model.get_bounds_for(prefA) if self.peak_model else None
                if not (bA and bA.is_set()):
                    params_all[ParamRef_to_key(prefA)].min = 0.0

            # global fixed flags (globals are pid=-1 now)
            params_all[ParamRef_to_key(global_ref(sid, "mult"))].set(vary=not self.fix_mult,   value=self.multiplier)
            params_all[ParamRef_to_key(global_ref(sid, "phi0_deg"))].set(vary=not self.fix_phi0, value=self.phi0_deg)
            params_all[ParamRef_to_key(global_ref(sid, "offset"))].set(vary=not self.fix_offset, value=self.offset)

            # per-peak fixed flags
            for pid in range(len(peaks)):
                fpos, famp, flor, fgauss = fix_flags[pid]
                params_all[ParamRef_to_key(ParamRef(sid, pid, "pos"))].vary   = not fpos
                params_all[ParamRef_to_key(ParamRef(sid, pid, "amp"))].vary   = not famp
                params_all[ParamRef_to_key(ParamRef(sid, pid, "lor"))].vary   = not flor
                params_all[ParamRef_to_key(ParamRef(sid, pid, "gauss"))].vary = not fgauss

            slice_ctxs.append((sid, ctx, [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in peaks], fix_flags))

        if not slice_ctxs:
            report["errors"].append("No slices with peaks to fit.")
            return params_all, slice_ctxs, report

        # 2b) collect links (store ParamRef, not strings)
        pending_linear: list[tuple["ParamRef", "ParamRef", float, float]] = []
        # (tgt_ref, drv_ref, a, b)
        pending_exp: list[tuple["ParamRef", "ParamRef", float, float, str, float]] = []
        # (tgt_ref, drv_ref, A_val, t_sec, k_term, C_val)

        t_by_kname: dict[str, list[float]] = {} # k__T_name → list of t_sec used
        Tname_for_k: dict[str, str] = {} # k__T_name → T_name

        for sid in joint_slices:
            try:
                reg = self._build_registry_for_slice(sid)
                _seed_external_drivers_into_registry(
                    registry=reg,
                    slice_states=self.slice_states,
                    current_slice_id=sid,
                    links=self.link_store,
                    strict=True,
                )
                links_subset = _links_for_target_slice(self.link_store, sid)

                for pref_tgt, _entry in reg.items():
                    if int(pref_tgt.slice_id) != sid:
                        continue
                    if not self.link_store.is_linked(pref_tgt):
                        continue

                    tgt_name = canon_name(pref_tgt.name)
                    if tgt_name not in ("pos", "amp", "lor", "gauss"):
                        continue

                    tgt_ref = ParamRef(slice_id=int(pref_tgt.slice_id),
                                       peak_id=int(pref_tgt.peak_id),
                                       name=tgt_name)

                    expr = links_subset.get(pref_tgt)
                    if not expr or expr.driver is None:
                        continue

                    drv = expr.driver
                    drv_name = canon_name(drv.name)
                    if drv_name not in ("pos", "amp", "lor", "gauss"):
                        report["errors"].append(
                            f"Link target {ParamRef_to_key(tgt_ref)} has unsupported driver name '{drv.name}'."
                        )
                        return params_all, slice_ctxs, report

                    drv_ref = ParamRef(slice_id=int(drv.slice_id),
                                       peak_id=int(drv.peak_id),
                                       name=drv_name)

                    if expr.type == LinkType.LINEAR:
                        a = float(expr.args.get("a", 1.0))
                        b = float(expr.args.get("b", 0.0))
                        pending_linear.append((tgt_ref, drv_ref, a, b))
                        continue

                    if expr.type == LinkType.RELAX_EXP:
                        A_val = float(expr.args.get("A", 1.0))

                        # time term
                        if "t_override" in expr.args:
                            t_sec = float(expr.args["t_override"])
                        else:
                            if slice_times is None:
                                report["errors"].append(
                                    f"Exponential link on {ParamRef_to_key(tgt_ref)} requires time data "
                                    "(load time_echo.txt or set t_override)."
                                )
                                return params_all, slice_ctxs, report
                            try:
                                t_sec = float(slice_times[sid])
                            except Exception:
                                report["errors"].append(
                                    f"Exponential link on {ParamRef_to_key(tgt_ref)} cannot find time for slice {sid}."
                                )
                                return params_all, slice_ctxs, report

                        # k term (shared fitted k__T_name, or numeric literal)
                        if "T_name" in expr.args:
                            T_name = str(expr.args["T_name"]).strip()
                            if not T_name:
                                report["errors"].append(f"Exponential link on {ParamRef_to_key(tgt_ref)} has empty T_name.")
                                return params_all, slice_ctxs, report

                            k_name = f"k__{T_name}"
                            Tname_for_k[k_name] = T_name
                            t_by_kname.setdefault(k_name, []).append(float(t_sec))

                            if k_name not in params_all:
                                params_all.add(k_name, value=1000.0, min=1e-9)
                                params_all[k_name].set(vary=True)

                            seed_for_T = _get_seed(self, T_name)
                            if isinstance(seed_for_T, dict):
                                k_lo, k_hi = _T_bounds_to_k_bounds(seed_for_T.get("T_lo_s"), seed_for_T.get("T_hi_s"))
                                apply_Tbounds_to_param(params_all[k_name], k_lo, k_hi)

                            report["T_results"].setdefault(T_name, []).append(ParamRef_to_key(tgt_ref))
                            k_term = k_name

                        elif "T_value" in expr.args:
                            T_val = float(expr.args["T_value"])
                            if T_val <= 0:
                                report["errors"].append(f"Exponential link on {ParamRef_to_key(tgt_ref)} has non-positive T_value.")
                                return params_all, slice_ctxs, report
                            k_term = f"{(1.0 / T_val):.12g}"

                        elif "T" in expr.args:
                            T_val = float(expr.args["T"])
                            if T_val <= 0:
                                report["errors"].append(f"Exponential link on {ParamRef_to_key(tgt_ref)} has non-positive T.")
                                return params_all, slice_ctxs, report
                            k_term = f"{(1.0 / T_val):.12g}"

                        else:
                            report["errors"].append(f"Exponential link on {ParamRef_to_key(tgt_ref)} has no T_name nor T_value.")
                            return params_all, slice_ctxs, report

                        C_val = float(expr.args.get("C", 0.0))
                        pending_exp.append((tgt_ref, drv_ref, A_val, float(t_sec), str(k_term), float(C_val)))
                        continue

            except Exception as e:
                report["errors"].append(f"Link error (slice {sid}): {e}")
                return params_all, slice_ctxs, report

        # --- finalize per-T_name bounds and seed k initial values  ---
        eps = 0.001
        for k_name, t_list in t_by_kname.items():
            t_pos = [abs(float(t)) for t in t_list if t is not None]
            if not t_pos:
                continue

            t_min = min(t_pos)
            t_max = max(t_pos)

            k_min_data = eps / t_max
            k_max_data = 14.0 / t_min

            if k_name not in params_all:
                continue
            par = params_all[k_name]

            old_min = par.min
            old_max = par.max

            new_min = max(0.0, float(k_min_data))
            new_max = float(k_max_data)

            if old_min is not None:
                try: new_min = max(new_min, float(old_min))
                except Exception: pass
            if old_max is not None:
                try: new_max = min(new_max, float(old_max))
                except Exception: pass

            if not np.isfinite(new_min) or not np.isfinite(new_max):
                if np.isfinite(old_min): new_min = float(old_min)
                if np.isfinite(old_max): new_max = float(old_max)

            if new_max <= new_min:
                span = max(abs(new_min), 1.0) * 1e-3
                new_max = new_min + span

            par.min = new_min
            par.max = new_max

            T_name = Tname_for_k.get(k_name, "")
            seed = _get_seed(self, T_name) if T_name else None
            if isinstance(seed, dict):
                fixed = bool(seed.get("fixed", False))
                T_seed_s = seed.get("T_seed_s", None)
                if T_seed_s and T_seed_s > 0:
                    k_seed = 1.0 / float(T_seed_s)
                    if par.min is not None: k_seed = max(k_seed, float(par.min))
                    if par.max is not None: k_seed = min(k_seed, float(par.max))
                    par.set(value=k_seed)
                par.set(vary=not fixed)
                report.setdefault("T_seed_applied", {})[T_name] = {
                    "fixed": fixed,
                    "T_seed": float(T_seed_s) if T_seed_s else None,
                    "k_bounds": (par.min, par.max),
                }

        # 3) Apply links (only here convert ParamRef -> lmfit name)
        try:
            missing = []

            # 3a) linear
            for tgt_ref, drv_ref, a, b in pending_linear:
                tgt_key = ParamRef_to_key(tgt_ref)
                drv_key = ParamRef_to_key(drv_ref)

                if tgt_key not in params_all:
                    missing.append(f"target {tgt_key}")
                    continue
                if drv_key not in params_all:
                    missing.append(f"driver {drv_key}")
                    continue

                par = params_all[tgt_key]
                par.set(expr=f"{a}*{drv_key}+{b}")
                report["applied_linear"].append((tgt_key, drv_key))
                try:
                    par.min = -np.inf
                    par.max =  np.inf
                except Exception:
                    pass

            # 3b) exponential
            for tgt_ref, drv_ref, A_val, t_sec, k_term, C_val in pending_exp:
                tgt_key = ParamRef_to_key(tgt_ref)
                drv_key = ParamRef_to_key(drv_ref)

                if tgt_key not in params_all:
                    missing.append(f"target {tgt_key} (exponential)")
                    continue
                if drv_key not in params_all:
                    missing.append(f"driver {drv_key} (exponential)")
                    continue

                par = params_all[tgt_key]
                par.set(expr=f"{drv_key}*({A_val})*exp(-({t_sec})*{k_term})+{C_val:.6g}")
                try:
                    par.min = -np.inf
                    par.max =  np.inf
                except Exception:
                    pass

            if missing:
                report["errors"].append("Cannot apply links; missing: " + "; ".join(missing))
                return params_all, slice_ctxs, report

            # sanity: linked targets should be derived (expr set) and not vary
            bad = []
            for tgt_ref, _drv_ref, _a, _b in pending_linear:
                tgt_key = ParamRef_to_key(tgt_ref)
                if tgt_key in params_all:
                    tpar = params_all[tgt_key]
                    if not getattr(tpar, "expr", ""):
                        bad.append(f"{tgt_key} (no expr)")
                    elif getattr(tpar, "vary", True):
                        bad.append(f"{tgt_key} (vary=True)")
            if bad:
                report["errors"].append("Some linked targets are not derived parameters: " + "; ".join(bad))
                return params_all, slice_ctxs, report

        except Exception as e:
            report["errors"].append(f"Joint Fit – Expression error: {e}")
            return params_all, slice_ctxs, report

        return params_all, slice_ctxs, report

    
    def on_show_statistics(self):
        stats = getattr(self, "fit_stats", None)
        if not stats:
            QtWidgets.QMessageBox.information(
                self, "Statistics", "No fit statistics available."
            )
            return

        dlg = StatsView(self)
        dlg.set_stats(stats)
        dlg.exec_()


    def on_seed_saved(self, T_name: str, T_seconds: float, use_flag: bool):
        # LinkEditor may choose to stash a seed; table remains source of truth.
        if not hasattr(self, "_TSeedRegistry"):
            self._TSeedRegistry = {}
        rec = self._TSeedRegistry.setdefault(T_name, {"fixed": False, "T_seed_s": None, "T_result_s": None})
        # We store the provided seed; 'use_flag' from old API maps to not-fixed seeding.
        rec["T_seed_s"] = float(T_seconds) if (T_seconds and T_seconds > 0) else None
        # If the user asked to "use for next fits", keep varying (fixed=False); else consider fixing.
        rec["fixed"] = False if bool(use_flag) else rec.get("fixed", False)
        rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
        # reflect in open dialog
        dlg = getattr(self, "_tseed_dlg", None)
        if dlg: 
            try: dlg.model.refresh()
            except Exception: pass



   
    def on_simulate(self):
        try:
            # 0) guards
            if self.x is None or self.y is None:
                QMessageBox.information(self, 'Simulate', 'Load a spectrum first.')
                return

            # 1) read UI → state (mult, phi0, offset, axis, SW, etc.)
            self.current_settings()

            # 2) sync peakTableModel → self.peaks (read pos/amp/lor/gauss and fix flags)
            self.Model_to_Peaks()

            if not self.peaks:
                QMessageBox.information(self, 'Simulate', 'No peaks to simulate. Add peaks or Auto-Pick first.')
                # still refresh to clear any stale model curve
                self.refresh_plot(show_model=False, preserve_view=True)
                return

            # 3) simply re-draw; refresh_plot already builds model_spectrum and applies φ0/offset
            self._save_slice_state(self.slice_index, has_fit=True)
            st = self.slice_states.get(self.slice_index)
            if st is not None:
                self.refresh_plot_from_state(st, preserve_view=True)

            self.status.showMessage('Simulated model updated', 2500)
        
        except ValueError as e:
            QMessageBox.warning(self, "Input Required", str(e))
            return
          
    def on_copy_params(self):
        self.slice_index = getattr(self, "slice_index", 0)
        src_state = self.slice_states[self.slice_index]
        n_traces = int(self.data2d.shape[0])        
        for i in range(n_traces):
            dst_state = self.slice_states[i]
            dst_state.peaks = [Peak(p.pos, p.amp, p.lor_hz, p.gauss_disp) for p in src_state.peaks]
            dst_state.fix_flags = [tuple(f) for f in src_state.fix_flags]

    def on_copy_bounds(self) -> None:
        """
        Copy all peak-parameter bounds (pos/amp/lor/gauss) from the current slice
        to every other slice.

        Uses the PeakTableModel._bounds store keyed by ParamRef(slice, peak, name).
        """
        mdl = getattr(self, "peak_model", None)
        data2d = getattr(self, "data2d", None)

        if mdl is None or not hasattr(mdl, "_bounds"):
            return

        src_sid = int(getattr(self, "slice_index", 0))

        # Determine how many slices we have (same logic as on_copy_params)
        if data2d is not None and hasattr(data2d, "shape") and data2d.ndim == 2:
            n_traces = int(data2d.shape[0])
        else:
            # Fallback: use slice_states dict
            states = getattr(self, "slice_states", {}) or {}
            n_traces = max(1, len(states) or 1)

        if n_traces <= 1:
            # Nothing to copy in pure 1D mode
            return

        # Collect all *set* bounds for the source slice
        src_bounds: list[tuple[ParamRef, ParamBounds]] = []
        for pref, b in mdl._bounds.items():
            try:
                sid = int(pref.slice_id)
            except Exception:
                continue
            if sid != src_sid:
                continue
            if not (b and b.is_set()):
                continue
            src_bounds.append((pref, b))

        if not src_bounds:
            QtWidgets.QMessageBox.information(
                self, "Copy bounds",
                "No bounds defined in the current slice to copy."
            )
            return

        # Replicate those bounds into every other slice
        for dst_sid in range(n_traces):
            if dst_sid == src_sid:
                continue
            for pref_src, b in src_bounds:
                new_pref = ParamRef(
                    slice_id=dst_sid,
                    peak_id=int(pref_src.peak_id),
                    name=str(pref_src.name),
                )
                mdl.set_bounds_for(new_pref, lo=b.lo, hi=b.hi)

        # Let the model/view know something changed (tooltips, etc.)
        mdl.layoutChanged.emit()


    def on_export_ascii(self):
        """
        Export current spectrum, model, and time-domain FID to ASCII text.
        Columns: x_hz, x_ppm, y_model, y_data, t_s, fid_real, fid_imag
        """
        # --- Guards ---
        if getattr(self, "x", None) is None or getattr(self, "y", None) is None:
            QtWidgets.QMessageBox.information(self, "Export ASCII", "No data to export.")
            return
        if not hasattr(self, "ref") or not self.ref:
            QtWidgets.QMessageBox.warning(self, "Export ASCII", "Missing ref; cannot compute ppm.")
            return
        if not hasattr(self, "sw_hz") or not self.sw_hz or self.sw_hz <= 0:
            QtWidgets.QMessageBox.warning(self, "Export ASCII", "Missing/invalid SW (Hz).")
            return
        # --- Suggested filename ---
        base = getattr(self, "data_file", None) or getattr(self, "data_path", None) or getattr(self, "dataset_label", None) or "spectrum"
        try:
            stem = Path(str(base)).stem if isinstance(base, (str, Path)) else "spectrum"
        except Exception:
            stem = "spectrum"
        suggested_name = f"{stem}_export_spectrum.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Export ASCII", suggested_name, "Text files (*.txt)")
        if not path:
            return
        # --- Ensure UI state synced ---
        if hasattr(self, "Model_to_Peaks"):
            self.Model_to_Peaks()
        if hasattr(self, "current_settings"):
            try:
                self.current_settings()
            except Exception:
                pass
        # --- Axes & data on DISPLAY frame ---
        x_disp = np.asarray(self.x, dtype=float)
        is_ppm = (getattr(self, "axis_mode", "hz").lower() == "ppm")
        MHz = float(self.ref)
        x_hz = x_disp * MHz if is_ppm else x_disp
        x_ppm = x_hz / MHz
        y_data = np.asarray(self.y, dtype=float)
        # --- Build t, fid, spec via orchestrator (math only) ---
        # (No phasing/offset here; matches residual/preview core.)
        t_s, fid, spec_c = model_spectrum(
           peaks=self.peaks,
           axis_mode=getattr(self, "axis_mode", "hz"),
           ref=self.ref,                   # MHz
           sw_hz=float(self.sw_hz),        # Hz
           N=len(x_disp),
           x=x_disp,                       # orchestrator will canonicalize if ppm
           multiplier=float(getattr(self, "multiplier", 1.0)),
           return_fid=True
       )
       # --- Presentation parity: φ0 + offset only (φ1 removed) ---
        phi0_deg = float(getattr(self, "phi0_deg", getattr(self, "phi0", 0.0)))
        offset   = float(getattr(self, "offset", 0.0))
        y_model  = apply_phase_and_offset(spec_c, phi0_deg, offset)
        if y_model.shape != y_data.shape:
            QtWidgets.QMessageBox.warning(self, "Export ASCII", "Model/data size mismatch after reconstruction.")
            return
        # --- Metadata ---
        sw_grid_hz = float(abs(x_hz[-1] - x_hz[0])) if x_hz.size > 1 else ""
        meta = [
            "# exporter = v14 ascii-phi1-free",
            f"# datetime_utc = {datetime.utcnow().isoformat()}Z",
            f"# ref_MHz = {self.ref}",
            f"# axis_mode_at_export = {getattr(self, 'axis_mode', '')}",
            f"# npoints = {x_hz.size}",
            f"# sw_grid_hz = {sw_grid_hz}",
            f"# sw_acq_hz = {float(self.sw_hz)}",
            f"# time_unit = s",
            f"# fid_source = model_spectrum -> build_fid_from_peaks + peakSim_kernel",
            f"# multiplier = {getattr(self, 'multiplier', '')}",
            f"# offset = {offset}",
            f"# phi0_deg = {phi0_deg}",
            "# phi1_deg_per_Hz = 0.0 (disabled)",
        ]
        if hasattr(self, "excluded_regions"):
            meta.append(f"# excluded_regions = {self.excluded_regions}")
        # --- Write ---
        try:
            with open(path, "w", encoding="utf-8-sig") as f:
                for line in meta:
                    f.write(line + "\n")
                f.write("x_hz\tx_ppm\ty_model\ty_data\tt_s\tfid_real\tfid_imag\n")
                for hz, ppm, ym, yd, t, fr, fi in zip(
                    x_hz, x_ppm, y_model, y_data, t_s, np.real(fid), np.imag(fid)
                ):
                    f.write(f"{hz:.6f}\t{ppm:.6f}\t{ym:.6e}\t{yd:.6e}\t{t:.9e}\t{fr:.6e}\t{fi:.6e}\n")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export ASCII", f"Failed to save: {e}")
            return
        QtWidgets.QMessageBox.information(self, "Export ASCII", f"Saved: {path}")

    def on_export_peak_table(self):
        def get_slice_for_export(idx: int):
            """Return a state-like object with attributes used by the exporter."""
            st = getattr(self, "slice_states", {}).get(idx, None)
            if st is not None:
                return st
            # Fallback for current slice: synthesize from UI + model (no direct view pokes)
            if idx == getattr(self, "slice_index", 0):
                # sync globals from UI widgets
                try:
                    self.offset = float(self.edt_offset.text() or 0.0)
                except Exception:
                    self.offset = float(getattr(self, "offset", 0.0))
                try:
                    self.multiplier = float(self.edt_mult.text() or 1.0)
                except Exception:
                    self.multiplier = float(getattr(self, "multiplier", 1.0))
                try:
                    self.phi0_deg = float(self.spn_phi0.value())
                except Exception:
                    self.phi0_deg = float(getattr(self, "phi0_deg", 0.0))
                # Read from model (single source of truth)
                peaks_now, fix_now, redchi = self.peak_model.return_model()
                class _Now: ...
                now = _Now()
                now.peaks = list(peaks_now)
                now.fix_flags = list(fix_now)
                now.redchi = float(redchi)
                now.axis_mode = str(getattr(self, "axis_mode", "ppm")).lower()
                now.mult = float(self.multiplier)
                now.offset = float(self.offset)
                now.phi0_deg = float(self.phi0_deg)
                now.excluded = list(getattr(self, "excluded", []))
                return now
            return None

        def _build_rows_from_state(state):
            """Build exporter rows; prefer state.fix_flags if available."""
            rows = []
            # if state carries fix flags, use them; else read from current model
            state_fix = getattr(state, "fix_flags", None)
            if state_fix is None:
                _, state_fix, _ = self.peak_model.return_model()
            for i, pk in enumerate(getattr(state, "peaks", []) or []):
                if i < len(state_fix):
                    fix_pos, fix_amp, fix_lor, fix_gauss = state_fix[i]
                else:
                    fix_pos, fix_amp, fix_lor, fix_gauss = (False, False, True, False)
                rows.append({
                    "id": f"p{i}",
                    "pos": float(pk.pos),
                    "pos_fix": 1 if fix_pos else 0,
                    "amp": float(pk.amp),
                    "amp_fix": 1 if fix_amp else 0,
                    "lor_hz": float(pk.lor_hz),
                    "lor_fix": 1 if fix_lor else 0,
                    "gauss_disp": float(pk.gauss_disp),
                    "gauss_fix": 1 if fix_gauss else 0,
                })
            return rows

        # --- dialog (same UX as v5) ---
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export Peak Tables")
        vbox = QtWidgets.QVBoxLayout(dlg)
        r_all = QtWidgets.QRadioButton("All slices (batch)")
        r_one = QtWidgets.QRadioButton("One slice")
        r_one.setChecked(True)
        vbox.addWidget(r_all)
        vbox.addWidget(r_one)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Slice index:"))
        spn = QtWidgets.QSpinBox()
        n_slices = (self.data2d.shape[0] if getattr(self, "data2d", None) is not None else
                    max(1, len(getattr(self, "slice_states", {})) or 1))
        spn.setRange(0, max(0, n_slices - 1))
        spn.setValue(getattr(self, "slice_index", 0))
        spn.setEnabled(True)
        r_one.toggled.connect(spn.setEnabled)
        h.addWidget(spn)
        vbox.addLayout(h)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        vbox.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        label = getattr(self, "dataset_label", "") or "peaks"
        base_dir = getattr(self, "default_save_dir", str(Path.home()))
        data_file = getattr(self, "data_file", "")
        ref_mhz = float(getattr(self, "ref", 0.0) or 0.0)

        try:
            if r_all.isChecked():
                out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", base_dir)
                if not out_dir:
                    return

                indices = sorted(getattr(self, "slice_states", {}).keys()) or list(range(n_slices))
                wrote_any = False
                for idx in indices:
                    st = get_slice_for_export(idx)
                    if st is None or not getattr(st, "peaks", None):
                        continue

                    rows = _build_rows_from_state(st)
                    excluded_tmp = getattr(st, "excluded", None)
                    if not excluded_tmp:
                        excluded_tmp = getattr(self, "excluded", []) or []
                    excluded_exported = [{"x_min": float(lo), "x_max": float(hi)} for (lo, hi) in excluded_tmp]

                    globals_fix_flags = {
                        "offset":     1 if getattr(self, "chk_fix_offset", None) and self.chk_fix_offset.isChecked() else 0,
                        "multiplier": 1 if getattr(self, "chk_fix_mult", None)   and self.chk_fix_mult.isChecked()   else 0,
                        "phi0_deg":   1 if getattr(self, "chk_fix_phi0", None)   and self.chk_fix_phi0.isChecked()   else 0,
                    }

                    fname = f"{label}__s{idx:02d}_fit.txt"
                    path = os.path.join(out_dir, fname)

                    export_peak_table(
                        path=path,
                        peaks=rows,                   # keep v5 key for backward compat
                        ref=ref_mhz,
                        globals_meta={
                            "offset": float(getattr(st, "offset", getattr(self, "offset", 0.0))),
                            "multiplier": float(getattr(st, "mult", getattr(self, "multiplier", 1.0))),
                            "phi0_deg": float(getattr(st, "phi0_deg", getattr(self, "phi0_deg", 0.0))),
                        },
                        globals_fix_flags=globals_fix_flags,
                        excluded=excluded_exported,
                        fit_stats=getattr(self, "fit_stats", None),
                        data_file=data_file,
                        program_version=getattr(self, "program_version", "mpFit_v6"),
                        extra_meta={
                            "AxisMode": getattr(st, "axis_mode", str(getattr(self, "axis_mode", "hz"))).lower(),
                            "SliceIndex": str(idx), "Redchi": f"{getattr(st, 'redchi', 'nan')}",
                        },
                    )
                    wrote_any = True

                QtWidgets.QMessageBox.information(self, "Export Peak Tables",
                                                  "Batch export completed." if wrote_any else "No slices to export.")

            else:
                idx = int(spn.value())
                st = get_slice_for_export(idx)
                if st is None or not getattr(st, "peaks", None):
                    QtWidgets.QMessageBox.information(self, "Export Peak Table", f"No peaks for slice {idx}.")
                    return

                suggested = f"{label}__s{idx:02d}_fit.txt"
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Export Peak Table", os.path.join(base_dir, suggested),
                    "Fit text (*_fit.txt);;Text (*.txt);;All files (*)"
                )
                if not path:
                    return

                rows = _build_rows_from_state(st)
                excluded_tmp = getattr(st, "excluded", None)
                if not excluded_tmp:
                    excluded_tmp = getattr(self, "excluded", []) or []
                excluded_exported = [{"x_min": float(lo), "x_max": float(hi)} for (lo, hi) in excluded_tmp]
                globals_fix_flags = {
                    "offset":     1 if getattr(self, "chk_fix_offset", None) and self.chk_fix_offset.isChecked() else 0,
                    "multiplier": 1 if getattr(self, "chk_fix_mult", None)   and self.chk_fix_mult.isChecked()   else 0,
                    "phi0_deg":   1 if getattr(self, "chk_fix_phi0", None)   and self.chk_fix_phi0.isChecked()   else 0,
                }

                export_peak_table(
                    path=path,
                    peaks=rows,                   # keep v5 key for backward compat
                    ref=ref_mhz,
                    globals_meta={
                        "offset": float(getattr(st, "offset", getattr(self, "offset", 0.0))),
                        "multiplier": float(getattr(st, "mult", getattr(self, "multiplier", 1.0))),
                        "phi0_deg": float(getattr(st, "phi0_deg", getattr(self, "phi0_deg", 0.0))),
                    },
                    globals_fix_flags=globals_fix_flags,
                    excluded=excluded_exported,
                    fit_stats=getattr(self, "fit_stats", None),
                    data_file=data_file,
                    program_version=getattr(self, "program_version", "mpFit_v6"),
                    extra_meta={
                        "AxisMode": getattr(st, "axis_mode", str(getattr(self, "axis_mode", "hz"))).lower(),
                        "SliceIndex": str(idx), "Redchi": f"{getattr(st, 'redchi', 'nan')}",
                    },
                )
                QtWidgets.QMessageBox.information(self, "Export Peak Table", "Peak table saved.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Peak Table", f"Failed to save:\n{e}")

    def on_import_peak_table(self):
        """
        Import peaks + per-peak fix flags into the QAbstractTableModel.
        Uses peak_table_io_v3.i
        """
        start_dir = getattr(self, "default_open_dir", "") or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Peak Table…", start_dir, "txt(*.txt);;CSV (*.csv);;All files (*.*)"
        )
        if not path:
            return

        try:
            rows, meta = import_peak_table(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Peaks", str(e))
            return

        # Apply axis mode (respect ppm only if MHz available)
        axis_mode = str(meta.get("axis_mode", getattr(self, "axis_mode", "hz"))).lower()
        if axis_mode == "ppm" and self.has_mhz_if_needed():
            try:
                self.axis_mode = "ppm"
                self.cmb_axis.setCurrentText("ppm")
            except Exception:
                self.axis_mode = "ppm"
        else:
            try:
                self.axis_mode = "hz"
                self.cmb_axis.setCurrentText("hz")
            except Exception:
                self.axis_mode = "hz"

        # Update globals (values)
        g = meta.get("globals", {})
        try:
            self.multiplier = float(g.get("multiplier", getattr(self, "multiplier", 1.0)))
            self.offset     = float(g.get("offset", getattr(self, "offset", 0.0)))
            self.phi0_deg   = float(g.get("phi0_deg", getattr(self, "phi0_deg", 0.0)))
            if hasattr(self, "edt_mult"):   self.edt_mult.setText(f"{self.multiplier:.3f}")
            if hasattr(self, "edt_offset"): self.edt_offset.setText(f"{self.offset:.3f}")
            if hasattr(self, "spn_phi0"):   self.spn_phi0.setValue(self.phi0_deg)
        except Exception:
            pass

        # Update global fix flags (checkboxes)
        gf = meta.get("globals_fix", {})
        try:
            self.fix_mult   = bool(gf.get("fix_mult", getattr(self, "fix_mult", True)))
            self.fix_phi0   = bool(gf.get("fix_phi0", getattr(self, "fix_phi0", True)))
            self.fix_offset = bool(gf.get("fix_offset", getattr(self, "fix_offset", True)))
            if hasattr(self, "chk_fix_mult"):   self.chk_fix_mult.setChecked(self.fix_mult)
            if hasattr(self, "chk_fix_phi0"):   self.chk_fix_phi0.setChecked(self.fix_phi0)
            if hasattr(self, "chk_fix_offset"): self.chk_fix_offset.setChecked(self.fix_offset)
        except Exception:
            pass

        # Excluded regions (already in display units)
        try:
            exc = meta.get("excluded", []) or []
            self.excluded = [(float(d["x_min"]), float(d["x_max"])) for d in exc if "x_min" in d and "x_max" in d]
        except Exception:
            self.excluded = []

        # Build Peak objects directly from rows (handles gauss_hz → display units)
        # Accept both "fix_pos" or "pos_fix" naming, same for others.
        new_peaks = []
        fix_flags = []
        redchi = None

        ref_mhz = float(getattr(self, "ref_Hz", 0.0)) if hasattr(self, "ref_Hz") and self.ref is not None else 0.0
        for i, r in enumerate(rows):
            try:
                pos = float(r.get("pos"))
                amp = float(r.get("amp"))
                lor_hz = float(r.get("lor_hz"))
                gauss_disp = float(r.get("gauss_disp"))

                new_peaks.append(Peak(pos=pos, amp=amp, lor_hz=lor_hz, gauss_disp=gauss_disp))
                # Fix flags: accept both "fix_pos" and "pos_fix" keys.
                fpos   = bool(r.get("fix_pos", r.get("pos_fix", False)))
                famp   = bool(r.get("fix_amp", r.get("amp_fix", False)))
                flor   = bool(r.get("fix_lor", r.get("lor_fix", True)))
                fgauss = bool(r.get("fix_gauss", r.get("gauss_fix", False)))
                fix_flags.append((fpos, famp, flor, fgauss))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Import Peaks", f"Bad row #{i+1}: {e}")
                return
        # Store and mirror into the model
        self.peaks = new_peaks
        self.peak_model.replace_data(self.peaks, fix_flags, redchi)
        self._fix_flags_cache = [tuple(ff) for ff in fix_flags]

        # Save slice state and redraw with preserved view
        self._save_slice_state(int(getattr(self, "slice_index", 0)), has_fit=bool(meta.get("has_fit", False)))
        st = self.slice_states.get(int(getattr(self, "slice_index", 0)))
        if st is not None:
            self.refresh_plot_from_state(st, preserve_view=True)
        else:
            self.refresh_plot(preserve_view=True)

        self.update_ui_state()
        self.statusBar().showMessage(f"Peak table imported: {Path(path).name}", 3500)

    def _normalize_range(self, a: float, b: float):
        lo, hi = (float(a), float(b))
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def on_toggle_exclude_mode(self, checked: bool):
        """Enable/disable drag-to-exclude mode; also neutralize other tools."""
        
        if not self.has_data():
            self.btn_excluded.setChecked(False)
        self._bind_spanselector()
        self._span_exc.set_active(bool(checked))
        if checked:
            # turn off Add-Peaks and toolbar modes so drag works
            if hasattr(self, 'tgl_add'):
                self.tgl_add.setChecked(False)
            if hasattr(self, 'nav') and hasattr(self.nav, 'mode'):
                self.nav.mode = ''
                for act in self.nav.actions():
                    try:
                        t = act.text().lower()
                    except Exception:
                        t = ''
                    if 'zoom' in t or 'pan' in t:
                        act.setChecked(False)
        self.status.showMessage("Exclude: drag on plot to add range" if checked else "Exclude mode off")
    
    def _bind_spanselector(self):
        ax = getattr(self.canvas, "ax", None)
        ss = getattr(self, "_span_exc", None)
        # If no selector yet, or it's attached to a different axes → rebuild
        need_new = (ss is None) or (getattr(ss, "ax", None) is not ax)
        if need_new:
            try:
                if ss is not None:
                    ss.set_active(False)
                    ss.disconnect_events()
            except Exception:
                pass
            self._span_exc = SpanSelector(
                ax,
                onselect=self.on_span_exclude,
                direction="horizontal",
                useblit=True,
                interactive=False,
            )
        toggle_on = bool(getattr(self, "btn_excluded", None) and self.btn_excluded.isChecked())
        self._span_exc.set_active(toggle_on)

    def on_span_exclude(self, x_min, x_max):
        """Callback from SpanSelector when user drags a region on the plot."""
        if self.x is None or self.y is None:
            return
        lo, hi = self._normalize_range(x_min, x_max)
        if hi - lo <= 0:
            return
        self.excluded.append((lo, hi))        
        self.refresh_plot(preserve_view=True)
        self._update_excluded_tooltip()

    def on_exclude_add_manual(self):
        """Add an exclusion by typing min/max."""
        if self.x is None:
            return
        lo, ok1 = QtWidgets.QInputDialog.getDouble(self, "Exclude Range", "Min (x):", float(self.x[0]))
        if not ok1: return
        hi, ok2 = QtWidgets.QInputDialog.getDouble(self, "Exclude Range", "Max (x):", float(self.x[-1]))
        if not ok2: return
        lo, hi = self._normalize_range(lo, hi)
        self.excluded.append((lo, hi))        
        self.refresh_plot(preserve_view=True)
        self._update_excluded_tooltip()

    def on_exclude_remove_selected(self):
        
        if self.excluded:
            self.excluded.pop()        
        self.refresh_plot(preserve_view=True)
        self._update_excluded_tooltip()

    def on_exclude_clear(self):
        self.excluded.clear()
        self.refresh_plot(preserve_view=True)
        self._update_excluded_tooltip()

    def _update_excluded_tooltip(self):
        """Show tooltip summary of excluded regions."""
        n = len(getattr(self, "excluded", []) or [])
        count_txt = f"{n} excluded range(s)" if n else "No excluded regions" 
        if n==0:
            tip = "No excluded regions"
        else:
            regions = [(min(a, b), max(a, b)) for a, b in self.excluded]
            parts = [f"({lo:.3f}, {hi:.3f})" for lo, hi in regions[:6]]
            if len(regions) > 8:
                parts.append(f"... +{len(regions)-6} more")
            tip = f"{len(regions)} excluded regions — " + "; ".join(parts)
            if hasattr(self, "axis_mode"):
                tip += f"  [{self.axis_mode.upper()}]" 
        if hasattr(self, "btn_excluded"):
            self.btn_excluded.setToolTip(tip)

        # ---------- helper to draw excluded spans (single source of truth) ----------
    def _draw_excluded(self, ax, spans):
        if not spans:
            return
        for lo, hi in spans:
            try:
                ax.axvspan(float(lo), float(hi), alpha=0.25, color="0.8", zorder=-5)
            except Exception:
                pass

    def _mask_from_excluded(self, x: np.ndarray) -> np.ndarray:
        """
        Return a weight array w with 1.0 outside excluded regions and 0.0 inside.
        x is in display units matching self.axis_mode.
        """
        if not self.excluded:
            return np.ones_like(x, dtype=float)
        w = np.ones_like(x, dtype=float)
        for (lo, hi) in self.excluded:
            w[(x >= lo) & (x <= hi)] = 0.0
        return w
    
        # Predicates to avoid crashing when an input is invalid
    def _val_positive(self, v) -> bool:
        try:
            return v is not None and float(v) > 0.0
        except Exception:
            return False

    def has_data(self) -> bool:

        x = getattr(self, "x", None)
        y = getattr(self, "y", None)
        return (isinstance(x, np.ndarray) and x.size > 1 and
                isinstance(y, np.ndarray) and y.size == x.size)

    def has_sw(self) -> bool:
        return self._val_positive(getattr(self, "sw_hz", None))

    def has_mhz_if_needed(self) -> bool:
        mode = str(getattr(self, "axis_mode", "hz")).lower()
        if mode == "ppm":
            return self._val_positive(getattr(self, "ref", None))
        return True  # Hz axis needs no MHz

    def has_peaks(self) -> bool:
        peaks = getattr(self, "peaks", None)
        return bool(peaks)  # simulate preview often needs peaks

    def ready_for_fit(self) -> bool:
        # Fit can operate without peaks in some flows???
        return self.has_data() and self.has_sw() and self.has_mhz_if_needed()

    def ready_for_sim(self) -> bool:
        # Simulate preview typically needs peaks
        return self.ready_for_fit() and self.has_peaks()

    def _mark_invalid(self, lineedit, invalid: bool):
        if lineedit is None:
            return
        try:
            lineedit.setStyleSheet("background-color: #ffcccc;" if invalid else "")
            if invalid:
                lineedit.setToolTip("Required and must be > 0")
            else:
                lineedit.setToolTip("")
        except Exception:
            pass

    def _set_enabled_if_exists(self, name: str, enabled: bool):
        w = getattr(self, name, None)
        if w is None:
            return
        try:
            w.setEnabled(enabled)
        except Exception:
            pass
    
    def _begin_busy(self):
        self._ui_busy += 1
        self.update_ui_state()
    def _end_busy(self):
            self._ui_busy = max(0, self._ui_busy - 1)
            self.update_ui_state()

    def update_ui_state(self):
        """
        Central place to:
          - compute readiness
          - update field highlights
          - enable/disable buttons or actions
          - set hz_per_ppm only when MHz is valid
          - disable when fit is running
        """
        # 1) Re-read current text boxes into the model (non-throwing)
        self.sw_hz = self._read_line_float(getattr(self, "edt_sw", None)) if hasattr(self, "edt_sw") else getattr(self, "sw_hz", None)
        self.ref = self._read_line_float(getattr(self, "edt_ref", None)) if hasattr(self, "edt_ref") else getattr(self, "ref", None)

        # 2) Compute readiness
        fit_ready = self.ready_for_fit()
        sim_ready = self.ready_for_sim()
        
        is_busy = getattr(self, "_ui_busy", 0) > 0
        
        # 3) Visual feedback for invalid/missing fields
        edt_sw = getattr(self, "edt_sw", None)
        edt_ref = getattr(self, "edt_ref", None)

        need_mhz = (str(getattr(self, "axis_mode", "hz")).lower() == "ppm")

        self._mark_invalid(edt_sw, not self.has_sw())
        self._mark_invalid(edt_ref, need_mhz and not self.has_mhz_if_needed())

        have_mhz = self._val_positive(self.ref)
        self._set_enabled_if_exists("btn_add", not need_mhz or have_mhz)
        self._set_enabled_if_exists("btn_auto", not need_mhz or have_mhz)

        # 5) Enable/disable UI controls
        # Try a few common names; only those that exist will be toggled.
        self._set_enabled_if_exists("btn_sim", sim_ready and not is_busy)
        self._set_enabled_if_exists("btn_fit", fit_ready and not is_busy)
        n_slices = (self.data2d.shape[0] if (getattr(self, "data2d", None) is not None and getattr(self.data2d, "ndim", 0) == 2) else 1)
        fit_sel_ready = fit_ready and n_slices > 1 and bool(getattr(self, "chk_2d_mode", None))
        
        self._set_enabled_if_exists("act_fit_selected", fit_sel_ready and not is_busy)
        self._set_enabled_if_exists("tgl_add", not need_mhz or have_mhz)
        self._set_enabled_if_exists("act_drag", self.has_data())

        self._set_enabled_if_exists("export_act", self.peak_model.rowCount() > 0)
        self._set_enabled_if_exists("import_act", True)

        # Optional: status hint (short, not a popup)
        try:
            if hasattr(self, "statusBar") and callable(self.statusBar):
                sb = self.statusBar()
                if not fit_ready:
                    if not self.has_data():
                        sb.showMessage("Load a spectrum to continue…", 3000)
                    elif not self.has_sw():
                        sb.showMessage("SW (Hz) required (> 0).", 3000)
                    elif need_mhz and not self.has_mhz_if_needed():
                        sb.showMessage("Spectrometer MHz required for ppm axis.", 3000)
                    else:
                        sb.clearMessage()
                else:
                    sb.clearMessage()
        except Exception:
            pass

class SlicePickerDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, N: int, labels: list[str], prechecked: list[int]):
        super().__init__(parent)
        self._parent = parent  
        self.setWindowTitle("Select Slices")
        self.setModal(True)
        self.resize(420, 520)
        self._N = int(N)
        self.labels: Optional[List[str]] = labels or []   # (minor: correct type hint)
        self.prechecked: set[int] = set(prechecked or [])
        self._validated_indices = None  # type: Optional[list[int]]
        self._mode = "sequential"  # or "joint"

        self._list = QtWidgets.QListWidget(self)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._list.setUniformItemSizes(True)
        for i in range(self._N):
            txt = f"[{i}]  {labels[i] if i < len(labels) else ''}".rstrip()
            it = QtWidgets.QListWidgetItem(txt, self._list)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if i in self.prechecked else QtCore.Qt.Unchecked)

        btn_all  = QtWidgets.QPushButton("Select All", self)
        btn_none = QtWidgets.QPushButton("Select None", self)
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)

        # New action row
        self.lbl_summary = QtWidgets.QLabel("", self)
        self.btn_validate = QtWidgets.QPushButton("Validate", self)
        self.btn_seq   = QtWidgets.QPushButton("Sequential Fitting", self)
        self.btn_joint = QtWidgets.QPushButton("Joint Fitting", self)
        self.btn_cancel = QtWidgets.QPushButton("Close", self)
        self.btn_validate.setDefault(True)
        self.btn_validate.clicked.connect(self._on_validate)
        self.btn_seq.clicked.connect(self._on_seq)     # will call worker
        self.btn_joint.clicked.connect(self._on_joint) # will call worker
        self.btn_cancel.clicked.connect(self.reject)

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(btn_all)
        top.addWidget(btn_none)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._list)
        lay.addWidget(self.lbl_summary)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_validate)
        row.addWidget(self.btn_seq)
        row.addWidget(self.btn_joint)
        lay.addLayout(row)

        # Disable OK if empty selection
        self._update_action_enabled()
        self._list.itemChanged.connect(self._update_action_enabled)

    # --- NEW: tiny helper, reused by both paths ---
    def _ensure_2d_mode(self) -> bool:
        data2d = getattr(self._parent, "data2d", None)
        if not (hasattr(data2d, "shape") and getattr(data2d, "ndim", 0) == 2 and data2d.shape[0] > 1):
            QtWidgets.QMessageBox.information(self, "Fit Selected…", "No 2D data with multiple slices.")
            return False
        if not getattr(self._parent, "chk_2d_mode", None).isChecked():
            self._parent.statusBar().showMessage("2D mode was OFF. Turning it ON to run multi-slice fit.", 4000)
            self._parent.chk_2d_mode.setChecked(True)
            if hasattr(self._parent, "on_toggle_2d_mode"):
                self._parent.on_toggle_2d_mode(True)
        return True

    def _select_all(self):
        for i in range(self._list.count()):
            it = self._list.item(i)
            it.setCheckState(QtCore.Qt.Checked)

    def _select_none(self):
        for i in range(self._list.count()):
            it = self._list.item(i)
            it.setCheckState(QtCore.Qt.Unchecked)

    def _update_action_enabled(self):
        has_any = len(self.checked_indices()) > 0
        self.btn_validate.setEnabled(has_any)
        self.btn_seq.setEnabled(has_any)
        self.btn_joint.setEnabled(has_any)

    def _on_validate(self):
        self._validated_indices = self.checked_indices()
        self.lbl_summary.setText(_compact_index_ranges(self._validated_indices))

    # --- CHANGED: call the workers directly, then close ---
    def _on_seq(self):
        if not self._ensure_2d_mode():
            return
        self._mode = "sequential"
        indices = self._validated_indices or self.checked_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "Sequential Fit", "No slices selected.")
            return
        # remember selection on parent for UX
        setattr(self._parent, "_last_slice_selection", sorted(set(indices)))
        # fire the worker and close
        try:
            self.accept()
            self._parent.on_fit_sequential(indices)
        finally:
            self.accept()

    def _on_joint(self):
        if not self._ensure_2d_mode():
            return
        self._mode = "joint"
        indices = self._validated_indices or self.checked_indices()
        if not indices:
            QtWidgets.QMessageBox.information(self, "Joint Fit", "No slices selected.")
            return
        setattr(self._parent, "_last_slice_selection", sorted(set(indices)))
        try:
            self.accept()
            self._parent.on_fit_jointed(indices)
        finally:
            self.accept()

    def checked_indices(self) -> List[int]:
        out: List[int] = []
        for i in range(self._list.count()):
            if self._list.item(i).checkState() == QtCore.Qt.Checked:
                out.append(i)
        return out

    # kept for backward-compatibility (now rarely used)
    def result_payload(self) -> tuple[list[int], str | None]:
        idx = self._validated_indices or self.checked_indices()
        return (idx, self._mode)


def _compact_index_ranges(indices: List[int]) -> str:
    """Return a compact '0, 3–5, 9' style summary for sorted ints."""
    if not indices:
        return ""
    xs = sorted(set(int(i) for i in indices))
    ranges = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((start, prev))
        start = prev = x
    ranges.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}–{b}" for a, b in ranges]
    return ", ".join(parts)



## ------------------------------- main --------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
#app = QApplication(sys.argv)
#w = MainWindow()
#print(w.ref)
#print(vars(FitContext))
#print(w.x_Hz[0])
