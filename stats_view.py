from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

from PyQt5 import QtCore, QtWidgets
from stats_extract import save_stats_bundle


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from stats_plots import (
    make_corr_heatmap_fig,
    make_absr_distribution_fig,
    make_amp_width_scatter_fig,
    write_stats_figures
)



# =========================
# Parameter Table Model
# =========================

class ParamStatsTableModel(QtCore.QAbstractTableModel):
    HEADERS = [
        "Name", "Value", "StdErr", "%Err",
        "Vary", "Min", "Max", "Expr", "Init"
    ]

    def __init__(self, rows: List[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self._rows = rows or []

    def set_rows(self, rows: List[Dict[str, Any]]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.HEADERS)

    def headerData(self, section, orientation, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.HEADERS[section]
        return section + 1

    def data(self, index, role):
        if not index.isValid():
            return None
        if role != QtCore.Qt.DisplayRole:
            return None

        row = self._rows[index.row()]
        col = index.column()

        key_map = [
            "name", "value", "stderr", "spercent",
            "vary", "min", "max", "expr", "init_values",
        ]
        key = key_map[col]
        val = row.get(key)

        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)


# =========================
# Correlation Pair Model
# =========================

class CorrPairTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["Param i", "Param j", "r", "|r|"]

    def __init__(self, pairs: List[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self._pairs = pairs or []

    def set_pairs(self, pairs: List[Dict[str, Any]]):
        self.beginResetModel()
        self._pairs = pairs
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._pairs)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.HEADERS)

    def headerData(self, section, orientation, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.HEADERS[section]
        return section + 1

    def data(self, index, role):
        if not index.isValid():
            return None
        if role != QtCore.Qt.DisplayRole:
            return None

        pair = self._pairs[index.row()]
        col = index.column()

        if col == 0:
            return pair["name_i"]
        if col == 1:
            return pair["name_j"]
        if col == 2:
            return f"{pair['r']:.3f}"
        if col == 3:
            return f"{pair['abs_r']:.3f}"
        return None

# =========================
# Main Statistics View
# =========================

class StatsView(QtWidgets.QDialog):
    """
    UI-only statistics viewer.
    Input must come from stats_extract (dict), NOT MinimizerResult.
    """

    # Display order + user-facing labels
    SUMMARY_FIELDS = [
        ("Mode",   "Fitting Mode"),
        ("Slices", "Slices"),
        ("Fitting method", "Fitting method"),
        ("Degree of freedoms", "Degree of freedoms"),
        ("Data points", "Number of data points"),
        ("Number of free variables", "Number of free variables"),
        ("Function evals",   "Function evals"),
        ("chi-square", "Chi-square"),
        ("reduced chi-square", "Reduced chi-square"),
        ("Akaike info crit", "Akaike info crit"),
        ("Bayesian info crit", "Bayesian info crit"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Statistics")
        self.resize(900, 600)

        self.param_model = ParamStatsTableModel()
        self.corr_model = CorrPairTableModel()

        self._summary_value_labels: Dict[str, QtWidgets.QLabel] = {}
        self._stats_payload: Dict[str, Any] = {}
        # Current display unit for frequency-like params (Hz or ppm)
        self._display_unit: str = "Hz"
        # ref frequency (MHz) needed for Hz<->ppm conversion (if available)
        self._ref_MHz: Optional[float] = None
        self._corr_view_key: str = "pairs_filtered" # default


        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- Summary box (top), 2 columns ---
        gb = QtWidgets.QGroupBox("Summary")
        grid = QtWidgets.QGridLayout(gb)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self._summary_value_labels = {}

        # 6 items left, 5 items right
        left_n = 6

        for idx, (key, label) in enumerate(self.SUMMARY_FIELDS):
            col_block = 0 if idx < left_n else 2   # 0/1 = left label/value, 2/3 = right label/value
            row = idx if idx < left_n else (idx - left_n)

            lab_key = QtWidgets.QLabel(f"{label}:")
            lab_key.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            lab_val = QtWidgets.QLabel("—")
            lab_val.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lab_val.setWordWrap(True)

            self._summary_value_labels[key] = lab_val

            grid.addWidget(lab_key, row, col_block)
            grid.addWidget(lab_val, row, col_block + 1)

        layout.addWidget(gb)

        # --- Units toggle row ---
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addStretch(1)
        unit_row.addWidget(QtWidgets.QLabel("Display units:"))
        self.cmb_units = QtWidgets.QComboBox()
        self.cmb_units.addItems(["Hz", "ppm"])
        self.cmb_units.setCurrentText(self._display_unit)
        self.cmb_units.currentTextChanged.connect(self._on_units_changed)
        unit_row.addWidget(self.cmb_units)
        layout.addLayout(unit_row)

        # --- Correlation view selector row ---
        corr_row = QtWidgets.QHBoxLayout()
        corr_row.addStretch(1)
        corr_row.addWidget(QtWidgets.QLabel("Correlation view:"))

        self.cmb_corr_view = QtWidgets.QComboBox()
        self.cmb_corr_view.setToolTip("Choose which correlation pair list to display.")
        self.cmb_corr_view.currentIndexChanged.connect(self._on_corr_view_changed)
        corr_row.addWidget(self.cmb_corr_view)

        layout.addLayout(corr_row)

        
        # --- Splitter (params + corr) ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Parameter table
        self.tbl_params = QtWidgets.QTableView()
        self.tbl_params.setModel(self.param_model)
        self.tbl_params.setSortingEnabled(True)
        splitter.addWidget(self.tbl_params)

        # --- Correlation section (label + table) ---
        corr_box = QtWidgets.QWidget()
        corr_layout = QtWidgets.QVBoxLayout(corr_box)
        corr_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_corr_view = QtWidgets.QLabel("Correlations: All pairs")
        self.lbl_corr_view.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_corr_view.setStyleSheet("font-weight: bold;")
        corr_layout.addWidget(self.lbl_corr_view)

        self.tbl_corr = QtWidgets.QTableView()
        self.tbl_corr.setModel(self.corr_model)
        self.tbl_corr.setSortingEnabled(True)
        corr_layout.addWidget(self.tbl_corr)

        splitter.addWidget(corr_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2) 


        layout.addWidget(splitter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        
        self.btn_show_fig = QtWidgets.QPushButton("Show Figures…")
        self.btn_show_fig.clicked.connect(self.on_show_figures)
        btn_row.addWidget(self.btn_show_fig)

        self.btn_save_fig = QtWidgets.QPushButton("Save Figures…")
        self.btn_save_fig.clicked.connect(self.on_save_figures)
        btn_row.addWidget(self.btn_save_fig)

        self.btn_save = QtWidgets.QPushButton("Save…")
        self.btn_save.clicked.connect(self.on_save)
        btn_row.addWidget(self.btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

    # =========================
    # Unit conversion helpers
    # =========================

    @staticmethod
    def _is_frequency_like_param(name: str) -> bool:
        """
        Heuristic classifier:
        Convert only parameters that represent frequency/linewidth/position-like quantities.
        Keep amplitude/phase/offset/etc. untouched.
        """
        s = (name or "").lower()
        tokens = [t for t in s.replace("-", "_").split("_") if t]

        # include (frequency-like)
        include = {
            "pos", "ppm", "hz",
            "lor", "lorentz", "gauss", "gau",
        }
        # exclude (non-frequency)
        exclude = {
            "amp", "ampl", "amplitude",
            "phi0",
            "offset",
            "mult",
            "k", "t",
        }

        if any(t in exclude for t in tokens):
            return False
        return any(t in include for t in tokens)

    @staticmethod
    def _try_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v))
        except Exception:
            return None

    @staticmethod
    def _get_ref_MHz(stats: Dict[str, Any]) -> Optional[float]:
        """
        Try multiple locations to find the reference frequency (MHz).
        Prefer stats['meta']['ref_MHz'] but accept common alternatives.
        """
        if not stats:
            return None
        meta = stats.get("meta", {}) or {}
        for key in ("ref_MHz", "ref_mhz", "obs_MHz", "sfo1_MHz"):
            v = StatsView._try_float(meta.get(key, None))
            if v:
                return v

    def _convert_params_for_display(
        self,
        rows_hz: List[Dict[str, Any]],
        unit: str,
        ref_MHz: Optional[float],
    ) -> List[Dict[str, Any]]:
        """
        Return display rows derived from canonical Hz rows.
        Only converts frequency-like params and only for numeric fields.
        """
        if unit == "Hz" or not ref_MHz:
            return rows_hz

        scale = 1.0 / float(ref_MHz)  # Hz -> ppm
        # keys used by stats_extract/Param rows (lowercase)
        numeric_keys = ("value", "stderr", "min", "max", "init_values", "spercent")

        out: List[Dict[str, Any]] = []
        for r in (rows_hz or []):
            rr = dict(r)
            name = str(rr.get("name", ""))
            if self._is_frequency_like_param(name):
                for k in numeric_keys:
                    v = rr.get(k, None)
                    if isinstance(v, (int, float)):
                        rr[k] = float(v) * scale
            out.append(rr)
        return out

    def _apply_display_unit(self) -> None:
        """
        Refresh UI from canonical payload using current display unit.
        """
        stats = self._stats_payload or {}
        self._ref_MHz = self._get_ref_MHz(stats)

        # If ppm requested but no ref available, force Hz and disable ppm choice
        if self._display_unit == "ppm" and not self._ref_MHz:
            self._display_unit = "Hz"
            if hasattr(self, "cmb_units"):
                self.cmb_units.blockSignals(True)
                self.cmb_units.setCurrentText("Hz")
                self.cmb_units.blockSignals(False)
                self.cmb_units.setToolTip("ppm display requires ref_MHz/obs_MHz metadata.")

        # ---- Summary (top) ----
        summary = stats.get("summary", {}) or {}

        def _fmt_value(v: Any) -> str:
            if v is None:
                return "—"
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return "—"
                return ", ".join(str(x) for x in v)
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        for key, _label in self.SUMMARY_FIELDS:
            lab = self._summary_value_labels.get(key, None)
            if lab is None:
                continue
            lab.setText(_fmt_value(summary.get(key, None)))

        # ---- Params ----
        rows_hz = stats.get("params", []) or []
        rows_disp = self._convert_params_for_display(rows_hz, self._display_unit, self._ref_MHz)
        self.param_model.set_rows(rows_disp)

        # ---- Correlations ---- (never convert)
        self._refresh_corr_view_selector()
        self.corr_model.set_pairs(self._get_selected_corr_pairs())

            
    def _on_units_changed(self, unit_txt: str) -> None:
        unit_txt = (unit_txt or "Hz").strip()
        if unit_txt not in ("Hz", "ppm"):
            unit_txt = "Hz"
        self._display_unit = unit_txt
        self._apply_display_unit()

    def _corr_view_options(self) -> List[Tuple[str, str]]:
        """
        Return list of (label, key) options.
        key is either:
          - "pairs_filtered" (raw)
          - "bundle:<bundle_key>" (subset)
        """
        corr = (self._stats_payload or {}).get("corr", {}) or {}
        bundle = corr.get("bundle_filtered", {}) or {}

        opts: List[Tuple[str, str]] = [("Filtered pairs", "pairs_filtered")]

        if isinstance(bundle, dict):
            # Show the most useful subsets first
            preferred = [
                ("Top |r| (bundle)", "top_all"),
                ("Same-slice", "top_same_slice"),
                ("Cross-slice", "top_cross_slice"),
                ("Intra-class", "top_intra_class"),
                ("Inter-class", "top_inter_class"),
                ("Amp vs width (amp-lor/gauss)", "top_amp_width"),
                ("Annotated (all)", "annotated_pairs"),
            ]
            for label, k in preferred:
                if k in bundle and isinstance(bundle.get(k), list):
                    opts.append((label, f"bundle:{k}"))

        return opts

    def _get_selected_corr_pairs(self) -> List[Dict[str, Any]]:
        stats = self._stats_payload or {}
        corr = stats.get("corr", {}) or {}
        key = (self._corr_view_key or "pairs_filtered").strip()

        if key == "pairs_filtered":
            return corr.get("pairs_filtered", corr.get("pairs", [])) or []

        if key == "pairs_all":
            return corr.get("pairs_all", corr.get("pairs", [])) or []

        if key.startswith("bundle:"):
            bundle = corr.get("bundle_filtered", corr.get("bundle", {})) or {}
            subkey = key.split(":", 1)[1]
            pairs = bundle.get(subkey, []) or []
            return [p for p in pairs if isinstance(p, dict)]

        return corr.get("pairs_filtered", corr.get("pairs", [])) or []

    def _refresh_corr_view_selector(self) -> None:
        """
        Populate the dropdown based on current payload.
        Keeps current selection if still available.
        """
        if not hasattr(self, "cmb_corr_view"):
            return

        current = self._corr_view_key or "pairs_filtered"
        opts = self._corr_view_options()

        self.cmb_corr_view.blockSignals(True)
        self.cmb_corr_view.clear()

        # Fill combo with labels, store key in itemData
        for label, key in opts:
            self.cmb_corr_view.addItem(label, key)

        # Restore selection
        idx = self.cmb_corr_view.findData(current)
        if idx < 0:
            idx = 0
            self._corr_view_key = opts[0][1]
        self.cmb_corr_view.setCurrentIndex(idx)
        self.cmb_corr_view.blockSignals(False)

    def _on_corr_view_changed(self, _idx: int) -> None:
        if not hasattr(self, "cmb_corr_view"):
            return

        # itemData stores the internal key (e.g. "pairs" or "bundle:top_cross_slice")
        self._corr_view_key = str(self.cmb_corr_view.currentData() or "pairs_filtered")

        # Update table
        pairs = self._get_selected_corr_pairs()
        self.corr_model.set_pairs(pairs)

        # Update label text (optionally include count)
        view_label = self.cmb_corr_view.currentText()
        if hasattr(self, "lbl_corr_view"):
            self.lbl_corr_view.setText(f"Correlations: {view_label} ({len(pairs)} pairs)")

    def on_show_figures(self):
        if not self._stats_payload:
            QtWidgets.QMessageBox.information(self, "Figures", "No statistics to plot.")
            return
        try:
            dlg = FiguresDialog(self._stats_payload, parent=self)
            dlg.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Figures failed", str(e))


    def on_save(self):
        if not self._stats_payload:
            QtWidgets.QMessageBox.information(self, "Save", "No statistics to save.")
            return
        # default filename
        s = (self._stats_payload.get("summary", {}) or {})
        mode = s.get("Mode", "fit")
        slices = s.get("Slices", None)
        if isinstance(slices, (list, tuple)) and slices:
            slice_tag = f"s{min(slices)}-s{max(slices)}"
        else:
            slice_tag = "slices"
        default_name = f"fit_stats_{mode}_{slice_tag}.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save fit statistics",
            default_name,
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            written = save_stats_bundle(path, self._stats_payload)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
            return
        msg = (
            "Saved:\n"
            f"- {written.get('json','')}\n"
            f"- {written.get('report_txt','')}\n"
        )
        QtWidgets.QMessageBox.information(self, "Save", msg)

    def on_save_figures(self):
        if not self._stats_payload:
            QtWidgets.QMessageBox.information(self, "Save Figures", "No statistics to plot.")
            return

        # Choose output directory
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose folder to save figures",
            "",
        )
        if not out_dir:
            return

        # Build a stable prefix like JSON export
        s = (self._stats_payload.get("summary", {}) or {})
        mode = s.get("Mode", "fit")
        slices = s.get("Slices", None)
        if isinstance(slices, (list, tuple)) and slices:
            slice_tag = f"s{min(slices)}-s{max(slices)}"
        else:
            slice_tag = "slices"
        prefix = f"fit_stats_{mode}_{slice_tag}"

        try:
            written = write_stats_figures(out_dir, self._stats_payload, prefix=prefix, include_pdf=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Figures failed", str(e))
            return

        msg = "Saved figures:\n" + "\n".join(f"- {p}" for p in written.values())
        QtWidgets.QMessageBox.information(self, "Save Figures", msg)



    # =========================
    # Public API
    # =========================

    def set_stats(self, stats: Dict[str, Any]):
        """
        stats = output of extract_FitResult_corr_and_sum()
        """
        self._stats_payload = stats or {}
        self._refresh_corr_view_selector()

        if hasattr(self, "cmb_corr_view") and hasattr(self, "lbl_corr_view"):
            txt = self.cmb_corr_view.currentText()
            self.lbl_corr_view.setText(f"Correlations: {txt}")
        # ---- Summary (top) ----
        summary = stats.get("summary", {}) or {}

        # enable/disable ppm option depending on availability of ref_MHz
        ref = self._get_ref_MHz(self._stats_payload)
        if hasattr(self, "cmb_units"):
            self.cmb_units.blockSignals(True)
            self.cmb_units.model().item(1).setEnabled(bool(ref))  # "ppm" entry
            if not ref and self.cmb_units.currentText() == "ppm":
                self.cmb_units.setCurrentText("Hz")
                self._display_unit = "Hz"
            self.cmb_units.blockSignals(False)
            self.cmb_units.setToolTip("" if ref else "ppm display requires ref_MHz/obs_MHz metadata.")

        # Apply display unit which updates summary, params and correlations
        self._apply_display_unit()


        def _fmt_value(v: Any) -> str:
            if v is None:
                return "—"
            # allow list/tuple of slices
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return "—"
                return ", ".join(str(x) for x in v)
            # floats: compact but stable
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        for key, _label in self.SUMMARY_FIELDS:
            lab = self._summary_value_labels.get(key, None)
            if lab is None:
                continue
            lab.setText(_fmt_value(summary.get(key, None)))

        # ---- Params ----
        self.param_model.set_rows(stats.get("params", []) or [])

class FiguresDialog(QtWidgets.QDialog):
    def __init__(self, stats: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figures")
        self.resize(1000, 700)
        self._stats = stats or {}

        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self._build_tabs()

    def _add_figure_tab(self, title: str, fig):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)

        v.addWidget(toolbar)
        v.addWidget(canvas)
        self.tabs.addTab(w, title)

    def _build_tabs(self):
        # 1) Heatmap
        fig1 = make_corr_heatmap_fig(self._stats, cmap="RdBu_r")
        self._add_figure_tab("Heatmap", fig1)

        # 2) |r| CDF
        fig2 = make_absr_distribution_fig(self._stats, kind="cdf")
        self._add_figure_tab("|r| CDF", fig2)

        # 3) Amp vs width scatter
        fig3 = make_amp_width_scatter_fig(self._stats, width_kind="both")
        self._add_figure_tab("Amp vs width", fig3)


