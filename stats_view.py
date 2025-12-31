from __future__ import annotations

from typing import List, Dict, Any, Optional

from PyQt5 import QtCore, QtWidgets
from stats_extract import save_stats_bundle



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

        # --- Splitter (params + corr) ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Parameter table
        self.tbl_params = QtWidgets.QTableView()
        self.tbl_params.setModel(self.param_model)
        self.tbl_params.setSortingEnabled(True)
        splitter.addWidget(self.tbl_params)

        # Correlation table
        self.tbl_corr = QtWidgets.QTableView()
        self.tbl_corr.setModel(self.corr_model)
        self.tbl_corr.setSortingEnabled(True)
        splitter.addWidget(self.tbl_corr)

        layout.addWidget(splitter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_save = QtWidgets.QPushButton("Save…")
        self.btn_save.clicked.connect(self.on_save)
        btn_row.addWidget(self.btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

            
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


    # =========================
    # Public API
    # =========================

    def set_stats(self, stats: Dict[str, Any]):
        """
        stats = output of extract_FitResult_corr_and_sum()
        """
        self._stats_payload = stats or {}
        # ---- Summary (top) ----
        summary = stats.get("summary", {}) or {}

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

        # ---- Correlations ----
        corr = stats.get("corr", {}) or {}
        self.corr_model.set_pairs(corr.get("pairs", []) or [])
