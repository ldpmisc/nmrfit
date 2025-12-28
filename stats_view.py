from __future__ import annotations

from typing import List, Dict, Any, Optional

from PyQt5 import QtCore, QtWidgets


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
            "vary", "min", "max", "expr", "init_value",
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Statistics")
        self.resize(900, 600)

        self.param_model = ParamStatsTableModel()
        self.corr_model = CorrPairTableModel()

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- Summary label ---
        self.lbl_summary = QtWidgets.QLabel()
        self.lbl_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_summary)

        # --- Splitter ---
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

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    # =========================
    # Public API
    # =========================

    def set_stats(self, stats: Dict[str, Any]):
        """
        stats = output of extract_FitResult_and_corr()
        """
        # ---- Params ----
        self.param_model.set_rows(stats.get("params", []))

        # ---- Correlations ----
        corr = stats.get("corr", {})
        self.corr_model.set_pairs(corr.get("pairs", []))

        # ---- Summary ----
        summary_lines = []
        for k in ("chisqr", "redchi", "ndata", "nvarys", "nfree", "method", "nfev"):
            if k in stats:
                summary_lines.append(f"{k}: {stats[k]}")
        self.lbl_summary.setText(" | ".join(summary_lines))
