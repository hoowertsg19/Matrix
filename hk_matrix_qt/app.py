from __future__ import annotations
import sys
import os
from PySide6.QtGui import QFontDatabase, QFont
from sympy import symbols as _SYM_symbols, sympify as _SYM_sympify, lambdify as _SYM_lambdify
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np
from sympy import Matrix as SMatrix

from hk_matrix.logic.core import (
    fmt_matrix, fmt_num,
    add_steps, sub_steps, multiply_steps,
    rref_steps, upper_triangular_steps,
    transpose_steps, inverse_steps,
    determinant_steps, cramer_steps,
)

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QSpinBox, QLabel, QTableWidget, QTableWidgetItem, QTextEdit, QLineEdit,
        QListWidget, QListWidgetItem, QComboBox, QSplitter, QScrollArea, QSizePolicy,
        QDialog, QAbstractItemView, QCheckBox, QDoubleSpinBox, QToolButton,
        QProgressBar, QFrame, QGraphicsDropShadowEffect      # <-- añadido
    )
    from PySide6.QtCore import Qt, QObject, Slot, Signal
    from PySide6.QtGui import QClipboard, QShortcut, QKeySequence, QColor, QFont, QIcon, QPixmap
    from PySide6.QtCore import QTimer, QEasingCurve, QPropertyAnimation
    # WebEngine eliminado: evitamos importar Qt WebEngine/WebChannel para prevenir
    # cierres nativos en algunos entornos. El editor visual se retiró por compatibilidad.
except Exception as e:
    raise ImportError('PySide6 is required for the Qt UI. Install via `pip install PySide6`.') from e


class MatrixTable(QTableWidget):
    def __init__(self, rows=3, cols=3, parent=None):
        super().__init__(rows, cols, parent)
        self.setRowCount(rows); self.setColumnCount(cols)
        self._refresh_headers()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _refresh_headers(self):
        self.setHorizontalHeaderLabels([str(i+1) for i in range(self.columnCount())])
        self.setVerticalHeaderLabels([str(i+1) for i in range(self.rowCount())])

    def set_headers(self, col_headers: list[str] | None = None, row_headers: list[str] | None = None):
        if col_headers is not None:
            # ensure length matches
            if len(col_headers) != self.columnCount():
                self.setColumnCount(len(col_headers))
            self.setHorizontalHeaderLabels(col_headers)
        if row_headers is not None:
            if len(row_headers) != self.rowCount():
                self.setRowCount(len(row_headers))
            self.setVerticalHeaderLabels(row_headers)

    def set_size(self, rows: int, cols: int):
        self.setRowCount(rows); self.setColumnCount(cols); self._refresh_headers()

    def set_matrix(self, mat: np.ndarray):
        r, c = mat.shape
        self.set_size(r, c)
        for i in range(r):
            for j in range(c):
                v = float(mat[i, j])
                s = str(int(v)) if float(v).is_integer() else str(v)
                item = QTableWidgetItem(s)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, item)

    def get_matrix(self) -> np.ndarray:
        r = self.rowCount(); c = self.columnCount()
        out = np.zeros((r, c), dtype=float)
        for i in range(r):
            for j in range(c):
                it = self.item(i, j)
                txt = it.text().strip() if it else ''
                out[i, j] = 0.0 if txt == '' else float(txt)
        return out

    def fill_random(self, low: int = -9, high: int = 9):
        r = self.rowCount(); c = self.columnCount()
        M = np.random.randint(low, high + 1, size=(r, c)).astype(float)
        self.set_matrix(M)


MATH_FONT_STACK = "'Garamond Math','EB Garamond Math','Cambria Math','STIX Two Math','Latin Modern Math','Times New Roman',serif"

# Simple LaTeX renderer using matplotlib's mathtext to display pretty formulas in labels
# Removed LaTeX rendering via matplotlib.mathtext to reduce dependencies

# SpinBox que evita ceros de relleno (muestra 0,1 en lugar de 0,100000)
class TrimDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox que:
    - No usa notación científica en pantalla.
    - No rellena con ceros; respeta lo que el usuario escribió (coma o punto) tras validar.
    - Para valores puestos por código, muestra número plano sin ceros ni notación científica.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inicializar primero los atributos antes de llamar a setters que
        # pueden invocar textFromValue internamente.
        self._last_user_text: str | None = None
        self._last_user_value: float | None = None
        self.setDecimals(12)
        try:
            self.lineEdit().textEdited.connect(self._on_text_edited)  # type: ignore
            self.editingFinished.connect(self._on_edit_finished)      # type: ignore
            self.valueChanged.connect(self._on_value_changed)         # type: ignore
        except Exception:
            pass

    def setValue(self, val: float) -> None:  # type: ignore[override]
        # Al establecer por código, olvidamos el último texto del usuario
        self._last_user_text = None
        self._last_user_value = None
        return super().setValue(val)

    def _on_text_edited(self, text: str):
        self._last_user_text = text

    def _on_edit_finished(self):
        # Guardamos el valor final para que textFromValue pueda restaurar el texto original
        try:
            self._last_user_value = float(self.value())
        except Exception:
            self._last_user_value = None

    def _on_value_changed(self, _):
        # Si el cambio fue por flechas/rueda, no conservar el texto previo
        if not self.hasFocus():
            self._last_user_text = None
            self._last_user_value = None

    def textFromValue(self, value: float) -> str:  # type: ignore[override]
        # Si el valor actual coincide con el que el usuario escribió, devolvemos su texto sin tocar
        if self._last_user_value is not None and abs(value - self._last_user_value) <= (10 ** -self.decimals()):
            if isinstance(self._last_user_text, str) and self._last_user_text.strip() != '':
                return self._last_user_text

        # Formateo plano sin notación científica ni ceros sobrantes
        s = f"{float(value):.12f}"
        # recortar ceros y punto final
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        # aplicar separador decimal de la locale
        dp = self.locale().decimalPoint()
        if dp != '.':
            s = s.replace('.', dp)
        return s

# def latex_to_pixmap(...) removed (unused)


def set_table_preview(table: QTableWidget, mat: np.ndarray):
    r, c = mat.shape
    table.setRowCount(r); table.setColumnCount(c)
    table.setAlternatingRowColors(True)
    table.setStyleSheet(f"QTableWidget{{gridline-color:#444;}} QTableWidget::item{{padding:4px; font-family: {MATH_FONT_STACK};}} QTableWidget::item:selected{{background: transparent; color: inherit;}}")
    # Make preview tables non-selectable to avoid random blue highlights
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.setFocusPolicy(Qt.NoFocus)
    for i in range(r):
        for j in range(c):
            v = float(mat[i, j])
            s = str(int(v)) if float(v).is_integer() else f"{v:.2f}"
            item = QTableWidgetItem(s)
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignCenter)
            f = item.font(); f.setFamily('Consolas'); item.setFont(f)
            table.setItem(i, j, item)


class StepsDialog(QDialog):
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Paso a paso')
        self.resize(940, 620)
        self.steps = steps

        # Root layout
        root = QHBoxLayout(self)
        left = QVBoxLayout(); right = QVBoxLayout()
        root.addLayout(left, 1); root.addLayout(right, 2)

        # Left: step list
        self.listbox = QListWidget(); left.addWidget(self.listbox)
        for i, (desc, _) in enumerate(steps):
            QListWidgetItem(f"{i+1}. {desc}", self.listbox)

        # Right: header toolbar
        header = QHBoxLayout(); right.addLayout(header)
        self.step_title = QLabel(''); self.step_title.setStyleSheet('font-weight:600; font-size:14px;')
        header.addWidget(self.step_title, 1)
        header.addWidget(QLabel('Decimales:'))
        self.decimals = QSpinBox(); self.decimals.setRange(0, 8); self.decimals.setValue(2)
        header.addWidget(self.decimals)
        self.only_changes = QCheckBox('Solo cambios'); header.addWidget(self.only_changes)
        self.manual_mode = QCheckBox('Modo manual'); header.addWidget(self.manual_mode)
        self.copy_btn = QPushButton('📋 Copiar matriz'); header.addWidget(self.copy_btn)

        # Right: preview table
        self.preview = QTableWidget();
        self.preview.setAlternatingRowColors(True)
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview.setSelectionMode(QAbstractItemView.NoSelection)
        self.preview.setFocusPolicy(Qt.NoFocus)
        self.preview.setStyleSheet("QTableWidget::item:selected{background: transparent; color: inherit;}")
        right.addWidget(self.preview, 1)
        # Manual explanation panel
        self.explain = QLabel('')
        self.explain.setWordWrap(True)
        self.explain.setStyleSheet(f"font-family: {MATH_FONT_STACK}; color:#cfd8dc;")
        right.addWidget(self.explain)

        # Info row
        self.stats = QLabel(''); self.stats.setStyleSheet('color:#888;')
        right.addWidget(self.stats)

        # Navigation
        nav = QHBoxLayout(); right.addLayout(nav)
        self.prev_btn = QPushButton('◀ Anterior'); self.next_btn = QPushButton('Siguiente ▶')
        nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn); nav.addStretch(1)

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._move(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._move(+1))

        # Wire up
        self.listbox.currentRowChanged.connect(self._on_select)
        self.prev_btn.clicked.connect(lambda: self._move(-1))
        self.next_btn.clicked.connect(lambda: self._move(+1))
        self.decimals.valueChanged.connect(lambda _=None: self._on_select(self.listbox.currentRow()))
        self.only_changes.toggled.connect(lambda _=None: self._on_select(self.listbox.currentRow()))
        self.manual_mode.toggled.connect(lambda _=None: self._on_select(self.listbox.currentRow()))
        self.copy_btn.clicked.connect(self._copy_current)

        # Initial selection
        if steps:
            self.listbox.setCurrentRow(0)

        # Subtle styling
        self.setStyleSheet('QListWidget::item { padding:6px; } QTableWidget { gridline-color:#444; }')

    def _move(self, delta):
        i = max(0, min(self.listbox.count()-1, self.listbox.currentRow()+delta))
        self.listbox.setCurrentRow(i)
        if 0 <= i < self.listbox.count():
            try:
                self.step_title.setText(self.listbox.item(i).text())
            except Exception:
                pass
        self._on_select(i)

    def _on_select(self, row):
        if row < 0 or row >= len(self.steps):
            return
        desc, mat = self.steps[row]
        self.step_title.setText(desc)
        arr = np.array(mat.tolist(), dtype=float)
        prev = None
        if row > 0:
            prev = np.array(self.steps[row-1][1].tolist(), dtype=float)
        # Parse description for manual-like hints
        hl_rows, hl_cols, pretty = self._parse_step_description(desc)
        self._render_matrix(arr, prev, hl_rows=hl_rows, hl_cols=hl_cols)
        self.explain.setText(pretty if self.manual_mode.isChecked() else '')
        # Stats
        changed = 0
        if prev is not None:
            changed = int(np.sum(np.round(arr, self.decimals.value()) != np.round(prev, self.decimals.value())))
        self.stats.setText(f"Tamaño: {arr.shape[0]} × {arr.shape[1]}  •  Celdas cambiadas: {changed}")

    def _render_matrix(self, arr: np.ndarray, prev: np.ndarray | None, hl_rows: set[int] | None = None, hl_cols: set[int] | None = None):
        d = self.decimals.value()
        r, c = arr.shape
        self.preview.setRowCount(r); self.preview.setColumnCount(c)
        for i in range(r):
            for j in range(c):
                v = float(arr[i, j])
                s = f"{v:.{d}f}" if not float(v).is_integer() or d > 0 else str(int(v))
                item = QTableWidgetItem(s)
                item.setFlags(Qt.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight if changed vs previous
                if prev is not None:
                    pv = float(prev[i, j])
                    if round(v, d) != round(pv, d):
                        # Highlight changed cells with requested celeste and bold text
                        item.setBackground(QColor('#0099a8'))
                        f = item.font(); f.setBold(True); item.setFont(f)
                        item.setForeground(QColor('white'))
                    elif self.only_changes.isChecked():
                        # Dim unchanged cells when 'Solo cambios' is enabled
                        item.setForeground(QColor('#777777'))
                # Manual-mode row/col soft highlight
                if hl_rows and i in hl_rows:
                    # only set if not already strong-highlighted
                    if item.background().color().name().lower() in ('#000000', '#00000000'):
                        item.setBackground(QColor(0, 153, 168, 40))  # soft accent
                if hl_cols and j in hl_cols:
                    if item.background().color().name().lower() in ('#000000', '#00000000'):
                        item.setBackground(QColor(0, 153, 168, 30))
                self.preview.setItem(i, j, item)
        # Ensure no selection remains
        self.preview.clearSelection()

    def _parse_step_description(self, desc: str):
        """Parse Spanish step descriptions to extract affected rows/cols and a prettier manual-style text.
        Returns (hl_rows, hl_cols, pretty_html_or_text).
        """
        import re
        hl_rows: set[int] = set()
        hl_cols: set[int] = set()
        pretty = desc
        d = desc.strip()
        # Intercambio de filas
        m = re.search(r"Intercambiar\s+fila\s+(\d+)\s+con\s+fila\s+(\d+)", d, re.IGNORECASE)
        if m:
            i, j = int(m.group(1))-1, int(m.group(2))-1
            hl_rows.update({i, j})
            pretty = f"Operación por filas: R{m.group(1)} ↔ R{m.group(2)}"
            return hl_rows, hl_cols, pretty
        # Dividir fila k por x
        m = re.search(r"Dividir\s+fila\s+(\d+)\s+por\s+([\-\d\./]+)", d, re.IGNORECASE)
        if m:
            i = int(m.group(1))-1; x = m.group(2)
            hl_rows.add(i)
            pretty = f"Escalado: R{m.group(1)} ← R{m.group(1)}/{x}"
            return hl_rows, hl_cols, pretty
        # Rk <- Rk ± (coef)*Rj variantes
        m = re.search(r"R\s*(\d+)\s*<-\s*R\s*\1\s*([+\-])\s*\(?([\-\d\./]+)\)?\s*\*?\s*R\s*(\d+)", d)
        if m:
            k, sign, coef, j = m.groups()
            k_i = int(k)-1; j_i = int(j)-1
            hl_rows.update({k_i, j_i})
            op = '+' if sign == '+' else '−'
            pretty = f"Operación elemental: R{k} ← R{k} {op} ({coef})·R{j}"
            return hl_rows, hl_cols, pretty
        # Calcular C[i,j] cases => show as-is but monospace
        if d.lower().startswith('calcular'):
            pretty = d
        return hl_rows, hl_cols, pretty

    def _copy_current(self):
        row = self.listbox.currentRow()
        if 0 <= row < len(self.steps):
            _, mat = self.steps[row]
            arr = np.array(mat.tolist(), dtype=float)
            self.parent().copy_to_clipboard(fmt_matrix(arr, self.decimals.value()))

class BisectionResultDialog(QDialog):
    def __init__(self, expr_text: str, xi: float, xu: float, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Método de Bisección — Detalles')
        self.resize(960, 640)

        lay = QVBoxLayout(self)
        title = QLabel('<b>Método de bisección</b>'); lay.addWidget(title)
        formula = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_text}</span>")
        lay.addWidget(formula)

        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            sgn = '+' if xr >= 0 else ''
            summary = QLabel(f"El método converge en <b>{n}</b> iteraciones.  LA RAÍZ ES: <b>{sgn}{xr:.6f}</b>")
            lay.addWidget(summary)

        tbl = QTableWidget(); lay.addWidget(tbl)
        headers = ['iteración','xi','xu','xr','Ea','yi','yu','yr']
        tbl.setColumnCount(len(headers)); tbl.setHorizontalHeaderLabels(headers)
        tbl.setRowCount(len(rows))
        def fmt(v):
            return ('+' if v>=0 else '') + f"{v:.6f}"
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                item = QTableWidgetItem(fmt(v) if isinstance(v, float) or isinstance(v, np.floating) else str(v))
                item.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                tbl.setItem(r,c,item)
        tbl.resizeColumnsToContents()
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet(f"QTableWidget::item{{font-family:{MATH_FONT_STACK}; padding:4px;}}")

        try:
            fig = Figure(figsize=(6,3.5), dpi=100)
            ax = fig.add_subplot(111)
            x = _SYM_symbols('x'); expr = _SYM_sympify(expr_text); fcall = _SYM_lambdify(x, expr, 'numpy')
            xs = np.linspace(xi, xu, 400)
            ys = fcall(xs)
            ax.axhline(0, color='#666', lw=1)
            ax.plot(xs, ys, color='#4fc3f7', label='f(x)')
            if rows:
                xr = rows[-1][3]; yr = rows[-1][7]
                ax.plot([xr],[yr],'o', color='#e05d5d', label='xr')
            ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(frameon=False)
            canvas = FigureCanvasQTAgg(fig)
            lay.addWidget(canvas)
        except Exception:
            pass


class MatrixQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Matrix (Qt)')
        # Establecer icono de ventana (también lo aplicamos a nivel de QApplication en run())
        try:
            self.setWindowIcon(QIcon(_resource_path('logo.png')))
        except Exception:
            pass
        self.resize(1280, 800)

        central = QWidget(); self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Sidebar
        sidebar = QVBoxLayout(); layout.addLayout(sidebar, 0)
        btn_style = {}
        self.btn_ops = QPushButton('🧮 Operaciones'); sidebar.addWidget(self.btn_ops)
        self.btn_ind = QPushButton('🧭 Independencia'); sidebar.addWidget(self.btn_ind)
        self.btn_triu = QPushButton('🔺 Triangular U'); sidebar.addWidget(self.btn_triu)
        self.btn_rref = QPushButton('🧱 RREF'); sidebar.addWidget(self.btn_rref)
        self.btn_ti = QPushButton('🔁 Transpuesta/Inversa'); sidebar.addWidget(self.btn_ti)
        self.btn_det = QPushButton('🧾 Determinante'); sidebar.addWidget(self.btn_det)
        self.btn_cramer = QPushButton('📐 Método de Cramer'); sidebar.addWidget(self.btn_cramer)
        self.btn_bis = QPushButton('📉 Bisección'); sidebar.addWidget(self.btn_bis)
        sidebar.addStretch(1)

        # Center and Right using splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        self.center = QWidget(); splitter.addWidget(self.center)
        cgrid = QVBoxLayout(self.center)
        self.center_title = QLabel(''); self.center_title.setObjectName('pageTitle'); cgrid.addWidget(self.center_title)
        self.center_body = QWidget(); self.center_layout = QVBoxLayout(self.center_body)
        cgrid.addWidget(self.center_body, 1)

        # Right results panel in scroll area
        self.right_scroll = QScrollArea(); self.right_scroll.setWidgetResizable(True)
        self.right_container = QWidget(); self.right_layout = QVBoxLayout(self.right_container)
        self.right_scroll.setWidget(self.right_container)
        splitter.addWidget(self.right_scroll)
        splitter.setSizes([850, 430])

        # connect
        self.btn_ops.clicked.connect(self.show_ops)
        self.btn_ind.clicked.connect(self.show_ind)
        self.btn_triu.clicked.connect(self.show_triu)
        self.btn_rref.clicked.connect(self.show_rref)
        self.btn_ti.clicked.connect(self.show_ti)
        self.btn_det.clicked.connect(self.show_det)
        self.btn_cramer.clicked.connect(self.show_cramer)
        self.btn_bis.clicked.connect(self.show_bisection)

        # state
        self.current_view = None
        self._result_widgets = []
        self._dialogs = []
        self.show_ops()
        # fonts and theme
        self._init_fonts()
        self.apply_theme()

    def _init_fonts(self):
        """Load bundled math fonts if present so they can be used in the app.
        Searches in a local 'fonts' folder next to logo.png.
        """
        try:
            base = os.path.dirname(_resource_path('logo.png'))
            fonts_dir = os.path.join(base, 'fonts')
            candidates = [
                'Garamond-Math.ttf', 'GaramondMath.ttf', 'Garamond-Math.otf',
                'EBGaramond-Math.otf', 'EBGaramondMath.otf', 'STIXTwoMath-Regular.ttf',
                'latinmodern-math.otf', 'LatinModernMath-Regular.otf'
            ]
            for name in candidates:
                p = os.path.join(fonts_dir, name)
                if os.path.exists(p):
                    try:
                        QFontDatabase.addApplicationFont(p)
                    except Exception:
                        pass
        except Exception:
            pass

    def apply_theme(self):
        """Apply a unified dark theme and accent color to make UI more aesthetic."""
        app = QApplication.instance()
        if app:
            try:
                app.setStyle('Fusion')
            except Exception:
                pass
        accent = '#0099a8'
        self.setStyleSheet(f"""
            QWidget#resultCard {{
                background-color: #1e1f22;
                border: 1px solid #2b2d31;
                border-radius: 8px;
                padding: 8px;
            }}
            QWidget#errorCard {{
                background-color: #251d1d;
                border: 1px solid #3a2b2b;
                border-left: 4px solid #e05d5d;
                border-radius: 8px;
                padding: 8px;
            }}
            QLabel#pageTitle {{
                font-size: 18px;
                font-weight: 600;
                padding: 4px 0 8px 0;
            }}
            QPushButton {{
                background: #2b2d31;
                border: 1px solid #3a3d41;
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                border-color: {accent};
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background: #1f2225;
            }}
            QTableWidget {{
                background: #15171a;
                selection-background-color: rgba(0,153,168,0.35);
                font-family: {MATH_FONT_STACK};
            }}
            QHeaderView::section {{
                background: #202225;
                color: #dddddd;
                padding: 4px;
                border: 0px;
                border-right: 1px solid #2b2d31;
            }}
            QListWidget {{
                background: #15171a;
                border: 1px solid #2b2d31;
                font-family: {MATH_FONT_STACK};
            }}
            QSpinBox, QDoubleSpinBox {{
                font-family: {MATH_FONT_STACK};
            }}
        """)

    # helpers
    def clear_center(self):
        for i in reversed(range(self.center_layout.count())):
            w = self.center_layout.itemAt(i).widget()
            if w: w.setParent(None)

    def clear_results(self):
        for i in reversed(range(self.right_layout.count())):
            w = self.right_layout.itemAt(i).widget()
            if w: w.setParent(None)
        self._result_widgets.clear()

    def copy_to_clipboard(self, text: str):
        QApplication.clipboard().setText(text, QClipboard.Clipboard)

    def push_result(self, title: str, matrix: np.ndarray | None, description: str = '', steps=None, accent: str | None = None):
        card = QWidget(); card.setObjectName('resultCard'); lay = QVBoxLayout(card)
        lay.addWidget(QLabel(f"<b>{title}</b>"))
        if matrix is not None:
            tbl = QTableWidget(); set_table_preview(tbl, np.array(matrix, dtype=float))
            tbl.setAlternatingRowColors(True)
            tbl.setStyleSheet(f"QTableWidget{{gridline-color:#444;}} QTableWidget::item{{padding:4px; font-family: {MATH_FONT_STACK};}}")
            lay.addWidget(tbl)
        if description:
            lbl = QLabel(description)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"font-family: {MATH_FONT_STACK};")
            lay.addWidget(lbl)
        row = QHBoxLayout(); lay.addLayout(row)
        btn_copy = QPushButton('📋 Copiar'); row.addWidget(btn_copy)
        btn_steps = QPushButton('👣 Ver pasos'); row.addWidget(btn_steps)
        btn_delete = QPushButton('🗑️ Quitar'); row.addWidget(btn_delete)
        row.addStretch(1)
        btn_copy.clicked.connect(lambda: self.copy_to_clipboard(description if matrix is None else fmt_matrix(np.array(matrix, dtype=float), 2)))
        if steps:
            btn_steps.setEnabled(True)
            btn_steps.clicked.connect(lambda: StepsDialog(steps, self).exec())
        else:
            btn_steps.setEnabled(False)
        # delete handler
        def _remove():
            card.setParent(None)
            if card in self._result_widgets:
                self._result_widgets.remove(card)
        btn_delete.clicked.connect(_remove)
        self.right_layout.addWidget(card)
        self._result_widgets.append(card)
        return card

    def push_error(self, message: str):
        """Push a compact error card at the TOP of results with only a delete action."""
        card = QWidget(); card.setObjectName('errorCard'); lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 8, 8, 8)
        title = QLabel('<b>Error</b>'); lay.addWidget(title)
        lbl = QLabel(message); lbl.setWordWrap(True); lay.addWidget(lbl)
        row = QHBoxLayout(); lay.addLayout(row)
        btn_delete = QPushButton('🗑️ Quitar'); row.addWidget(btn_delete)
        row.addStretch(1)
        def _remove():
            card.setParent(None)
            if card in self._result_widgets:
                self._result_widgets.remove(card)
        btn_delete.clicked.connect(_remove)
        # Insert at top
        self.right_layout.insertWidget(0, card)
        self._result_widgets.append(card)
        return card

    # views
    def show_ops(self):
        self.current_view = 'ops'
        self.center_title.setText('Operaciones con matrices')
        self.clear_center()
        cont = QWidget(); gl = QGridLayout(cont)
        self.gridA = MatrixTable(3,3); self.gridB = MatrixTable(3,3)
        # A header and size controls
        gl.addWidget(QLabel('<b>Matriz A</b>'), 0, 0)
        sizeA = QHBoxLayout();
        sizeA.addWidget(QLabel('Filas:'))
        a_rows = QSpinBox(); a_rows.setRange(1, 20); a_rows.setValue(3); sizeA.addWidget(a_rows)
        sizeA.addWidget(QLabel('Columnas:'))
        a_cols = QSpinBox(); a_cols.setRange(1, 20); a_cols.setValue(3); sizeA.addWidget(a_cols)
        btn_setA = QPushButton('Aplicar tamaño A'); sizeA.addWidget(btn_setA)
        # random button for A
        btn_randA = QPushButton('🎲 Aleatoria A'); sizeA.addWidget(btn_randA)
        gl.addLayout(sizeA, 1, 0)
        gl.addWidget(self.gridA, 2, 0)
        # B header and size controls
        gl.addWidget(QLabel('<b>Matriz B</b>'), 3, 0)
        sizeB = QHBoxLayout();
        sizeB.addWidget(QLabel('Filas:'))
        b_rows = QSpinBox(); b_rows.setRange(1, 20); b_rows.setValue(3); sizeB.addWidget(b_rows)
        sizeB.addWidget(QLabel('Columnas:'))
        b_cols = QSpinBox(); b_cols.setRange(1, 20); b_cols.setValue(3); sizeB.addWidget(b_cols)
        btn_setB = QPushButton('Aplicar tamaño B'); sizeB.addWidget(btn_setB)
        btn_randB = QPushButton('🎲 Aleatoria B'); sizeB.addWidget(btn_randB)
        gl.addLayout(sizeB, 4, 0)
        gl.addWidget(self.gridB, 5, 0)
        bar = QHBoxLayout()
        self.op_selector = QComboBox(); self.op_selector.addItems(['Suma (A + B)','Resta (A - B)','Producto (A · B)','Combinación (α·A + β·B)'])
        bar.addWidget(self.op_selector)
        calc = QPushButton('⚙️ Calcular'); bar.addWidget(calc)
        self.compat_label = QLabel('Compatibilidad: —'); bar.addWidget(self.compat_label); bar.addStretch(1)
        gl.addLayout(bar, 6, 0)
        # Combination controls (hidden unless selected)
        comb_wrap = QWidget(); comb_lay = QVBoxLayout(comb_wrap)
        coef_bar = QHBoxLayout(); comb_lay.addLayout(coef_bar)
        coef_bar.addWidget(QLabel('α:'))
        alpha_spin = QDoubleSpinBox(); alpha_spin.setDecimals(6); alpha_spin.setRange(-1e9, 1e9); alpha_spin.setValue(1.0); coef_bar.addWidget(alpha_spin)
        alpha_det = QCheckBox('α = det(C)'); coef_bar.addWidget(alpha_det)
        coef_bar.addSpacing(12)
        coef_bar.addWidget(QLabel('β:'))
        beta_spin = QDoubleSpinBox(); beta_spin.setDecimals(6); beta_spin.setRange(-1e9, 1e9); beta_spin.setValue(1.0); coef_bar.addWidget(beta_spin)
        beta_det = QCheckBox('β = det(C)'); coef_bar.addWidget(beta_det)
        # Matrix C for determinant
        csize = QHBoxLayout(); comb_lay.addLayout(csize)
        csize.addWidget(QLabel('Matriz C (para det): Filas:'))
        c_rows = QSpinBox(); c_rows.setRange(1, 12); c_rows.setValue(3); csize.addWidget(c_rows)
        csize.addWidget(QLabel('Columnas:'))
        c_cols = QSpinBox(); c_cols.setRange(1, 12); c_cols.setValue(3); csize.addWidget(c_cols)
        btn_c_set = QPushButton('Aplicar tamaño C'); csize.addWidget(btn_c_set)
        self.gridC = MatrixTable(3,3); comb_lay.addWidget(self.gridC)
        cacts = QHBoxLayout(); comb_lay.addLayout(cacts)
        btn_c_rand = QPushButton('🎲 Aleatoria C'); cacts.addWidget(btn_c_rand); cacts.addStretch(1)
        comb_wrap.setVisible(False)
        gl.addWidget(comb_wrap, 7, 0)
        self.center_layout.addWidget(cont)

        def update_compat():
            A = (self.gridA.rowCount(), self.gridA.columnCount()); B = (self.gridB.rowCount(), self.gridB.columnCount())
            op = self.op_selector.currentText()
            if op.startswith('Producto'):
                ok = (self.gridA.columnCount() == self.gridB.rowCount())
            else:
                ok = (A == B)
            # toggle combination controls
            comb_wrap.setVisible(op.startswith('Combinación'))
            self.compat_label.setText(f"Compatibilidad: {'OK' if ok else 'No compatible'} (A: {A[0]}×{A[1]}, B: {B[0]}×{B[1]})")
        self.op_selector.currentIndexChanged.connect(lambda _: update_compat())
        btn_setA.clicked.connect(lambda: (self.gridA.set_size(int(a_rows.value()), int(a_cols.value())), update_compat()))
        btn_setB.clicked.connect(lambda: (self.gridB.set_size(int(b_rows.value()), int(b_cols.value())), update_compat()))
        btn_randA.clicked.connect(lambda: (self.gridA.fill_random(), update_compat()))
        btn_randB.clicked.connect(lambda: (self.gridB.fill_random(), update_compat()))
        btn_c_set.clicked.connect(lambda: self.gridC.set_size(int(c_rows.value()), int(c_cols.value())))
        btn_c_rand.clicked.connect(lambda: self.gridC.fill_random())
        update_compat()

        def do_calc():
            op = self.op_selector.currentText()
            A = self.gridA.get_matrix(); B = self.gridB.get_matrix()
            try:
                if op.startswith('Suma'):
                    if A.shape != B.shape:
                        self.push_error('A y B deben tener la misma forma')
                        return
                    C = A + B; steps = add_steps(A,B)
                    self.push_result('Suma de matrices (A + B)', np.round(C,2), 'Resultado de A + B', steps)
                elif op.startswith('Resta'):
                    if A.shape != B.shape:
                        self.push_error('A y B deben tener la misma forma')
                        return
                    C = A - B; steps = sub_steps(A,B)
                    self.push_result('Resta de matrices (A - B)', np.round(C,2), 'Resultado de A - B', steps)
                elif op.startswith('Producto'):
                    if A.shape[1] != B.shape[0]:
                        self.push_error('Dimensiones incompatibles para multiplicación (cols de A ≠ filas de B)')
                        return
                    C = A @ B; steps = multiply_steps(A,B)
                    self.push_result('Producto de matrices (A · B)', np.round(C,2), 'Resultado de A · B', steps)
                else:
                    # Combinación lineal α·A + β·B
                    if A.shape != B.shape:
                        self.push_error('A y B deben tener la misma forma para la combinación lineal')
                        return
                    alpha = float(alpha_spin.value()); beta = float(beta_spin.value())
                    steps = []
                    # If any coefficient is set to det(C), compute it
                    if alpha_det.isChecked() or beta_det.isChecked():
                        Cmat = self.gridC.get_matrix()
                        if Cmat.shape[0] != Cmat.shape[1]:
                            self.push_error('Para usar det(C), C debe ser cuadrada')
                            return
                        det_steps = determinant_steps(Cmat)
                        # last det is number, but determinant_steps returns steps of matrices; we can compute value here
                        det_val = float(np.linalg.det(Cmat))
                        steps.extend(det_steps)
                        if alpha_det.isChecked():
                            alpha = det_val
                        if beta_det.isChecked():
                            beta = det_val
                    # Build steps for scaling and sum
                    A1 = alpha * A; B1 = beta * B
                    from sympy import Matrix as _SM
                    steps.append((f"Escalar α·A (α = {alpha})", _SM(A1.tolist())))
                    steps.append((f"Escalar β·B (β = {beta})", _SM(B1.tolist())))
                    R = A1 + B1
                    steps.append(("Suma α·A + β·B", _SM(R.tolist())))
                    self.push_result('Combinación lineal', np.round(R, 6), f"α·A + β·B (α={alpha}, β={beta})", steps)
            except Exception as e:
                self.push_error(str(e))
        calc.clicked.connect(do_calc)

    def show_ind(self):
        self.current_view = 'ind'
        self.center_title.setText('Independencia de vectores')
        self.clear_center()
        box = QVBoxLayout()
        wrap = QWidget(); wrap.setLayout(box)
        box.addWidget(QLabel('Introduce vectores (cada vector como fila):'))
        size_bar = QHBoxLayout(); box.addLayout(size_bar)
        size_bar.addWidget(QLabel('Filas:'))
        vi_rows = QSpinBox(); vi_rows.setRange(1, 20); vi_rows.setValue(3); size_bar.addWidget(vi_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        vi_cols = QSpinBox(); vi_cols.setRange(1, 20); vi_cols.setValue(3); size_bar.addWidget(vi_cols)
        btn_set = QPushButton('Aplicar tamaño'); size_bar.addWidget(btn_set)
        self.vgrid = MatrixTable(3,3)
        box.addWidget(self.vgrid)
        btn = QPushButton('🧭 Evaluar independencia'); box.addWidget(btn)
        self.center_layout.addWidget(wrap)

        btn_rand = QPushButton('🎲 Aleatoria'); box.addWidget(btn_rand)
        btn_set.clicked.connect(lambda: self.vgrid.set_size(int(vi_rows.value()), int(vi_cols.value())))
        btn_rand.clicked.connect(lambda: self.vgrid.fill_random())

        def calcular():
            try:
                M = self.vgrid.get_matrix(); mat = M.T
                rank = SMatrix(mat.tolist()).rank(); n_vecs = mat.shape[1]; dim = mat.shape[0]
                indep = rank == n_vecs
                txt = f"Dimensión del espacio: {dim}\nNúmero de vectores: {n_vecs}\nRango: {rank}\nConclusión: {'INDEPENDIENTES' if indep else 'DEPENDIENTES'}"
                steps = rref_steps(mat)
                self.push_result('Independencia de vectores', np.round(mat,2), txt, steps)
            except Exception as e:
                self.push_result('Error', None, str(e))
        btn.clicked.connect(calcular)

    def show_triu(self):
        self.current_view = 'triu'
        self.center_title.setText('Triangular Superior (U)')
        self.clear_center()
        box = QVBoxLayout(); wrap = QWidget(); wrap.setLayout(box)
        # size controls
        size_bar = QHBoxLayout(); box.addLayout(size_bar)
        size_bar.addWidget(QLabel('Filas:'))
        triu_rows = QSpinBox(); triu_rows.setRange(1, 12); triu_rows.setValue(4); size_bar.addWidget(triu_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        triu_cols = QSpinBox(); triu_cols.setRange(1, 12); triu_cols.setValue(4); size_bar.addWidget(triu_cols)
        btn_set = QPushButton('Aplicar tamaño'); size_bar.addWidget(btn_set)
        self.triu_grid = MatrixTable(4,4); box.addWidget(self.triu_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        btn_rand = QPushButton('🎲 Aleatoria'); acts.addWidget(btn_rand)
        btn = QPushButton('🔺 Calcular U'); acts.addWidget(btn)
        self.center_layout.addWidget(wrap)

        btn_set.clicked.connect(lambda: self.triu_grid.set_size(int(triu_rows.value()), int(triu_cols.value())))
        btn_rand.clicked.connect(lambda: self.triu_grid.fill_random())

        def calcular():
            try:
                A = self.triu_grid.get_matrix(); steps = upper_triangular_steps(A)
                final = np.round(np.array(steps[-1][1].tolist(), dtype=float), 2)
                self.push_result('Triangular superior (U)', final, 'Matriz U', steps)
            except Exception as e:
                self.push_result('Error', None, str(e))
        btn.clicked.connect(calcular)

    def show_rref(self):
        self.current_view = 'rref'
        self.center_title.setText('Forma Escalonada Reducida (RREF)')
        self.clear_center()
        box = QVBoxLayout(); wrap = QWidget(); wrap.setLayout(box)
        size_bar = QHBoxLayout(); box.addLayout(size_bar)
        size_bar.addWidget(QLabel('Filas:'))
        rref_rows = QSpinBox(); rref_rows.setRange(1, 12); rref_rows.setValue(3); size_bar.addWidget(rref_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        rref_cols = QSpinBox(); rref_cols.setRange(1, 12); rref_cols.setValue(4); size_bar.addWidget(rref_cols)
        btn_set = QPushButton('Aplicar tamaño'); size_bar.addWidget(btn_set)
        self.rref_grid = MatrixTable(3,4); box.addWidget(self.rref_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        btn_rand = QPushButton('🎲 Aleatoria'); acts.addWidget(btn_rand)
        btn = QPushButton('🧱 Calcular RREF'); acts.addWidget(btn)
        self.center_layout.addWidget(wrap)

        btn_set.clicked.connect(lambda: self.rref_grid.set_size(int(rref_rows.value()), int(rref_cols.value())))
        btn_rand.clicked.connect(lambda: self.rref_grid.fill_random())

        def calcular():
            try:
                A = self.rref_grid.get_matrix(); steps = rref_steps(A)
                final = np.round(np.array(steps[-1][1].tolist(), dtype=float), 2)
                self.push_result('RREF', final, f"Rango: {SMatrix(final.tolist()).rank()}", steps)
            except Exception as e:
                self.push_result('Error', None, str(e))
        btn.clicked.connect(calcular)

    def show_ti(self):
        self.current_view = 'ti'
        self.center_title.setText('Transpuesta e Inversa')
        self.clear_center()
        box = QVBoxLayout(); wrap = QWidget(); wrap.setLayout(box)
        size_bar = QHBoxLayout(); box.addLayout(size_bar)
        size_bar.addWidget(QLabel('Filas:'))
        ti_rows = QSpinBox(); ti_rows.setRange(1, 20); ti_rows.setValue(3); size_bar.addWidget(ti_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        ti_cols = QSpinBox(); ti_cols.setRange(1, 20); ti_cols.setValue(3); size_bar.addWidget(ti_cols)
        btn_set = QPushButton('Aplicar tamaño'); size_bar.addWidget(btn_set)
        self.ti_grid = MatrixTable(3,3); box.addWidget(self.ti_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        btn_rand = QPushButton('🎲 Aleatoria'); acts.addWidget(btn_rand)
        btn = QPushButton('🔁 Calcular'); acts.addWidget(btn)
        self.center_layout.addWidget(wrap)

        btn_set.clicked.connect(lambda: self.ti_grid.set_size(int(ti_rows.value()), int(ti_cols.value())))
        btn_rand.clicked.connect(lambda: self.ti_grid.fill_random())

        def calcular():
            try:
                A = self.ti_grid.get_matrix(); T = np.round(A.T, 2)
                tsteps = transpose_steps(A)
                self.push_result('Transpuesta', T, 'Matriz transpuesta.', tsteps)
                if A.shape[0] == A.shape[1]:
                    isteps = inverse_steps(A)
                    try:
                        inv = np.linalg.inv(A); invr = np.round(inv, 2)
                        self.push_result('Inversa', invr, 'Matriz inversa (si existe).', isteps)
                    except np.linalg.LinAlgError:
                        self.push_result('Inversa', None, 'La matriz no es invertible.', isteps)
                else:
                    self.push_result('Transpuesta / Inversa', T, 'La matriz no es cuadrada, no existe inversa.')
            except Exception as e:
                self.push_result('Error', None, str(e))
        btn.clicked.connect(calcular)

    def show_det(self):
        self.current_view = 'det'
        self.center_title.setText('Determinante')
        self.clear_center()
        box = QVBoxLayout(); wrap = QWidget(); wrap.setLayout(box)
        size_bar = QHBoxLayout(); box.addLayout(size_bar)
        size_bar.addWidget(QLabel('Filas:'))
        det_rows = QSpinBox(); det_rows.setRange(1, 12); det_rows.setValue(3); size_bar.addWidget(det_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        det_cols = QSpinBox(); det_cols.setRange(1, 12); det_cols.setValue(3); size_bar.addWidget(det_cols)
        btn_set = QPushButton('Aplicar tamaño'); size_bar.addWidget(btn_set)
        self.det_grid = MatrixTable(3,3); box.addWidget(self.det_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        btn_rand = QPushButton('🎲 Aleatoria'); acts.addWidget(btn_rand)
        btn = QPushButton('🧾 Calcular det'); acts.addWidget(btn)
        self.center_layout.addWidget(wrap)

        btn_set.clicked.connect(lambda: self.det_grid.set_size(int(det_rows.value()), int(det_cols.value())))
        btn_rand.clicked.connect(lambda: self.det_grid.fill_random())

        def calcular():
            try:
                A = self.det_grid.get_matrix()
                if A.shape[0] != A.shape[1]:
                    self.push_result('Determinante', None, 'Solo para matrices cuadradas'); return
                steps = determinant_steps(A)
                d = float(np.linalg.det(A))
                self.push_result('Determinante', None, f'det(A) = {fmt_num(d, 6)}', steps)
            except Exception as e:
                self.push_result('Error', None, str(e))
        btn.clicked.connect(calcular)

    def show_cramer(self):
        self.current_view = 'cramer'
        self.center_title.setText('Método de Cramer')
        self.clear_center()
        box = QVBoxLayout(); wrap = QWidget(); wrap.setLayout(box)
        top = QHBoxLayout(); box.addLayout(top)
        top.addWidget(QLabel('Dimensión del sistema (n):'))
        self.cramer_n = QSpinBox(); self.cramer_n.setRange(1, 10); self.cramer_n.setValue(3)
        top.addWidget(self.cramer_n)
        btn_create = QPushButton('Crear sistema'); top.addWidget(btn_create)
        # placeholder: no grid until user creates it
        self.cramer_grid = None
        self.cramer_grid_host = QWidget(); self.cramer_grid_layout = QVBoxLayout(self.cramer_grid_host)
        box.addWidget(self.cramer_grid_host)
        # actions area under grid (created after grid exists)
        self.cramer_actions_host = QWidget(); self.cramer_actions_layout = QHBoxLayout(self.cramer_actions_host)
        box.addWidget(self.cramer_actions_host)
        btn_calc = QPushButton('📐 Resolver'); btn_calc.setEnabled(False); box.addWidget(btn_calc)
        self.center_layout.addWidget(wrap)

        def crear():
            # Remove previous grid if present
            while self.cramer_grid_layout.count():
                w = self.cramer_grid_layout.takeAt(0).widget()
                if w:
                    w.setParent(None)
            n = int(self.cramer_n.value())
            self.cramer_grid = MatrixTable(n, n+1)
            # set headers: x1..xn and b
            col_headers = [f"x{i+1}" for i in range(n)] + ['b']
            row_headers = [str(i+1) for i in range(n)]
            self.cramer_grid.set_headers(col_headers=col_headers, row_headers=row_headers)
            self.cramer_grid_layout.addWidget(self.cramer_grid)
            # reset actions layout and add random button
            while self.cramer_actions_layout.count():
                w = self.cramer_actions_layout.takeAt(0).widget()
                if w: w.setParent(None)
            btn_rand = QPushButton('🎲 Aleatoria [A|b]')
            self.cramer_actions_layout.addWidget(btn_rand)
            btn_rand.clicked.connect(lambda: self.cramer_grid.fill_random())
            btn_calc.setEnabled(True)
        btn_create.clicked.connect(crear)

        def calcular():
            try:
                if self.cramer_grid is None:
                    self.push_result('Cramer', None, 'Primero crea el sistema con la dimensión n.'); return
                M = self.cramer_grid.get_matrix(); n = M.shape[0]
                if M.shape[1] != n+1: raise ValueError('La matriz debe ser de tamaño n × (n+1).')
                A = M[:, :n]; b = M[:, n:]
                sol, steps, detA, det_cols, sol_exacta = cramer_steps(A, b)
                if sol is None:
                    self.push_result('Cramer', None, 'det(A) = 0 ⇒ no existe solución única.', steps)
                    return
                detalle = [f"det(A) = {detA}"]
                for idx, det_col in enumerate(det_cols):
                    detalle.append(f"x{idx+1} = det(A_{idx+1})/det(A) = {det_col}/{detA} = {sol_exacta[idx]}")
                detalle.append('Vector solución (aprox)')
                # La matriz (vector solución) se muestra arriba en la tabla; evitamos duplicarlo en texto
                self.push_result('Método de Cramer', np.round(sol, 4), '\n'.join(detalle), steps)
            except Exception as e:
                self.push_result('Error', None, str(e))
        btn_calc.clicked.connect(calcular)

    def show_bisection(self):
        self.current_view = 'bisection'
        self.center_title.setText('Método de Bisección')
        self.clear_center()
        box = QVBoxLayout(); box.setContentsMargins(8, 4, 8, 6); box.setSpacing(6)
        wrap = QWidget(); wrap.setLayout(box)
        # Limitar ancho de esta vista para dejar más espacio a resultados a la derecha
        wrap.setMaximumWidth(720)

        # Function input
        fn_row = QHBoxLayout(); box.addLayout(fn_row)
        lbl_fn = QLabel('f(x) =')
        lbl_fn.setStyleSheet(f"font-family: {MATH_FONT_STACK}; font-size:14px;")
        fn_row.addWidget(lbl_fn)
        # Campo de función sin valor por defecto
        expr_edit = QLineEdit(""); expr_edit.setPlaceholderText("Expresión en x, p.ej. x**3 - x - 2")
        expr_edit.setStyleSheet(f"font-family: {MATH_FONT_STACK}; font-size:13px;")
        expr_edit.setFixedHeight(28)
        fn_row.addWidget(expr_edit, 1)

        # (WYSIWYG retirado) — eliminamos el recuadro informativo para ahorrar espacio

        # Teclado matemático compacto justo debajo de la función
        keys_panel = QWidget(); kb = QGridLayout(keys_panel)
        kb.setContentsMargins(0, 4, 0, 0); kb.setHorizontalSpacing(6); kb.setVerticalSpacing(6)
        box.addWidget(keys_panel)

        def add_key(text, insert, r, c):
            btn = QToolButton(); btn.setText(text)
            btn.setFixedSize(40, 34)
            btn.setStyleSheet(
                "QToolButton{background:#2b2d31; border:1px solid #3a3d41; border-radius:6px; font-family:"+MATH_FONT_STACK+"; font-size:13px;}"
                "QToolButton:hover{border-color:#0099a8}"
                "QToolButton:pressed{background:#1f2225}"
            )
            btn.setToolTip(insert)
            kb.addWidget(btn, r, c)
            def on_click():
                expr_edit.insert(insert)
                if insert.endswith('()'):
                    expr_edit.setCursorPosition(expr_edit.cursorPosition()-1)
            btn.clicked.connect(on_click)

        # Fila funciones/símbolos (8 columnas)
        row = 0
        for idx,(t,i) in enumerate([
            ('x','x'), ('x^2','**2'), ('x^3','**3'), ('^','**'), ('(','('), (')',')'), ('|x|','Abs()'), ('√','sqrt()')
        ]):
            add_key(t,i,row,idx)
        # Fila trig/log/const
        row = 1
        for idx,(t,i) in enumerate([
            ('sen','sin()'), ('cos','cos()'), ('tg','tan()'), ('ln','ln()'), ('log','log()'), ('exp','exp()'), ('π','pi'), ('e','E')
        ]):
            add_key(t,i,row,idx)
        # Números y operaciones en 4 columnas
        for r, cols in enumerate([
            [('7','7'),('8','8'),('9','9'),('÷','/')],
            [('4','4'),('5','5'),('6','6'),('×','*')],
            [('1','1'),('2','2'),('3','3'),('−','-')],
            [('0','0'),('.','.'),(', ',','),('+','+')],
        ], start=2):
            for c,(t,i) in enumerate(cols):
                add_key(t,i,r,c)

        fmt_lbl = QLabel('Formato: usa ** para potencias; puedes escribir funciones SymPy (sin, cos, exp, log, ...).')
        fmt_lbl.setStyleSheet('color:#9aa4aa; font-size:12px;')
        box.addWidget(fmt_lbl)

        # Parameters — diseño compacto con etiquetas arriba y sin valores por defecto
        p_row = QHBoxLayout(); p_row.setSpacing(10); box.addLayout(p_row)
        def make_col(label_text, widget):
            col = QVBoxLayout();
            lab = QLabel(label_text); lab.setStyleSheet('color:#9aa4aa; font-size:12px;')
            col.addWidget(lab)
            col.addWidget(widget)
            return col

        # xi
        xi_spin = TrimDoubleSpinBox(); xi_spin.setDecimals(8); xi_spin.setRange(-1e12, 1e12)
        xi_spin.setSpecialValueText(''); xi_spin.setValue(xi_spin.minimum()); xi_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Intervalo inferior (xi)', xi_spin))
        # xu
        xu_spin = TrimDoubleSpinBox(); xu_spin.setDecimals(8); xu_spin.setRange(-1e12, 1e12)
        xu_spin.setSpecialValueText(''); xu_spin.setValue(xu_spin.minimum()); xu_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Intervalo superior (xu)', xu_spin))
        # epsilon
        eps_spin = TrimDoubleSpinBox(); eps_spin.setDecimals(8); eps_spin.setRange(1e-12, 1.0); eps_spin.setSingleStep(1e-4)
        eps_spin.setMinimum(0.0); eps_spin.setSpecialValueText(''); eps_spin.setValue(0.0); eps_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Error de convergencia (ε)', eps_spin))
        # iteraciones
        itmax = QSpinBox(); itmax.setRange(1, 1000); itmax.setMinimum(0); itmax.setSpecialValueText(''); itmax.setValue(0); itmax.setFixedWidth(100)
        p_row.addLayout(make_col('Iter máx', itmax))
        # botones a la derecha
        rand_btn = QPushButton('🎲 Aleatoria'); rand_btn.setToolTip('Rellenar con función e intervalo aleatorios válidos')
        p_row.addWidget(rand_btn)
        calc_btn = QPushButton('⚙️ Calcular'); p_row.addWidget(calc_btn)

        # (Vista previa f(xi), f(xu) retirada a petición del usuario)

        # (Botón 'Buscar intervalo' retirado a petición del usuario)

        # Host for results will go to right panel as a card
        self.center_layout.addWidget(wrap)

        def calc():
            try:
                expr_text = expr_edit.text().strip()
                if not expr_text:
                    self.push_error('Escribe una expresión para f(x).'); return
                # Validar que los campos numéricos no estén "vacíos"
                def _is_empty_spin(sp):
                    return (sp.specialValueText()=='' and sp.value()==sp.minimum())
                if any([_is_empty_spin(x) for x in (xi_spin, xu_spin, eps_spin, itmax)]):
                    self.push_error('Completa xi, xu, ε e iter máx.'); return
                x = _SYM_symbols('x')
                expr = _SYM_sympify(expr_text)
                f = _SYM_lambdify(x, expr, 'numpy')
                xi = float(xi_spin.value()); xu = float(xu_spin.value())
                if not (xi < xu):
                    self.push_error('Debe cumplirse xi < xu.'); return
                yi = float(f(xi)); yu = float(f(xu))
                if np.isnan(yi) or np.isnan(yu) or np.isinf(yi) or np.isinf(yu):
                    self.push_error('f(xi) o f(xu) no es válido. Revisa la expresión/intervalo.'); return
                if yi * yu > 0:
                    self.push_error('No hay cambio de signo en [xi, xu]. Elige otro intervalo.'); return
                eps = float(eps_spin.value()); max_iter = int(itmax.value())
                if max_iter <= 0:
                    self.push_error('Iteraciones máximas debe ser > 0.'); return
                rows = []  # (iter, xi, xu, xr, Ea, yi, yu, yr)
                xr_old = None
                xi_c, xu_c, yi_c, yu_c = xi, xu, yi, yu
                for it in range(1, max_iter+1):
                    xr = 0.5*(xi_c + xu_c)
                    yr = float(f(xr))
                    Ea = 0.0 if xr_old is None else abs(xr - xr_old)
                    rows.append((it, xi_c, xu_c, xr, Ea, yi_c, yu_c, yr))
                    if xr_old is not None and Ea <= eps:
                        break
                    # Decide subinterval
                    if yi_c * yr < 0:
                        xu_c, yu_c = xr, yr
                    else:
                        xi_c, yi_c = xr, yr
                    xr_old = xr

                # Build summary card and open detailed dialog
                self._push_bisect_summary_card(expr_text, xi, xu, rows)
                self._show_bisect_dialog(expr_text, xi, xu, rows)
            except Exception as e:
                self.push_error(str(e))

        calc_btn.clicked.connect(calc)

        # --- Generador aleatorio coherente ---
        def fill_random():
            try:
                x = _SYM_symbols('x')

                def build_expr_text():
                    # Seleccionamos un tipo de función sencillo y robusto
                    choice = np.random.choice(['poly', 'sin', 'poly_sin', 'exp'])
                    if choice == 'poly':
                        deg = int(np.random.randint(2, 5))
                        coeffs = list(np.random.randint(-5, 6, size=deg+1))
                        while coeffs[0] == 0:
                            coeffs[0] = int(np.random.randint(-5, 6))
                        # Construir string: a*x**n + b*x**(n-1) + ... + c
                        terms = []
                        p = deg
                        for c in coeffs:
                            if p > 1:
                                terms.append(f"{c}*x**{p}")
                            elif p == 1:
                                terms.append(f"{c}*x")
                            else:
                                terms.append(f"{c}")
                            p -= 1
                        return ' + '.join(terms).replace('+ -', '- ')
                    elif choice == 'sin':
                        a = int(np.random.randint(1, 4)); b = int(np.random.randint(1, 4)); d = int(np.random.randint(-2, 3))
                        return f"{a}*sin({b}*x) + {d}"
                    elif choice == 'poly_sin':
                        a = int(np.random.randint(-3, 4)); b = int(np.random.randint(1, 4)); c = int(np.random.randint(-2, 3)); d = int(np.random.randint(-2, 3))
                        if a == 0: a = 1
                        return f"{a}*x**2 + {b}*sin(x) + {c}*x + {d}"
                    else:  # 'exp'
                        a = int(np.random.randint(1, 4)); b = float(np.random.choice([0.3, 0.5, 1.0]))
                        cst = int(np.random.randint(0, 4))
                        return f"{a}*exp({b}*x) - {cst}"

                # Intentamos varias veces hasta lograr cambio de signo en [-5, 5]
                for _ in range(12):
                    expr_text = build_expr_text()
                    expr = _SYM_sympify(expr_text)
                    f = _SYM_lambdify(x, expr, 'numpy')
                    xs = np.linspace(-5.0, 5.0, 400)
                    ys = f(xs)
                    ys = np.asarray(ys, dtype=float)
                    finite = np.isfinite(ys)
                    found = False
                    for i in range(len(xs)-1):
                        if not (finite[i] and finite[i+1]):
                            continue
                        if ys[i] == 0:
                            xi_val = float(xs[i] - 0.5*(xs[1]-xs[0]))
                            xu_val = float(xs[i] + 0.5*(xs[1]-xs[0]))
                            found = True
                            break
                        if ys[i] * ys[i+1] < 0:
                            xi_val = float(xs[i])
                            xu_val = float(xs[i+1])
                            found = True
                            break
                    if found:
                        # Elegimos epsilon razonable y máx iteraciones
                        eps_val = float(np.random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
                        it_val = int(np.random.randint(18, 45))
                        expr_edit.setText(expr_text)
                        xi_spin.setValue(xi_val)
                        xu_spin.setValue(xu_val)
                        eps_spin.setValue(eps_val)
                        itmax.setValue(it_val)
                        return
                # Fallback conocido
                expr_edit.setText("x**3 - x - 2")
                xi_spin.setValue(1.0)
                xu_spin.setValue(2.0)
                eps_spin.setValue(1e-4)
                itmax.setValue(25)
            except Exception as e:
                self.push_error(str(e))

        rand_btn.clicked.connect(fill_random)

    def _push_bisect_summary_card(self, expr_text: str, xi: float, xu: float, rows):
        card = QWidget(); card.setObjectName('resultCard'); lay = QVBoxLayout(card)
        title = QLabel('<b>Método de bisección</b>'); lay.addWidget(title)
        formula = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_text}</span>")
        lay.addWidget(formula)
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            sgn = '+' if xr >= 0 else ''
            summary = QLabel(f"El método converge en <b>{n}</b> iteraciones.<br>LA RAÍZ ES: <b>{sgn}{xr:.6f}</b>")
            lay.addWidget(summary)

        # Actions
        row = QHBoxLayout(); lay.addLayout(row)
        btn_open = QPushButton('🔍 Ver detalles…'); row.addWidget(btn_open)
        btn_delete = QPushButton('🗑️ Quitar'); row.addWidget(btn_delete); row.addStretch(1)

        def _open():
            self._show_bisect_dialog(expr_text, xi, xu, rows)
        def _remove():
            card.setParent(None)
            if card in self._result_widgets:
                self._result_widgets.remove(card)
        btn_open.clicked.connect(_open)
        btn_delete.clicked.connect(_remove)
        self.right_layout.addWidget(card)
        self._result_widgets.append(card)

    def _show_bisect_dialog(self, expr_text: str, xi: float, xu: float, rows):
        try:
            dlg = BisectionResultDialog(expr_text, xi, xu, rows, self)
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            self._dialogs.append(dlg)
            def _cleanup():
                if dlg in self._dialogs:
                    self._dialogs.remove(dlg)
            dlg.destroyed.connect(lambda *_: _cleanup())
            dlg.show()
        except Exception as e:
            self.push_error(str(e))


# +++++++++++++++++++++++++++
# Splash Screen (pantalla de carga)
# +++++++++++++++++++++++++++
def _resource_path(name: str) -> str:
    """Busca 'name' en: cwd, carpeta del módulo y raíz del proyecto."""
    here = os.path.dirname(__file__)
    # Cuando se ejecuta como ejecutable (PyInstaller onefile), los recursos se
    # extraen temporalmente en sys._MEIPASS. Lo probamos primero si existe.
    meipass = getattr(sys, '_MEIPASS', None)
    candidates = [
        os.path.join(meipass, name) if meipass else None,
        os.path.join(os.getcwd(), name),
        os.path.join(here, name),
        os.path.join(os.path.dirname(here), name),
    ]
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            return p
    return name  # Qt intentará resolverlo igualmente

class SplashScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.SplashScreen | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Contenedor con esquinas redondeadas y sombra
        frame = QFrame(self)
        frame.setObjectName("splashFrame")
        frame.setStyleSheet("""
            QFrame#splashFrame {
                background-color: #1e1f22;
                border: 1px solid #2b2d31;
                border-radius: 18px;
            }
            QLabel#splashTitle { font-size: 16px; font-weight: 700; color: #e6eef2; }
            QLabel#splashSub   { font-size: 11px; color: #99a1a7; }
            QProgressBar {
                background-color: #101214;
                border: 1px solid #30343a;
                border-radius: 6px;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #0099a8;
                border-radius: 6px;
            }
        """)
        eff = QGraphicsDropShadowEffect(self)
        eff.setBlurRadius(28); eff.setOffset(0, 8); eff.setColor(QColor(0, 0, 0, 180))
        frame.setGraphicsEffect(eff)

        lay = QVBoxLayout(frame); lay.setContentsMargins(22, 20, 22, 18); lay.setSpacing(10)

        # Logo
        logo = QLabel()
        pix = QPixmap(_resource_path("logo.png"))
        if not pix.isNull():
            pix = pix.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pix)
        logo.setAlignment(Qt.AlignCenter)
        lay.addWidget(logo, 0, Qt.AlignCenter)

        title = QLabel("Matrix")
        title.setObjectName("splashTitle")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        sub = QLabel("Cargando componentes…")
        sub.setObjectName("splashSub")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # indeterminado
        lay.addWidget(self.bar)

        # Tamaño y centrado
        frame.resize(320, 260)
        self.resize(frame.size())
        frame.move(0, 0)
        scr = QApplication.primaryScreen().geometry()
        self.move(int(scr.center().x() - self.width()/2), int(scr.center().y() - self.height()/2))

        # Animaciones
        self.setWindowOpacity(0.0)
        self._fade_in = QPropertyAnimation(self, b"windowOpacity")
        self._fade_in.setDuration(300)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.OutCubic)

        self._fade_out = QPropertyAnimation(self, b"windowOpacity")
        self._fade_out.setDuration(250)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out.finished.connect(self.close)

    def start(self):
        self.show()
        self._fade_in.start()

    def finish(self, after: callable | None = None):
        def done():
            if callable(after):
                after()
        self._fade_out.finished.connect(done)
        self._fade_out.start()


def run():
    # En Windows, establece un AppUserModelID explícito para que la barra de tareas
    # use el icono de la ventana (logo.png) en lugar del icono de python.exe y para
    # que el agrupado sea independiente si el usuario fija la app.
    if sys.platform == 'win32':
        try:
            import ctypes  # lazy import para evitar dependencia en otros SO
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'hk_matrix.matrix_app'
            )
        except Exception:
            pass
    app = QApplication(sys.argv)

    # Icono global para que Windows muestre el logo en la barra de tareas y miniaturas
    try:
        app.setWindowIcon(QIcon(_resource_path('logo.png')))
    except Exception:
        pass

    # Mostrar splash
    splash = SplashScreen()
    splash.start()
    app.processEvents()

    # Construir la ventana principal mientras el splash está visible
    w = MatrixQtApp()

    # Cerrar splash con fade-out y mostrar la app
    def _show_main():
        w.show()
        splash.finish()
    QTimer.singleShot(700, _show_main)  # breve espera para sensación fluida

    sys.exit(app.exec())


if __name__ == '__main__':
    run()
