from __future__ import annotations
import sys
import os
import html
from PySide6.QtGui import QFontDatabase, QFont
from sympy import (
    symbols as _SYM_symbols,
    sympify as _SYM_sympify,
    lambdify as _SYM_lambdify,
    diff as _SYM_diff,
)
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np
from sympy import Matrix as SMatrix


def _safe_sec(x):
    return 1.0 / np.cos(x)


def _safe_csc(x):
    return 1.0 / np.sin(x)


def _safe_cot(x):
    return 1.0 / np.tan(x)


_LAMBDA_EXTRA_FUNCS = {
    'Abs': np.abs,
    'sec': _safe_sec,
    'csc': _safe_csc,
    'cot': _safe_cot,
    'cotg': _safe_cot,
    'sech': lambda x: 1.0 / np.cosh(x),
    'csch': lambda x: 1.0 / np.sinh(x),
    'coth': lambda x: 1.0 / np.tanh(x),
    'asec': lambda x: np.arccos(1.0 / x),
    'acsc': lambda x: np.arcsin(1.0 / x),
    'acot': lambda x: np.arctan(1.0 / x),
}

_LAMBDA_MODULES = [_LAMBDA_EXTRA_FUNCS, 'numpy']

SECANT_DECIMALS = 6

# -------------------------------
# Design system base (Fase 1)
# -------------------------------
BG_MAIN = "#1e1e2e"          # Fondo principal muy oscuro, azulado
BG_PANEL = "#252535"         # Paneles ligeramente más claros
ACCENT_PRIMARY = "#7f5af0"   # Violeta vibrante para acciones principales
ACCENT_SECONDARY = "#2cb67d" # Verde neón suave para estados OK
TEXT_PRIMARY = "#fffffe"     # Blanco cálido
TEXT_MUTED = "#94a1b2"       # Gris suave

from hk_matrix.logic.core import (
    fmt_matrix, fmt_num,
    add_steps, sub_steps, multiply_steps,
    rref_steps, upper_triangular_steps,
    transpose_steps, inverse_steps,
    determinant_steps, cramer_steps,
)
from PySide6.QtCore import Qt, QTimer, QEasingCurve, QPropertyAnimation, QUrl, QLocale, QPoint
from PySide6.QtGui import (
    QIcon, QColor, QKeySequence, QShortcut, QPixmap, QClipboard, QDesktopServices
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QSpinBox, QLabel, QTableWidget, QTableWidgetItem, QLineEdit,
    QListWidget, QListWidgetItem, QComboBox, QSplitter, QScrollArea,
    QDialog, QAbstractItemView, QCheckBox, QDoubleSpinBox, QToolButton,
    QProgressBar, QFrame, QGraphicsDropShadowEffect, QStackedWidget,
    QSizePolicy, QMessageBox, QButtonGroup, QGroupBox, QRadioButton
)
from urllib.parse import quote_plus as _url_quote_plus

# Tipografía matemática monoespaciada utilizada en tablas y fórmulas sencillas
MATH_FONT_STACK = "'Consolas','DejaVu Sans Mono','Courier New',monospace"


class TitleBar(QWidget):
    """Barra de título personalizada para la ventana principal."""

    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self._window = parent
        self._mouse_pos = QPoint()

        self.setFixedHeight(32)
        self.setObjectName('titleBar')
        self.setAutoFillBackground(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        # Icono de la app
        icon_label = QLabel()
        try:
            icon = QIcon(_resource_path('logo.png'))
            pix = icon.pixmap(20, 20)
            icon_label.setPixmap(pix)
        except Exception:
            icon_label.setText('🔢')
        icon_label.setFixedSize(20, 20)
        layout.addWidget(icon_label)

        # Título
        title_label = QLabel('Matrix')
        title_label.setStyleSheet('color:#ffffff; font-weight:600; letter-spacing:0.5px;')
        layout.addWidget(title_label)

        layout.addStretch(1)

        def make_btn(text: str, tooltip: str = '') -> QPushButton:
            btn = QPushButton(text)
            btn.setFixedSize(32, 22)
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(
                "QPushButton{background:transparent; color:#c3c8d4; border:none;}"
                "QPushButton:hover{background-color:rgba(255,255,255,0.08);}"
            )
            return btn

        btn_min = make_btn('−', 'Minimizar')
        btn_max = make_btn('□', 'Maximizar / Restaurar')
        btn_close = make_btn('✕', 'Cerrar')
        btn_close.setStyleSheet(
            "QPushButton{background:transparent; color:#c3c8d4; border:none;}"
            "QPushButton:hover{background-color:#ff4b4b; color:#ffffff;}"
        )

        layout.addWidget(btn_min)
        layout.addWidget(btn_max)
        layout.addWidget(btn_close)

        btn_min.clicked.connect(self._on_minimize)
        btn_max.clicked.connect(self._on_maximize_restore)
        btn_close.clicked.connect(self._on_close)

        self.setStyleSheet(
            "#titleBar{background-color:#1e1e2e; border-bottom:1px solid #333333;}"
        )

    def _on_minimize(self):
        self._window.showMinimized()

    def _on_maximize_restore(self):
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def _on_close(self):
        self._window.close()

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._mouse_pos = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if event.buttons() & Qt.LeftButton:
            pos = event.globalPosition().toPoint() - self._mouse_pos
            self._window.move(pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)


class WelcomeScreen(QMainWindow):
    """Pantalla de bienvenida a pantalla casi completa antes de abrir MatrixQtApp."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(900, 600)

        self.setObjectName('welcomeWindow')
        try:
            self.setWindowIcon(QIcon(_resource_path('logo.png')))
        except Exception:
            pass

        # Contenedor con fondo de imagen
        outer = QWidget(self)
        self.setCentralWidget(outer)

        frame = QFrame()
        frame.setObjectName('welcomeFrame')
        frame.setStyleSheet(self._build_stylesheet())

        layout = QVBoxLayout(outer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(frame)

        inner_layout = QVBoxLayout(frame)
        inner_layout.setContentsMargins(24, 20, 24, 24)
        inner_layout.setSpacing(18)

        # Barra superior con controles de ventana
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)

        def make_top_button(text: str, tooltip: str = '', is_close: bool = False, is_settings: bool = False) -> QToolButton:
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.PointingHandCursor)
            # Ajustar tamaño: más grande para settings, estándar para otros
            if is_settings:
                btn.setFixedSize(36, 36)
            else:
                btn.setFixedSize(30, 30)
            base = (
                "QToolButton {"
                "  background: transparent;"
                "  border: none;"
                "  border-radius: 18px;"
                "  color: #f5f5ff;"
                "  font-size: 16px;"
                "}"
            )
            if is_close:
                hover = (
                    "QToolButton:hover {"
                    "  background-color: #ff5555;"
                    "  color: #ffffff;"
                    "}"
                )
            else:
                hover = (
                    "QToolButton:hover {"
                    "  background-color: rgba(255,255,255,0.15);"
                    "  color: #ffffff;"
                    "}"
                )
            btn.setStyleSheet(base + hover)
            return btn

        # Botón Configuración (más grande y visible)
        btn_settings = make_top_button("⚙", "Configuración", is_settings=True)
        btn_settings.clicked.connect(self.open_settings)
        top_bar.addWidget(btn_settings)

        top_bar.addStretch(1)

        # Botón Minimizar
        btn_min = make_top_button("−", "Minimizar")
        btn_min.clicked.connect(self.showMinimized)
        top_bar.addWidget(btn_min)

        # Botón Cerrar
        btn_close = make_top_button("✕", "Cerrar", is_close=True)
        btn_close.clicked.connect(self.close)
        top_bar.addWidget(btn_close)

        inner_layout.addLayout(top_bar)

        inner_layout.addStretch(1)

        # Título principal
        title = QLabel('MATRIX')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #ffffff; font-size: 64px; font-weight: 800;"
            "letter-spacing: 6px;"
        )
        # Efecto glow mediante QGraphicsDropShadowEffect (en lugar de text-shadow)
        try:
            glow = QGraphicsDropShadowEffect(title)
            glow.setBlurRadius(30)
            glow.setColor(QColor("#7f5af0"))
            glow.setOffset(0, 0)
            title.setGraphicsEffect(glow)
        except Exception:
            pass
        inner_layout.addWidget(title)

        # Subtítulo
        subtitle = QLabel('Linear Algebra & Numerical Methods Suite')
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "color: #a78bfa; font-size: 18px; letter-spacing: 2px;"
        )
        inner_layout.addWidget(subtitle)

        # Separador sutil
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: rgba(255,255,255,0.16);")
        inner_layout.addWidget(sep)

        # Botón principal de entrada
        btn_enter = QPushButton('INICIAR SISTEMA')
        btn_enter.setCursor(Qt.PointingHandCursor)
        btn_enter.setFixedHeight(52)
        btn_enter.setStyleSheet(
            "QPushButton {"
            "  color: #ffffff;"
            "  font-size: 16px; font-weight: 600;"
            "  padding: 12px 36px;"
            "  border-radius: 24px;"
            "  border: 2px solid #7f5af0;"
            "  background-color: transparent;"
            "}"
            "QPushButton:hover {"
            "  background-color: #7f5af0;"
            "  color: #ffffff;"
            "  border-color: #9b6bff;"
            "}"
        )
        inner_layout.addWidget(btn_enter, 0, Qt.AlignHCenter)

        inner_layout.addStretch(2)

        # Créditos en la esquina inferior derecha
        credits_row = QHBoxLayout()
        credits_row.addStretch(1)
        credits = QLabel('Created by Nelson Lacayo')
        credits.setStyleSheet("color:#b0b8c8; font-size:11px;")
        credits_row.addWidget(credits)
        inner_layout.addLayout(credits_row)

        # Conectar botón
        btn_enter.clicked.connect(self._on_enter_clicked)

        # Centrar en pantalla
        scr = QApplication.primaryScreen().geometry()
        self.move(int(scr.center().x() - self.width()/2), int(scr.center().y() - self.height()/2))

    def _build_stylesheet(self) -> str:
        bg_path = _resource_path(os.path.join('assets', 'welcome_bg.png'))
        if os.path.exists(bg_path):
            # Fondo con imagen PNG
            return (
                "#welcomeFrame {"
                "  border-radius: 20px;"
                f"  background-image: url('{bg_path.replace(chr(92), '/')}');"
                "  background-position: center;"
                "  background-repeat: no-repeat;"
                "}"
            )
        # Fallback si no existe la imagen
        return (
            "#welcomeFrame {"
            "  border-radius: 20px;"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            "    stop:0 #1e1e2e, stop:1 #0f0f1a);"
            "}"
        )

    def _on_enter_clicked(self):
        # Placeholder: la lógica real de transición se implementa en run()
        self.close()

    def open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configuración")
        self.setModal(True)
        self.resize(420, 260)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #f5f5ff;
            }
            QGroupBox {
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #a78bfa;
            }
            QPushButton {
                background-color: #7f5af0;
                color:white;
                border-radius: 6px;
                padding: 6px 14px;
                border:none;
            }
            QPushButton:hover {
                background-color: #9b6bff;
            }
        """)

        # --- Contenido principal ---
        # Grupo Tema
        theme_group = QGroupBox("Tema")
        theme_layout = QHBoxLayout(theme_group)
        self.radio_dark = QRadioButton("Oscuro (actual)")
        self.radio_light = QRadioButton("Claro (próximamente)")
        self.radio_dark.setChecked(True)
        self.radio_light.setEnabled(False)
        theme_layout.addWidget(self.radio_dark)
        theme_layout.addWidget(self.radio_light)
        root.addWidget(theme_group)

        # Grupo Idioma
        lang_group = QGroupBox("Idioma")
        lang_layout = QHBoxLayout(lang_group)
        lang_label = QLabel("Idioma de la interfaz:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Español", "English"])
        self.lang_combo.setCurrentIndex(0)
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        root.addWidget(lang_group)

        # Grupo Experiencia
        exp_group = QGroupBox("Experiencia")
        exp_layout = QVBoxLayout(exp_group)
        self.animations_check = QCheckBox("Activar animaciones suaves")
        self.animations_check.setChecked(True)
        exp_layout.addWidget(self.animations_check)
        root.addWidget(exp_group)

        # Botón Guardar y cerrar
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        save_btn = QPushButton("Guardar y cerrar")
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

        # Overlay "Próximamente" para indicar que la configuración aún no está activa
        overlay = QLabel(self.tr("Próximamente"), self)
        overlay.setAlignment(Qt.AlignCenter)
        overlay.setStyleSheet(
            "QLabel{"
            "  background-color: rgba(0, 0, 0, 180);"
            "  color: #ffffff;"
            "  font-size: 18px;"
            "  font-weight: 700;"
            "  border-radius: 10px;"
            "}"
        )
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        overlay.resize(self.size())
        overlay.move(0, 0)
        overlay.raise_()


class ResultCard(QFrame):
    """Tarjeta de resultado elegante para el panel derecho.

    Puede construirse a partir de una matriz + descripción + pasos (modo clásico)
    o recibiendo un `content_widget` arbitrario ya maquetado para el cuerpo.
    """

    def __init__(
        self,
        title: str,
        main_window: "MatrixQtApp",
        matrix=None,
        description: str = "",
        steps=None,
        details_callback=None,
        copy_text: str | None = None,
        content_widget: QWidget | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._main = main_window
        self._matrix = matrix
        self._description = description or ""
        self._steps = steps
        self._copy_text = copy_text
        self._content_widget = content_widget
        self._details_callback = details_callback

        self.setObjectName("resultCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Estilo base por código (además del QSS existente)
        self.setStyleSheet("""
            QFrame#resultCard {
                background-color: #252535;
                border: 1px solid #333333;
                border-radius: 12px;
            }
        """)

        try:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(18)
            shadow.setOffset(0, 6)
            shadow.setColor(QColor(0, 0, 0, 130))
            self.setGraphicsEffect(shadow)
        except Exception:
            pass

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(8)

        # Encabezado
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        icon_label = QLabel("📊")
        icon_label.setStyleSheet("font-size: 15px;")
        header.addWidget(icon_label)

        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("color: #ffffff; font-size: 13px;")
        header.addWidget(title_label)

        header.addStretch(1)

        close_btn = QToolButton()
        close_btn.setText("✕")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QToolButton {
                background: transparent;
                color: #888;
                font-size: 12px;
                padding: 0 4px;
                border: none;
            }
            QToolButton:hover {
                color: #ffffff;
            }
        """)
        close_btn.clicked.connect(self._on_close)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Cuerpo: contenido arbitrario o matriz/descripcion
        if self._content_widget is not None:
            layout.addWidget(self._content_widget)
        else:
            if matrix is not None:
                tbl = QTableWidget()
                set_table_preview(tbl, np.array(matrix, dtype=float))
                tbl.setAlternatingRowColors(True)
                tbl.setStyleSheet(
                    f"QTableWidget{{gridline-color:#444;}} "
                    f"QTableWidget::item{{padding:4px; font-family: {MATH_FONT_STACK};}}"
                )
                layout.addWidget(tbl)

            if description:
                lbl = QLabel(description)
                lbl.setWordWrap(True)
                lbl.setStyleSheet(f"font-family: {MATH_FONT_STACK}; font-size: 12px; color: #d0d4e4;")
                layout.addWidget(lbl)

        # Footer con acciones
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 4, 0, 0)
        footer.setSpacing(6)

        btn_copy = QToolButton()
        btn_copy.setText("📋 Copiar")
        btn_copy.setCursor(Qt.PointingHandCursor)
        btn_copy.setStyleSheet("""
            QToolButton {
                background: transparent;
                color: #a78bfa;
                border: none;
                font-size: 11px;
                padding: 2px 6px;
            }
            QToolButton:hover {
                background-color: rgba(127, 90, 240, 0.12);
                color: #ffffff;
            }
        """)
        btn_copy.clicked.connect(self._on_copy)
        footer.addWidget(btn_copy)

        btn_steps = QToolButton()
        btn_steps.setText("🔍 Ver detalles")
        btn_steps.setCursor(Qt.PointingHandCursor)
        btn_steps.setEnabled(self._steps is not None or self._details_callback is not None)
        btn_steps.setStyleSheet("""
            QToolButton {
                background: transparent;
                color: #a78bfa;
                border: none;
                font-size: 11px;
                padding: 2px 6px;
            }
            QToolButton:disabled {
                color: #555a70;
            }
            QToolButton:hover:!disabled {
                background-color: rgba(127, 90, 240, 0.12);
                color: #ffffff;
            }
        """)
        if self._details_callback is not None:
            btn_steps.clicked.connect(self._details_callback)
        elif self._steps is not None:
            btn_steps.clicked.connect(self._on_steps)
        footer.addWidget(btn_steps)

        footer.addStretch(1)
        layout.addLayout(footer)

        self._run_appear_animation()

    # Slots
    def _on_close(self):
        self._run_disappear_animation()

    def _on_copy(self):
        if self._copy_text is not None:
            text = self._copy_text
        else:
            text = self._description if self._matrix is None else fmt_matrix(np.array(self._matrix, dtype=float), 2)
        self._main.copy_to_clipboard(text)

    def _on_steps(self):
        if self._steps is not None:
            StepsDialog(self._steps, self._main).exec()

    def _run_appear_animation(self):
        try:
            self.setWindowOpacity(0.0)
            anim = QPropertyAnimation(self, b"windowOpacity", self)
            anim.setDuration(220)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._anim_in = anim
            anim.start(QPropertyAnimation.DeleteWhenStopped)
        except Exception:
            pass

    def _run_disappear_animation(self):
        try:
            anim = QPropertyAnimation(self, b"windowOpacity", self)
            anim.setDuration(180)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.InCubic)

            def _on_finished():
                self.setParent(None)

            anim.finished.connect(_on_finished)
            self._anim_out = anim
            anim.start(QPropertyAnimation.DeleteWhenStopped)
        except Exception:
            # fallback sin animación
            self.setParent(None)


class TrimDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox que evita notación científica y ceros de relleno al mostrar."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLocale(QLocale.c())  # fuerza separador decimal '.'

    def textFromValue(self, value: float) -> str:  # type: ignore[override]
        # Representación con número de decimales configurado y sin ceros innecesarios
        try:
            s = f"{value:.{self.decimals()}f}"
            if '.' in s:
                s = s.rstrip('0').rstrip('.')
            # Si es -0, mostrar 0
            if s in ('-0', '-0.0'):
                s = '0'
            return s
        except Exception:
            return super().textFromValue(value)

    def validate(self, text: str, pos: int):  # type: ignore[override]
        if ',' in text and '.' not in text:
            text = text.replace(',', '.')
        return super().validate(text, pos)

    def valueFromText(self, text: str) -> float:  # type: ignore[override]
        return super().valueFromText(text.replace(',', '.'))


class MatrixTable(QTableWidget):
    """Tabla simple para edición de matrices con utilidades de tamaño, aleatorio y extracción."""
    def __init__(self, rows: int, cols: int, parent=None):
        super().__init__(rows, cols, parent)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._ensure_items()

    def _ensure_items(self):
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                if not self.item(i, j):
                    self.setItem(i, j, QTableWidgetItem('0'))

    def set_size(self, rows: int, cols: int):
        self.setRowCount(rows)
        self.setColumnCount(cols)
        self._ensure_items()
        self.resizeColumnsToContents()

    def set_headers(self, row_headers=None, col_headers=None):
        if col_headers is not None:
            self.setColumnCount(len(col_headers))
            self.setHorizontalHeaderLabels([str(x) for x in col_headers])
        if row_headers is not None:
            self.setRowCount(len(row_headers))
            self.setVerticalHeaderLabels([str(x) for x in row_headers])
        self._ensure_items()

    def get_matrix(self) -> np.ndarray:
        r, c = self.rowCount(), self.columnCount()
        arr = np.zeros((r, c), dtype=float)
        for i in range(r):
            for j in range(c):
                it = self.item(i, j)
                txt = it.text().strip() if it else '0'
                try:
                    arr[i, j] = float(txt.replace(',', '.'))
                except Exception:
                    arr[i, j] = 0.0
        return arr

    def fill_random(self, low: int = -5, high: int = 6):
        r, c = self.rowCount(), self.columnCount()
        vals = np.random.randint(low, high, size=(r, c))
        # Evitar que toda la matriz sea cero
        if not np.any(vals):
            vals[0, 0] = 1
        for i in range(r):
            for j in range(c):
                self.setItem(i, j, QTableWidgetItem(str(int(vals[i, j]))))
        self.resizeColumnsToContents()


def set_table_preview(tbl: QTableWidget, arr: np.ndarray, decimals: int = 2):
    """Pinta en una QTableWidget la matriz arr solo para vista previa."""
    arr = np.array(arr, dtype=float)
    r, c = arr.shape
    tbl.setRowCount(r)
    tbl.setColumnCount(c)
    for i in range(r):
        for j in range(c):
            v = float(arr[i, j])
            if float(v).is_integer():
                s = str(int(v))
            else:
                s = f"{v:.{decimals}f}"
            it = QTableWidgetItem(s)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tbl.setItem(i, j, it)
    tbl.resizeColumnsToContents()

class StepsDialog(QDialog):
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Paso a paso')
        self.resize(940, 620)
        self.steps = steps

        root = QHBoxLayout(self)
        left = QVBoxLayout(); right = QVBoxLayout()
        root.addLayout(left, 1); root.addLayout(right, 2)

        self.listbox = QListWidget(); left.addWidget(self.listbox)
        for i, (desc, _) in enumerate(steps):
            QListWidgetItem(f"{i+1}. {desc}", self.listbox)

        header = QHBoxLayout(); right.addLayout(header)
        self.step_title = QLabel(''); self.step_title.setStyleSheet('font-weight:600; font-size:14px;')
        header.addWidget(self.step_title, 1)
        header.addWidget(QLabel('Decimales:'))
        self.decimals = QSpinBox(); self.decimals.setRange(0,8); self.decimals.setValue(2); header.addWidget(self.decimals)
        self.only_changes = QCheckBox('Solo cambios'); header.addWidget(self.only_changes)
        self.manual_mode = QCheckBox('Modo manual'); header.addWidget(self.manual_mode)
        self.copy_btn = QPushButton('📋 Copiar matriz'); header.addWidget(self.copy_btn)

        self.preview = QTableWidget(); self.preview.setAlternatingRowColors(True)
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview.setSelectionMode(QAbstractItemView.NoSelection)
        self.preview.setFocusPolicy(Qt.NoFocus)
        self.preview.setStyleSheet('QTableWidget::item:selected{background:transparent; color:inherit;}')
        right.addWidget(self.preview, 1)

        self.explain = QLabel(''); self.explain.setWordWrap(True)
        self.explain.setStyleSheet(f"font-family:{MATH_FONT_STACK}; color:#cfd8dc;")
        right.addWidget(self.explain)

        self.stats = QLabel(''); self.stats.setStyleSheet('color:#888;'); right.addWidget(self.stats)

        nav = QHBoxLayout(); right.addLayout(nav)
        self.prev_btn = QPushButton('◀ Anterior'); self.next_btn = QPushButton('Siguiente ▶')
        nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn); nav.addStretch(1)

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._move(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._move(+1))

        self.listbox.currentRowChanged.connect(self._on_select)
        self.prev_btn.clicked.connect(lambda: self._move(-1))
        self.next_btn.clicked.connect(lambda: self._move(+1))
        self.decimals.valueChanged.connect(lambda _=None: self._on_select(self.listbox.currentRow()))
        self.only_changes.toggled.connect(lambda _=None: self._on_select(self.listbox.currentRow()))
        self.manual_mode.toggled.connect(lambda _=None: self._on_select(self.listbox.currentRow()))
        self.copy_btn.clicked.connect(self._copy_current)

        if steps:
            self.listbox.setCurrentRow(0)
        self.setStyleSheet('QListWidget::item{padding:6px;} QTableWidget{gridline-color:#444;}')

    def _move(self, delta: int):
        i = max(0, min(self.listbox.count()-1, self.listbox.currentRow()+delta))
        self.listbox.setCurrentRow(i)
        if 0 <= i < self.listbox.count():
            self.step_title.setText(self.listbox.item(i).text())
        self._on_select(i)

    def _on_select(self, row: int):
        if row < 0 or row >= len(self.steps): return
        desc, mat = self.steps[row]; self.step_title.setText(desc)
        arr = np.array(mat.tolist(), dtype=float)
        prev = np.array(self.steps[row-1][1].tolist(), dtype=float) if row>0 else None
        hl_rows, hl_cols, pretty = self._parse_step_description(desc)
        self._render_matrix(arr, prev, hl_rows=hl_rows, hl_cols=hl_cols)
        self.explain.setText(pretty if self.manual_mode.isChecked() else '')
        d = self.decimals.value(); changed = 0
        if prev is not None:
            changed = int(np.sum(np.round(arr,d) != np.round(prev,d)))
        self.stats.setText(f"Tamaño: {arr.shape[0]} × {arr.shape[1]}  •  Celdas cambiadas: {changed}")

    def _render_matrix(self, arr: np.ndarray, prev: np.ndarray | None, hl_rows: set[int] | None = None, hl_cols: set[int] | None = None):
        d = self.decimals.value(); r,c = arr.shape
        self.preview.setRowCount(r); self.preview.setColumnCount(c)
        for i in range(r):
            for j in range(c):
                v = float(arr[i,j]); s = f"{v:.{d}f}" if (not float(v).is_integer() or d>0) else str(int(v))
                item = QTableWidgetItem(s); item.setFlags(Qt.ItemIsEnabled); item.setTextAlignment(Qt.AlignCenter)
                if prev is not None:
                    pv = float(prev[i,j])
                    if round(v,d) != round(pv,d):
                        item.setBackground(QColor('#0099a8')); f = item.font(); f.setBold(True); item.setFont(f); item.setForeground(QColor('white'))
                    elif self.only_changes.isChecked():
                        item.setForeground(QColor('#777777'))
                if hl_rows and i in hl_rows and item.background().color().alpha() == 0:
                    item.setBackground(QColor(0,153,168,40))
                if hl_cols and j in hl_cols and item.background().color().alpha() == 0:
                    item.setBackground(QColor(0,153,168,30))
                self.preview.setItem(i,j,item)
        self.preview.clearSelection()

    def _parse_step_description(self, desc: str):
        import re
        hl_rows: set[int] = set(); hl_cols: set[int] = set(); pretty = desc; d = desc.strip()
        m = re.search(r"Intercambiar\s+fila\s+(\d+)\s+con\s+fila\s+(\d+)", d, re.IGNORECASE)
        if m:
            i,j = int(m.group(1))-1, int(m.group(2))-1; hl_rows.update({i,j}); pretty = f"Operación por filas: R{m.group(1)} ↔ R{m.group(2)}"; return hl_rows, hl_cols, pretty
        m = re.search(r"Dividir\s+fila\s+(\d+)\s+por\s+([\-\d\./]+)", d, re.IGNORECASE)
        if m:
            i = int(m.group(1))-1; x = m.group(2); hl_rows.add(i); pretty = f"Escalado: R{m.group(1)} ← R{m.group(1)}/{x}"; return hl_rows, hl_cols, pretty
        m = re.search(r"R\s*(\d+)\s*<-\s*R\s*\1\s*([+\-])\s*\(?([\-\d\./]+)\)?\s*\*?\s*R\s*(\d+)", d)
        if m:
            k, sign, coef, j = m.groups(); k_i = int(k)-1; j_i = int(j)-1; hl_rows.update({k_i,j_i}); op = '+' if sign == '+' else '−'; pretty = f"Operación elemental: R{k} ← R{k} {op} ({coef})·R{j}"; return hl_rows, hl_cols, pretty
        if d.lower().startswith('calcular'):
            pretty = d
        return hl_rows, hl_cols, pretty

    def _copy_current(self):
        row = self.listbox.currentRow()
        if 0 <= row < len(self.steps):
            _, mat = self.steps[row]; arr = np.array(mat.tolist(), dtype=float)
            self.parent().copy_to_clipboard(fmt_matrix(arr, self.decimals.value()))

class BisectionResultDialog(QDialog):
    def __init__(self, expr_text: str, xi: float, xu: float, rows, epsilon_text: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Método de Bisección — Detalles')
        self.resize(960, 640)

        lay = QVBoxLayout(self)
        title = QLabel('<b>Método de bisección</b>'); lay.addWidget(title)
        # Encabezado con métricas solicitadas
        header = QWidget(); grid = QGridLayout(header)
        grid.setContentsMargins(0,0,0,0); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(4)
        grid.addWidget(QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_text}</span>"), 0, 0, 1, 4)
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            ea = float(rows[-1][4])
            residual = abs(float(rows[-1][7]))
            sgn = '+' if xr >= 0 else ''
            grid.addWidget(QLabel(f"Iteraciones: <b>{n}</b>"), 1, 0)
            grid.addWidget(QLabel(f"Raíz: <b>{sgn}{xr:.6f}</b>"), 1, 1)
            grid.addWidget(QLabel(f"Error (Ea): <b>{ea*100:.2f}%</b>"), 1, 2)
            grid.addWidget(QLabel(f"Error raíz |f(r)|: <b>{residual:.6g}</b>"), 1, 3)
            if epsilon_text is not None and epsilon_text != '':
                grid.addWidget(QLabel(f"Tolerancia: <b>{epsilon_text}</b>"), 2, 0)
        # estilizar header
        for i in range(grid.count()):
            w = grid.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setStyleSheet("color:#ddd;")
        lay.addWidget(header)

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
            x = _SYM_symbols('x'); expr = _SYM_sympify(expr_text); fcall = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
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
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            lay.addWidget(canvas)
        except Exception:
            pass


class MatrixQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Matrix (Qt)')
        # Ventana principal moderna sin marco del sistema
        try:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self.setAttribute(Qt.WA_TranslucentBackground)
        except Exception:
            pass
        # Establecer icono de ventana (también lo aplicamos a nivel de QApplication en run())
        try:
            self.setWindowIcon(QIcon(_resource_path('logo.png')))
        except Exception:
            pass
        self.resize(1280, 800)
        self.setObjectName('rootWindow')

        # Contenedor raíz con borde/sombra sutil y barra de título personalizada
        outer = QWidget(); outer.setObjectName('outerContainer'); self.setCentralWidget(outer)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(0)

        # Marco principal que simula el marco de ventana
        frame = QFrame(); frame.setObjectName('windowFrame')
        frame.setStyleSheet(
            "#windowFrame{background-color:#1e1e2e; border:1px solid #333333; "
            "border-radius:8px;}"
        )
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        outer_layout.addWidget(frame)

        # Barra de título personalizada
        self.title_bar = TitleBar(self)
        frame_layout.addWidget(self.title_bar)

        # Contenido principal bajo la barra de título
        central = QWidget(); central.setObjectName('centralWidget')
        frame_layout.addWidget(central, 1)

        # Root layout interno para la app
        layout = QHBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(18)

        # Sidebar enmarcado en un QFrame para integrarlo visualmente
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName('sidebarFrame')
        self.sidebar_frame.setMinimumWidth(210)
        sidebar_widget = QWidget(self.sidebar_frame)
        sidebar_widget.setObjectName('sidebarContainer')
        sidebar = QVBoxLayout(sidebar_widget)
        sidebar.setContentsMargins(0, 0, 16, 0)
        sidebar.setSpacing(10)
        frame_layout = QVBoxLayout(self.sidebar_frame)
        frame_layout.setContentsMargins(14, 14, 10, 14)
        frame_layout.setSpacing(10)
        frame_layout.addWidget(sidebar_widget)
        layout.addWidget(self.sidebar_frame, 0)

        # Cabecera del menú lateral
        title_label = QLabel('MATRIX')
        title_label.setObjectName('sidebarTitle')
        sidebar.addWidget(title_label)

        # Botón para volver a la pantalla de bienvenida
        btn_back_home = QPushButton(self.tr('Volver al inicio'))
        btn_back_home.setCursor(Qt.PointingHandCursor)
        btn_back_home.setFixedHeight(28)
        btn_back_home.setStyleSheet(
            "QPushButton {"
            "  background-color: transparent;"
            "  color: #a78bfa;"
            "  border: 1px solid rgba(167,139,250,0.4);"
            "  border-radius: 14px;"
            "  font-size: 11px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(127,90,240,0.16);"
            "  color: #ffffff;"
            "}"
        )
        btn_back_home.clicked.connect(self._return_to_welcome)
        sidebar.addWidget(btn_back_home)
        # Grupo exclusivo para que siempre haya una sección activa
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        def make_nav_button(text: str) -> QPushButton:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty('navButton', True)
            btn.setCursor(Qt.PointingHandCursor)
            self._nav_group.addButton(btn)
            return btn

        self.btn_ops = make_nav_button('🧮 Operaciones'); sidebar.addWidget(self.btn_ops)
        self.btn_ind = make_nav_button('🧭 Independencia'); sidebar.addWidget(self.btn_ind)
        self.btn_triu = make_nav_button('🔺 Triangular U'); sidebar.addWidget(self.btn_triu)
        self.btn_rref = make_nav_button('🧱 RREF'); sidebar.addWidget(self.btn_rref)
        self.btn_ti = make_nav_button('🔁 Transpuesta/Inversa'); sidebar.addWidget(self.btn_ti)
        self.btn_det = make_nav_button('🧾 Determinante'); sidebar.addWidget(self.btn_det)
        self.btn_cramer = make_nav_button('📐 Método de Cramer'); sidebar.addWidget(self.btn_cramer)
        self.btn_bis = make_nav_button('📉 Bisección'); sidebar.addWidget(self.btn_bis)
        self.btn_fp = make_nav_button('📈 Falsa posición'); sidebar.addWidget(self.btn_fp)
        self.btn_secant = make_nav_button('📐 Secante'); sidebar.addWidget(self.btn_secant)
        self.btn_newton = make_nav_button('⚡ Newton-Raphson'); sidebar.addWidget(self.btn_newton)
        sidebar.addStretch(1)

        # Center and Right using splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        self.center = QWidget(); splitter.addWidget(self.center)
        cgrid = QVBoxLayout(self.center)
        cgrid.setContentsMargins(4, 0, 12, 0)
        cgrid.setSpacing(16)
        self.center_title = QLabel(''); self.center_title.setObjectName('pageTitle'); cgrid.addWidget(self.center_title)
        self.center_scroll = QScrollArea(); self.center_scroll.setWidgetResizable(True)
        self.center_scroll.setObjectName('centerScroll')
        self.center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.center_body = QWidget()
        self.center_layout = QVBoxLayout(self.center_body)
        self.center_layout.setContentsMargins(20, 16, 20, 20)
        self.center_layout.setSpacing(20)
        self.center_scroll.setWidget(self.center_body)
        cgrid.addWidget(self.center_scroll, 1)

        # Right results panel in scroll area
        self.right_scroll = QScrollArea(); self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_container = QWidget(); self.right_container.setObjectName('rightPanel')
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(8, 8, 8, 8)
        self.right_layout.setSpacing(14)
        self.right_scroll.setWidget(self.right_container)
        splitter.addWidget(self.right_scroll)
        splitter.setSizes([850, 430])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # connect
        self.btn_ops.clicked.connect(self.show_ops)
        self.btn_ind.clicked.connect(self.show_ind)
        self.btn_triu.clicked.connect(self.show_triu)
        self.btn_rref.clicked.connect(self.show_rref)
        self.btn_ti.clicked.connect(self.show_ti)
        self.btn_det.clicked.connect(self.show_det)
        self.btn_cramer.clicked.connect(self.show_cramer)
        self.btn_bis.clicked.connect(self.show_bisection)
        self.btn_fp.clicked.connect(self.show_false_position)
        self.btn_secant.clicked.connect(self.show_secant)
        self.btn_newton.clicked.connect(self.show_newton)

        # state
        self.current_view = None
        self._result_widgets = []
        self._dialogs = []
        self.show_ops()
        # Marcar como activo el botón inicial para que siempre haya uno seleccionado
        self.btn_ops.setChecked(True)
        # fonts and theme
        self._init_fonts()
        self.apply_theme()

    def _return_to_welcome(self):
        # Volver a la pantalla de bienvenida sin cerrar la aplicación completa
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, WelcomeScreen):
                widget.show()
                break
        self.hide()

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
        """Aplicar el design system oscuro tipo "dark glass/cyberpunk" (fase 1).

        Aquí definimos solo la base global: colores, tipografía, barras de
        scroll y tooltips. Los widgets específicos (botones, tablas, etc.) se
        refinarán en fases posteriores.
        """
        app = QApplication.instance()
        if app:
            try:
                app.setStyle('Fusion')  # base neutra para QSS personalizado
            except Exception:
                pass

            # Tipografía global: intentamos cargar una fuente moderna si existe,
            # y si no, caemos a Segoe UI/Helvetica.
            base_font = QFont('Segoe UI', 10)
            try:
                # Si el usuario agrega una carpeta "fonts" con .ttf/.otf
                fonts_dir = os.path.join(os.path.dirname(_resource_path('logo.png')), 'fonts')
                for family in os.listdir(fonts_dir) if os.path.exists(fonts_dir) else []:
                    if family.lower().endswith(('.ttf', '.otf')):
                        QFontDatabase.addApplicationFont(os.path.join(fonts_dir, family))
                # Si hubiera cargado "Inter" o "Roboto", se podría usar aquí.
            except Exception:
                pass
            app.setFont(base_font)

        # Matplotlib coherente con el fondo oscuro
        try:
            mpl.rcParams['figure.facecolor'] = BG_PANEL
            mpl.rcParams['axes.facecolor'] = BG_PANEL
            mpl.rcParams['axes.edgecolor'] = TEXT_MUTED
            mpl.rcParams['axes.labelcolor'] = TEXT_PRIMARY
            mpl.rcParams['xtick.color'] = TEXT_MUTED
            mpl.rcParams['ytick.color'] = TEXT_MUTED
            mpl.rcParams['text.color'] = TEXT_PRIMARY
            mpl.rcParams['grid.color'] = '#44475a'
        except Exception:
            pass

        # QSS global: base visual + estilos principales de dashboard
        self.setStyleSheet(f"""
            QMainWindow#rootWindow, QWidget#centralWidget {{
                background-color: {BG_MAIN};
                color: {TEXT_PRIMARY};
            }}

            QWidget {{
                background-color: transparent;
                color: {TEXT_PRIMARY};
            }}

            QLabel {{
                color: {TEXT_PRIMARY};
            }}

            QLabel#pageTitle {{
                font-size: 20px;
                font-weight: 600;
                padding: 4px 4px 10px 4px;
                color: {TEXT_PRIMARY};
            }}

            QWidget#rightPanel {{
                background-color: transparent;
                border-radius: 14px;
            }}

            QScrollArea#centerScroll, QScrollArea {{
                border: none;
                background: transparent;
            }}

            QFrame#resultCard {{
                background-color: #252535;
                border-radius: 12px;
                border: 1px solid rgba(148,161,178,0.35);
                padding: 12px 14px;
            }}

            /* Checkboxes y radio buttons sobre fondo oscuro */
            QCheckBox, QRadioButton {{
                color: #e0e0e0;
                spacing: 8px;
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid #5b5b70;
                border-radius: 4px;
                background: #2a2a3e;
            }}
            QCheckBox::indicator:checked {{
                background-color: #7f5af0;
                border-color: #7f5af0;
                image: none;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid #5b5b70;
                border-radius: 10px;
                background: #2a2a3e;
            }}
            QRadioButton::indicator:checked {{
                background-color: #7f5af0;
                border-color: #7f5af0;
            }}

            QWidget#errorCard {{
                background-color: rgba(120, 40, 40, 0.95);
                border-radius: 12px;
                border: 1px solid rgba(255, 120, 120, 0.65);
                padding: 16px;
            }}

            QFrame#methodCard {{
                background-color: {BG_PANEL};
                border-radius: 12px;
                border: 1px solid rgba(148,161,178,0.35);
                padding: 20px;
            }}

            QLabel#cardTitle {{
                font-size: 15px;
                font-weight: 600;
                color: {TEXT_PRIMARY};
            }}

            QLabel#sectionTitle {{
                font-size: 18px;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 16px;
            }}

            QLabel#helperLabel {{
                color: {TEXT_MUTED};
                font-size: 12px;
            }}

            /* Checkboxes y radio buttons globales */
            QCheckBox, QRadioButton {{
                color: #e0e0e0;
                spacing: 8px;
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid #5b5b70;
                border-radius: 4px;
                background: #2a2a3e;
            }}
            QCheckBox::indicator:checked {{
                background-color: #7f5af0;
                border-color: #7f5af0;
                image: none;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid #5b5b70;
                border-radius: 10px;
                background: #2a2a3e;
            }}
            QRadioButton::indicator:checked {{
                background-color: #7f5af0;
                border-color: #7f5af0;
            }}

            /* Tablas tipo dashboard */
            QTableWidget {{
                background-color: #2a2a3e;
                alternate-background-color: #24243a;
                gridline-color: #2a2a3e;
                border: 1px solid #2a2a3e;
                border-radius: 8px;
                selection-background-color: {ACCENT_PRIMARY};
                selection-color: {TEXT_PRIMARY};
            }}
            QTableWidget::item {{
                padding: 8px 8px;
                color: {TEXT_PRIMARY};
                font-family: {MATH_FONT_STACK};
            }}
            QHeaderView::section {{
                background-color: {BG_MAIN};
                color: {TEXT_MUTED};
                padding: 8px 8px;
                border: none;
                border-right: 1px solid #333344;
                font-weight: 600;
            }}

            QTableCornerButton::section {{
                background-color: {BG_MAIN};
                border: none;
                border-right: 1px solid #333344;
            }}

            /* Sidebar frame y título */
            QFrame#sidebarFrame {{
                background-color: {BG_PANEL};
                border-right: 1px solid #333444;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
                border-top-right-radius: 18px;
                border-bottom-right-radius: 18px;
            }}

            QLabel#sidebarTitle {{
                font-size: 22px;
                font-weight: 800;
                letter-spacing: 2px;
                color: {TEXT_PRIMARY};
                margin-bottom: 4px;
            }}

            /* Botones de navegación planos */
            QPushButton[navButton="true"] {{
                background-color: transparent;
                border: none;
                text-align: left;
                padding: 10px 18px;
                border-radius: 8px;
                font-size: 14px;
                color: {TEXT_MUTED};
            }}
            QPushButton[navButton="true"]:hover {{
                background-color: #32324a;
                color: {TEXT_PRIMARY};
            }}
            QPushButton[navButton="true"]:checked {{
                background-color: {ACCENT_PRIMARY};
                color: {TEXT_PRIMARY};
                font-weight: 600;
            }}

            /* Barras de scroll estilo móvil */
            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 4px 2px 4px 2px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(148,161,178,0.7);
                border-radius: 5px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                background: none;
                height: 0px;
            }}

            QScrollBar:horizontal {{
                background: transparent;
                height: 10px;
                margin: 2px 4px 2px 4px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: rgba(148,161,178,0.7);
                border-radius: 5px;
                min-width: 24px;
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                background: none;
                width: 0px;
            }}

            /* Tooltips oscuros con borde acentuado */
            QToolTip {{
                background-color: {BG_PANEL};
                color: {TEXT_PRIMARY};
                border: 1px solid {ACCENT_PRIMARY};
                padding: 6px 8px;
                border-radius: 6px;
            }}

            /* Campos de entrada */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: #2a2a3e;
                border: 1px solid #444455;
                border-radius: 6px;
                padding: 6px 10px;
                color: {TEXT_PRIMARY};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {ACCENT_PRIMARY};
            }}

            /* Botones de incremento/decremento de SpinBox */
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background-color: #32324a;
                border-left: 1px solid #444455;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 0px;
                padding: 0px;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background-color: #32324a;
                border-left: 1px solid #444455;
                border-bottom-right-radius: 6px;
                border-top-right-radius: 0px;
                padding: 0px;
            }}

            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: #3e3e55;
            }}

            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background-color: {ACCENT_PRIMARY};
            }}

            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                width: 0;
                height: 0;
                margin: 0px 4px 0px 4px;
            }}
            /* Triángulos claros con borders CSS */
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 6px solid #94a1b2;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #94a1b2;
            }}

            QComboBox {{
                min-height: 36px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 26px;
                border: none;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
                margin-right: 4px;
                image: none;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow:!editable {{
                border-left: none;
            }}
            QComboBox::down-arrow {{
                border: none;
            }}
            QComboBox::drop-down:pressed {{
                background: transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {BG_PANEL};
                border: 1px solid #444455;
                selection-background-color: {ACCENT_PRIMARY};
                selection-color: {TEXT_PRIMARY};
            }}

            /* Botones de acción (excluye los del sidebar por propiedad navButton) */
            QPushButton:not([navButton="true"]) {{
                font-size: 14px;
                border-radius: 8px;
                padding: 8px 18px;
                min-height: 40px;
            }}

            /* Botones dentro de tarjetas de resultado y diálogos secundarios */
            QFrame#resultCard QPushButton,
            QWidget#resultCard QPushButton {{
                background-color: transparent;
                border: 1px solid #7f5af0;
                color: #a78bfa;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QFrame#resultCard QPushButton:hover,
            QWidget#resultCard QPushButton:hover {{
                background-color: rgba(127, 90, 240, 0.12);
                color: #ffffff;
            }}

            /* Primario: usado para acciones "Calcular" y similares */
            QPushButton#btnPrimary {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                 stop:0 #8b5cf6, stop:1 #7c3aed);
                color: {TEXT_PRIMARY};
                font-weight: 700;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                min-height: 40px;
                box-shadow: 0 4px 12px rgba(127, 90, 240, 0.3);
            }}
            QPushButton#btnPrimary:hover {{
                background-color: #9370db;
                box-shadow: 0 6px 16px rgba(127, 90, 240, 0.4);
            }}
            QPushButton#btnPrimary:pressed {{
                background-color: #6a4fc9;
                box-shadow: 0 2px 8px rgba(127, 90, 240, 0.25);
            }}

            /* Secundario: outline para "Aplicar tamaño", "Aleatoria", etc. */
            QPushButton#btnSecondary {{
                background-color: transparent;
                color: {ACCENT_PRIMARY};
                border: 2px solid {ACCENT_PRIMARY};
                border-radius: 8px;
                font-weight: 600;
                padding: 8px 16px;
                min-height: 36px;
            }}
            QPushButton#btnSecondary:hover {{
                background-color: {ACCENT_PRIMARY};
                color: {TEXT_PRIMARY};
            }}
            QPushButton#btnSecondary:pressed {{
                background-color: #6a4fc9;
                color: {TEXT_PRIMARY};
            }}
        """)

    # helpers
    def clear_center(self):
        while self.center_layout.count():
            item = self.center_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def clear_results(self):
        for i in reversed(range(self.right_layout.count())):
            w = self.right_layout.itemAt(i).widget()
            if w: w.setParent(None)
        self._result_widgets.clear()

    def copy_to_clipboard(self, text: str):
        QApplication.clipboard().setText(text, QClipboard.Clipboard)

    def push_result(self, title: str, matrix: np.ndarray | None, description: str = '', steps=None, accent: str | None = None):
        """API clásica para resultados matriciales (usa matriz + descripción)."""
        card = ResultCard(title, self, matrix=matrix, description=description, steps=steps)
        self.right_layout.insertWidget(0, card)
        self._result_widgets.append(card)
        return card

    def add_result_card(self, title: str, content_widget: QWidget, steps=None, copy_text: str | None = None, details_callback=None):
        """Crea una ResultCard usando un widget de contenido ya construido.

        Útil para métodos numéricos que generan resúmenes más ricos (gráficas,
        métricas, etc.) sin duplicar botones de acción.
        """
        card = ResultCard(title, self, content_widget=content_widget, steps=steps, copy_text=copy_text, details_callback=details_callback)
        self.right_layout.insertWidget(0, card)
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

    def _build_method_card(self, title: str, subtitle: str | None = None):
        card = QFrame(); card.setObjectName('methodCard')
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        # Sombra suave tipo tarjeta flotante
        try:
            shadow = QGraphicsDropShadowEffect(card)
            shadow.setBlurRadius(18)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(0, 0, 0, 90))
            card.setGraphicsEffect(shadow)
        except Exception:
            pass
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)
        header = QLabel(f"<b>{title}</b>"); header.setObjectName('cardTitle')
        layout.addWidget(header)
        if subtitle:
            sub = QLabel(subtitle); sub.setObjectName('helperLabel'); sub.setWordWrap(True)
            layout.addWidget(sub)
        return card, layout

    def _build_math_keyboard(self, target_edit: QLineEdit) -> QWidget:
        panel = QWidget(); panel.setObjectName('mathKeyboard')
        grid = QGridLayout(panel)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        def add_key(text: str, insert: str, row: int, col: int):
            btn = QToolButton(); btn.setText(text)
            btn.setMinimumSize(36, 32)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setToolTip(insert)
            btn.clicked.connect(lambda _=None, payload=insert: self._insert_text(target_edit, payload))
            # Estilo compacto tipo teclado para funciones matemáticas
            btn.setStyleSheet("""
                QToolButton {
                    background-color: #2a2a3e;
                    border: 1px solid #444;
                    border-radius: 6px;
                    color: #ddd;
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    padding: 2px 6px;
                }
                QToolButton:hover {
                    background-color: #7f5af0;
                    color: white;
                    border-color: #7f5af0;
                }
                QToolButton:pressed {
                    background-color: #6a4fc9;
                }
            """)
            grid.addWidget(btn, row, col)

        rows = [
            [('x', 'x'), ('x^2', '**2'), ('x^3', '**3'), ('^', '**'), ('(', '('), (')', ')'), ('|x|', 'Abs()'), ('√', 'sqrt()')],
            [('sen', 'sin()'), ('cos', 'cos()'), ('tg', 'tan()'), ('sec', 'sec()'), ('csc', 'csc()'), ('cot', 'cot()'), ('ln', 'ln()'), ('log', 'log()')],
            [('exp', 'exp()'), ('π', 'pi'), ('e', 'E'), ('sinh', 'sinh()'), ('cosh', 'cosh()'), ('tanh', 'tanh()'), ('sech', 'sech()'), ('csch', 'csch()')],
        ]
        for row_idx, row in enumerate(rows):
            for col_idx, (text, insert) in enumerate(row):
                add_key(text, insert, row_idx, col_idx)
        return panel

    def _insert_text(self, target: QLineEdit, payload: str):
        cursor_pos = target.cursorPosition()
        target.insert(payload)
        if payload.endswith('()'):
            target.setCursorPosition(cursor_pos + len(payload) - 1)

    def _build_format_hint_label(self, extra: str | None = None) -> QLabel:
        base = ('Formato: usa ** para potencias y * para multiplicar. '
                'Funciones: sin, cos, tan, sec, csc, cot, exp, log, Abs, π, E.')
        if extra:
            base = f"{base} {extra}"
        lbl = QLabel(base); lbl.setObjectName('helperLabel'); lbl.setWordWrap(True)
        return lbl

    def _build_equation_examples_label(self) -> QLabel:
        lbl = QLabel("Ej.: x**3 - 4*x + sec(x/2) mezcla polinomio y trigonometría.")
        lbl.setObjectName('helperLabel'); lbl.setWordWrap(True)
        return lbl

    def _style_button(self, btn, tipo: str = 'secondary'):
        """Aplica estilos modernos DIRECTAMENTE al botón para asegurar que se vean bien"""
        if tipo == 'primary':
            # Estilo para botón CALCULAR (Violeta sólido)
            btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #8b5cf6, stop:1 #7c3aed);
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
                border: none;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #9370db;
            }
            QPushButton:pressed {
                background-color: #6a4fc9;
                padding-top: 12px; /* Efecto de hundirse */
            }
        """)
        else:
            # Estilo para botones SECUNDARIOS (Outline violeta)
            btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #a78bfa;
                font-weight: 600;
                border-radius: 8px;
                padding: 8px 16px;
                border: 2px solid #7c3aed;
            }
            QPushButton:hover {
                background-color: #7c3aed;
                color: white;
            }
            QPushButton:pressed {
                background-color: #5b21b6;
                border-color: #5b21b6;
            }
        """)

        # Añadir sombra suave
        shadow = QGraphicsDropShadowEffect(btn)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(124, 58, 237, 80))
        shadow.setOffset(0, 4)
        btn.setGraphicsEffect(shadow)

        # Cursor de mano
        btn.setCursor(Qt.PointingHandCursor)

    def _style_spinbox_arrows(self, *spinboxes):
        """Apply a localized stylesheet to spinboxes to ensure clear triangular arrows."""
        css = '''
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background-color: #32324a;
                border-left: 1px solid #444455;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 0px;
                padding: 0px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background-color: #32324a;
                border-left: 1px solid #444455;
                border-bottom-right-radius: 6px;
                border-top-right-radius: 0px;
                padding: 0px;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                width: 0;
                height: 0;
                margin: 0px 4px 0px 4px;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 6px solid #94a1b2;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #94a1b2;
            }
        '''
        for sp in spinboxes:
            try:
                if sp is None:
                    continue
                sp.setStyleSheet(css)
            except Exception:
                pass

    def _build_geogebra_url(self, expr_text: str, x0: float | None = None, root: float | None = None) -> str:
        expr_clean = expr_text.replace('^', '**')
        commands = [f"f(x) = {expr_clean}"]
        def _finite(val: float | None) -> bool:
            return val is not None and np.isfinite(val)
        if _finite(x0):
            commands.append(f"x0 = {x0}")
            commands.append("P = (x0, f(x0))")
            commands.append("Q = (x0, 0)")
            commands.append("L = Line(P, Q)")
            commands.append("SetLineStyle(L, 2)")
            commands.append("SetPointStyle(P, 2)")
            commands.append("SetColor(P, 0.85, 0.33, 0.31)")
        if _finite(root):
            commands.append(f"root = {root}")
            commands.append("R = (root, 0)")
            commands.append("SetPointStyle(R, 2)")
            commands.append("SetColor(R, 0.2, 0.76, 0.8)")
        cmd_block = ','.join(f'"{cmd}"' for cmd in commands)
        script = f"Execute[{{{cmd_block}}}]"
        encoded = _url_quote_plus(script)
        return f"https://www.geogebra.org/graphing?command={encoded}&lang=es"

    def _open_geogebra(self, expr_text: str, x0: float | None = None, root: float | None = None):
        url = QUrl(self._build_geogebra_url(expr_text, x0, root))
        QDesktopServices.openUrl(url)

    # views
    def show_ops(self):
        self.current_view = 'ops'
        self.center_title.setText('Operaciones con matrices')
        self.clear_center()
        cont = QWidget(); gl = QGridLayout(cont)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setHorizontalSpacing(20)
        gl.setVerticalSpacing(12)
        self.gridA = MatrixTable(3,3); self.gridB = MatrixTable(3,3)
        # A header and size controls
        headerA = QLabel('Matriz A'); headerA.setObjectName('sectionTitle'); gl.addWidget(headerA, 0, 0)
        sizeA = QHBoxLayout();
        sizeA.setSpacing(16)
        sizeA.setContentsMargins(0, 0, 0, 8)
        sizeA.addWidget(QLabel('Filas:'))
        a_rows = QSpinBox(); a_rows.setRange(1, 20); a_rows.setValue(3); sizeA.addWidget(a_rows)
        sizeA.addWidget(QLabel('Columnas:'))
        a_cols = QSpinBox(); a_cols.setRange(1, 20); a_cols.setValue(3); sizeA.addWidget(a_cols)
        btn_setA = QPushButton('Aplicar tamaño A'); btn_setA.setObjectName('btnSecondary'); sizeA.addWidget(btn_setA)
        # random button for A
        btn_randA = QPushButton('🎲 Aleatoria A'); btn_randA.setObjectName('btnSecondary'); sizeA.addWidget(btn_randA)
        sizeA.addStretch(1)
        gl.addLayout(sizeA, 1, 0)
        gl.addWidget(self.gridA, 2, 0)
        # B header and size controls
        headerB = QLabel('Matriz B'); headerB.setObjectName('sectionTitle'); gl.addWidget(headerB, 3, 0)
        sizeB = QHBoxLayout();
        sizeB.setSpacing(16)
        sizeB.setContentsMargins(0, 0, 0, 8)
        sizeB.addWidget(QLabel('Filas:'))
        b_rows = QSpinBox(); b_rows.setRange(1, 20); b_rows.setValue(3); sizeB.addWidget(b_rows)
        sizeB.addWidget(QLabel('Columnas:'))
        b_cols = QSpinBox(); b_cols.setRange(1, 20); b_cols.setValue(3); sizeB.addWidget(b_cols)
        btn_setB = QPushButton('Aplicar tamaño B'); btn_setB.setObjectName('btnSecondary'); sizeB.addWidget(btn_setB)
        btn_randB = QPushButton('🎲 Aleatoria B'); btn_randB.setObjectName('btnSecondary'); sizeB.addWidget(btn_randB)
        sizeB.addStretch(1)
        gl.addLayout(sizeB, 4, 0)
        gl.addWidget(self.gridB, 5, 0)
        bar = QHBoxLayout()
        self.op_selector = QComboBox(); self.op_selector.addItems(['Suma (A + B)','Resta (A - B)','Producto (A · B)','Combinación (α·A + β·B)'])
        bar.addWidget(self.op_selector)
        calc = QPushButton('⚙️ Calcular'); calc.setObjectName('btnPrimary'); bar.addWidget(calc)
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
        btn_c_set = QPushButton('Aplicar tamaño C'); btn_c_set.setObjectName('btnSecondary'); csize.addWidget(btn_c_set)
        self.gridC = MatrixTable(3,3); comb_lay.addWidget(self.gridC)
        cacts = QHBoxLayout(); comb_lay.addLayout(cacts)
        btn_c_rand = QPushButton('🎲 Aleatoria C'); btn_c_rand.setObjectName('btnSecondary'); cacts.addWidget(btn_c_rand); cacts.addStretch(1)
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
        try:
            self._style_button(btn_setA, 'secondary')
            self._style_button(btn_randA, 'secondary')
            self._style_button(btn_setB, 'secondary')
            self._style_button(btn_randB, 'secondary')
            self._style_button(calc, 'primary')
            self._style_button(btn_c_set, 'secondary')
            self._style_button(btn_c_rand, 'secondary')
            self._style_spinbox_arrows(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, alpha_spin, beta_spin)
        except Exception:
            pass

    def show_ind(self):
        self.current_view = 'ind'
        self.center_title.setText('Independencia de vectores')
        self.clear_center()
        box = QVBoxLayout()
        wrap = QWidget(); wrap.setLayout(box)
        box.addWidget(QLabel('Introduce vectores (cada vector como fila):'))
        size_bar = QHBoxLayout(); box.addLayout(size_bar)
        size_bar.setSpacing(16)
        size_bar.setContentsMargins(0, 0, 0, 8)
        size_bar.addWidget(QLabel('Filas:'))
        vi_rows = QSpinBox(); vi_rows.setRange(1, 20); vi_rows.setValue(3); size_bar.addWidget(vi_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        vi_cols = QSpinBox(); vi_cols.setRange(1, 20); vi_cols.setValue(3); size_bar.addWidget(vi_cols)
        btn_set = QPushButton('Aplicar tamaño'); btn_set.setObjectName('btnSecondary'); size_bar.addWidget(btn_set)
        size_bar.addStretch(1)
        self.vgrid = MatrixTable(3,3)
        box.addWidget(self.vgrid)
        btn = QPushButton('🧭 Evaluar independencia'); btn.setObjectName('btnPrimary'); box.addWidget(btn)
        self.center_layout.addWidget(wrap)

        btn_rand = QPushButton('🎲 Aleatoria'); btn_rand.setObjectName('btnSecondary'); box.addWidget(btn_rand)
        # Estilos de botones
        try:
            self._style_button(btn_set, 'secondary')
            self._style_button(btn_rand, 'secondary')
            self._style_button(btn, 'primary')
            self._style_spinbox_arrows(vi_rows, vi_cols)
        except Exception:
            pass
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
        size_bar.setSpacing(16)
        size_bar.setContentsMargins(0, 0, 0, 8)
        size_bar.addWidget(QLabel('Filas:'))
        triu_rows = QSpinBox(); triu_rows.setRange(1, 12); triu_rows.setValue(4); size_bar.addWidget(triu_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        triu_cols = QSpinBox(); triu_cols.setRange(1, 12); triu_cols.setValue(4); size_bar.addWidget(triu_cols)
        btn_set = QPushButton('Aplicar tamaño'); btn_set.setObjectName('btnSecondary'); size_bar.addWidget(btn_set)
        size_bar.addStretch(1)
        self.triu_grid = MatrixTable(4,4); box.addWidget(self.triu_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        acts.setSpacing(16)
        acts.setContentsMargins(0, 0, 0, 8)
        btn_rand = QPushButton('🎲 Aleatoria'); btn_rand.setObjectName('btnSecondary'); acts.addWidget(btn_rand)
        btn = QPushButton('🔺 Calcular U'); btn.setObjectName('btnPrimary'); acts.addWidget(btn)
        acts.addStretch(1)
        self.center_layout.addWidget(wrap)

        # Estilos de botones
        try:
            self._style_button(btn_set, 'secondary')
            self._style_button(btn_rand, 'secondary')
            self._style_button(btn, 'primary')
            self._style_spinbox_arrows(triu_rows, triu_cols)
        except Exception:
            pass

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
        size_bar.setSpacing(16)
        size_bar.setContentsMargins(0, 0, 0, 8)
        size_bar.addWidget(QLabel('Filas:'))
        rref_rows = QSpinBox(); rref_rows.setRange(1, 12); rref_rows.setValue(3); size_bar.addWidget(rref_rows)
        size_bar.addWidget(QLabel('Columnas:'))
        rref_cols = QSpinBox(); rref_cols.setRange(1, 12); rref_cols.setValue(4); size_bar.addWidget(rref_cols)
        btn_set = QPushButton('Aplicar tamaño'); btn_set.setObjectName('btnSecondary'); size_bar.addWidget(btn_set)
        size_bar.addStretch(1)
        self.rref_grid = MatrixTable(3,4); box.addWidget(self.rref_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        acts.setSpacing(16)
        acts.setContentsMargins(0, 0, 0, 8)
        btn_rand = QPushButton('🎲 Aleatoria'); btn_rand.setObjectName('btnSecondary'); acts.addWidget(btn_rand)
        btn = QPushButton('🧱 Calcular RREF'); btn.setObjectName('btnPrimary'); acts.addWidget(btn)
        acts.addStretch(1)
        self.center_layout.addWidget(wrap)

        # Estilos de botones
        try:
            self._style_button(btn_set, 'secondary')
            self._style_button(btn_rand, 'secondary')
            self._style_button(btn, 'primary')
            self._style_spinbox_arrows(rref_rows, rref_cols)
        except Exception:
            pass

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
        btn_set = QPushButton('Aplicar tamaño'); btn_set.setObjectName('btnSecondary'); size_bar.addWidget(btn_set)
        self.ti_grid = MatrixTable(3,3); box.addWidget(self.ti_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        acts.setSpacing(16)
        acts.setContentsMargins(0, 0, 0, 8)
        btn_rand = QPushButton('🎲 Aleatoria'); btn_rand.setObjectName('btnSecondary'); acts.addWidget(btn_rand)
        btn = QPushButton('🔁 Calcular'); btn.setObjectName('btnPrimary'); acts.addWidget(btn)
        acts.addStretch(1)
        self.center_layout.addWidget(wrap)

        # Estilos de botones
        try:
            self._style_button(btn_set, 'secondary')
            self._style_button(btn_rand, 'secondary')
            self._style_button(btn, 'primary')
            self._style_spinbox_arrows(ti_rows, ti_cols)
        except Exception:
            pass

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
        btn_set = QPushButton('Aplicar tamaño'); btn_set.setObjectName('btnSecondary'); size_bar.addWidget(btn_set)
        self.det_grid = MatrixTable(3,3); box.addWidget(self.det_grid)
        acts = QHBoxLayout(); box.addLayout(acts)
        acts.setSpacing(16)
        acts.setContentsMargins(0, 0, 0, 8)
        btn_rand = QPushButton('🎲 Aleatoria'); btn_rand.setObjectName('btnSecondary'); acts.addWidget(btn_rand)
        btn = QPushButton('🧾 Calcular det'); btn.setObjectName('btnPrimary'); acts.addWidget(btn)
        acts.addStretch(1)
        self.center_layout.addWidget(wrap)

        # Estilos de botones
        try:
            self._style_button(btn_set, 'secondary')
            self._style_button(btn_rand, 'secondary')
            self._style_button(btn, 'primary')
            self._style_spinbox_arrows(det_rows, det_cols)
        except Exception:
            pass

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
        btn_create = QPushButton('Crear sistema'); btn_create.setObjectName('btnSecondary'); top.addWidget(btn_create)
        # placeholder: no grid until user creates it
        self.cramer_grid = None
        self.cramer_grid_host = QWidget(); self.cramer_grid_layout = QVBoxLayout(self.cramer_grid_host)
        box.addWidget(self.cramer_grid_host)
        # actions area under grid (created after grid exists)
        self.cramer_actions_host = QWidget(); self.cramer_actions_layout = QHBoxLayout(self.cramer_actions_host)
        box.addWidget(self.cramer_actions_host)
        btn_calc = QPushButton('📐 Resolver'); btn_calc.setObjectName('btnPrimary'); btn_calc.setEnabled(False); box.addWidget(btn_calc)
        # Estilos de botones
        try:
            self._style_button(btn_create, 'secondary')
            self._style_button(btn_calc, 'primary')
            self._style_spinbox_arrows(self.cramer_n)
        except Exception:
            pass
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
            btn_rand = QPushButton('🎲 Aleatoria [A|b]'); btn_rand.setObjectName('btnSecondary')
            self.cramer_actions_layout.addWidget(btn_rand)
            try:
                self._style_button(btn_rand, 'secondary')
            except Exception:
                pass
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
        try:
            self._style_button(btn_create, 'secondary')
            self._style_button(btn_calc, 'primary')
            self._style_spinbox_arrows(self.cramer_n)
        except Exception:
            pass

    def show_bisection(self):
        self.current_view = 'bisection'
        self.center_title.setText('Método de Bisección')
        self.clear_center()
        panel, box = self._build_method_card(
            'Configura f(x) e intervalo',
            'El método requiere un intervalo [xi, xu] con cambio de signo (f(xi)·f(xu) < 0).'
        )

        fn_row = QHBoxLayout(); box.addLayout(fn_row)
        lbl_fn = QLabel('f(x) ='); lbl_fn.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:14px;")
        fn_row.addWidget(lbl_fn)
        expr_edit = QLineEdit(''); expr_edit.setPlaceholderText('Expresión en x, p.ej. x**3 - x - 2')
        expr_edit.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:13px;")
        expr_edit.setFixedHeight(28)
        fn_row.addWidget(expr_edit, 1)

        box.addWidget(self._build_math_keyboard(expr_edit))
        box.addWidget(self._build_format_hint_label('Recuerda validar el signo de f(x) en los extremos.'))
        box.addWidget(self._build_equation_examples_label())

        p_row = QHBoxLayout(); p_row.setSpacing(12); box.addLayout(p_row)
        def make_col(label_text, widget):
            col = QVBoxLayout(); lab = QLabel(label_text); lab.setStyleSheet('color:#9aa4aa; font-size:12px;')
            col.addWidget(lab); col.addWidget(widget); return col
        xi_spin = TrimDoubleSpinBox(); xi_spin.setDecimals(8); xi_spin.setRange(-1e12,1e12); xi_spin.setSpecialValueText(''); xi_spin.setValue(xi_spin.minimum()); xi_spin.setFixedWidth(120)
        xu_spin = TrimDoubleSpinBox(); xu_spin.setDecimals(8); xu_spin.setRange(-1e12,1e12); xu_spin.setSpecialValueText(''); xu_spin.setValue(xu_spin.minimum()); xu_spin.setFixedWidth(120)
        eps_spin = TrimDoubleSpinBox(); eps_spin.setDecimals(8); eps_spin.setRange(1e-12,1.0); eps_spin.setSingleStep(1e-4); eps_spin.setMinimum(0.0); eps_spin.setSpecialValueText(''); eps_spin.setValue(0.0); eps_spin.setFixedWidth(120)
        itmax = QSpinBox(); itmax.setRange(1,1000); itmax.setMinimum(0); itmax.setSpecialValueText(''); itmax.setValue(0); itmax.setFixedWidth(100)
        p_row.addLayout(make_col('Intervalo inferior (xi)', xi_spin))
        p_row.addLayout(make_col('Intervalo superior (xu)', xu_spin))
        p_row.addLayout(make_col('Error de convergencia (ε)', eps_spin))
        p_row.addLayout(make_col('Iter máx', itmax))
        rand_btn = QPushButton('🎲 Aleatoria'); rand_btn.setToolTip('Rellenar con función e intervalo aleatorios válidos'); rand_btn.setObjectName('btnSecondary'); p_row.addWidget(rand_btn)
        calc_btn = QPushButton('⚙️ Calcular'); calc_btn.setObjectName('btnPrimary'); p_row.addWidget(calc_btn)

        # Estilos de botones
        try:
            self._style_button(rand_btn, 'secondary')
            self._style_button(calc_btn, 'primary')
            self._style_spinbox_arrows(xi_spin, xu_spin, eps_spin, itmax)
        except Exception:
            pass

        self.center_layout.addWidget(panel)
        def calc():
            try:
                expr_text = expr_edit.text().strip()
                if not expr_text:
                    self.push_error('Escribe una expresión para f(x).'); return
                def _is_empty(sp):
                    return (sp.specialValueText()=='' and sp.value()==sp.minimum())
                if any(_is_empty(sp) for sp in (xi_spin, xu_spin, eps_spin, itmax)):
                    self.push_error('Completa xi, xu, ε e iter máx.'); return
                x = _SYM_symbols('x'); expr = _SYM_sympify(expr_text); f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                xi = float(xi_spin.value()); xu = float(xu_spin.value())
                if not (xi < xu):
                    self.push_error('Debe cumplirse xi < xu.'); return
                yi = float(f(xi)); yu = float(f(xu))
                if not all(np.isfinite([yi, yu])):
                    self.push_error('f(xi) o f(xu) no es válido.'); return
                if yi * yu > 0:
                    self.push_error('No hay cambio de signo en [xi, xu].'); return
                eps_text = eps_spin.text().strip(); eps = float(eps_spin.value()); max_iter = int(itmax.value())
                if max_iter <= 0:
                    self.push_error('Iteraciones máximas debe ser > 0.'); return
                rows = []; xr_old = None; xi_c, xu_c, yi_c, yu_c = xi, xu, yi, yu
                for it in range(1, max_iter+1):
                    xr = 0.5*(xi_c + xu_c); yr = float(f(xr))
                    Ea = 0.0 if xr_old is None else (abs((xr - xr_old)/xr) if xr != 0 else abs(xr - xr_old))
                    rows.append((it, xi_c, xu_c, xr, Ea, yi_c, yu_c, yr))
                    if xr_old is not None and Ea <= eps:
                        break
                    if yi_c * yr < 0:
                        xu_c, yu_c = xr, yr
                    else:
                        xi_c, yi_c = xr, yr
                    xr_old = xr
                self._push_bisect_summary_card(expr_text, xi, xu, rows, eps_text)
                self._show_bisect_dialog(expr_text, xi, xu, rows, eps_text)
            except Exception as e:
                self.push_error(str(e))
        calc_btn.clicked.connect(calc)
        def fill_random():
            try:
                x = _SYM_symbols('x')
                def build_expr_text():
                    choice = np.random.choice(['poly','sin','poly_sin','exp'])
                    if choice == 'poly':
                        deg = int(np.random.randint(2,5)); coeffs = list(np.random.randint(-5,6,size=deg+1))
                        while coeffs[0] == 0:
                            coeffs[0] = int(np.random.randint(-5,6))
                        terms=[]; p=deg
                        for c in coeffs:
                            if p>1: terms.append(f"{c}*x**{p}")
                            elif p==1: terms.append(f"{c}*x")
                            else: terms.append(f"{c}")
                            p-=1
                        return ' + '.join(terms).replace('+ -','- ')
                    elif choice == 'sin':
                        a=int(np.random.randint(1,4)); b=int(np.random.randint(1,4)); d=int(np.random.randint(-2,3)); return f"{a}*sin({b}*x) + {d}"
                    elif choice == 'poly_sin':
                        a=int(np.random.randint(-3,4)); b=int(np.random.randint(1,4)); c=int(np.random.randint(-2,3)); d=int(np.random.randint(-2,3));
                        if a==0: a=1; return f"{a}*x**2 + {b}*sin(x) + {c}*x + {d}"
                    else:
                        a=int(np.random.randint(1,4)); b=float(np.random.choice([0.3,0.5,1.0])); cst=int(np.random.randint(0,4)); return f"{a}*exp({b}*x) - {cst}"
                for _ in range(12):
                    expr_text = build_expr_text(); expr = _SYM_sympify(expr_text); f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                    xs = np.linspace(-5.0,5.0,400); ys = np.asarray(f(xs), dtype=float); finite = np.isfinite(ys); found=False
                    for i in range(len(xs)-1):
                        if not (finite[i] and finite[i+1]): continue
                        if ys[i] == 0:
                            xi_val = float(xs[i] - 0.5*(xs[1]-xs[0])); xu_val = float(xs[i] + 0.5*(xs[1]-xs[0])); found=True; break
                        if ys[i]*ys[i+1] < 0:
                            xi_val = float(xs[i]); xu_val = float(xs[i+1]); found=True; break
                    if found:
                        eps_val = float(np.random.choice([1e-2,5e-3,1e-3,5e-4,1e-4])); it_val = int(np.random.randint(18,45))
                        expr_edit.setText(expr_text); xi_spin.setValue(xi_val); xu_spin.setValue(xu_val); eps_spin.setValue(eps_val); itmax.setValue(it_val); return
                expr_edit.setText('x**3 - x - 2'); xi_spin.setValue(1.0); xu_spin.setValue(2.0); eps_spin.setValue(1e-4); itmax.setValue(25)
            except Exception as e:
                self.push_error(str(e))
        rand_btn.clicked.connect(fill_random)

    def _push_bisect_summary_card(self, expr_text: str, xi: float, xu: float, rows, eps_text: str | None = None):
        """Crea una tarjeta estándar de Bisección usando ResultCard."""
        content = QWidget(); lay = QVBoxLayout(content); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        title = QLabel('<b>Método de bisección</b>'); lay.addWidget(title)
        formula = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_text}</span>")
        lay.addWidget(formula)
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            ea = float(rows[-1][4])
            residual = abs(float(rows[-1][7]))
            sgn = '+' if xr >= 0 else ''

            def add_frame_label(text: str):
                fr = QFrame(); fr.setObjectName('resultFrame')
                vl = QVBoxLayout(fr); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
                lab = QLabel(text); lab.setStyleSheet("color:#ddd; font-size:12pt;")
                vl.addWidget(lab)
                lay.addWidget(fr)

            add_frame_label(f"El método CONVERGE en <b>{n}</b> iteraciones.")
            add_frame_label(f"LA RAÍZ ES: <b>{sgn}{xr:.6f}</b> | Error (Ea): <b>{ea*100:.2f}%</b>")
            tol_show = eps_text if (eps_text is not None and eps_text != '') else ''
            add_frame_label(f"Error raíz |f(r)|: <b>{residual:.6g}</b> | Tolerancia: <b>{tol_show}</b>")

        copy_lines = []
        copy_lines.append(f"Método de Bisección")
        copy_lines.append(f"f(x) = {expr_text}")
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            ea = float(rows[-1][4])
            residual = abs(float(rows[-1][7]))
            tol_show = eps_text if (eps_text is not None and eps_text != '') else ''
            copy_lines.append(f"Iteraciones: {n}")
            copy_lines.append(f"Raíz ≈ {xr:.10g}")
            copy_lines.append(f"Ea ≈ {ea*100:.4f}%")
            copy_lines.append(f"|f(r)| ≈ {residual:.6g}")
            if tol_show:
                copy_lines.append(f"Tolerancia: {tol_show}")
        copy_text = "\n".join(copy_lines)

        # Usa la tarjeta unificada. El botón "Ver detalles" abre el diálogo
        # específico de bisección ya existente.
        self.add_result_card(
            'Método de Bisección',
            content,
            steps=None,
            copy_text=copy_text,
            details_callback=lambda: self._show_bisect_dialog(expr_text, xi, xu, rows, eps_text),
        )

    def _show_bisect_dialog(self, expr_text: str, xi: float, xu: float, rows, eps_text: str | None = None):
        try:
            dlg = BisectionResultDialog(expr_text, xi, xu, rows, eps_text, self)
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            self._dialogs.append(dlg)
            def _cleanup():
                if dlg in self._dialogs:
                    self._dialogs.remove(dlg)
            dlg.destroyed.connect(lambda *_: _cleanup())
            dlg.show()
        except Exception as e:
            self.push_error(str(e))

    # -------------------------------
    # Método de Falsa Posición (Regla Falsa)
    # -------------------------------
    def show_false_position(self):
        self.current_view = 'false_position'
        self.center_title.setText('Método de Falsa Posición (Regla Falsa)')
        self.clear_center()
        wrap, box = self._build_method_card(
            'Configura f(x) e intervalo',
            'Selecciona un intervalo [xi, xu] con f(xi)·f(xu) < 0 para aplicar Regla Falsa.'
        )

        # Reducir el espacio entre el título y los controles
        if isinstance(wrap.layout(), QVBoxLayout):
            wrap.layout().setContentsMargins(20, 8, 20, 18)

        # Function input
        fn_row = QHBoxLayout(); box.addLayout(fn_row)
        lbl_fn = QLabel('f(x) =')
        lbl_fn.setStyleSheet(f"font-family: {MATH_FONT_STACK}; font-size:14px;")
        fn_row.addWidget(lbl_fn)
        expr_edit = QLineEdit(""); expr_edit.setPlaceholderText("Expresión en x, p.ej. x**3 - x - 2")
        expr_edit.setStyleSheet(f"font-family: {MATH_FONT_STACK}; font-size:13px;")
        expr_edit.setFixedHeight(28)
        fn_row.addWidget(expr_edit, 1)

        box.addWidget(self._build_math_keyboard(expr_edit))
        box.addWidget(self._build_format_hint_label('Secante, cosecante y cotangente están disponibles mediante sec(x), csc(x) y cot(x).'))
        box.addWidget(self._build_equation_examples_label())

        # Parámetros
        p_row = QHBoxLayout(); p_row.setSpacing(10); box.addLayout(p_row)
        def make_col(label_text, widget):
            col = QVBoxLayout(); lab = QLabel(label_text); lab.setStyleSheet('color:#9aa4aa; font-size:12px;')
            col.addWidget(lab); col.addWidget(widget); return col

        xi_spin = TrimDoubleSpinBox(); xi_spin.setDecimals(8); xi_spin.setRange(-1e12, 1e12)
        xi_spin.setSpecialValueText(''); xi_spin.setValue(xi_spin.minimum()); xi_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Intervalo inferior (xi)', xi_spin))
        xu_spin = TrimDoubleSpinBox(); xu_spin.setDecimals(8); xu_spin.setRange(-1e12, 1e12)
        xu_spin.setSpecialValueText(''); xu_spin.setValue(xu_spin.minimum()); xu_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Intervalo superior (xu)', xu_spin))
        eps_spin = TrimDoubleSpinBox(); eps_spin.setDecimals(8); eps_spin.setRange(1e-12, 1.0); eps_spin.setSingleStep(1e-4)
        eps_spin.setMinimum(0.0); eps_spin.setSpecialValueText(''); eps_spin.setValue(0.0); eps_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Error de convergencia (ε)', eps_spin))
        itmax = QSpinBox(); itmax.setRange(1, 1000); itmax.setMinimum(0); itmax.setSpecialValueText(''); itmax.setValue(0); itmax.setFixedWidth(100)
        p_row.addLayout(make_col('Iter máx', itmax))
        rand_btn = QPushButton('🎲 Aleatoria'); rand_btn.setToolTip('Rellenar con función e intervalo aleatorios válidos'); rand_btn.setObjectName('btnSecondary')
        p_row.addWidget(rand_btn)
        calc_btn = QPushButton('⚙️ Calcular'); calc_btn.setObjectName('btnPrimary'); p_row.addWidget(calc_btn)

        error_label = QLabel('')
        error_label.setStyleSheet('color:#ff9999; font-size:11px;')
        box.addWidget(error_label)

        self.center_layout.addWidget(wrap)

        # Estilos de botones
        try:
            self._style_button(rand_btn, 'secondary')
            self._style_button(calc_btn, 'primary')
            self._style_spinbox_arrows(xi_spin, xu_spin, eps_spin, itmax)
        except Exception:
            pass

        def calc():
            try:
                expr_text = expr_edit.text().strip()
                if not expr_text:
                    self.push_error('Escribe una expresión para f(x).'); return
                def _is_empty_spin(sp):
                    return (sp.specialValueText()=='' and sp.value()==sp.minimum())
                if any([_is_empty_spin(x) for x in (xi_spin, xu_spin, eps_spin, itmax)]):
                    self.push_error('Completa xi, xu, ε e iter máx.'); return
                x = _SYM_symbols('x')
                expr = _SYM_sympify(expr_text)
                f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                xi = float(xi_spin.value()); xu = float(xu_spin.value())
                if not (xi < xu):
                    self.push_error('Debe cumplirse xi < xu.'); return
                yi = float(f(xi)); yu = float(f(xu))
                if np.isnan(yi) or np.isnan(yu) or np.isinf(yi) or np.isinf(yu):
                    self.push_error('f(xi) o f(xu) no es válido. Revisa la expresión/intervalo.'); return
                if yi * yu > 0:
                    self.push_error('No hay cambio de signo en [xi, xu]. Elige otro intervalo.'); return
                eps_text = eps_spin.text().strip()
                eps = float(eps_spin.value()); max_iter = int(itmax.value())
                if max_iter <= 0:
                    self.push_error('Iteraciones máximas debe ser > 0.'); return

                rows = []  # (iter, xi, xu, xr, Ea, yi, yu, yr)
                xi_c, xu_c, yi_c, yu_c = xi, xu, yi, yu
                xr_old = None
                for it in range(1, max_iter+1):
                    denom = (yi_c - yu_c)
                    if denom == 0:
                        # intervalo degenerado; detenemos
                        break
                    xr = xu_c - yu_c * ((xi_c - xu_c) / denom)
                    yr = float(f(xr))
                    if xr_old is None:
                        Ea = 0.0
                    else:
                        Ea = abs((xr - xr_old) / xr) if xr != 0 else abs(xr - xr_old)
                    rows.append((it, xi_c, xu_c, xr, Ea, yi_c, yu_c, yr))
                    if xr_old is not None and Ea <= eps:
                        break
                    # Actualizar intervalo por regla falsa
                    if yi_c * yr < 0:
                        xu_c, yu_c = xr, yr
                    else:
                        xi_c, yi_c = xr, yr
                    xr_old = xr

                self._push_falsepos_summary_card(expr_text, xi, xu, rows, eps_text)
                self._show_falsepos_dialog(expr_text, xi, xu, rows, eps_text)
            except Exception as e:
                self.push_error(str(e))

        calc_btn.clicked.connect(calc)

        def fill_random():
            try:
                x = _SYM_symbols('x')
                def build_expr_text():
                    choice = np.random.choice(['poly', 'sin', 'poly_sin', 'exp'])
                    if choice == 'poly':
                        deg = int(np.random.randint(2, 5))
                        coeffs = list(np.random.randint(-5, 6, size=deg+1))
                        while coeffs[0] == 0:
                            coeffs[0] = int(np.random.randint(-5, 6))
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
                    else:
                        a = int(np.random.randint(1, 4)); b = float(np.random.choice([0.3, 0.5, 1.0]))
                        cst = int(np.random.randint(0, 4))
                        return f"{a}*exp({b}*x) - {cst}"

                for _ in range(12):
                    expr_text = build_expr_text()
                    expr = _SYM_sympify(expr_text)
                    f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
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
                        eps_val = float(np.random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
                        it_val = int(np.random.randint(18, 45))
                        expr_edit.setText(expr_text)
                        xi_spin.setValue(xi_val)
                        xu_spin.setValue(xu_val)
                        eps_spin.setValue(eps_val)
                        itmax.setValue(it_val)
                        return
                # Fallback
                expr_edit.setText("x**3 - x - 2")
                xi_spin.setValue(1.0)
                xu_spin.setValue(2.0)
                eps_spin.setValue(1e-4)
                itmax.setValue(25)
            except Exception as e:
                self.push_error(str(e))

        rand_btn.clicked.connect(fill_random)

    # -------------------------------
    # Método de la Secante
    # -------------------------------
    def show_secant(self):
        self.current_view = 'secant'
        self.center_title.setText('Método de la Secante')
        self.clear_center()

        wrap, box = self._build_method_card(
            'Configura f(x) y valores iniciales',
            'La secante usa dos aproximaciones iniciales x0 y x1; no exige cambio de signo en f(x).'
        )

        # Acerca un poco el contenido al título para reducir el espacio
        if isinstance(wrap.layout(), QVBoxLayout):
            wrap.layout().setContentsMargins(20, 8, 20, 18)

        fn_row = QHBoxLayout(); box.addLayout(fn_row)
        lbl_fn = QLabel('f(x) ='); lbl_fn.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:14px;")
        fn_row.addWidget(lbl_fn)
        expr_edit = QLineEdit(''); expr_edit.setPlaceholderText('Ejemplo: x**3 - x - 2')
        expr_edit.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:13px;")
        expr_edit.setFixedHeight(28)
        fn_row.addWidget(expr_edit, 1)

        box.addWidget(self._build_math_keyboard(expr_edit))
        box.addWidget(self._build_format_hint_label('No se requiere que f(x) cambie de signo entre x0 y x1.'))
        box.addWidget(self._build_equation_examples_label())

        p_row = QHBoxLayout(); p_row.setSpacing(10); box.addLayout(p_row)
        def make_col(label_text, widget):
            col = QVBoxLayout(); lbl = QLabel(label_text)
            lbl.setStyleSheet('font-size:12px;')
            col.addWidget(lbl)
            col.addWidget(widget)
            return col

        x0_spin = TrimDoubleSpinBox(); x0_spin.setDecimals(8); x0_spin.setRange(-1e12, 1e12)
        x0_spin.setSpecialValueText(''); x0_spin.setValue(x0_spin.minimum()); x0_spin.setFixedWidth(140)
        p_row.addLayout(make_col('x₀ (primera aproximación)', x0_spin))

        x1_spin = TrimDoubleSpinBox(); x1_spin.setDecimals(8); x1_spin.setRange(-1e12, 1e12)
        x1_spin.setSpecialValueText(''); x1_spin.setValue(x1_spin.minimum()); x1_spin.setFixedWidth(140)
        p_row.addLayout(make_col('x₁ (segunda aproximación)', x1_spin))

        eps_spin = TrimDoubleSpinBox(); eps_spin.setDecimals(8); eps_spin.setRange(1e-12, 1.0); eps_spin.setSingleStep(1e-4)
        eps_spin.setMinimum(0.0); eps_spin.setSpecialValueText(''); eps_spin.setValue(0.0); eps_spin.setFixedWidth(120)
        p_row.addLayout(make_col('Error de convergencia (ε)', eps_spin))

        itmax = QSpinBox(); itmax.setRange(1, 1000); itmax.setMinimum(0); itmax.setSpecialValueText(''); itmax.setValue(0); itmax.setFixedWidth(100)
        p_row.addLayout(make_col('Iter máx', itmax))

        rand_btn = QPushButton('🎲 Aleatoria'); rand_btn.setToolTip('Rellenar con función y valores iniciales sugeridos'); rand_btn.setObjectName('btnSecondary')
        p_row.addWidget(rand_btn)
        calc_btn = QPushButton('⚙️ Calcular'); calc_btn.setObjectName('btnPrimary'); p_row.addWidget(calc_btn)

        error_label = QLabel('')
        error_label.setStyleSheet('color:#ff9999; font-size:11px;')
        box.addWidget(error_label)

        self.center_layout.addWidget(wrap)

        # Estilos de botones
        try:
            self._style_button(rand_btn, 'secondary')
            self._style_button(calc_btn, 'primary')
            self._style_spinbox_arrows(x0_spin, x1_spin, eps_spin, itmax)
        except Exception:
            pass

        def _is_empty(spin: TrimDoubleSpinBox | QSpinBox):
            if isinstance(spin, TrimDoubleSpinBox):
                return spin.specialValueText() == '' and spin.value() == spin.minimum()
            return spin.specialValueText() == '' and spin.value() == spin.minimum()

        def calc():
            try:
                calc_btn.setEnabled(False)
                error_label.setText('')
                expr_text = expr_edit.text().strip()
                if not expr_text:
                    error_label.setText('Escribe una expresión para f(x).')
                    return
                if any(_is_empty(sp) for sp in (x0_spin, x1_spin, eps_spin, itmax)):
                    error_label.setText('Completa x₀, x₁, ε e iter máx.')
                    return
                x = _SYM_symbols('x')
                expr = _SYM_sympify(expr_text)
                f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)

                x0 = float(x0_spin.value()); x1 = float(x1_spin.value())
                if not (np.isfinite(x0) and np.isfinite(x1)):
                    error_label.setText('x₀ y x₁ deben ser números finitos.')
                    return
                if x0 == x1:
                    error_label.setText('x₀ y x₁ deben ser distintos para la secante.')
                    return

                eps = float(eps_spin.value()); eps_text = eps_spin.text().strip()
                max_iter = int(itmax.value())
                if eps <= 0:
                    error_label.setText('El error objetivo ε debe ser > 0.')
                    return
                if max_iter <= 0:
                    error_label.setText('Iteraciones máximas debe ser > 0.')
                    return

                rows = []
                x_prev, x_curr = x0, x1
                root_estimate = x_curr
                residual = abs(float(f(x_curr)))

                for it in range(1, max_iter + 1):
                    f_prev = float(f(x_prev))
                    f_curr = float(f(x_curr))
                    if not (np.isfinite(f_prev) and np.isfinite(f_curr)):
                        raise ValueError('f(x) no es finito en la iteración actual. Revisa x₀, x₁ o la función.')
                    denom = (f_curr - f_prev)
                    if denom == 0:
                        raise ValueError('f(x₁) - f(x₀) = 0 produce división no válida. Prueba con otros valores iniciales.')
                    x_next = x_curr - f_curr * (x_curr - x_prev) / denom
                    ea = 0.0 if it == 1 else (abs((x_next - x_curr) / x_next) if x_next != 0 else abs(x_next - x_curr))
                    residual = abs(float(f(x_next)))
                    rows.append((it, x_prev, x_curr, f_prev, f_curr, x_next, ea, residual))
                    root_estimate = x_next
                    if residual <= eps or (it > 1 and ea <= eps):
                        break
                    x_prev, x_curr = x_curr, x_next

                if not rows:
                    self._push_secant_summary_card(expr_text, x0, x1, rows, eps_text, root_estimate, residual, converged=False)
                    return

                converged = residual <= eps or any(r[6] <= eps for r in rows[1:])
                self._push_secant_summary_card(expr_text, x0, x1, rows, eps_text, root_estimate, residual, converged=converged)
                self._show_secant_dialog(expr_text, x0, x1, rows, eps_text)
            except Exception as e:
                error_label.setText(str(e))
            finally:
                calc_btn.setEnabled(True)

        calc_btn.clicked.connect(calc)

        def fill_random():
            try:
                x = _SYM_symbols('x')
                def build_expr_text():
                    choice = np.random.choice(['poly','sin','exp'])
                    if choice == 'poly':
                        a = int(np.random.randint(-3, 4) or 1)
                        b = int(np.random.randint(-5, 6))
                        c = int(np.random.randint(-5, 6))
                        return f"{a}*x**2 + {b}*x + {c}"
                    if choice == 'sin':
                        k = int(np.random.randint(1, 4))
                        return f"sin({k}*x) - 0.5"
                    return "exp(x) - 3"

                expr_text = build_expr_text()
                expr = _SYM_sympify(expr_text)
                f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                # Buscar dos puntos cercanos con valores distintos
                x0_val = float(np.random.uniform(-3, 0))
                x1_val = x0_val + float(np.random.uniform(0.5, 2.0))
                expr_edit.setText(expr_text)
                x0_spin.setValue(x0_val)
                x1_spin.setValue(x1_val)
                eps_spin.setValue(1e-4)
                itmax.setValue(30)
            except Exception:
                expr_edit.setText('x**3 - x - 2')
                x0_spin.setValue(1.0)
                x1_spin.setValue(2.0)
                eps_spin.setValue(1e-4)
                itmax.setValue(30)

        rand_btn.clicked.connect(fill_random)

    def mostrar_grafica_secante(self, iteraciones: list[tuple]):
        """Dibuja opcionalmente la gráfica del método de la secante.

        `iteraciones` es la lista de tuplas
        (it, x_prev, x_curr, f_prev, f_curr, x_next, ea, residual)
        ya calculadas en el slot de la secante. Aquí solo se usan
        para representar gráficamente el proceso.
        """
        if not iteraciones:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np  # por si no está en el ámbito local
        except Exception:
            return

        # Extraer datos de iteración
        iters = np.array([row[0] for row in iteraciones], dtype=float)
        x_prev = np.array([row[1] for row in iteraciones], dtype=float)
        x_curr = np.array([row[2] for row in iteraciones], dtype=float)
        fx_prev = np.array([row[3] for row in iteraciones], dtype=float)
        fx_curr = np.array([row[4] for row in iteraciones], dtype=float)
        x_next = np.array([row[5] for row in iteraciones], dtype=float)

        # Ventana para representar f(x) y las rectas secantes
        fig, (ax_fun, ax_iter) = plt.subplots(1, 2, figsize=(9, 4))

        # Dominio aproximado a partir de las aproximaciones obtenidas
        xs_all = np.concatenate([x_prev, x_curr, x_next])
        xmin = float(xs_all.min())
        xmax = float(xs_all.max())
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        margin = 0.15 * (xmax - xmin)
        xmin -= margin
        xmax += margin

        # Como aquí no tenemos directamente f(x), aproximamos su forma
        # con los puntos (x_prev, f_prev) y (x_curr, f_curr) que ya están
        # evaluados en cada iteración.
        # Para dar una idea global, usamos todos esos puntos dispersos.
        ax_fun.scatter(x_prev, fx_prev, s=18, color='#58a6ff', label='f(xₙ₋₁)')
        ax_fun.scatter(x_curr, fx_curr, s=18, color='#f2cc60', label='f(xₙ)')

        # Dibujar las rectas secantes de cada iteración
        for xp, xc, fp, fc in zip(x_prev, x_curr, fx_prev, fx_curr):
            if xc == xp:
                continue
            m = (fc - fp) / (xc - xp)
            xs_line = np.linspace(xp, xc, 2)
            ys_line = m * (xs_line - xp) + fp
            ax_fun.plot(xs_line, ys_line, color='#8b949e', alpha=0.8)

        # Marcar la última aproximación de la raíz
        root_est = float(x_next[-1])
        ax_fun.axvline(root_est, color='#d18616', linestyle='--', alpha=0.9,
                       label='aprox. raíz')

        ax_fun.set_xlim(xmin, xmax)
        ax_fun.set_xlabel('x')
        ax_fun.set_ylabel('f(x) / rectas secantes')
        ax_fun.set_title('Secante: función y rectas')
        ax_fun.grid(True, linestyle='--', alpha=0.3)
        ax_fun.legend(loc='best', fontsize=8)

        # Segundo panel: evolución de xₙ
        ax_iter.plot(iters, x_next, marker='o', linestyle='-', color='#58a6ff')
        ax_iter.set_xlabel('Iteración')
        ax_iter.set_ylabel('xₙ₊₁')
        ax_iter.set_title('Evolución de las aproximaciones')
        ax_iter.grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.show()

    def show_newton(self):
        self.current_view = 'newton'
        self.center_title.setText('Método de Newton-Raphson')
        self.clear_center()

        config_card, box = self._build_method_card(
            'Configura f(x) y parámetros',
            'Newton-Raphson necesita un valor inicial x₀ cercano a la raíz y una derivada distinta de cero.'
        )

        fn_row = QHBoxLayout(); box.addLayout(fn_row)
        lbl_fn = QLabel('f(x) ='); lbl_fn.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:14px;")
        fn_row.addWidget(lbl_fn)
        expr_edit = QLineEdit(''); expr_edit.setPlaceholderText('Ejemplo: x**3 - x - 2')
        expr_edit.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:13px;")
        expr_edit.setFixedHeight(28)
        fn_row.addWidget(expr_edit, 1)

        box.addWidget(self._build_math_keyboard(expr_edit))
        box.addWidget(self._build_format_hint_label('Asegúrate de que f\'(x) no se anule en el vecindario de la raíz.'))
        box.addWidget(self._build_equation_examples_label())

        p_grid = QGridLayout(); p_grid.setHorizontalSpacing(16); p_grid.setVerticalSpacing(10); box.addLayout(p_grid)
        p_grid.setColumnStretch(0, 1)
        p_grid.setColumnStretch(1, 1)
        def add_field(idx: int, label_text: str, widget: QWidget):
            wrap = QWidget(); wrap_layout = QVBoxLayout(wrap); wrap_layout.setContentsMargins(0,0,0,0); wrap_layout.setSpacing(4)
            lab = QLabel(label_text); lab.setStyleSheet('color:#9aa4aa; font-size:12px;')
            wrap_layout.addWidget(lab)
            wrap_layout.addWidget(widget)
            row = idx // 2
            col = idx % 2
            p_grid.addWidget(wrap, row, col)

        x0_spin = TrimDoubleSpinBox(); x0_spin.setDecimals(8); x0_spin.setRange(-1e12, 1e12)
        x0_spin.setSpecialValueText(''); x0_spin.setValue(x0_spin.minimum())
        x0_spin.setMinimumWidth(160)
        add_field(0, 'Valor inicial (x₀)', x0_spin)

        eps_spin = TrimDoubleSpinBox(); eps_spin.setDecimals(8); eps_spin.setRange(1e-12, 1.0)
        eps_spin.setSingleStep(1e-4); eps_spin.setMinimum(0.0)
        eps_spin.setSpecialValueText(''); eps_spin.setValue(0.0)
        add_field(1, 'Error de convergencia (ε)', eps_spin)

        itmax = QSpinBox(); itmax.setRange(1, 1000); itmax.setMinimum(0)
        itmax.setSpecialValueText(''); itmax.setValue(0)
        add_field(2, 'Iter máx', itmax)

        buttons_wrap = QWidget(); btn_layout = QHBoxLayout(buttons_wrap); btn_layout.setContentsMargins(0,0,0,0); btn_layout.setSpacing(12)
        rand_btn = QPushButton('🎲 Aleatoria'); rand_btn.setToolTip('Rellenar con función y x₀ sugeridos'); rand_btn.setObjectName('btnSecondary')
        calc_btn = QPushButton('⚙️ Calcular'); calc_btn.setObjectName('btnPrimary')
        # Estilos de botones principales
        try:
            self._style_button(rand_btn, 'secondary')
            self._style_button(calc_btn, 'primary')
            self._style_spinbox_arrows(x0_spin, eps_spin, itmax)
        except Exception:
            pass
        btn_layout.addStretch(1)
        btn_layout.addWidget(rand_btn)
        btn_layout.addWidget(calc_btn)
        p_grid.addWidget(buttons_wrap, 2, 0, 1, 2)

        self.center_layout.addWidget(config_card)

        deriv_card, deriv_box = self._build_method_card(
            'Derivada de f(x)',
            'Puedes dejar que Matrix calcule f\'(x) automáticamente o activar el modo manual cuando necesites control total.'
        )
        mode_switch = QCheckBox("Ingresar f'(x) manualmente")
        deriv_box.addWidget(mode_switch)

        stack = QStackedWidget()
        stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        deriv_box.addWidget(stack)

        auto_page = QWidget(); auto_layout = QVBoxLayout(auto_page); auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(6)
        auto_preview = QLabel("f'(x) = —"); auto_preview.setStyleSheet(f"font-family:{MATH_FONT_STACK}; background:#15171a; border:1px solid #2b2d31; border-radius:8px; padding:10px;")
        auto_preview.setTextFormat(Qt.RichText)
        auto_preview.setWordWrap(True)
        auto_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        auto_layout.addWidget(auto_preview)
        auto_layout.addWidget(self._build_format_hint_label('El motor usa SymPy.diff, así que obtienes la derivada simbólica exacta.'))

        manual_page = QWidget(); manual_layout = QVBoxLayout(manual_page); manual_layout.setContentsMargins(0,0,0,0)
        manual_layout.setSpacing(6)
        manual_edit = QLineEdit(''); manual_edit.setPlaceholderText("Escribe aquí f'(x)")
        manual_edit.setStyleSheet(f"font-family:{MATH_FONT_STACK}; font-size:13px;")
        manual_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        manual_layout.addWidget(manual_edit)
        manual_layout.addWidget(self._build_math_keyboard(manual_edit))
        manual_hint = QLabel("Ejemplo manual: 3*x**2 - sec(x/2). Usa la misma sintaxis que en f(x)."); manual_hint.setObjectName('helperLabel'); manual_hint.setWordWrap(True)
        manual_layout.addWidget(manual_hint)
        copy_btn = QPushButton('⬇️ Copiar derivada automática al campo'); copy_btn.setObjectName('btnSecondary')
        manual_layout.addWidget(copy_btn, 0)

        # Estilo para botón auxiliar de copia
        try:
            self._style_button(copy_btn, 'secondary')
        except Exception:
            pass

        stack.addWidget(auto_page)
        stack.addWidget(manual_page)

        def toggle_manual(checked: bool):
            stack.setCurrentIndex(1 if checked else 0)
        mode_switch.toggled.connect(toggle_manual)
        toggle_manual(False)

        self.center_layout.addWidget(deriv_card)

        auto_state = {'expr': None, 'text': ''}

        def refresh_auto():
            text = expr_edit.text().strip()
            if not text:
                auto_state['expr'] = None
                auto_state['text'] = ''
                auto_preview.setText("f'(x) = —")
                return
            try:
                x = _SYM_symbols('x')
                expr = _SYM_sympify(text)
                derivative = _SYM_diff(expr, x)
                auto_state['expr'] = derivative
                auto_state['text'] = str(derivative)
                escaped = html.escape(auto_state['text'])
                auto_preview.setText(f"f'(x) = <span style=\"font-family:{MATH_FONT_STACK}; color:#d6f4ff;\">{escaped}</span>")
            except Exception:
                auto_state['expr'] = None
                auto_state['text'] = ''
                auto_preview.setText("f'(x) = (expresión inválida)")

        expr_edit.textChanged.connect(refresh_auto)
        refresh_auto()

        def copy_auto_to_manual():
            if auto_state['text']:
                manual_edit.setText(auto_state['text'])
        copy_btn.clicked.connect(copy_auto_to_manual)

        def _is_empty(spin: TrimDoubleSpinBox | QSpinBox):
            if isinstance(spin, TrimDoubleSpinBox):
                return spin.specialValueText() == '' and spin.value() == spin.minimum()
            return spin.specialValueText() == '' and spin.value() == spin.minimum()

        def calc():
            try:
                expr_text = expr_edit.text().strip()
                if not expr_text:
                    self.push_error('Escribe una expresión para f(x).'); return
                if any(_is_empty(sp) for sp in (x0_spin, eps_spin, itmax)):
                    self.push_error('Completa x₀, ε e iter máx.'); return
                x = _SYM_symbols('x')
                expr = _SYM_sympify(expr_text)
                f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                if mode_switch.isChecked():
                    d_text = manual_edit.text().strip()
                    if not d_text:
                        self.push_error('Ingresa la expresión de f\'(x) en modo manual.'); return
                    d_expr = _SYM_sympify(d_text)
                else:
                    if auto_state['expr'] is None:
                        auto_state['expr'] = _SYM_diff(expr, x)
                        auto_state['text'] = str(auto_state['expr'])
                    d_expr = auto_state['expr']
                df = _SYM_lambdify(x, d_expr, _LAMBDA_MODULES)
                x0_value = float(x0_spin.value())
                if not np.isfinite(x0_value):
                    self.push_error('x₀ debe ser un número finito.'); return
                x_curr = x0_value
                eps = float(eps_spin.value()); eps_text = eps_spin.text().strip()
                max_iter = int(itmax.value())
                if eps <= 0:
                    self.push_error('El error objetivo ε debe ser > 0.'); return
                if max_iter <= 0:
                    self.push_error('Iteraciones máximas debe ser > 0.'); return
                rows = []
                root_estimate = x_curr
                residual_next = abs(float(f(x_curr)))
                for it in range(1, max_iter + 1):
                    fx = float(f(x_curr))
                    if not np.isfinite(fx):
                        raise ValueError('f(x) no es finito en la iteración actual. Revisa x₀ o la función.')
                    dfx = float(df(x_curr))
                    if dfx == 0:
                        raise ValueError("f'(x) = 0 produce una división no válida. Prueba con otro x₀.")
                    x_next = x_curr - fx / dfx
                    ea = 0.0 if it == 1 else (abs((x_next - x_curr) / x_next) if x_next != 0 else abs(x_next - x_curr))
                    residual_next = abs(float(f(x_next)))
                    rows.append((it, x_curr, fx, dfx, x_next, ea, residual_next))
                    root_estimate = x_next
                    if residual_next <= eps or (it > 1 and ea <= eps):
                        break
                    x_curr = x_next
                if not rows:
                    self.push_error('No se pudo generar ninguna iteración. Ajusta los parámetros.'); return
                manual_mode = mode_switch.isChecked()
                self._push_newton_summary_card(
                    expr_text,
                    str(d_expr),
                    rows,
                    eps_text,
                    root_estimate,
                    residual_next,
                    manual_mode,
                    x0_value
                )
                self._show_newton_dialog(
                    expr_text,
                    str(d_expr),
                    rows,
                    eps_text,
                    manual_mode,
                    root_estimate,
                    x0_value
                )
            except Exception as e:
                self.push_error(str(e))

        calc_btn.clicked.connect(calc)

        def fill_random():
            try:
                x = _SYM_symbols('x')
                def build_expr_text():
                    choice = np.random.choice(['poly','sin','poly_sin','exp'])
                    if choice == 'poly':
                        deg = int(np.random.randint(2, 5))
                        coeffs = list(np.random.randint(-5, 6, size=deg+1))
                        while coeffs[0] == 0:
                            coeffs[0] = int(np.random.randint(-5, 6))
                        terms = []; power = deg
                        for c in coeffs:
                            if power > 1:
                                terms.append(f"{c}*x**{power}")
                            elif power == 1:
                                terms.append(f"{c}*x")
                            else:
                                terms.append(f"{c}")
                            power -= 1
                        return ' + '.join(terms).replace('+ -', '- ')
                    if choice == 'sin':
                        a = int(np.random.randint(1, 4)); b = int(np.random.randint(1, 4)); d = int(np.random.randint(-2, 3))
                        return f"{a}*sin({b}*x) + {d}"
                    if choice == 'poly_sin':
                        a = int(np.random.randint(-3, 4)); b = int(np.random.randint(1, 4)); c = int(np.random.randint(-2, 3)); d = int(np.random.randint(-2, 3))
                        if a == 0:
                            a = 1
                        return f"{a}*x**2 + {b}*sin(x) + {c}*x + {d}"
                    a = int(np.random.randint(1, 4)); b = float(np.random.choice([0.3, 0.5, 1.0])); cst = int(np.random.randint(0, 4))
                    return f"{a}*exp({b}*x) - {cst}"

                for _ in range(12):
                    expr_text = build_expr_text()
                    expr = _SYM_sympify(expr_text)
                    f = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                    xs = np.linspace(-5.0, 5.0, 400)
                    ys = np.asarray(f(xs), dtype=float)
                    finite = np.isfinite(ys)
                    found = False
                    for i in range(len(xs)-1):
                        if not (finite[i] and finite[i+1]):
                            continue
                        if ys[i] == 0:
                            x0_val = float(xs[i])
                            found = True
                            break
                        if ys[i] * ys[i+1] < 0:
                            x0_val = float(0.5 * (xs[i] + xs[i+1]))
                            found = True
                            break
                    if found:
                        eps_val = float(np.random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
                        it_val = int(np.random.randint(12, 35))
                        expr_edit.setText(expr_text)
                        x0_spin.setValue(x0_val)
                        eps_spin.setValue(eps_val)
                        itmax.setValue(it_val)
                        return
                expr_edit.setText('x**3 - x - 2')
                x0_spin.setValue(1.5)
                eps_spin.setValue(1e-4)
                itmax.setValue(25)
            except Exception as e:
                self.push_error(str(e))

        rand_btn.clicked.connect(fill_random)

    def _push_falsepos_summary_card(self, expr_text: str, xi: float, xu: float, rows, eps_text: str | None = None):
        content = QWidget(); lay = QVBoxLayout(content); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        title = QLabel('<b>Falsa Posición</b>'); lay.addWidget(title)
        formula = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_text}</span>")
        lay.addWidget(formula)
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            ea = float(rows[-1][4])
            residual = abs(float(rows[-1][7]))
            sgn = '+' if xr >= 0 else ''
            tol_show = eps_text if (eps_text is not None and eps_text != '') else ''

            for text in (
                f"El método CONVERGE en <b>{n}</b> iteraciones.",
                f"LA RAÍZ ES: <b>{sgn}{xr:.6f}</b> | Error (Ea): <b>{ea*100:.2f}%</b>",
                f"Error raíz |f(r)|: <b>{residual:.6g}</b> | Tolerancia: <b>{tol_show}</b>",
            ):
                lab = QLabel(text); lab.setStyleSheet("color:#ddd;")
                lay.addWidget(lab)

        copy_lines = []
        copy_lines.append("Método de Falsa Posición")
        copy_lines.append(f"f(x) = {expr_text}")
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            ea = float(rows[-1][4])
            residual = abs(float(rows[-1][7]))
            tol_show = eps_text if (eps_text is not None and eps_text != '') else ''
            copy_lines.append(f"Iteraciones: {n}")
            copy_lines.append(f"Raíz ≈ {xr:.10g}")
            copy_lines.append(f"Ea ≈ {ea*100:.4f}%")
            copy_lines.append(f"|f(r)| ≈ {residual:.6g}")
            if tol_show:
                copy_lines.append(f"Tolerancia: {tol_show}")
        copy_text = "\n".join(copy_lines)

        # Igual que en bisección: el botón "Ver detalles" abre el diálogo
        # dedicado de Falsa Posición.
        self.add_result_card(
            'Método de Falsa Posición',
            content,
            steps=None,
            copy_text=copy_text,
            details_callback=lambda: self._show_falsepos_dialog(expr_text, xi, xu, rows, eps_text),
        )

    def _show_falsepos_dialog(self, expr_text: str, xi: float, xu: float, rows, eps_text: str | None = None):
        try:
            dlg = FalsePositionResultDialog(expr_text, xi, xu, rows, eps_text, self)
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            self._dialogs.append(dlg)
            def _cleanup():
                if dlg in self._dialogs:
                    self._dialogs.remove(dlg)
            dlg.destroyed.connect(lambda *_: _cleanup())
            dlg.show()
        except Exception as e:
            self.push_error(str(e))

    def _push_secant_summary_card(self, expr_text: str, x0: float, x1: float, rows, eps_text: str | None, root_estimate: float, residual: float, converged: bool):
        content = QWidget(); lay = QVBoxLayout(content); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lay.addWidget(QLabel('<b>Método de la Secante</b>'))
        lay.addWidget(QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{html.escape(expr_text)}</span>"))
        lay.addWidget(QLabel(f"Valores iniciales: x₀ = {x0:.6f}, x₁ = {x1:.6f}"))
        if rows:
            iterations = len(rows)
            ea = float(rows[-1][6])
            status_text = "CONVERGE" if converged else "NO CONVERGE (máx. iteraciones alcanzado)"
            info = QLabel(
                f"Iteraciones: <b>{iterations}</b> • Estado: <b>{status_text}</b><br>"
                f"Raíz aproximada = <b>{root_estimate:.6f}</b><br>"
                f"Error aprox. (Ea) = <b>{ea*100:.2f}%</b><br>"
                f"Error de la raíz |f(c)| = <b>{residual:.6g}</b>"
            )
            info.setStyleSheet('color:#ddd;')
            lay.addWidget(info)
            tol_show = eps_text if eps_text else ''
            tol = QLabel(f"Tolerancia = <b>{tol_show}</b>")
            tol.setStyleSheet('color:#ddd;')
            lay.addWidget(tol)

        copy_lines = []
        copy_lines.append("Método de la Secante")
        copy_lines.append(f"f(x) = {expr_text}")
        copy_lines.append(f"x0 = {x0:.10g}, x1 = {x1:.10g}")
        if rows:
            iterations = len(rows)
            ea = float(rows[-1][6])
            tol_show = eps_text if eps_text else ''
            copy_lines.append(f"Iteraciones: {iterations}")
            copy_lines.append(f"Raíz aproximada ≈ {root_estimate:.10g}")
            copy_lines.append(f"Ea ≈ {ea*100:.4f}%")
            copy_lines.append(f"|f(c)| ≈ {residual:.6g}")
            if tol_show:
                copy_lines.append(f"Tolerancia: {tol_show}")
            copy_lines.append(f"Estado: {'CONVERGE' if converged else 'NO CONVERGE'}")
        copy_text = "\n".join(copy_lines)

        # La tarjeta muestra un resumen y el botón "Ver detalles" abre el
        # SecantResultDialog correspondiente.
        self.add_result_card(
            'Método de la Secante',
            content,
            steps=None,
            copy_text=copy_text,
            details_callback=lambda: self._show_secant_dialog(expr_text, x0, x1, rows, eps_text),
        )

    def _show_secant_dialog(self, expr_text: str, x0: float, x1: float, rows, eps_text: str | None = None):
        try:
            dlg = SecantResultDialog(expr_text, x0, x1, rows, eps_text, self)
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            self._dialogs.append(dlg)
            def _cleanup():
                if dlg in self._dialogs:
                    self._dialogs.remove(dlg)
            dlg.destroyed.connect(lambda *_: _cleanup())
            dlg.show()
        except Exception as e:
            self.push_error(str(e))

    def _push_newton_summary_card(self, expr_text: str, deriv_text: str, rows, eps_text: str | None, root_estimate: float, residual: float, manual_derivative: bool, x0_value: float | None):
        content = QWidget(); lay = QVBoxLayout(content); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lay.addWidget(QLabel('<b>Newton-Raphson</b>'))
        expr_html = html.escape(expr_text)
        deriv_html = html.escape(deriv_text)
        lay.addWidget(QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_html}</span>"))
        lay.addWidget(QLabel(f"f'(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{deriv_html}</span>"))
        if rows:
            try:
                fig = Figure(figsize=(4.8, 2.2), dpi=110)
                ax = fig.add_subplot(111)
                x = _SYM_symbols('x'); expr = _SYM_sympify(expr_text); fcall = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
                xs_iter = [row[1] for row in rows] + [rows[-1][4]]
                if xs_iter:
                    min_x, max_x = min(xs_iter), max(xs_iter)
                    span = max(1.0, abs(max_x - min_x))
                    pad = span * 0.5
                    xs = np.linspace(min_x - pad, max_x + pad, 300)
                else:
                    xs = np.linspace(-5.0, 5.0, 300)
                ys = np.asarray(fcall(xs), dtype=float)
                ax.axhline(0, color='#666', lw=1)
                ax.plot(xs, ys, color='#4fc3f7')
                points = np.array([row[1] for row in rows], dtype=float)
                ax.scatter(points, np.zeros_like(points), color='#ffd54f', marker='x')
                ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True, linestyle='--', alpha=0.25)
                fig.tight_layout(pad=0.6)
                canvas = FigureCanvasQTAgg(fig)
                canvas.setMinimumHeight(170)
                canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                lay.addWidget(canvas)
            except Exception:
                pass
        link = QPushButton('🌐 Ver gráfica completa'); link.setObjectName('btnSecondary')
        link.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        link.setStyleSheet("""
            QPushButton {
                background-color: rgba(40, 40, 60, 200);
                border: 1px solid #7f5af0;
                color: #ffffff;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #7f5af0;
            }
        """)
        lay.addWidget(link)
        if rows:
            iterations = len(rows)
            ea = float(rows[-1][5])
            info_frame = QFrame(); info_frame.setObjectName('resultFrame'); info_layout = QVBoxLayout(info_frame)
            info_layout.setContentsMargins(8, 4, 8, 4)
            info_layout.addWidget(QLabel(f"Iteraciones: <b>{iterations}</b>"))
            info_layout.addWidget(QLabel(f"Raíz aproximada: <b>{root_estimate:.6f}</b>"))
            info_layout.addWidget(QLabel(f"Error (Ea): <b>{ea*100:.2f}%</b> | |f(x)|: <b>{residual:.6g}</b>"))
            tol_show = eps_text if eps_text else ''
            info_layout.addWidget(QLabel(f"Tolerancia: <b>{tol_show}</b> | Derivada: <b>{'Manual' if manual_derivative else 'Automática'}</b>"))
            for i in range(info_layout.count()):
                w = info_layout.itemAt(i).widget()
                if isinstance(w, QLabel):
                    w.setStyleSheet('color:#ddd;')
            lay.addWidget(info_frame)

        def _geogebra():
            self._open_geogebra(expr_text, x0_value, root_estimate)

        link.clicked.connect(_geogebra)

        copy_lines = []
        copy_lines.append("Método de Newton-Raphson")
        copy_lines.append(f"f(x) = {expr_text}")
        copy_lines.append(f"f'(x) = {deriv_text}")
        if x0_value is not None:
            copy_lines.append(f"x0 = {x0_value:.10g}")
        if rows:
            iterations = len(rows)
            ea = float(rows[-1][5])
            tol_show = eps_text if eps_text else ''
            copy_lines.append(f"Iteraciones: {iterations}")
            copy_lines.append(f"Raíz aproximada ≈ {root_estimate:.10g}")
            copy_lines.append(f"Ea ≈ {ea*100:.4f}%")
            copy_lines.append(f"|f(x)| ≈ {residual:.6g}")
            if tol_show:
                copy_lines.append(f"Tolerancia: {tol_show}")
            copy_lines.append(f"Derivada: {'Manual' if manual_derivative else 'Automática'}")
        copy_text = "\n".join(copy_lines)

        # En Newton-Raphson el botón "Ver detalles" abre el diálogo completo
        # ya existente.
        self.add_result_card(
            'Newton-Raphson',
            content,
            steps=None,
            copy_text=copy_text,
            details_callback=lambda: self._show_newton_dialog(
                expr_text,
                deriv_text,
                rows,
                eps_text,
                manual_derivative,
                root_estimate,
                x0_value,
            ),
        )

    def _show_newton_dialog(self, expr_text: str, deriv_text: str, rows, eps_text: str | None, manual_derivative: bool, root_estimate: float | None, x0_value: float | None):
        try:
            geo_url = self._build_geogebra_url(expr_text, x0_value, root_estimate)
            dlg = NewtonResultDialog(expr_text, deriv_text, rows, eps_text, manual_derivative, geo_url, self)
            dlg.setAttribute(Qt.WA_DeleteOnClose, True)
            self._dialogs.append(dlg)
            def _cleanup():
                if dlg in self._dialogs:
                    self._dialogs.remove(dlg)
            dlg.destroyed.connect(lambda *_: _cleanup())
            dlg.show()
        except Exception as e:
            self.push_error(str(e))

class FalsePositionResultDialog(QDialog):
    def __init__(self, expr_text: str, xi: float, xu: float, rows, epsilon_text: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Método de Falsa Posición — Detalles')
        self.resize(960, 640)

        lay = QVBoxLayout(self)
        title = QLabel('<b>Método de falsa posición</b>'); lay.addWidget(title)

        # 1) Ecuación sola (encabezado)
        eq_label = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{expr_text}</span>")
        eq_label.setTextFormat(Qt.RichText)
        eq_label.setStyleSheet("color:#ddd;")
        lay.addWidget(eq_label)

        # 2) Gráfica
        canvas = None
        try:
            fig = Figure(figsize=(6,3.5), dpi=100)
            ax = fig.add_subplot(111)
            x = _SYM_symbols('x'); expr = _SYM_sympify(expr_text); fcall = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
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
            canvas = None

        # 3) Intervalos, errores, tolerancia + tabla
        header = QWidget(); grid = QGridLayout(header)
        grid.setContentsMargins(0,0,0,0); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(4)
        metrics_row = 0
        if rows:
            n = len(rows)
            xr = float(rows[-1][3])
            ea = float(rows[-1][4])
            residual = abs(float(rows[-1][7]))
            sgn = '+' if xr >= 0 else ''
            grid.addWidget(QLabel(f"Iteraciones: <b>{n}</b>"), metrics_row, 0)
            grid.addWidget(QLabel(f"Raíz: <b>{sgn}{xr:.6f}</b>"), metrics_row, 1)
            grid.addWidget(QLabel(f"Error (Ea): <b>{ea*100:.2f}%</b>"), metrics_row, 2)
            grid.addWidget(QLabel(f"Error raíz |f(r)|: <b>{residual:.6g}</b>"), metrics_row, 3)
            metrics_row += 1
        if epsilon_text is not None and epsilon_text != '':
            grid.addWidget(QLabel(f"Tolerancia: <b>{epsilon_text}</b>"), metrics_row, 0)
            metrics_row += 1
        for i in range(grid.count()):
            w = grid.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setStyleSheet("color:#ddd;")

        tbl = QTableWidget()
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

        lay.addWidget(header)
        lay.addWidget(tbl)


class SecantResultDialog(QDialog):
    def __init__(self, expr_text: str, x0: float, x1: float, rows, epsilon_text: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Método de la Secante — Detalles')
        self.resize(960, 640)
        self._rows = rows or []
        self._parent_app = parent

        lay = QVBoxLayout(self)
        title = QLabel('<b>Método de la Secante</b>'); lay.addWidget(title)

        eq_label = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{html.escape(expr_text)}</span>")
        eq_label.setTextFormat(Qt.RichText)
        eq_label.setStyleSheet('color:#ddd;')
        lay.addWidget(eq_label)

        header = QWidget(); grid = QGridLayout(header)
        grid.setContentsMargins(0,0,0,0); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(4)
        metrics_row = 0
        grid.addWidget(QLabel(f"x₀: <b>{x0:.6f}</b>"), metrics_row, 0)
        grid.addWidget(QLabel(f"x₁: <b>{x1:.6f}</b>"), metrics_row, 1)
        if rows:
            n = len(rows)
            c = float(rows[-1][5])
            ea = float(rows[-1][6])
            residual = float(rows[-1][7])
            grid.addWidget(QLabel(f"Iteraciones: <b>{n}</b>"), metrics_row, 2)
            metrics_row += 1
            grid.addWidget(QLabel(f"Raíz aproximada: <b>{c:.6f}</b>"), metrics_row, 0)
            grid.addWidget(QLabel(f"Error (Ea): <b>{ea*100:.2f}%</b>"), metrics_row, 1)
            grid.addWidget(QLabel(f"Error raíz |f(c)|: <b>{residual:.6g}</b>"), metrics_row, 2)
            metrics_row += 1
        if epsilon_text is not None and epsilon_text != '':
            grid.addWidget(QLabel(f"Tolerancia: <b>{epsilon_text}</b>"), metrics_row, 0)
        for i in range(grid.count()):
            w = grid.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setStyleSheet('color:#ddd;')
        lay.addWidget(header)

        # Botón para mostrar gráfica opcionalmente una vez resuelto el ejercicio
        btn_plot = QPushButton('📈 Ver gráfica')
        btn_plot.setObjectName('btnSecondary')
        btn_plot.setEnabled(bool(self._rows))
        def _show_plot():
            app = self._parent_app
            if app is None or not hasattr(app, 'mostrar_grafica_secante'):
                return
            try:
                app.mostrar_grafica_secante(self._rows)
            except Exception as e:
                QMessageBox.warning(self, 'Error al mostrar gráfica', str(e))
        btn_plot.clicked.connect(_show_plot)
        lay.addWidget(btn_plot)

        tbl = QTableWidget(); lay.addWidget(tbl)
        headers = ['iteración','x_{n-1}','x_n','f(x_{n-1})','f(x_n)','x_{n+1}','Error aprox. (%)','|f(x_{n+1})|']
        tbl.setColumnCount(len(headers)); tbl.setHorizontalHeaderLabels(headers)
        tbl.setRowCount(len(rows))
        def fmt(v):
            return ('+' if v >= 0 else '') + f"{v:.{SECANT_DECIMALS}f}"
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                text = fmt(v) if isinstance(v, (float, np.floating)) else str(v)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if r == len(rows) - 1:
                    item.setBackground(QColor(0, 80, 120, 160))
                    item.setForeground(QColor('#ffffff'))
                    font = item.font(); font.setBold(True); item.setFont(font)
                tbl.setItem(r, c, item)
        tbl.resizeColumnsToContents()
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet(f"QTableWidget::item{{font-family:{MATH_FONT_STACK}; padding:4px;}}")


class NewtonResultDialog(QDialog):
    def __init__(self, expr_text: str, deriv_text: str, rows, epsilon_text: str | None, manual_derivative: bool, geogebra_url: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Método de Newton-Raphson — Detalles')
        self.resize(980, 660)
        self._geo_url = geogebra_url

        lay = QVBoxLayout(self)
        title = QLabel('<b>Método de Newton-Raphson</b>'); lay.addWidget(title)

        eq_label = QLabel(f"f(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{html.escape(expr_text)}</span>")
        eq_label.setTextFormat(Qt.RichText); eq_label.setStyleSheet('color:#ddd;')
        lay.addWidget(eq_label)
        deriv_label = QLabel(f"f'(x) = <span style=\"font-family:{MATH_FONT_STACK}\">{html.escape(deriv_text)}</span> ({'manual' if manual_derivative else 'automática'})")
        deriv_label.setTextFormat(Qt.RichText); deriv_label.setStyleSheet('color:#bbb;')
        lay.addWidget(deriv_label)

        canvas = None
        try:
            fig = Figure(figsize=(6,3.4), dpi=100)
            ax = fig.add_subplot(111)
            x = _SYM_symbols('x'); expr = _SYM_sympify(expr_text); fcall = _SYM_lambdify(x, expr, _LAMBDA_MODULES)
            xs_iter = [row[1] for row in rows] + ([rows[-1][4]] if rows else [])
            if xs_iter:
                min_x, max_x = min(xs_iter), max(xs_iter)
                span = max(1.0, abs(max_x - min_x))
                padding = span * 0.5
                xs = np.linspace(min_x - padding, max_x + padding, 400)
            else:
                xs = np.linspace(-5.0, 5.0, 400)
            ys = fcall(xs)
            ax.axhline(0, color='#666', lw=1)
            ax.plot(xs, ys, color='#4fc3f7', label='f(x)')
            if rows:
                xr = rows[-1][4]
                ax.plot([xr], [0], 'o', color='#e05d5d', label='Raíz aprox.')
                points = np.array([row[1] for row in rows], dtype=float)
                ax.scatter(points, np.zeros_like(points), color='#ffd54f', marker='x', label='Iteraciones')
            ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(frameon=False)
            canvas = FigureCanvasQTAgg(fig)
            lay.addWidget(canvas)
        except Exception:
            canvas = None

        metrics = QWidget(); grid = QGridLayout(metrics)
        grid.setContentsMargins(0,0,0,0); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(4)
        if rows:
            iterations = len(rows)
            xr = float(rows[-1][4])
            ea = float(rows[-1][5])
            residual = float(rows[-1][6])
            grid.addWidget(QLabel(f"Iteraciones: <b>{iterations}</b>"), 0, 0)
            grid.addWidget(QLabel(f"Raíz aproximada: <b>{xr:.6f}</b>"), 0, 1)
            grid.addWidget(QLabel(f"Error (Ea): <b>{ea*100:.2f}%</b>"), 1, 0)
            grid.addWidget(QLabel(f"|f(x)|: <b>{residual:.6g}</b>"), 1, 1)
        if epsilon_text:
            grid.addWidget(QLabel(f"Tolerancia: <b>{epsilon_text}</b>"), 2, 0)
        for i in range(grid.count()):
            w = grid.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setStyleSheet('color:#ddd;')
        lay.addWidget(metrics)

        link_row = QHBoxLayout()
        link_btn = QPushButton('🌐 Ver gráfica completa'); link_btn.setObjectName('btnSecondary')
        link_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        link_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(40, 40, 60, 200);
                border: 1px solid #7f5af0;
                color: #ffffff;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #7f5af0;
            }
        """)
        link_row.addStretch(1)
        link_row.addWidget(link_btn)
        lay.addLayout(link_row)

        tbl = QTableWidget(); columns = ['iteración','x_n','f(x_n)','f\'(x_n)','x_{n+1}','Ea','|f(x_{n+1})|']
        tbl.setColumnCount(len(columns)); tbl.setHorizontalHeaderLabels(columns)
        tbl.setRowCount(len(rows))
        def fmt(v):
            return ('+' if v >= 0 else '') + f"{v:.6f}"
        for r, row in enumerate(rows):
            data = [row[0], row[1], row[2], row[3], row[4], row[5], row[6]]
            for c, val in enumerate(data):
                text = fmt(val) if isinstance(val, (float, np.floating)) else str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tbl.setItem(r, c, item)
        tbl.resizeColumnsToContents()
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet(f"QTableWidget::item{{font-family:{MATH_FONT_STACK}; padding:4px;}}")
        lay.addWidget(tbl)

        link_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._geo_url)))


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
                background-color: #1e1e2e;
                border: 1px solid #2b2d31;
                border-radius: 18px;
            }
            QLabel#splashTitle {
                font-size: 18px;
                font-weight: 800;
                color: #ffffff;
                letter-spacing: 3px;
            }
            QLabel#splashSub   {
                font-size: 11px;
                color: #a78bfa;
            }
            QProgressBar {
                background-color: #2a2a3e;
                border: 1px solid #32324a;
                border-radius: 4px;
                height: 10px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #7f5af0, stop:1 #9b6bff);
                border-radius: 4px;
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

    # Mantener referencias para poder alternar entre WelcomeScreen y la app principal
    main_window: MatrixQtApp | None = None

    # 1) Mostrar pantalla de bienvenida
    welcome = WelcomeScreen()
    welcome.show()

    # 2) Cuando el usuario pulse ENTRAR, se mostrará el splash de carga y luego la app
    def launch_main_after_welcome():
        nonlocal main_window, welcome

        # Splash de carga
        splash = SplashScreen()
        splash.start()
        app.processEvents()

        # Construir la ventana principal si aún no existe
        if main_window is None:
            main_window = MatrixQtApp()

        # Cerrar splash con fade-out y mostrar la app
        def _show_main():
            if main_window is not None:
                main_window.showMaximized()
            splash.finish()
        QTimer.singleShot(700, _show_main)

    # Sobrescribimos el handler del botón de entrada para encadenar la transición
    def _on_enter():
        welcome.hide()
        launch_main_after_welcome()

    welcome._on_enter_clicked = _on_enter  # type: ignore[attr-defined]

    sys.exit(app.exec())


if __name__ == '__main__':
    run()
