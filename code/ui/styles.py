class AppTheme:
    BACKGROUND = "#1A202C"
    PANELS = "#2D3748"
    ACCENT = "#4FD1C5"
    ACCENT_HOVER = "#38B2AC"
    TEXT_PRIMARY = "#E2E8F0"
    TEXT_SECONDARY = "#A0AEC0"
    SUCCESS = "#68D391"
    ERROR = "#FC8181"
    WARNING = "#F6E05E"
    BORDER = "#4A5568"

    GLOBAL_STYLE = f"""
        QMainWindow {{
            background-color: {BACKGROUND};
        }}
        QWidget {{
            color: {TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            font-size: 14px;
        }}
        QToolTip {{
            background-color: {PANELS};
            color: {TEXT_PRIMARY};
            border: 1px solid {ACCENT};
            border-radius: 8px;
            padding: 8px;
        }}
        QScrollBar:vertical {{
            background: {PANELS};
            width: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background: {ACCENT};
            border-radius: 6px;
            min-height: 30px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
    """

    MAIN_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {ACCENT};
            color: {BACKGROUND};
            border: none;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {ACCENT_HOVER};
        }}
        QPushButton:disabled {{
            background-color: #4A5568;
            color: #A0AEC0;
        }}
    """

    SECONDARY_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PANELS};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
        }}
        QPushButton:hover {{
            background-color: {BORDER};
            border-color: {ACCENT};
        }}
    """

    COMBOBOX_STYLE = f"""
        QComboBox {{
            background-color: {PANELS};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 8px 12px;
        }}
        QComboBox:hover {{
            border-color: {ACCENT};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QComboBox::down-arrow {{
            image: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {PANELS};
            selection-background-color: {ACCENT};
            color: {TEXT_PRIMARY};
            selection-color: {BACKGROUND};
            border: 1px solid {BORDER};
        }}
    """

    SPINBOX_STYLE = f"""
        QSpinBox {{
            background-color: {PANELS};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 4px 10px;
            color: {TEXT_PRIMARY};
        }}
        QSpinBox:hover {{
            border-color: {ACCENT};
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background: transparent;
            border: none;
            width: 16px;
        }}
        QSpinBox::up-arrow, QSpinBox::down-arrow {{
            width: 8px;
            height: 8px;
        }}
    """

    GROUPBOX_STYLE = f"""
        QGroupBox {{
            color: {TEXT_PRIMARY};
            font-weight: bold;
            font-size: 15px;
            border: 1px solid {BORDER};
            border-radius: 12px;
            margin-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 15px;
            padding: 0 5px;
            background-color: {BACKGROUND};
        }}
    """

    RADIOBUTTON_STYLE = f"QRadioButton {{ color: {TEXT_SECONDARY}; }}"
