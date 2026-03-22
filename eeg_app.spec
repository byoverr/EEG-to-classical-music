# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec для сборки EEG → Classical Music GUI в .app / .exe.

Сборка:
    pyinstaller eeg_app.spec

Результат:
    dist/EEG-Music-Analyzer/  (или dist/EEG-Music-Analyzer.app на macOS)
"""
import sys
from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(SPECPATH)

a = Analysis(
    [str(PROJECT_ROOT / 'gui' / 'app.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        # Данные проекта, необходимые во время выполнения
        (str(PROJECT_ROOT / 'src'), 'src'),
        (str(PROJECT_ROOT / 'scripts'), 'scripts'),
        (str(PROJECT_ROOT / 'data' / 'FluidR3_GM.sf2'), 'data'),
    ],
    hiddenimports=[
        # PySide6
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Проектные модули
        'src.config',
        'src.deap_loader',
        'src.eeg_preprocessing',
        'src.eeg_processing',
        'src.midi_utils',
        'src.MIDIComparator',
        'src.track_features',
        'src.emopia_loader',
        'src.maestro_loader',
        'src.html_generator',
        'src.audio_converter',
        'scripts.run_comparison',
        # Зависимости
        'numpy',
        'pandas',
        'scipy',
        'scipy.signal',
        'scipy.stats',
        'scipy.spatial',
        'sklearn',
        'sklearn.decomposition',
        'sklearn.preprocessing',
        'mido',
        'music21',
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'seaborn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'jupyter',
        'notebook',
        'ipykernel',
        'IPython',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EEG-Music-Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # GUI-приложение, без консоли
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EEG-Music-Analyzer',
)

# macOS: создаём .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='EEG-Music-Analyzer.app',
        icon=None,   # Можно указать .icns файл
        bundle_identifier='ru.leti.eeg-music-analyzer',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleDisplayName': 'EEG Music Analyzer',
            'CFBundleShortVersionString': '1.0.0',
        },
    )
