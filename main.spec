# main.spec
import os
from PyInstaller.utils.hooks import get_package_paths

block_cipher = None

# Get the path of llama_cpp
_, llama_cpp_pkg_path = get_package_paths('llama_cpp')

# Define the path for llama.dll
llama_dll_path = os.path.join(llama_cpp_pkg_path, 'lib', 'llama.dll')

# Ensure the DLL exists before proceeding
if not os.path.exists(llama_dll_path):
    raise FileNotFoundError(f"llama.dll not found at {llama_dll_path}")

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        (llama_dll_path, '.')  # Place the DLL in the same folder as the .exe
    ],
    datas=[
        ('cmd.ico', '.'),  # Include icon file
        (llama_dll_path, '.')  # Ensure llama.dll is copied
    ],
    hiddenimports=[
        'llama_cpp._llama',  # Explicitly include llama_cpp dependencies
        'llama_cpp',
        'art',
        'pyperclip',
        'tqdm'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=True,  # Enable debug mode to catch errors
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX compression to prevent DLL issues
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='cmd.ico',
)
