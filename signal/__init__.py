"""
SIGNAL — Substance Intelligence through Grounded Narrative Analysis of Language

Compatibility note: This package name shadows Python's stdlib 'signal' module.
We forward ALL stdlib signal attributes (both C-level from '_signal' and
Python-level from signal.py: Signals enum, Handlers enum, etc.) so that
anyio, torch, multiprocessing, and other libraries continue to work correctly.
"""
import os as _os
import sys as _sys
import importlib.util as _util

# Load stdlib signal.py by absolute path (avoids 'import signal' recursion).
_stdlib_lib = _os.path.dirname(_os.__file__)  # e.g. /Library/.../python3.11
_signal_py = _os.path.join(_stdlib_lib, "signal.py")

if _os.path.exists(_signal_py):
    _spec = _util.spec_from_file_location("_signal_stdlib_wrapper", _signal_py)
    _mod = _util.module_from_spec(_spec)
    # Register under a different name so exec_module doesn't recurse into us
    _sys.modules["_signal_stdlib_wrapper"] = _mod
    try:
        _spec.loader.exec_module(_mod)
        _g = globals()
        for _attr in dir(_mod):
            if not _attr.startswith("__"):
                _g[_attr] = getattr(_mod, _attr)
    except Exception:
        # Fallback: at minimum forward C-level constants from _signal
        try:
            import _signal as _c
            for _attr in dir(_c):
                if not _attr.startswith("__"):
                    globals()[_attr] = getattr(_c, _attr)
        except Exception:
            pass
    finally:
        _sys.modules.pop("_signal_stdlib_wrapper", None)
