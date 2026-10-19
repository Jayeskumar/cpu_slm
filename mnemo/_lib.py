"""Locate (and if necessary build) the C engine and declare its signatures."""
import ctypes
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAMES = {"darwin": "libmnemo.dylib", "win32": "mnemo.dll"}
LIBPATH = os.path.join(ROOT, "build", _NAMES.get(sys.platform, "libmnemo.so"))


class Config(ctypes.Structure):
    _fields_ = [
        ("table_bits", ctypes.c_int),
        ("hist_bits", ctypes.c_int),
        ("match_bits", ctypes.c_int),
        ("class_bits", ctypes.c_int),
        ("lr", ctypes.c_float),
        ("lr0", ctypes.c_float),
        ("ctx_mask", ctypes.c_uint32),
        ("lex_order", ctypes.c_int),
        ("lex_spaced", ctypes.c_int),
    ]


def _build():
    try:
        subprocess.run(["make", "-C", ROOT, "build/libmnemo.so"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"could not build the C engine ({e}); run `make` in {ROOT}") from e


def load():
    if not os.path.exists(LIBPATH):
        _build()
    lib = ctypes.CDLL(LIBPATH)
    P, D, I, U32, U64, SZ = (ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_uint32,
                            ctypes.c_uint64, ctypes.c_size_t)
    BUF = ctypes.c_char_p
    sig = {
        "mn_default_config": (None, [ctypes.POINTER(Config)]),
        "mn_new": (P, [ctypes.POINTER(Config)]),
        "mn_free": (None, [P]),
        "mn_memory": (SZ, [P]),
        "mn_set_learning": (None, [P, I]),
        "mn_get_learning": (I, [P]),
        "mn_observe": (D, [P, BUF, SZ]),
        "mn_observe_byte": (D, [P, I]),
        "mn_next_byte_dist": (None, [P, ctypes.POINTER(D)]),
        "mn_word_hash": (U32, [BUF, SZ]),
        "mn_set_class": (None, [P, U32, U32]),
        "mn_get_class": (U32, [P, U32]),
        "mn_save": (I, [P, BUF]),
        "mn_load": (P, [BUF]),
        "mn_snapshot": (P, [P]),
        "mn_restore": (None, [P, P]),
        "mn_generate": (SZ, [P, ctypes.c_char_p, SZ, D, U64, I]),
        "mn_lexicon": (P, [P]),
        "mn_num_inputs": (I, [P]),
        "lex_num_words": (U32, [P]),
        "lex_word": (ctypes.c_void_p, [P, U32, ctypes.POINTER(I)]),
        "lex_count": (U32, [P, U32]),
        "lex_find": (ctypes.c_int64, [P, BUF, I]),
        "lex_top_next": (I, [P, ctypes.POINTER(U32), ctypes.POINTER(ctypes.c_float), I]),
        "lex_state": (I, [P]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args
    return lib


LIB = None


def lib():
    global LIB
    if LIB is None:
        LIB = load()
    return LIB
