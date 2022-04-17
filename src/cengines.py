import ctypes

# Engines used in Checkerboard (http://www.fierz.ch/checkers.htm)
engines = {name : ctypes.WinDLL(f"..\\engines\\{name}64.dll") for name in ["cakeM", "easych", "simplech"]}

class coor(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
    ]

class CBmove(ctypes.Structure):
    _fields_ = [
        ("jumps", ctypes.c_int),
        ("newpiece", ctypes.c_int),
        ("oldpiece", ctypes.c_int),
        ("from", coor),
        ("to", coor),
        ("path", coor * 12),
        ("del", coor * 12),
        ("delpiece", ctypes.c_int * 12)
    ]

for engine in engines.values():
    engine.getmove.argtypes = [
        (ctypes.c_int * 8) * 8,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_char * 1024,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(CBmove),
    ]
