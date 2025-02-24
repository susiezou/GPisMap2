""" some utility for call C++ code"""
# Adopted from
# https://github.com/geek-ai/MAgent/blob/master/python/magent/c_lib.py

from __future__ import absolute_import

import os
import ctypes
import platform
import multiprocessing


def _load_lib():
    """ Load library in build/lib. """
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, os.path.dirname(cur_path))
    lib_path = os.path.join(lib_path, os.path.dirname(lib_path))

    if platform.system() == 'Linux':
        path_to_so_file = os.path.join(lib_path,'build/libgpismap.so')
    elif platform.system() == 'Windows':
        path_to_so_file = os.path.join(lib_path,'build/gpismap_Dll.dll')    # release version
    else:
        raise BaseException("unsupported system: " + platform.system())

    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


def as_float_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


_LIB = _load_lib()
