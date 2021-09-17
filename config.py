import os
from collections import namedtuple

import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt

from src.display import Disp


def _make_namedtuple(name, struct_dict):
    _cls = namedtuple(name, struct_dict.keys())
    return _cls(**struct_dict)


_root = os.path.dirname(os.path.abspath(__file__))
_data = os.path.join(_root, 'data')

_project_paths = {
    'ROOT': _root,
    'DATA': _data,
    'FONT': '/Users/dev/Fonts/Noto_Sans_KR/NotoSansKR-Regular.otf',
    'MODEL': os.path.join(_data, 'model'),
    'KSS': os.path.join(_data, 'korean-single-speaker', 'kss'),
    'KSSCSV':os.path.join(_data, 'korean-single-speaker', 'kss-csv'),
    'TRANSCRIPT': os.path.join(_data, 'korean-single-speaker', 'transcript.v.1.4.txt'),
    'TEMPLATE': os.path.join(_root, 'template'),
    'TEXTGRID': os.path.join(_data, 'korean-single-speaker', 'kss')
}

path = _make_namedtuple('Path', _project_paths)


def setup_font():
    return mfm.FontProperties(fname=path.FONT)


def setup_plot():
    plt.style.use('dark_background')
    return plt


def setup_display():
    return Disp(display)


_project = {
    'path': path,
    'setup_display': setup_display,
    'setup_font': setup_font,
    'setup_plot': setup_plot
}

Project = _make_namedtuple('Project', _project)