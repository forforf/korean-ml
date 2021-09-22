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
_template = os.path.join(_root, 'template')
_src = os.path.join(_root, 'src'),
_kss_event_rms = os.path.join(_root, 'kss-event-detection-rms')

_project_paths = {
    'ROOT': _root,
    'DATA': _data,
    'FONT': '/Users/dev/Fonts/Noto_Sans_KR/NotoSansKR-Regular.otf',
    'MODEL': os.path.join(_data, 'model'),
    'KSS': os.path.join(_data, 'korean-single-speaker', 'kss'),
    'KSS_EVENT_RMS': _kss_event_rms,
    'KSS_EVENT_RMS_MODEL': os.path.join(_kss_event_rms, 'models'),
    'KSSCSV':os.path.join(_data, 'korean-single-speaker', 'kss-csv'),
    'TRANSCRIPT': os.path.join(_data, 'korean-single-speaker', 'transcript.v.1.4.txt'),
    'TEMPLATE': _template,
    'NB_TEMPLATE': os.path.join(_template, 'notebook_templates'),
    'TEXTGRID': os.path.join(_data, 'korean-single-speaker', 'kss'),
    'SRC': _src
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