import os
from collections import namedtuple

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

_ProjectStruct = namedtuple('Project', ['path'])
_PathStruct = namedtuple('Path', _project_paths.keys())

path = _PathStruct(**_project_paths)
Project = _ProjectStruct(path=path)
