import src.display
from src.display import Disp


class CallArgs:

    @classmethod
    def get(cls, mock, call_num=0):
        call_list = mock.call_args_list
        call_i = call_list[call_num]
        return cls(call_i[0], call_i[1])

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def assert_mocker_arg(mock, expected_mocker_name, arg_pos=0):
    call_args = CallArgs.get(mock)
    mocked_callee = call_args.args[arg_pos]
    # This is a hack, but I don't know of a better way
    # noinspection PyProtectedMember
    assert mocked_callee._extract_mock_name() == expected_mocker_name


def test_disp_obj(mocker):
    mocker.patch('src.display.idisplay')
    mock_idisplay = src.display.idisplay
    spy_idisplay_markdown = mocker.spy(mock_idisplay, 'Markdown')

    disp = Disp(mock_idisplay)
    obj, label = 'obj', 'label'
    disp.obj(obj, label)

    mock_idisplay.assert_called_once_with(mocker.ANY, obj)
    spy_idisplay_markdown.assert_called_once_with(label)


def test_disp_code(mocker):
    mock_idisplay_code = mocker.patch('src.display.idisplay.Code')
    mock_disp_obj = mocker.patch('src.display.Disp.obj')

    disp = Disp(mocker.Mock())

    fname, label = 'path/to/file', 'a label'

    disp.code(fname, label=label)

    mock_idisplay_code.assert_called_once_with(filename=fname)
    mock_disp_obj.assert_called_once_with(mocker.ANY, label=label)

    # Validate disp.obj obj arguments (since mocker.ANY is too general)
    assert_mocker_arg(mock_disp_obj, 'Code()')


def test_audio_file(mocker):
    mock_idisplay_audio = mocker.patch('src.display.idisplay.Audio')
    mock_disp_obj = mocker.patch('src.display.Disp.obj')

    disp = Disp(mocker.Mock())

    fname, label = 'path/to/file', 'a label'

    disp.audio_file(fname, label=label)

    mock_idisplay_audio.assert_called_once_with(fname)
    mock_disp_obj.assert_called_once_with(mocker.ANY, label=label)

    # Validate disp.obj obj arguments (since mocker.ANY is too general)
    assert_mocker_arg(mock_disp_obj, 'Audio()')


def test_audio_raw(mocker):
    mock_idisplay_audio = mocker.patch('src.display.idisplay.Audio')
    mock_disp_obj = mocker.patch('src.display.Disp.obj')

    disp = Disp(mocker.Mock())

    wav, sr, label = 'wav binary data', 14400, 'a label'

    disp.audio_raw(wav, sr, label=label)

    mock_idisplay_audio.assert_called_once_with(wav, rate=sr)
    mock_disp_obj.assert_called_once_with(mocker.ANY, label=label)

    # Validate disp.obj obj arguments (since mocker.ANY is too general)
    assert_mocker_arg(mock_disp_obj, 'Audio()')


def test_audio_with_filename(mocker):
    mock_disp_audio_file = mocker.patch('src.display.Disp.audio_file')

    disp = Disp(mocker.Mock())

    fname, label = 'path/to/file', 'a label'

    disp.audio(filename=fname, label=label)

    mock_disp_audio_file.assert_called_once_with(mocker.ANY, label=label)

    audio_file_args, audio_file_kwargs = mock_disp_audio_file.call_args_list[0]

    assert fname == audio_file_args[0]
    assert label == audio_file_kwargs['label']


def test_audio_without_filename(mocker):
    mock_disp_audio_raw = mocker.patch('src.display.Disp.audio_raw')

    disp = Disp(mocker.Mock())

    data, sr, label = 'wav binary data', 14400, 'a label'

    disp.audio(data=data, rate=sr, label=label)

    mock_disp_audio_raw.assert_called_once_with(data, sr, label=label)

    audio_raw_args, audio_raw_kwargs = mock_disp_audio_raw.call_args_list[0]

    assert label == audio_raw_kwargs['label']
