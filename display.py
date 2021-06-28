from IPython import display as idisplay


class Disp:
    def __init__(self, display_fn):
        self.display = display_fn

    def obj(self, obj, label='', **kwargs):
        self.display(idisplay.Markdown(label), obj, **kwargs)

    def code(self, f, **kwargs):
        self.obj(idisplay.Code(filename=f), **kwargs)

    def audio_file(self, f, **kwargs):
        self.obj(idisplay.Audio(f), **kwargs)

    def audio_raw(self, wav, sr, **kwargs):
        print(kwargs)
        self.obj(idisplay.Audio(wav, rate=sr), **kwargs)

    def audio(self, filename=None, data=None, rate=None, **kwargs):
        if filename is None:
            self.audio_raw(data, rate, **kwargs)
        else:
            self.audio_file(filename, **kwargs)

