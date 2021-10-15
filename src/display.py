from IPython import display as idisplay

# Hack to get markdown working, try removing it and see if native markdown works.
from markdown import markdown

Markdown = lambda string: idisplay.HTML(markdown(string))


class Disp:
    def __init__(self, display_fn):
        self.display = display_fn

    def md(self, md, **kwargs):
        # # Native markdown doesn't work for some reason
        # self.display(idisplay.Markdown(md), **kwargs)
        self.display(idisplay.HTML(markdown(md)))

    def latex(self, latex, **kwargs):
        self.display(idisplay.Latex(latex), **kwargs)

    def obj(self, obj, label='', **kwargs):
        self.display(idisplay.Markdown(label), obj, **kwargs)

    def code(self, f, **kwargs):
        self.obj(idisplay.Code(filename=f), **kwargs)

    def audio_file(self, f, **kwargs):
        self.obj(idisplay.Audio(f), **kwargs)

    def audio_raw(self, wav, sr, **kwargs):
        self.obj(idisplay.Audio(wav, rate=sr), **kwargs)

    def audio(self, filename=None, data=None, rate=None, **kwargs):
        if filename is None:
            self.audio_raw(data, rate, **kwargs)
        else:
            self.audio_file(filename, **kwargs)

