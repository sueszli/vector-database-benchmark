import core.editor

class INIDoc(core.editor.EditorDocument):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        core.editor.EditorDocument.__init__(self)
        self._iniobj = None

class INIView(core.editor.EditorView):
    pass