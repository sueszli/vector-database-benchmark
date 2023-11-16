from functools import partial
from PyQt6 import QtWidgets
from picard import log
from picard.album import Album
from picard.cluster import Cluster, ClusterList
from picard.script import ScriptError, ScriptParser
from picard.track import Track
from picard.util import iter_unique

class ScriptsMenu(QtWidgets.QMenu):

    def __init__(self, scripts, *args):
        if False:
            return 10
        super().__init__(*args)
        for script in scripts:
            action = self.addAction(script[1])
            action.triggered.connect(partial(self._run_script, script))

    def _run_script(self, script):
        if False:
            i = 10
            return i + 15
        s_name = script[1]
        s_text = script[3]
        parser = ScriptParser()
        for obj in self._iter_unique_metadata_objects():
            try:
                parser.eval(s_text, obj.metadata)
                obj.update()
            except ScriptError as e:
                log.exception('Error running tagger script "%s" on object %r', s_name, obj)
                msg = N_('Script error in "%(script)s": %(message)s')
                mparms = {'script': s_name, 'message': str(e)}
                self.tagger.window.set_statusbar_message(msg, mparms)

    def _iter_unique_metadata_objects(self):
        if False:
            for i in range(10):
                print('nop')
        return iter_unique(self._iter_metadata_objects(self.tagger.window.selected_objects))

    def _iter_metadata_objects(self, objs):
        if False:
            print('Hello World!')
        for obj in objs:
            if hasattr(obj, 'metadata') and (not getattr(obj, 'special', False)):
                yield obj
            if isinstance(obj, Cluster) or isinstance(obj, Track):
                yield from self._iter_metadata_objects(obj.iterfiles())
            elif isinstance(obj, ClusterList):
                yield from self._iter_metadata_objects(obj)
            elif isinstance(obj, Album):
                yield from self._iter_metadata_objects(obj.tracks)
                yield from self._iter_metadata_objects(obj.unmatched_files.iterfiles())