"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""
from TDStoreTools import StorageManager, DependDict, DependList
import TDFunctions as TDF

class Highscores:
    """
	Functionality for storing & recalling highscores on a per round basis in the SystemFailed tracker/trackslot/score component  
	"""

    def __init__(self, ownerComp):
        if False:
            while True:
                i = 10
        self.ownerComp = ownerComp
        self.list = op('score/highscores')
        self.Highscoretotal = 0

    def CaptureHighscore(self, ident):
        if False:
            while True:
                i = 10
        ident = str(ident)
        new = int(self.ownerComp.par.Score.eval())
        prow = self.list.row(ident)
        if prow:
            prev = int(prow[1])
        else:
            self.list.appendRow([ident, 0, 0])
            prev = -1
        rated = new - 50
        self.list.replaceRow(ident, [ident, new, rated])
        self.ownerComp.par.Newhighscore.val = new
        self.UpdateHighscoretotal()

    def QueryHighscore(self, ident):
        if False:
            while True:
                i = 10
        prow = self.list.row(ident)
        if not prow:
            return [ident, -1, 0]
        else:
            return [prow[0], prow[1], prow[2]]

    def ClearHighscore(self):
        if False:
            while True:
                i = 10
        self.list.clear(keepFirstRow=True)
        self.ownerComp.par.Highscore.val = 0
        self.ownerComp.par.Newhighscore = 0

    def UpdateHighscoretotal(self):
        if False:
            i = 10
            return i + 15
        tmp = 0
        scores = self.list.col('highscore')
        scores.pop(0)
        for c in scores:
            tmp += int(c.val)
        self.Highscoretotal = tmp
        self.ownerComp.par.Highscore.val = tmp