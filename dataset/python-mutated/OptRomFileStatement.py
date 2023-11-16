from __future__ import absolute_import
import Common.LongFilePathOs as os
from .GenFdsGlobalVariable import GenFdsGlobalVariable

class OptRomFileStatement:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.FileName = None
        self.FileType = None
        self.OverrideAttribs = None

    def GenFfs(self, Dict=None, IsMakefile=False):
        if False:
            while True:
                i = 10
        if Dict is None:
            Dict = {}
        if self.FileName is not None:
            self.FileName = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.FileName)
        return self.FileName