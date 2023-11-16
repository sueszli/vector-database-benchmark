"""
DirectNotify module: this module contains the DirectNotify class
"""
from __future__ import annotations
from panda3d.core import StreamWriter
from . import Notifier
from . import Logger

class DirectNotify:
    """
    DirectNotify class: this class contains methods for creating
    mulitple notify categories via a dictionary of Notifiers.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        '\n        DirectNotify class keeps a dictionary of Notfiers\n        '
        self.__categories: dict[str, Notifier.Notifier] = {}
        self.logger = Logger.Logger()
        self.streamWriter: StreamWriter | None = None

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Print handling routine\n        '
        return 'DirectNotify categories: %s' % self.__categories

    def getCategories(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        '\n        Return list of category dictionary keys\n        '
        return list(self.__categories.keys())

    def getCategory(self, categoryName: str) -> Notifier.Notifier | None:
        if False:
            print('Hello World!')
        'getCategory(self, string)\n        Return the category with given name if present, None otherwise\n        '
        return self.__categories.get(categoryName, None)

    def newCategory(self, categoryName: str, logger: Logger.Logger | None=None) -> Notifier.Notifier:
        if False:
            while True:
                i = 10
        'newCategory(self, string)\n        Make a new notify category named categoryName. Return new category\n        if no such category exists, else return existing category\n        '
        if categoryName not in self.__categories:
            self.__categories[categoryName] = Notifier.Notifier(categoryName, logger)
            self.setDconfigLevel(categoryName)
        notifier = self.getCategory(categoryName)
        assert notifier is not None
        return notifier

    def setDconfigLevel(self, categoryName: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Check to see if this category has a dconfig variable\n        to set the notify severity and then set that level. You cannot\n        set these until config is set.\n        '
        from panda3d.core import ConfigVariableString
        dconfigParam = 'notify-level-' + categoryName
        cvar = ConfigVariableString(dconfigParam, '')
        level = cvar.getValue()
        if not level:
            cvar2 = ConfigVariableString('default-directnotify-level', 'info')
            level = cvar2.getValue()
        if not level:
            level = 'error'
        category = self.getCategory(categoryName)
        assert category is not None, f'failed to find category: {categoryName!r}'
        if level == 'error':
            category.setWarning(False)
            category.setInfo(False)
            category.setDebug(False)
        elif level == 'warning':
            category.setWarning(True)
            category.setInfo(False)
            category.setDebug(False)
        elif level == 'info':
            category.setWarning(True)
            category.setInfo(True)
            category.setDebug(False)
        elif level == 'debug':
            category.setWarning(True)
            category.setInfo(True)
            category.setDebug(True)
        else:
            print('DirectNotify: unknown notify level: ' + str(level) + ' for category: ' + str(categoryName))

    def setDconfigLevels(self) -> None:
        if False:
            print('Hello World!')
        for categoryName in self.getCategories():
            self.setDconfigLevel(categoryName)

    def setVerbose(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for categoryName in self.getCategories():
            category = self.getCategory(categoryName)
            assert category is not None
            category.setWarning(True)
            category.setInfo(True)
            category.setDebug(True)

    def popupControls(self, tl=None):
        if False:
            i = 10
            return i + 15
        import importlib
        NotifyPanel = importlib.import_module('direct.tkpanels.NotifyPanel')
        NotifyPanel.NotifyPanel(self, tl)

    def giveNotify(self, cls) -> None:
        if False:
            print('Hello World!')
        cls.notify = self.newCategory(cls.__name__)