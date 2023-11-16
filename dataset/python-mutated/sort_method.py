from abc import abstractmethod
from typing import Callable, Any, Iterable
from trashcli.compat import Protocol
from trashcli.restore.args import Sort
from trashcli.restore.trashed_file import TrashedFile

def sort_files(sort, trashed_files):
    if False:
        for i in range(10):
            print('nop')
    return sorter_for(sort).sort_files(trashed_files)

class Sorter(Protocol):

    @abstractmethod
    def sort_files(self, trashed_files):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class NoSorter(Sorter):

    def sort_files(self, trashed_files):
        if False:
            i = 10
            return i + 15
        return trashed_files

class SortFunction(Sorter):

    def __init__(self, sort_func):
        if False:
            i = 10
            return i + 15
        self.sort_func = sort_func

    def sort_files(self, trashed_files):
        if False:
            while True:
                i = 10
        return sorted(trashed_files, key=self.sort_func)

def sorter_for(sort):
    if False:
        i = 10
        return i + 15
    path_ranking = lambda x: x.original_location + str(x.deletion_date)
    date_rankking = lambda x: x.deletion_date
    return {Sort.ByPath: SortFunction(path_ranking), Sort.ByDate: SortFunction(date_rankking), Sort.DoNot: NoSorter}[sort]