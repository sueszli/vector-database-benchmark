from typing import TypeVar, Generic, List, NamedTuple, Callable
from six.moves import range
from trashcli.lib.my_input import Input
from trashcli.restore.index import Sequence
from trashcli.restore.output import Output
from trashcli.restore.output_event import Die, Exiting, OutputEvent, Quit
from trashcli.restore.range import Range
from trashcli.restore.sequences import Sequences
from trashcli.restore.single import Single
from trashcli.restore.trashed_file import TrashedFile
SelectedFiles = NamedTuple('SelectedFiles', [('files_to_restore', List[TrashedFile]), ('overwrite', bool)])
Context = NamedTuple('Context', [('trashed_files', List[TrashedFile]), ('overwrite', bool)])
InputRead = NamedTuple('InputRead', [('user_input', str), ('trashed_files', List[TrashedFile]), ('overwrite', bool)])

class RestoreAskingTheUser(object):

    def __init__(self, input, restorer, output):
        if False:
            return 10
        self.input = input
        self.restorer = restorer
        self.output = output

    def read_user_input(self, args):
        if False:
            for i in range(10):
                print('nop')
        try:
            user_input = self.input.read_input('What file to restore [0..%d]: ' % (len(args.trashed_files) - 1))
        except KeyboardInterrupt:
            return Left(Quit())
        except EOFError:
            return Left(Quit())
        else:
            if user_input == '':
                return Left(Exiting())
            else:
                return Right(InputRead(user_input, args.trashed_files, args.overwrite))

    def restore_asking_the_user(self, trashed_files, overwrite):
        if False:
            print('Hello World!')
        input = Right(Context(trashed_files, overwrite))
        compose(input, [self.read_user_input, trashed_files_to_restore, self.restore_selected_files]).on_error(lambda error: self.output.append_event(error))

    def restore_selected_files(self, selected_files):
        if False:
            while True:
                i = 10
        try:
            for trashed_file in selected_files.files_to_restore:
                self.restorer.restore_trashed_file(trashed_file, selected_files.overwrite)
            return Right(None)
        except IOError as e:
            return Left(Die(e))
Error = TypeVar('Error')
Value = TypeVar('Value')

def compose(input, funcs):
    if False:
        while True:
            i = 10
    for f in funcs:
        input = input.apply(f)
    return input

class Either(Generic[Error, Value]):

    def apply(self, f):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class Left(Either, Generic[Error, Value]):

    def __init__(self, error):
        if False:
            print('Hello World!')
        self.error = error

    def apply(self, f):
        if False:
            for i in range(10):
                print('nop')
        return self

    def on_error(self, f):
        if False:
            return 10
        return f(self.error)
OutputValue = TypeVar('OutputValue')

class Right(Either[Error, Value]):

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def apply(self, f):
        if False:
            print('Hello World!')
        return f(self.value)

    def on_error(self, f):
        if False:
            while True:
                i = 10
        return self

def trashed_files_to_restore(input_read):
    if False:
        for i in range(10):
            print('nop')
    try:
        sequences = parse_indexes(input_read.user_input, len(input_read.trashed_files))
        file_to_restore = [input_read.trashed_files[index] for index in sequences.all_indexes()]
        selected_files = SelectedFiles(file_to_restore, input_read.overwrite)
        return Right(selected_files)
    except InvalidEntry as e:
        return Left(Die('Invalid entry: %s' % e))

class InvalidEntry(Exception):
    pass

def parse_indexes(user_input, len_trashed_files):
    if False:
        while True:
            i = 10
    indexes = user_input.split(',')
    sequences = []
    for index in indexes:
        if '-' in index:
            (first, last) = index.split('-', 2)
            if first == '' or last == '':
                raise InvalidEntry('open interval: %s' % index)
            split = list(map(parse_int_index, (first, last)))
            sequences.append(Range(split[0], split[1]))
        else:
            int_index = parse_int_index(index)
            sequences.append(Single(int_index))
    result = Sequences(sequences)
    acceptable_values = range(0, len_trashed_files)
    for index in result.all_indexes():
        if not index in acceptable_values:
            raise InvalidEntry('out of range %s..%s: %s' % (acceptable_values[0], acceptable_values[-1], index))
    return result

def parse_int_index(text):
    if False:
        return 10
    try:
        return int(text)
    except ValueError:
        raise InvalidEntry('not an index: %s' % text)