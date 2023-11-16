import os
from robot.errors import DataError
from robot.output import LOGGER
from robot.utils import getshortdoc
from .arguments import EmbeddedArguments, UserKeywordArgumentParser
from .handlerstore import HandlerStore
from .userkeywordrunner import UserKeywordRunner, EmbeddedArgumentsRunner
from .usererrorhandler import UserErrorHandler

class UserLibrary:

    def __init__(self, resource, resource_file=True):
        if False:
            while True:
                i = 10
        source = resource.source
        basename = os.path.basename(source) if source else None
        self.name = os.path.splitext(basename)[0] if resource_file else None
        self.doc = resource.doc
        self.handlers = HandlerStore()
        self.source = source
        for kw in resource.keywords:
            try:
                handler = self._create_handler(kw)
            except DataError as error:
                handler = UserErrorHandler(error, kw.name, self.name, source, kw.lineno)
                self._log_creating_failed(handler, error)
            embedded = isinstance(handler, EmbeddedArgumentsHandler)
            try:
                self.handlers.add(handler, embedded)
            except DataError as error:
                self._log_creating_failed(handler, error)

    def _create_handler(self, kw):
        if False:
            return 10
        if kw.error:
            raise DataError(kw.error)
        if not kw.body:
            raise DataError('User keyword cannot be empty.')
        if not kw.name:
            raise DataError('User keyword name cannot be empty.')
        embedded = EmbeddedArguments.from_name(kw.name)
        if not embedded:
            return UserKeywordHandler(kw, self.name)
        return EmbeddedArgumentsHandler(kw, self.name, embedded)

    def _log_creating_failed(self, handler, error):
        if False:
            i = 10
            return i + 15
        LOGGER.error(f"Error in file '{self.source}' on line {handler.lineno}: Creating keyword '{handler.name}' failed: {error.message}")

    def handlers_for(self, name):
        if False:
            while True:
                i = 10
        return self.handlers.get_handlers(name)

class UserKeywordHandler:
    supports_embedded_args = False

    def __init__(self, keyword, owner):
        if False:
            while True:
                i = 10
        self.name = keyword.name
        self.owner = owner
        self.doc = keyword.doc
        self.source = keyword.source
        self.lineno = keyword.lineno
        self.tags = keyword.tags
        self.arguments = UserKeywordArgumentParser().parse(tuple(keyword.args), self.full_name)
        self.timeout = keyword.timeout
        self.body = keyword.body
        self.setup = keyword.setup if keyword.has_setup else None
        self.teardown = keyword.teardown if keyword.has_teardown else None

    @property
    def full_name(self):
        if False:
            return 10
        return f'{self.owner}.{self.name}' if self.owner else self.name

    @property
    def short_doc(self):
        if False:
            for i in range(10):
                print('nop')
        return getshortdoc(self.doc)

    @property
    def private(self):
        if False:
            i = 10
            return i + 15
        return bool(self.tags and self.tags.robot('private'))

    def create_runner(self, name, languages=None):
        if False:
            i = 10
            return i + 15
        return UserKeywordRunner(self)

class EmbeddedArgumentsHandler(UserKeywordHandler):
    supports_embedded_args = True

    def __init__(self, keyword, owner, embedded):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(keyword, owner)
        self.embedded = embedded

    def matches(self, name):
        if False:
            while True:
                i = 10
        return self.embedded.match(name) is not None

    def create_runner(self, name, languages=None):
        if False:
            for i in range(10):
                print('nop')
        return EmbeddedArgumentsRunner(self, name)