from __future__ import annotations
import pytest
pytest
import logging
from bokeh.document import Document
import bokeh.application.handlers.document_lifecycle as bahd

class MockSessionContext:

    def __init__(self, doc: Document) -> None:
        if False:
            return 10
        self._document = doc
        self.status = None
        self.counter = 0

class Test_DocumentLifecycleHandler:

    def test_document_bad_on_session_destroyed_signature(self) -> None:
        if False:
            return 10
        doc = Document()

        def destroy(a, b):
            if False:
                return 10
            pass
        with pytest.raises(ValueError):
            doc.on_session_destroyed(destroy)

    async def test_document_on_session_destroyed(self) -> None:
        doc = Document()
        handler = bahd.DocumentLifecycleHandler()

        def destroy(session_context):
            if False:
                while True:
                    i = 10
            assert doc is session_context._document
            session_context.status = 'Destroyed'
        doc.on_session_destroyed(destroy)
        session_context = MockSessionContext(doc)
        await handler.on_session_destroyed(session_context)
        assert session_context.status == 'Destroyed'
        assert set(session_context._document.session_destroyed_callbacks) == set()

    async def test_document_on_session_destroyed_calls_multiple(self) -> None:
        doc = Document()

        def increment(session_context):
            if False:
                while True:
                    i = 10
            session_context.counter += 1
        doc.on_session_destroyed(increment)

        def increment_by_two(session_context):
            if False:
                print('Hello World!')
            session_context.counter += 2
        doc.on_session_destroyed(increment_by_two)
        handler = bahd.DocumentLifecycleHandler()
        session_context = MockSessionContext(doc)
        await handler.on_session_destroyed(session_context)
        assert session_context.counter == 3, 'DocumentLifecycleHandler did not call all callbacks'

    async def test_document_on_session_destroyed_exceptions(self, caplog: pytest.LogCaptureFixture) -> None:
        doc = Document()

        def blowup(session_context):
            if False:
                return 10
            raise ValueError('boom!')
        doc.on_session_destroyed(blowup)
        handler = bahd.DocumentLifecycleHandler()
        session_context = MockSessionContext(doc)
        with caplog.at_level(logging.WARN):
            await handler.on_session_destroyed(session_context)
            assert len(caplog.records) == 1
            assert 'boom!' in caplog.text