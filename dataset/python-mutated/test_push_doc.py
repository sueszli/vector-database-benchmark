from __future__ import annotations
import pytest
pytest
import bokeh.document as document
from bokeh.core.properties import Instance, Int, Nullable
from bokeh.model import Model
from bokeh.protocol import Protocol
proto = Protocol()

class AnotherModelInTestPushDoc(Model):
    bar = Int(1)

class SomeModelInTestPushDoc(Model):
    foo = Int(2)
    child = Nullable(Instance(Model))

class TestPushDocument:

    def _sample_doc(self):
        if False:
            return 10
        doc = document.Document()
        another = AnotherModelInTestPushDoc()
        doc.add_root(SomeModelInTestPushDoc(child=another))
        doc.add_root(SomeModelInTestPushDoc())
        return doc

    def test_create(self) -> None:
        if False:
            return 10
        sample = self._sample_doc()
        proto.create('PUSH-DOC', sample)

    def test_create_then_parse(self) -> None:
        if False:
            while True:
                i = 10
        sample = self._sample_doc()
        msg = proto.create('PUSH-DOC', sample)
        copy = document.Document()
        msg.push_to_document(copy)
        assert len(sample.roots) == 2
        assert len(copy.roots) == 2