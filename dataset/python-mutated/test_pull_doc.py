from __future__ import annotations
import pytest
pytest
import bokeh.document as document
from bokeh.core.properties import Instance, Int, Nullable
from bokeh.model import Model
from bokeh.protocol import Protocol
proto = Protocol()

class AnotherModelInTestPullDoc(Model):
    bar = Int(1)

class SomeModelInTestPullDoc(Model):
    foo = Int(2)
    child = Nullable(Instance(Model))

class TestPullDocument:

    def _sample_doc(self):
        if False:
            while True:
                i = 10
        doc = document.Document()
        another = AnotherModelInTestPullDoc()
        doc.add_root(SomeModelInTestPullDoc(child=another))
        doc.add_root(SomeModelInTestPullDoc())
        return doc

    def test_create_req(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        proto.create('PULL-DOC-REQ')

    def test_create_reply(self) -> None:
        if False:
            while True:
                i = 10
        sample = self._sample_doc()
        proto.create('PULL-DOC-REPLY', 'fakereqid', sample)

    def test_create_reply_then_parse(self) -> None:
        if False:
            i = 10
            return i + 15
        sample = self._sample_doc()
        msg = proto.create('PULL-DOC-REPLY', 'fakereqid', sample)
        copy = document.Document()
        msg.push_to_document(copy)
        assert len(sample.roots) == 2
        assert len(copy.roots) == 2