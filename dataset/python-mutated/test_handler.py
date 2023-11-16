from __future__ import annotations
import pytest
pytest
import bokeh.application.handlers.handler as bahh

class Test_Handler:

    def test_create(self) -> None:
        if False:
            while True:
                i = 10
        h = bahh.Handler()
        assert h.failed is False
        assert h.url_path() is None
        assert h.static_path() is None
        assert h.error is None
        assert h.error_detail is None

    def test_modify_document_abstract(self) -> None:
        if False:
            return 10
        h = bahh.Handler()
        with pytest.raises(NotImplementedError):
            h.modify_document('doc')

    def test_default_server_hooks_return_none(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        h = bahh.Handler()
        assert h.on_server_loaded('context') is None
        assert h.on_server_unloaded('context') is None

    async def test_default_sesssion_hooks_return_none(self) -> None:
        h = bahh.Handler()
        assert await h.on_session_created('context') is None
        assert await h.on_session_destroyed('context') is None

    def test_static_path(self) -> None:
        if False:
            return 10
        h = bahh.Handler()
        assert h.static_path() is None
        h._static = 'path'
        assert h.static_path() == 'path'
        h._failed = True
        assert h.static_path() is None

    def test_process_request(self) -> None:
        if False:
            i = 10
            return i + 15
        h = bahh.Handler()
        assert h.process_request('request') == {}