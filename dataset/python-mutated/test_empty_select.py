import pytest
from textual.app import App
from textual.widgets import Select
from textual.widgets.select import EmptySelectError

async def test_empty_select_is_ok_with_blanks():

    class SelectApp(App[None]):

        def compose(self):
            if False:
                i = 10
                return i + 15
            yield Select([])
    app = SelectApp()
    async with app.run_test():
        assert app.query_one(Select).is_blank()

async def test_empty_set_options_is_ok_with_blanks():

    class SelectApp(App[None]):

        def compose(self):
            if False:
                i = 10
                return i + 15
            yield Select([(str(n), n) for n in range(3)], value=0)
    app = SelectApp()
    async with app.run_test():
        select = app.query_one(Select)
        assert not select.is_blank()
        select.set_options([])
        assert select.is_blank()

async def test_empty_select_raises_exception_if_allow_blank_is_false():

    class SelectApp(App[None]):

        def compose(self):
            if False:
                for i in range(10):
                    print('nop')
            yield Select([], allow_blank=False)
    app = SelectApp()
    with pytest.raises(EmptySelectError):
        async with app.run_test():
            pass

async def test_empty_set_options_raises_exception_if_allow_blank_is_false():

    class SelectApp(App[None]):

        def compose(self):
            if False:
                i = 10
                return i + 15
            yield Select([(str(n), n) for n in range(3)], allow_blank=False)
    app = SelectApp()
    async with app.run_test():
        select = app.query_one(Select)
        assert not select.is_blank()
        with pytest.raises(EmptySelectError):
            select.set_options([])