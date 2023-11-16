from typing import Type, Union

import pytest
from pytest_mock import MockerFixture

from litestar.datastructures import UploadFile
from litestar.datastructures.multi_dicts import FormMultiDict, ImmutableMultiDict, MultiDict


@pytest.mark.parametrize("multi_class", [MultiDict, ImmutableMultiDict])
def test_multi_to_dict(multi_class: Type[Union[MultiDict, ImmutableMultiDict]]) -> None:
    multi = multi_class([("key", "value"), ("key", "value2"), ("key2", "value3")])

    assert multi.dict() == {"key": ["value", "value2"], "key2": ["value3"]}


@pytest.mark.parametrize("multi_class", [MultiDict, ImmutableMultiDict])
def test_multi_multi_items(multi_class: Type[Union[MultiDict, ImmutableMultiDict]]) -> None:
    data = [("key", "value"), ("key", "value2"), ("key2", "value3")]
    multi = multi_class(data)

    assert sorted(multi.multi_items()) == sorted(data)


def test_multi_dict_as_immutable() -> None:
    data = [("key", "value"), ("key", "value2"), ("key2", "value3")]
    multi = MultiDict[str](data)  # pyright: ignore
    assert multi.immutable().dict() == ImmutableMultiDict(data).dict()


def test_immutable_multi_dict_as_mutable() -> None:
    data = [("key", "value"), ("key", "value2"), ("key2", "value3")]
    multi = ImmutableMultiDict[str](data)  # pyright: ignore
    assert multi.mutable_copy().dict() == MultiDict(data).dict()


async def test_form_multi_dict_close(mocker: MockerFixture) -> None:
    close = mocker.patch("litestar.datastructures.multi_dicts.UploadFile.close")

    multi = FormMultiDict(
        [
            ("foo", UploadFile(filename="foo", content_type="text/plain")),
            ("bar", UploadFile(filename="foo", content_type="text/plain")),
        ]
    )

    await multi.close()

    assert close.call_count == 2
