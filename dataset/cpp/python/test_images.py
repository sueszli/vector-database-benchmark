# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

import os

import pytest

from openai import OpenAI, AsyncOpenAI
from tests.utils import assert_matches_type
from openai.types import ImagesResponse
from openai._client import OpenAI, AsyncOpenAI

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")
api_key = "My API Key"


class TestImages:
    strict_client = OpenAI(base_url=base_url, api_key=api_key, _strict_response_validation=True)
    loose_client = OpenAI(base_url=base_url, api_key=api_key, _strict_response_validation=False)
    parametrize = pytest.mark.parametrize("client", [strict_client, loose_client], ids=["strict", "loose"])

    @parametrize
    def test_method_create_variation(self, client: OpenAI) -> None:
        image = client.images.create_variation(
            image=b"raw file contents",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_method_create_variation_with_all_params(self, client: OpenAI) -> None:
        image = client.images.create_variation(
            image=b"raw file contents",
            model="dall-e-2",
            n=1,
            response_format="url",
            size="1024x1024",
            user="user-1234",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_raw_response_create_variation(self, client: OpenAI) -> None:
        response = client.images.with_raw_response.create_variation(
            image=b"raw file contents",
        )
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        image = response.parse()
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_method_edit(self, client: OpenAI) -> None:
        image = client.images.edit(
            image=b"raw file contents",
            prompt="A cute baby sea otter wearing a beret",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_method_edit_with_all_params(self, client: OpenAI) -> None:
        image = client.images.edit(
            image=b"raw file contents",
            prompt="A cute baby sea otter wearing a beret",
            mask=b"raw file contents",
            model="dall-e-2",
            n=1,
            response_format="url",
            size="1024x1024",
            user="user-1234",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_raw_response_edit(self, client: OpenAI) -> None:
        response = client.images.with_raw_response.edit(
            image=b"raw file contents",
            prompt="A cute baby sea otter wearing a beret",
        )
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        image = response.parse()
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_method_generate(self, client: OpenAI) -> None:
        image = client.images.generate(
            prompt="A cute baby sea otter",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_method_generate_with_all_params(self, client: OpenAI) -> None:
        image = client.images.generate(
            prompt="A cute baby sea otter",
            model="dall-e-3",
            n=1,
            quality="standard",
            response_format="url",
            size="1024x1024",
            style="vivid",
            user="user-1234",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    def test_raw_response_generate(self, client: OpenAI) -> None:
        response = client.images.with_raw_response.generate(
            prompt="A cute baby sea otter",
        )
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        image = response.parse()
        assert_matches_type(ImagesResponse, image, path=["response"])


class TestAsyncImages:
    strict_client = AsyncOpenAI(base_url=base_url, api_key=api_key, _strict_response_validation=True)
    loose_client = AsyncOpenAI(base_url=base_url, api_key=api_key, _strict_response_validation=False)
    parametrize = pytest.mark.parametrize("client", [strict_client, loose_client], ids=["strict", "loose"])

    @parametrize
    async def test_method_create_variation(self, client: AsyncOpenAI) -> None:
        image = await client.images.create_variation(
            image=b"raw file contents",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_method_create_variation_with_all_params(self, client: AsyncOpenAI) -> None:
        image = await client.images.create_variation(
            image=b"raw file contents",
            model="dall-e-2",
            n=1,
            response_format="url",
            size="1024x1024",
            user="user-1234",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_raw_response_create_variation(self, client: AsyncOpenAI) -> None:
        response = await client.images.with_raw_response.create_variation(
            image=b"raw file contents",
        )
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        image = response.parse()
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_method_edit(self, client: AsyncOpenAI) -> None:
        image = await client.images.edit(
            image=b"raw file contents",
            prompt="A cute baby sea otter wearing a beret",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_method_edit_with_all_params(self, client: AsyncOpenAI) -> None:
        image = await client.images.edit(
            image=b"raw file contents",
            prompt="A cute baby sea otter wearing a beret",
            mask=b"raw file contents",
            model="dall-e-2",
            n=1,
            response_format="url",
            size="1024x1024",
            user="user-1234",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_raw_response_edit(self, client: AsyncOpenAI) -> None:
        response = await client.images.with_raw_response.edit(
            image=b"raw file contents",
            prompt="A cute baby sea otter wearing a beret",
        )
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        image = response.parse()
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_method_generate(self, client: AsyncOpenAI) -> None:
        image = await client.images.generate(
            prompt="A cute baby sea otter",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_method_generate_with_all_params(self, client: AsyncOpenAI) -> None:
        image = await client.images.generate(
            prompt="A cute baby sea otter",
            model="dall-e-3",
            n=1,
            quality="standard",
            response_format="url",
            size="1024x1024",
            style="vivid",
            user="user-1234",
        )
        assert_matches_type(ImagesResponse, image, path=["response"])

    @parametrize
    async def test_raw_response_generate(self, client: AsyncOpenAI) -> None:
        response = await client.images.with_raw_response.generate(
            prompt="A cute baby sea otter",
        )
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        image = response.parse()
        assert_matches_type(ImagesResponse, image, path=["response"])
