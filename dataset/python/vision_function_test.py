# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import flask
from google.cloud import vision
import pytest

import vision_function


# Create a fake "app" for generating test request contexts.
@pytest.fixture(scope="module")
def app() -> flask.Flask:
    return flask.Flask(__name__)


@mock.patch("vision_function.urllib.request")
@mock.patch("vision_function.vision")
def test_vision_function(
    mock_vision: object, mock_request: object, app: flask.Flask
) -> None:
    mock_request.urlopen = mock.Mock(read=mock.Mock(return_value=b"filedata"))
    label_detection_mock = mock.Mock(
        side_effect=[
            vision.AnnotateImageResponse(
                {"label_annotations": [{"description": "apple"}]}
            ),
            vision.AnnotateImageResponse(
                {"label_annotations": [{"description": "banana"}]}
            ),
        ]
    )
    mock_vision.ImageAnnotatorClient = mock.Mock(
        return_value=mock.Mock(label_detection=label_detection_mock)
    )
    mock_vision.AnnotateImageResponse = vision.AnnotateImageResponse
    with app.test_request_context(
        json={
            "calls": [
                ["https://storage.googleapis.com/bucket/apple"],
                ["https://storage.googleapis.com/bucket/banana"],
            ]
        }
    ):
        response = vision_function.label_detection(flask.request)
        assert response.status_code == 200
        assert len(response.get_json()["replies"]) == 2
        assert "apple" in str(response.get_json()["replies"][0])
        assert "banana" in str(response.get_json()["replies"][1])


@mock.patch("vision_function.urllib.request")
@mock.patch("vision_function.vision")
def test_vision_function_error(
    mock_vision: object, mock_request: object, app: flask.Flask
) -> None:
    mock_request.urlopen = mock.Mock(read=mock.Mock(return_value=b"filedata"))
    label_detection_mock = mock.Mock(side_effect=Exception("API error"))
    mock_vision.ImageAnnotatorClient = mock.Mock(
        return_value=mock.Mock(label_detection=label_detection_mock)
    )
    with app.test_request_context(
        json={
            "calls": [
                ["https://storage.googleapis.com/bucket/apple"],
                ["https://storage.googleapis.com/bucket/banana"],
            ]
        }
    ):
        response = vision_function.label_detection(flask.request)
        assert response.status_code == 400
        assert "API error" in str(response.get_data())
