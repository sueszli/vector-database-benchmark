# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from pathlib import Path

import testslide

from marshmallow import ValidationError

from ... import error

from .. import code_navigation_request, protocol as lsp


class CodeNavigationRequestsTest(testslide.TestCase):
    def test_serialize_request(self) -> None:
        hover_request = code_navigation_request.HoverRequest(
            path="/a/b.py",
            client_id="foo",
            position=lsp.PyrePosition(line=1, character=2),
        )
        self.assertEqual(
            hover_request.to_json(),
            [
                "Hover",
                {
                    "path": "/a/b.py",
                    "client_id": "foo",
                    "position": {"line": 1, "column": 2},
                },
            ],
        )

        hover_request = code_navigation_request.HoverRequest(
            path="/a/b.py",
            client_id="foo",
            position=lsp.PyrePosition(line=1, character=2),
        )
        self.assertEqual(
            hover_request.to_json(),
            [
                "Hover",
                {
                    "path": "/a/b.py",
                    "client_id": "foo",
                    "position": {"line": 1, "column": 2},
                },
            ],
        )
        definition_request = code_navigation_request.LocationOfDefinitionRequest(
            path="/a/b.py",
            client_id="foo",
            position=lsp.PyrePosition(line=1, character=2),
        )
        self.assertEqual(
            definition_request.to_json(),
            [
                "LocationOfDefinition",
                {
                    "path": "/a/b.py",
                    "client_id": "foo",
                    "position": {"line": 1, "column": 2},
                },
            ],
        )

        completion_request = lsp.CompletionRequest(
            path="/a/b.py",
            client_id="foo",
            position=lsp.PyrePosition(line=1, character=2),
        )
        self.assertEqual(
            completion_request.to_json(),
            [
                "Completion",
                {
                    "path": "/a/b.py",
                    "client_id": "foo",
                    "position": {"line": 1, "column": 2},
                },
            ],
        )

    def test_serialize_type_errors_request(self) -> None:
        request = code_navigation_request.TypeErrorsRequest(
            path="/a/b.py",
            client_id="foo",
        )
        self.assertEqual(
            request.to_json(),
            [
                "GetTypeErrors",
                {
                    "path": "/a/b.py",
                    "client_id": "foo",
                },
            ],
        )

    def test_parse_raw_response(self) -> None:
        raw_response = json.dumps(
            [
                "NotHover",
                {"contents": [{"kind": ["PlainText"], "value": "`int`"}]},
            ]
        )
        self.assertEqual(
            code_navigation_request.parse_raw_response(
                raw_response,
                expected_response_kind="Hover",
                response_type=code_navigation_request.HoverResponse,
            ),
            code_navigation_request.ErrorResponse(
                f"Invalid response {raw_response} to pyre code_navigation request."
            ),
        )

        raw_response = json.dumps(
            [
                "Hover",
                {"contents": [{"kind": ["PlainText"], "value": "`int`"}]},
                "ExtraField",
            ]
        )

        self.assertEqual(
            code_navigation_request.parse_raw_response(
                raw_response,
                expected_response_kind="Hover",
                response_type=code_navigation_request.HoverResponse,
            ),
            code_navigation_request.ErrorResponse(
                f"Invalid response {raw_response} to pyre code_navigation request."
            ),
        )

    def test_hover_response(self) -> None:
        response = {"contents": [{"value": "int", "docstring": "test docstring"}]}
        self.assertEqual(
            code_navigation_request.parse_response(
                response, response_type=code_navigation_request.HoverResponse
            ),
            code_navigation_request.HoverResponse(
                contents=[
                    lsp.PyreHoverResponse(value="int", docstring="test docstring")
                ]
            ),
        )

        # Note that there's a type error here in the TypedDict
        response = {"contents": [{"value": 32, "docstring": None}]}
        with self.assertRaises(ValidationError):
            code_navigation_request.parse_response(
                response, response_type=code_navigation_request.HoverResponse
            ),

    def test_definition_response(self) -> None:
        response = {
            "definitions": [
                {
                    "path": "/a/b.py",
                    "range": {
                        "start": {"line": 1, "column": 2},
                        "stop": {"line": 1, "column": 6},
                    },
                }
            ]
        }
        self.assertEqual(
            code_navigation_request.parse_response(
                response,
                response_type=code_navigation_request.LocationOfDefinitionResponse,
            ),
            code_navigation_request.LocationOfDefinitionResponse(
                definitions=[
                    code_navigation_request.DefinitionResponse(
                        path="/a/b.py",
                        range=code_navigation_request.CodeNavigationRange(
                            code_navigation_request.CodeNavigationPosition(
                                line=1, column=2
                            ),
                            code_navigation_request.CodeNavigationPosition(
                                line=1, column=6
                            ),
                        ),
                    )
                ]
            ),
        )

    def test_type_errors_response(self) -> None:
        response = {
            "errors": [
                {
                    "line": 7,
                    "column": 10,
                    "stop_line": 7,
                    "stop_column": 18,
                    "path": "test.py",
                    "code": 16,
                    "name": "Undefined attribute",
                    "description": "Undefined attribute [16]: `int` has no attribute `format`.",
                    "concise_description": "Undefined attribute [16]: `int` has no attribute `format`.",
                },
            ]
        }
        parsed_response = code_navigation_request.parse_response(
            response,
            response_type=code_navigation_request.TypeErrorsResponse,
        )
        if isinstance(parsed_response, code_navigation_request.ErrorResponse):
            self.fail()
        else:
            self.assertListEqual(
                parsed_response.to_errors_response(),
                [
                    error.Error(
                        line=7,
                        column=10,
                        stop_line=7,
                        stop_column=18,
                        path=Path("test.py"),
                        code=16,
                        name="Undefined attribute",
                        description="Undefined attribute [16]: `int` has no attribute `format`.",
                        concise_description="Undefined attribute [16]: `int` has no attribute `format`.",
                    )
                ],
            )

    def test_completion_response(self) -> None:
        response = {
            "completions": [
                {"label": "attribute", "kind": "SIMPLE", "detail": "object"},
                {"label": "attribute2", "kind": "METHOD", "detail": "object"},
                {"label": "attribute3", "kind": "PROPERTY", "detail": "object"},
                {"label": "attribute4", "kind": "VARIABLE", "detail": "object"},
            ]
        }
        self.assertEqual(
            code_navigation_request.parse_response(
                response,
                response_type=code_navigation_request.PyreCompletionsResponse,
            ),
            code_navigation_request.PyreCompletionsResponse(
                completions=[
                    code_navigation_request.PyreCompletionItem(
                        label="attribute",
                        kind=code_navigation_request.PyreCompletionItemKind.SIMPLE,
                        detail="object",
                    ),
                    code_navigation_request.PyreCompletionItem(
                        label="attribute2",
                        kind=code_navigation_request.PyreCompletionItemKind.METHOD,
                        detail="object",
                    ),
                    code_navigation_request.PyreCompletionItem(
                        label="attribute3",
                        kind=code_navigation_request.PyreCompletionItemKind.PROPERTY,
                        detail="object",
                    ),
                    code_navigation_request.PyreCompletionItem(
                        label="attribute4",
                        kind=code_navigation_request.PyreCompletionItemKind.VARIABLE,
                        detail="object",
                    ),
                ]
            ),
        )

    def test_register_client_json(self) -> None:
        register_client = code_navigation_request.RegisterClient(client_id="foo")
        self.assertEqual(
            register_client.to_json(),
            [
                "RegisterClient",
                {
                    "client_id": "foo",
                },
            ],
        )

    def test_dispose_client_json(self) -> None:
        dispose_client = code_navigation_request.DisposeClient(client_id="foo")
        self.assertEqual(
            dispose_client.to_json(),
            [
                "DisposeClient",
                {
                    "client_id": "foo",
                },
            ],
        )

    def test_local_update_json(self) -> None:
        local_update = code_navigation_request.LocalUpdate(
            path="/a/b.py",
            content="def foo() -> int: pass\n",
            client_id="/a/b.py 1234",
        )
        self.assertEqual(
            local_update.to_json(),
            [
                "LocalUpdate",
                {
                    "path": "/a/b.py",
                    "content": "def foo() -> int: pass\n",
                    "client_id": "/a/b.py 1234",
                },
            ],
        )

    def test_file_opened_json(self) -> None:
        local_update = code_navigation_request.FileOpened(
            path=Path("/a/b.py"),
            content="def foo() -> int: pass\n",
            client_id="/a/b.py 1234",
        )
        self.assertEqual(
            local_update.to_json(),
            [
                "FileOpened",
                {
                    "path": "/a/b.py",
                    "content": "def foo() -> int: pass\n",
                    "client_id": "/a/b.py 1234",
                },
            ],
        )

    def test_file_closed_json(self) -> None:
        local_update = code_navigation_request.FileClosed(
            path=Path("/a/b.py"),
            client_id="/a/b.py 1234",
        )
        self.assertEqual(
            local_update.to_json(),
            [
                "FileClosed",
                {
                    "path": "/a/b.py",
                    "client_id": "/a/b.py 1234",
                },
            ],
        )

    def test_superclasses_request_json(self) -> None:
        superclasses_request = code_navigation_request.SuperclassesRequest(
            class_=code_navigation_request.ClassExpression(
                module="a", qualified_name="C"
            ),
        )
        self.assertEqual(
            superclasses_request.to_json(),
            [
                "Superclasses",
                {
                    "class": {"module": "a", "qualified_name": "C"},
                },
            ],
        )

    def test_superclasses_response_from_json(self) -> None:
        superclasses_response = (
            code_navigation_request.SuperclassesResponse.cached_schema().load(
                {
                    "superclasses": [
                        {
                            "module": "typing",
                            "qualified_name": "Sequence",
                        },
                        {
                            "module": "typing",
                            "qualified_name": "Collection",
                        },
                    ]
                }
            )
        )
        self.assertIsInstance(
            superclasses_response, code_navigation_request.SuperclassesResponse
        )
        superclasses = superclasses_response.superclasses
        self.assertEqual(len(superclasses), 2)
        self.assertEqual(
            superclasses[0],
            code_navigation_request.ClassExpression("typing", "Sequence"),
        )

        # Invalid module kind.
        with self.assertRaises(ValidationError):
            code_navigation_request.SuperclassesResponse.cached_schema().load(
                {
                    "superclasses": [
                        {
                            "module": ["Invalid", "/a/b/typing.py"],
                            "qualified_name": "Sequence",
                        },
                    ]
                }
            )
