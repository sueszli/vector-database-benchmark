from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
if TYPE_CHECKING:
    import httpretty
    from cleo.testers.command_tester import CommandTester
    from tests.types import CommandTesterFactory
TESTS_DIRECTORY = Path(__file__).parent.parent.parent
FIXTURES_DIRECTORY = TESTS_DIRECTORY / 'repositories' / 'fixtures' / 'pypi.org' / 'search'

@pytest.fixture(autouse=True)
def mock_search_http_response(http: type[httpretty.httpretty]) -> None:
    if False:
        i = 10
        return i + 15
    with FIXTURES_DIRECTORY.joinpath('search.html').open(encoding='utf-8') as f:
        http.register_uri('GET', 'https://pypi.org/search', f.read())

@pytest.fixture
def tester(command_tester_factory: CommandTesterFactory) -> CommandTester:
    if False:
        i = 10
        return i + 15
    return command_tester_factory('search')

def test_search(tester: CommandTester, http: type[httpretty.httpretty]) -> None:
    if False:
        while True:
            i = 10
    tester.execute('sqlalchemy')
    expected = '\nsqlalchemy (1.3.10)\n Database Abstraction Library\n\nsqlalchemy-dao (1.3.1)\n Simple wrapper for sqlalchemy.\n\ngraphene-sqlalchemy (2.2.2)\n Graphene SQLAlchemy integration\n\nsqlalchemy-utcdatetime (1.0.4)\n Convert to/from timezone aware datetimes when storing in a DBMS\n\npaginate-sqlalchemy (0.3.0)\n Extension to paginate.Page that supports SQLAlchemy queries\n\nsqlalchemy-audit (0.1.0)\n sqlalchemy-audit provides an easy way to set up revision tracking for your data.\n\ntransmogrify-sqlalchemy (1.0.2)\n Feed data from SQLAlchemy into a transmogrifier pipeline\n\nsqlalchemy-schemadisplay (1.3)\n Turn SQLAlchemy DB Model into a graph\n\nsqlalchemy-traversal (0.5.2)\n UNKNOWN\n\nsqlalchemy-filters (0.10.0)\n A library to filter SQLAlchemy queries.\n\nsqlalchemy-wrap (2.1.7)\n Python wrapper for the CircleCI API\n\nsqlalchemy-nav (0.0.2)\n SQLAlchemy-Nav provides SQLAlchemy Mixins for creating navigation bars compatible with Bootstrap\n\nsqlalchemy-repr (0.0.1)\n Automatically generates pretty repr of a SQLAlchemy model.\n\nsqlalchemy-diff (0.1.3)\n Compare two database schemas using sqlalchemy.\n\nsqlalchemy-equivalence (0.1.1)\n Provides natural equivalence support for SQLAlchemy declarative models.\n\nbroadway-sqlalchemy (0.0.1)\n A broadway extension wrapping Flask-SQLAlchemy\n\njsonql-sqlalchemy (1.0.1)\n Simple JSON-Based CRUD Query Language for SQLAlchemy\n\nsqlalchemy-plus (0.2.0)\n Create Views and Materialized Views with SqlAlchemy\n\ncherrypy-sqlalchemy (0.5.3)\n Use SQLAlchemy with CherryPy\n\nsqlalchemy-sqlany (1.0.3)\n SAP Sybase SQL Anywhere dialect for SQLAlchemy\n'
    output = tester.io.fetch_output()
    assert output == expected