import os
import shutil
import re
from textwrap import dedent
import pytest
from dash.development._r_components_generation import make_namespace_exports

@pytest.fixture
def make_r_dir():
    if False:
        print('Hello World!')
    os.makedirs('R')
    yield
    shutil.rmtree('R')

def test_r_exports(make_r_dir):
    if False:
        for i in range(10):
            print('nop')
    extra_file = dedent('\n        # normal function syntax\n        my_func <- function(a, b) {\n            c <- a + b\n            nested_func <- function() { stop("no!") }\n            another_to_exclude = function(d) { d * d }\n            another_to_exclude(c)\n        }\n\n        # indented (no reason but we should allow) and using = instead of <-\n        # also braces in comments enclosing it {\n            my_func2 = function() {\n                s <- "unmatched closing brace }"\n                ignore_please <- function() { 1 }\n            }\n        # }\n\n        # real example from dash-table that should exclude FUN\n        df_to_list <- function(df) {\n          if(!(is.data.frame(df)))\n            stop("!")\n          setNames(lapply(split(df, seq(nrow(df))),\n                          FUN = function (x) {\n                            as.list(x)\n                          }), NULL)\n        }\n\n        # single-line compressed\n        util<-function(x){x+1}\n\n        # prefix with . to tell us to ignore\n        .secret <- function() { stop("You can\'t see me") }\n\n        # . in the middle is OK though\n        not.secret <- function() { 42 }\n    ')
    components = ['Component1', 'Component2']
    prefix = 'pre'
    expected_exports = [prefix + c for c in components] + ['my_func', 'my_func2', 'df_to_list', 'util', 'not.secret']
    mock_component_file = dedent('\n        nope <- function() { stop("we don\'t look in component files") }\n    ')
    with open(os.path.join('R', 'preComponent1.R'), 'w') as f:
        f.write(mock_component_file)
    with open(os.path.join('R', 'extras.R'), 'w') as f:
        f.write(extra_file)
    exports = make_namespace_exports(components, prefix)
    print(exports)
    matches = re.findall('export\\(([^()]+)\\)', exports.replace('\n', ' '))
    assert matches == expected_exports