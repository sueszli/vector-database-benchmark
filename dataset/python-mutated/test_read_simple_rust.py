from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook
import jupytext
from jupytext.compare import compare, compare_notebooks

def test_read_magics(text='// :vars\n'):
    if False:
        return 10
    nb = jupytext.reads(text, 'rs')
    compare_notebooks(nb, new_notebook(cells=[new_code_cell(':vars')]))
    compare(jupytext.writes(nb, 'rs'), text)

def test_read_simple_file(text='println!("Hello world");\neprintln!("Hello error");\nformat!("Hello {}", "world")\n\n// A Function\npub fn fib(x: i32) -> i32 {\n    if x <= 2 {0} else {fib(x - 2) + fib(x - 1)}\n}\n\n// This is a\n// Markdown cell\n\n// This is a magic instruction\n// :vars\n\n// This is a rust identifier\n::std::mem::drop\n'):
    if False:
        return 10
    nb = jupytext.reads(text, 'rs')
    compare_notebooks(nb, new_notebook(cells=[new_code_cell('println!("Hello world");\neprintln!("Hello error");\nformat!("Hello {}", "world")'), new_code_cell('// A Function\npub fn fib(x: i32) -> i32 {\n    if x <= 2 {0} else {fib(x - 2) + fib(x - 1)}\n}'), new_markdown_cell('This is a\nMarkdown cell'), new_code_cell('// This is a magic instruction\n:vars'), new_code_cell('// This is a rust identifier\n::std::mem::drop')]))
    compare(jupytext.writes(nb, 'rs'), text)

def test_read_write_script_with_metadata_241(no_jupytext_version_number, rsnb='#!/usr/bin/env scriptisto\n// ---\n// jupyter:\n//   jupytext:\n//     text_representation:\n//       extension: .rs\n//       format_name: light\n//   kernelspec:\n//     display_name: Rust\n//     language: rust\n//     name: rust\n// ---\n\nlet mut a: i32 = 2;\na += 1;\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(rsnb, 'rs')
    assert 'executable' in nb.metadata['jupytext']
    rsnb2 = jupytext.writes(nb, 'rs')
    compare(rsnb, rsnb2)