import jupytext
from jupytext.compare import compare

def test_read_simple_file(script=';; ---\n;; title: Simple file\n;; ---\n\n;; Here we have some text\n;; And below we have some code\n\n((fn\n  []\n  (println "Hello World")))\n'):
    if False:
        return 10
    nb = jupytext.reads(script, 'clj')
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: Simple file\n---'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[1].source == 'Here we have some text\nAnd below we have some code'
    assert nb.cells[2].cell_type == 'code'
    compare(nb.cells[2].source, '((fn\n  []\n  (println "Hello World")))')
    script2 = jupytext.writes(nb, 'clj')
    compare(script2, script)