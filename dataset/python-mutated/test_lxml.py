from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['lxml'])
def test_lxml(selenium):
    if False:
        return 10
    from lxml import etree
    root = etree.XML('<root>\n        <TEXT1 class="myitem">one</TEXT1>\n        <TEXT2 class="myitem">two</TEXT2>\n        <TEXT3 class="myitem">three</TEXT3>\n        <v-TEXT4 class="v-list">four</v-TEXT4>\n    </root>')
    items = root.xpath("//*[@class='myitem']")
    assert ['one', 'two', 'three'] == [item.text for item in items]