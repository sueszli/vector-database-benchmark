from __future__ import annotations
import pytest
pytest

@pytest.fixture
def test_plot() -> None:
    if False:
        print('Hello World!')
    from bokeh.plotting import figure
    test_plot = figure()
    test_plot.circle([1, 2], [2, 3])
    return test_plot
' XXX\nclass Test_notebook_content(object):\n\n    @patch(\'bokeh.embed.notebook.standalone_docs_json_and_render_items\')\n    def test_notebook_content(self, mock_sdjari: MagicMock, test_plot: MagicMock) -> None:\n        (docs_json, render_items) = ("DOC_JSON", [RenderItem(docid="foo", elementid="bar")])\n        mock_sdjari.return_value = (docs_json, render_items)\n\n        expected_script = DOC_NB_JS.render(docs_json=serialize_json(docs_json),\n                                        render_items=serialize_json(render_items))\n        expected_div = PLOT_DIV.render(elementid=render_items[0][\'elementid\'])\n\n        (script, div, _) = ben.notebook_content(test_plot)\n\n        assert script == expected_script\n        assert div == expected_div\n\n    @patch(\'bokeh.embed.notebook.standalone_docs_json_and_render_items\')\n    def test_notebook_content_with_notebook_comms_target(self, mock_sdjari: MagicMock, test_plot: MagicMock) -> None:\n        (docs_json, render_items) = ("DOC_JSON", [RenderItem(docid="foo", elementid="bar")])\n        mock_sdjari.return_value = (docs_json, render_items)\n        comms_target = "NOTEBOOK_COMMS_TARGET"\n\n        ## assert that NOTEBOOK_COMMS_TARGET is added to render_items bundle\n        assert \'notebook_comms_target\' not in render_items[0]\n        (script, _, _) = ben.notebook_content(test_plot, notebook_comms_target=comms_target)\n        assert \'notebook_comms_target\' in render_items[0]\n\n        ## assert that NOTEBOOK_COMMS_TARGET ends up in generated script\n        expected_script = DOC_NB_JS.render(docs_json=serialize_json(docs_json),\n                                        render_items=serialize_json(render_items))\n\n        assert script == expected_script\n'