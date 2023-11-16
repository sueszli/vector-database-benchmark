from unittest.mock import patch
from nose.tools import assert_greater, assert_in
from pyecharts.charts import Custom
from pyecharts.commons.utils import JsCode

@patch('pyecharts.render.engine.write_utf8_html_file')
def test_custom_base(fake_writer):
    if False:
        i = 10
        return i + 15
    c = Custom().add(series_name='', render_item=JsCode("\n            function (params, api) {\n                var categoryIndex = api.value(0);\n                var start = api.coord([api.value(1), categoryIndex]);\n                var end = api.coord([api.value(2), categoryIndex]);\n                var height = api.size([0, 1])[1] * 0.6;\n                var rectShape = echarts.graphic.clipRectByRect({\n                    x: start[0],\n                    y: start[1] - height / 2,\n                    width: end[0] - start[0],\n                    height: height\n                }, {\n                    x: params.coordSys.x,\n                    y: params.coordSys.y,\n                    width: params.coordSys.width,\n                    height: params.coordSys.height\n                });\n                return rectShape && {\n                    type: 'rect',\n                    shape: rectShape,\n                    style: api.style()\n                };\n            }\n            "), data=None)
    c.render()
    (_, content) = fake_writer.call_args[0]
    assert_greater(len(content), 2000)
    assert_in('renderItem', content)