from unittest import TestCase
import plotly.graph_objects as go

class TestUpdateLayout(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        import plotly.io as pio
        pio.templates.default = None

    def test_update_layout_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        fig = go.Figure()
        fig.layout.title.font.size = 10
        orig_fig = go.Figure(fig)
        fig.update_layout(title_font_family='Courier New')
        orig_fig.layout.update(title_font_family='Courier New')
        self.assertEqual(fig, orig_fig)

    def test_update_layout_dict(self):
        if False:
            return 10
        fig = go.Figure()
        fig.layout.title.font.size = 10
        orig_fig = go.Figure(fig)
        fig.update_layout(dict(title=dict(font=dict(family='Courier New'))))
        orig_fig.layout.update(title_font_family='Courier New')
        self.assertEqual(fig, orig_fig)

    def test_update_layout_overwrite(self):
        if False:
            for i in range(10):
                print('nop')
        fig = go.Figure(layout=go.Layout(annotations=[go.layout.Annotation(text='one'), go.layout.Annotation(text='two')]))
        fig.update_layout(overwrite=True, annotations=[go.layout.Annotation(width=10), go.layout.Annotation(width=20), go.layout.Annotation(width=30), go.layout.Annotation(width=40), go.layout.Annotation(width=50)])
        expected = {'annotations': [{'width': 10}, {'width': 20}, {'width': 30}, {'width': 40}, {'width': 50}]}
        fig.layout.pop('template')
        self.assertEqual(fig.layout.to_plotly_json(), expected)
        fig.update_layout(overwrite=True, annotations=None)
        self.assertEqual(fig.layout.annotations, ())