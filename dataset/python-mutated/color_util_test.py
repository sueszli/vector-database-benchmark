import unittest
from streamlit import color_util
valid_hex_colors = ['#123', '#1234', '#112233', '#11223344']
valid_css_rgb_colors = ['rgb(1, 2, 3)', 'rgba(1, 2, 3, 4)']
valid_color_tuples = [[0, 1, 2], [0, 1, 2, 4], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 0.5], (0, 1, 2), (0, 1, 2, 4), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0, 0.5)]
invalid_colors = ['#12345', '#12', [1, 2], [1, 2, 3, 4, 5], '#0z0', '# NAs', '0f0', [1, 'foo', 3], {1, 2, 3}, 100, 'red', 'rgb(1, 2, 3)']

class ColorUtilTest(unittest.TestCase):

    def test_to_int_color_tuple(self):
        if False:
            print('Hello World!')
        'Test to_int_color_tuple with good inputs'
        test_combinations = [('#0f0', (0, 255, 0, 255)), ('#0f08', (0, 255, 0, 136)), ('#00ff00', (0, 255, 0, 255)), ('#00ff0088', (0, 255, 0, 136)), ('#00FF00', (0, 255, 0, 255)), ([0, 255, 0], (0, 255, 0)), ((0, 255, 0), (0, 255, 0)), ([0, 255, 0, 128], (0, 255, 0, 128)), ((0, 255, 0, 128), (0, 255, 0, 128)), ([0.0, 0.2, 1.0], (0, 51, 255)), ((0.0, 0.2, 1.0), (0, 51, 255)), ([0.0, 0.2, 1.0, 0.2], (0, 51, 255, 51)), ((0.0, 0.2, 1.0, 0.2), (0, 51, 255, 51)), ([0, 255, 0, 0.2], (0, 255, 0, 51)), ((0, 255, 0, 0.2), (0, 255, 0, 51)), ([600, -100, 50], (255, 0, 50)), ((600, -100, 50), (255, 0, 50)), ([2.0, -1.0, 50], (255, 0, 50)), ((2.0, -1.0, 50), (255, 0, 50))]
        for (test_arg, expected_out) in test_combinations:
            out = color_util.to_int_color_tuple(test_arg)
            self.assertEqual(out, expected_out)

    def test_to_int_color_tuple_fails(self):
        if False:
            print('Hello World!')
        'Test to_int_color_tuple with bad inputs'
        for test_arg in invalid_colors:
            with self.assertRaises(color_util.InvalidColorException):
                color_util.to_int_color_tuple(test_arg)

    def test_to_css_color(self):
        if False:
            while True:
                i = 10
        'Test to_css_color with good inputs.'
        test_combinations = [('#0f0', '#0f0'), ('#0f08', '#0f08'), ('#00ff00', '#00ff00'), ('#00ff0088', '#00ff0088'), ('#00FF00', '#00FF00'), ([0, 255, 0], 'rgb(0, 255, 0)'), ([0, 255, 0, 51], 'rgba(0, 255, 0, 0.2)'), ([0.0, 0.2, 1.0], 'rgb(0, 51, 255)'), ([0.0, 0.2, 1.0, 0.2], 'rgba(0, 51, 255, 0.2)'), ([0, 255, 0, 0.2], 'rgba(0, 255, 0, 0.2)'), ([600, -100, 50], 'rgb(255, 0, 50)'), ([2.0, -1.0, 50], 'rgb(255, 0, 50)'), ((0, 255, 0), 'rgb(0, 255, 0)')]
        for (test_arg, expected_out) in test_combinations:
            out = color_util.to_css_color(test_arg)
            self.assertEqual(out, expected_out)

    def test_to_css_color_fails(self):
        if False:
            i = 10
            return i + 15
        'Test to_css_color with bad inputs.'
        test_args = list(invalid_colors)
        test_args.remove('#0z0')
        test_args.remove('rgb(1, 2, 3)')
        for test_arg in test_args:
            with self.assertRaises(color_util.InvalidColorException):
                color_util.to_css_color(test_arg)

    def test_is_hex_color_like_true(self):
        if False:
            i = 10
            return i + 15
        for test_arg in valid_hex_colors:
            out = color_util.is_hex_color_like(test_arg)
            self.assertTrue(out)

    def test_is_hex_color_like_false(self):
        if False:
            return 10
        test_args = list(invalid_colors)
        test_args.remove('#0z0')
        for test_arg in test_args:
            out = color_util.is_hex_color_like(test_arg)
            self.assertFalse(out)

    def test_is_css_color_like_true(self):
        if False:
            return 10
        for test_arg in [*valid_hex_colors, *valid_css_rgb_colors]:
            out = color_util.is_css_color_like(test_arg)
            self.assertTrue(out)

    def test_is_css_color_like_false(self):
        if False:
            i = 10
            return i + 15
        test_args = list(invalid_colors)
        test_args.remove('#0z0')
        test_args.remove('rgb(1, 2, 3)')
        for test_arg in test_args:
            out = color_util.is_css_color_like(test_arg)
            self.assertFalse(out)

    def test_is_color_tuple_like_true(self):
        if False:
            return 10
        for test_arg in valid_color_tuples:
            out = color_util.is_color_tuple_like(test_arg)
            self.assertTrue(out)

    def test_is_color_tuple_like_false(self):
        if False:
            print('Hello World!')
        for test_arg in invalid_colors:
            out = color_util.is_color_tuple_like(test_arg)
            self.assertFalse(out)

    def test_is_color_like_true(self):
        if False:
            while True:
                i = 10
        for test_arg in [*valid_color_tuples, *valid_hex_colors]:
            out = color_util.is_color_like(test_arg)
            self.assertTrue(out)

    def test_is_color_like_false(self):
        if False:
            for i in range(10):
                print('nop')
        test_args = list(invalid_colors)
        test_args.remove('#0z0')
        test_args.remove('rgb(1, 2, 3)')
        for test_arg in test_args:
            out = color_util.is_color_like(test_arg)
            self.assertFalse(out)