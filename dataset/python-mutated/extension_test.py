import json
import testslide
from ..exceptions import InvalidConfiguration
from ..extension import Element

class ElementTest(testslide.TestCase):

    def test_from_json(self) -> None:
        if False:
            return 10

        def assert_extension_equal(input: object, expected: Element) -> None:
            if False:
                return 10
            self.assertEqual(Element.from_json(input), expected)

        def assert_extension_raises(input: object) -> None:
            if False:
                for i in range(10):
                    print('nop')
            with self.assertRaises(InvalidConfiguration):
                Element.from_json(input)
        assert_extension_raises({})
        assert_extension_raises({'derp': 42})
        assert_extension_equal('.pyi', Element(suffix='.pyi'))
        assert_extension_equal({'suffix': '.pyi', 'include_suffix_in_module_qualifier': True}, Element(suffix='.pyi', include_suffix_in_module_qualifier=True))
        assert_extension_raises({'suffix': 42, 'include_suffix_in_module_qualifier': True})
        assert_extension_raises({'suffix': '.pyi', 'include_suffix_in_module_qualifier': []})

    def test_to_json(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(Element(suffix='.pyi', include_suffix_in_module_qualifier=True).to_json(), json.dumps({'suffix': '.pyi', 'include_suffix_in_module_qualifier': True}))