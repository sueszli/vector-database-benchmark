"""Convert CSS selectors into XPath selectors"""
from cssselect.xpath import GenericTranslator

class ConvertibleToCssTranslator(GenericTranslator):
    """An implementation of :py:class:`cssselect.GenericTranslator` with
    XPath output that more readily converts back to CSS selectors.
    The simplified examples in https://devhints.io/xpath were used as a
    reference here."""

    def css_to_xpath(self, css, prefix='//'):
        if False:
            while True:
                i = 10
        return super().css_to_xpath(css, prefix)

    def xpath_attrib_equals(self, xpath, name, value):
        if False:
            print('Hello World!')
        xpath.add_condition('%s=%s' % (name, self.xpath_literal(value)))
        return xpath

    def xpath_attrib_includes(self, xpath, name, value):
        if False:
            return 10
        from cssselect.xpath import is_non_whitespace
        if is_non_whitespace(value):
            xpath.add_condition('contains(%s, %s)' % (name, self.xpath_literal(value)))
        else:
            xpath.add_condition('0')
        return xpath

    def xpath_attrib_substringmatch(self, xpath, name, value):
        if False:
            return 10
        if value:
            xpath.add_condition('contains(%s, %s)' % (name, self.xpath_literal(value)))
        else:
            xpath.add_condition('0')
        return xpath

    def xpath_class(self, class_selector):
        if False:
            i = 10
            return i + 15
        xpath = self.xpath(class_selector.selector)
        return self.xpath_attrib_includes(xpath, '@class', class_selector.class_name)

    def xpath_descendant_combinator(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        'right is a child, grand-child or further descendant of left'
        return left.join('//', right)

def convert_css_to_xpath(css):
    if False:
        while True:
            i = 10
    'Convert CSS Selectors to XPath Selectors.\n    Example:\n        convert_css_to_xpath(\'button:contains("Next")\')\n        Output => "//button[contains(., \'Next\')]"\n    '
    xpath = ConvertibleToCssTranslator().css_to_xpath(css)
    return xpath