from pygments.formatters import HtmlFormatter

class HtmlListFormatter(HtmlFormatter):
    """A custom pygments class to format html. Returns a list of source code.
    Each element of the list corresponds to a line of (marked up) source code.
    """

    def get_marked_up_list(self, tokensource):
        if False:
            return 10
        'an updated version of pygments.formatter.format_unencoded'
        source = self._format_lines(tokensource)
        if self.hl_lines:
            source = self._highlight_lines(source)
        if not self.nowrap:
            if self.linenos == 2:
                source = self._wrap_inlinelinenos(source)
            if self.lineanchors:
                source = self._wrap_lineanchors(source)
            if self.linespans:
                source = self._wrap_linespans(source)
            if self.linenos == 1:
                source = self._wrap_tablelinenos(source)
        IS_CODE_INDEX = 0
        HTML_VALUE_INDEX = 1
        IS_CODE_VAL = 1
        source_list = [html_line[HTML_VALUE_INDEX] for html_line in self._wrap_div(self._wrap_pre(source)) if html_line[IS_CODE_INDEX] == IS_CODE_VAL]
        return source_list