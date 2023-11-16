from helium._impl.util.xpath import lower, replace_nbsp

class MatchType:

    def xpath(self, value, text):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def text(self, value, text):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class PREFIX_IGNORE_CASE(MatchType):

    def xpath(self, value, text):
        if False:
            return 10
        if not text:
            return ''
        if '*' in text:
            strip_asterisks = value
        else:
            strip_asterisks = "translate(%s, '*', '')" % value
        if "'" in text:
            text = "concat('%s')" % '\',"\'",\''.join(text.split("'"))
        else:
            text = "'%s'" % text
        return 'starts-with(normalize-space(%s), %s)' % (lower(replace_nbsp(strip_asterisks)), text.lower())

    def text(self, value, text):
        if False:
            return 10
        if not text:
            return True
        return value.lower().lstrip().startswith(text.lower())