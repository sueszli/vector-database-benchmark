import re

class FormatSpec(object):

    def __init__(self, string='', **kwargs):
        if False:
            print('Hello World!')
        self.fill = ''
        self.align = ''
        self.sign = ''
        self.altform = False
        self.minwidth = -1
        self.precision = -1
        self.formtype = ''
        if string != '':
            self.from_string(string)
        self.from_specs(**kwargs)

    def from_specs(self, fill=None, align=None, sign=None, altform=None, minwidth=None, precision=None, formtype=None):
        if False:
            return 10
        if fill is not None:
            self.fill = fill
        if align is not None:
            self.align = align
        if sign is not None:
            self.sign = sign
        if altform is not None:
            self.altform = altform
        if minwidth is not None:
            self.minwidth = minwidth
        if precision is not None:
            self.precision = precision
        if formtype is not None:
            self.formtype = formtype

    def from_string(self, formatspec):
        if False:
            print('Hello World!')
        regexp = '\\A(.[<>=^]|[<>=^])?([-+ ]|\\(\\))?(#?)(0?)(\\d*)(\\.\\d+)?(.)?\\Z'
        match = re.search(regexp, formatspec)
        if match is None:
            raise ValueError('Invalid format specification: ' + formatspec)
        if match.group(1):
            fillalign = match.group(1)
            if len(fillalign) > 1:
                self.fill = fillalign[0]
                self.align = fillalign[1]
            elif fillalign:
                self.align = fillalign
        if match.group(2):
            self.sign = match.group(2)
        if match.group(3):
            self.altform = len(match.group(3)) > 0
        if len(match.group(4)):
            if not self.fill:
                self.fill = '0'
                if not self.align:
                    self.align = '='
        if match.group(5):
            self.minwidth = int(match.group(5))
        if match.group(6):
            self.precision = int(match.group(6)[1:])
        if match.group(7):
            self.formtype = match.group(7)

    def to_string(self):
        if False:
            for i in range(10):
                print('nop')
        formatspec = ''
        if self.align:
            formatspec = self.fill + self.align
        formatspec += self.sign
        if self.sign == '(':
            formatspec += ')'
        if self.altform:
            formatspec += '#'
        if self.minwidth >= 0:
            formatspec += str(self.minwidth)
        if self.precision >= 0:
            formatspec += '.' + str(self.precision)
        formatspec += self.formtype
        return formatspec

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.to_string()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<FormatSpec (' + self.to_string() + ')>'