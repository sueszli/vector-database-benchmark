import copy
from ..core.pattern import Pattern
__all__ = ['TemplatablePattern']

class TemplateNames:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.django = False
        self.erb = False
        self.handlebars = False
        self.php = False
        self.smarty = False

class TemplatePatterns:

    def __init__(self, input_scanner):
        if False:
            print('Hello World!')
        pattern = Pattern(input_scanner)
        self.handlebars_comment = pattern.starting_with('{{!--').until_after('--}}')
        self.handlebars_unescaped = pattern.starting_with('{{{').until_after('}}}')
        self.handlebars = pattern.starting_with('{{').until_after('}}')
        self.php = pattern.starting_with('<\\?(?:[= ]|php)').until_after('\\?>')
        self.erb = pattern.starting_with('<%[^%]').until_after('[^%]%>')
        self.django = pattern.starting_with('{%').until_after('%}')
        self.django_value = pattern.starting_with('{{').until_after('}}')
        self.django_comment = pattern.starting_with('{#').until_after('#}')
        self.smarty_value = pattern.starting_with('{(?=[^}{\\s\\n])').until_after('}')
        self.smarty_comment = pattern.starting_with('{\\*').until_after('\\*}')
        self.smarty_literal = pattern.starting_with('{literal}').until_after('{/literal}')

class TemplatablePattern(Pattern):

    def __init__(self, input_scanner, parent=None):
        if False:
            print('Hello World!')
        Pattern.__init__(self, input_scanner, parent)
        self.__template_pattern = None
        self._disabled = TemplateNames()
        self._excluded = TemplateNames()
        if parent is not None:
            self.__template_pattern = self._input.get_regexp(parent.__template_pattern)
            self._disabled = copy.copy(parent._disabled)
            self._excluded = copy.copy(parent._excluded)
        self.__patterns = TemplatePatterns(input_scanner)

    def _create(self):
        if False:
            for i in range(10):
                print('nop')
        return TemplatablePattern(self._input, self)

    def _update(self):
        if False:
            while True:
                i = 10
        self.__set_templated_pattern()

    def read_options(self, options):
        if False:
            return 10
        result = self._create()
        for language in ['django', 'erb', 'handlebars', 'php', 'smarty']:
            setattr(result._disabled, language, not language in options.templating)
        result._update()
        return result

    def disable(self, language):
        if False:
            print('Hello World!')
        result = self._create()
        setattr(result._disabled, language, True)
        result._update()
        return result

    def exclude(self, language):
        if False:
            return 10
        result = self._create()
        setattr(result._excluded, language, True)
        result._update()
        return result

    def read(self):
        if False:
            while True:
                i = 10
        result = ''
        if bool(self._match_pattern):
            result = self._input.read(self._starting_pattern)
        else:
            result = self._input.read(self._starting_pattern, self.__template_pattern)
        next = self._read_template()
        while bool(next):
            if self._match_pattern is not None:
                next += self._input.read(self._match_pattern)
            else:
                next += self._input.readUntil(self.__template_pattern)
            result += next
            next = self._read_template()
        if self._until_after:
            result += self._input.readUntilAfter(self._until_after)
        return result

    def __set_templated_pattern(self):
        if False:
            while True:
                i = 10
        items = list()
        if not self._disabled.php:
            items.append(self.__patterns.php._starting_pattern.pattern)
        if not self._disabled.handlebars:
            items.append(self.__patterns.handlebars._starting_pattern.pattern)
        if not self._disabled.erb:
            items.append(self.__patterns.erb._starting_pattern.pattern)
        if not self._disabled.django:
            items.append(self.__patterns.django._starting_pattern.pattern)
            items.append(self.__patterns.django_value._starting_pattern.pattern)
            items.append(self.__patterns.django_comment._starting_pattern.pattern)
        if not self._disabled.smarty:
            items.append(self.__patterns.smarty._starting_pattern.pattern)
        if self._until_pattern:
            items.append(self._until_pattern.pattern)
        self.__template_pattern = self._input.get_regexp('(?:' + '|'.join(items) + ')')

    def _read_template(self):
        if False:
            while True:
                i = 10
        resulting_string = ''
        c = self._input.peek()
        if c == '<':
            peek1 = self._input.peek(1)
            if not self._disabled.php and (not self._excluded.php) and (peek1 == '?'):
                resulting_string = resulting_string or self.__patterns.php.read()
            if not self._disabled.erb and (not self._excluded.erb) and (peek1 == '%'):
                resulting_string = resulting_string or self.__patterns.erb.read()
        elif c == '{':
            if not self._disabled.handlebars and (not self._excluded.handlebars):
                resulting_string = resulting_string or self.__patterns.handlebars_comment.read()
                resulting_string = resulting_string or self.__patterns.handlebars_unescaped.read()
                resulting_string = resulting_string or self.__patterns.handlebars.read()
            if not self._disabled.django:
                if not self._excluded.django and (not self._excluded.handlebars):
                    resulting_string = resulting_string or self.__patterns.django_value.read()
                if not self._excluded.django:
                    resulting_string = resulting_string or self.__patterns.django_comment.read()
                    resulting_string = resulting_string or self.__patterns.django.read()
            if not self._disabled.smarty:
                if self._disabled.django and self._disabled.handlebars:
                    resulting_string = resulting_string or self.__patterns.smarty_comment.read()
                    resulting_string = resulting_string or self.__patterns.smarty_literal.read()
                    resulting_string = resulting_string or self.__patterns.smarty.read()
        return resulting_string