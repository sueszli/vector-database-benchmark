import fileinput
import os
import re
import string
import sys
if_else = lambda a, b, c: (a and [b] or [c])[0]
max_len = 79
ident = 4

def nearest_ident_pos(text):
    if False:
        print('Hello World!')
    return len(text) / ident * ident

def block_format(limits, text, first_sep='  ', sep=',', need_last_ident=1):
    if False:
        for i in range(10):
            print('nop')
    if sep == ',' and string.find(text, '<') != -1:
        sep = '%s ' % sep
    words = string.split(string.join(string.split(text), ' '), sep)
    s = ' ' * limits[0]
    max_len = limits[1]
    return '%s\n%s' % (reduce(lambda t, w, max_len=max_len, s=s, sep=sep: if_else(t[1] + len(w) < max_len, ('%s%s%s' % (t[0], t[2], w), t[1] + len(w) + len(t[2]), sep), ('%s\n%s%s%s' % (t[0], s, sep, w), len(s) + len(w) + len(sep), sep)), words, (s, len(s) + len(first_sep), first_sep))[0], if_else(need_last_ident, s, ''))

def handle_args(match):
    if False:
        return 10
    if re.compile('^\\s*(typedef|struct|static)\\s+.*?$').match(match.group(0)):
        return match.group(0)
    return '%s' % block_format((nearest_ident_pos(match.group(1)), max_len), match.group(3), match.group(2), ',', 0)

def handle_inline_args(match):
    if False:
        while True:
            i = 10
    if len(match.group(0)) < max_len:
        return match.group(0)
    if match.group(9) == None:
        return '%s%s<\n%s>\n' % (match.group(1), match.group(3), block_format((nearest_ident_pos(match.group(1)) + ident, max_len), match.group(4)))
    return '%s%s<\n%s>\n%s%s' % (match.group(1), match.group(3), block_format((nearest_ident_pos(match.group(1)) + ident, max_len - len(match.group(9))), match.group(4)), string.replace(match.group(1), ',', ' '), match.group(9))

def handle_simple_list(match):
    if False:
        while True:
            i = 10
    if match.group(1) == 'template':
        return match.group(0)
    single_arg = re.compile('^\\s*(\\w|\\d)+\\s*$').match(match.group(2))
    return if_else(single_arg, '%s<%s>', '%s< %s >') % (match.group(1), string.join(string.split(match.group(2)), ''))

def handle_static(match):
    if False:
        for i in range(10):
            print('nop')
    if len(match.group(0)) < max_len:
        return match.group(0)
    (first_sep, sep) = if_else(string.find(match.group(0), '+') == -1, (' ', ' '), ('  ', '+'))
    return '%s%s\n%s%s' % (match.group(1), string.join(string.split(match.group(2)), ' '), block_format((nearest_ident_pos(match.group(1)) + ident, max_len), match.group(4), first_sep, sep), match.group(5))

def handle_typedefs(match):
    if False:
        while True:
            i = 10
    if string.count(match.group(2), ';') == 1:
        return match.group(0)
    join_sep = ';\n%s' % match.group(1)
    return '%s%s\n' % (match.group(1), string.join(map(string.strip, string.split(match.group(2), ';')), join_sep))

def fix_angle_brackets(match):
    if False:
        for i in range(10):
            print('nop')
    return ' '.join(''.join(match.group(1).split(' '))) + match.group(3)

class pretty:

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.output = open(name, 'w')
        self.prev_line = ''
        self.re_copyright_start = re.compile('^// Copyright .*$')
        self.re_copyright_end = re.compile('^// See .* for documentation.$')
        self.reading_copyright = 0
        self.copyright = None
        self.re_header_name_comment = re.compile('^\\s*//\\s+\\$[I]d:\\s+(.*?%s\\.hpp)\\s+[^$]+[$]$' % os.path.splitext(name)[0])
        self.header_was_written = 0
        self.re_junk = re.compile('^\\s*(#|//[^/]|////).*$')
        self.re_c_comment_start = re.compile('^\\s*/\\*.*')
        self.re_c_comment_end = re.compile('^.*\\*/\\s*$')
        self.inside_c_comment = 0
        self.re_empty_line = re.compile('^\\s*$')
        self.re_comma = re.compile('(\\S+)\\s*,\\s*')
        self.re_assign = re.compile('(\\S+[^<|^!|^>])\\s*(=+)\\s*(\\S+)')
        self.re_marked_empty_comment = re.compile('^\\s*//\\s*$')
        self.re_typedef = re.compile('^\\s+typedef\\s+.*?;$')
        self.re_nsl = re.compile('^(\\s+typedef\\s+.*?;|\\s*(private|public):\\s*|\\s*{\\s*|\\s*(\\w|\\d|,)+\\s*)$')
        self.re_templ_decl = re.compile('^(\\s*template\\s*<\\s*.*?|\\s*(private|public):\\s*)$')
        self.re_type_const = re.compile('(const)\\s+((unsigned|signed)?(bool|char|short|int|long))')
        self.re_templ_args = re.compile('^(\\s*)(, | {2})((\\s*(\\w+)(\\s+|::)\\w+\\s*.*?,?)+)\\s*$')
        self.re_inline_templ_args = re.compile('^(\\s+(,|:\\s+)?|struct\\s+)(\\w+)\\s*<((\\s*(typename\\s+)?\\w+\\s*(=\\s*.*|<(\\s*\\w+\\s*,?)+>\\s*)?,?)+)\\s*>\\s+((struct|class).*?)?$')
        self.re_simple_list = re.compile('(\\w+)\\s*<((\\w|,| |-)+)>')
        self.re_static_const = re.compile('(\\s*)((BOOST_STATIC_CONSTANT\\(\\s*\\w+,\\s*|enum\\s*\\w*\\s*{\\s*)value\\s*=)(.*?)([}|\\)];)$')
        self.re_typedefs = re.compile('(\\s*)((\\s*typedef\\s*.*?;)+)\\s*$')
        self.re_fix_angle_brackets = re.compile('(>(\\s*>)+)(,|\\n$)')
        self.re_closing_curly_brace = re.compile('^(}|struct\\s+\\w+);\\s*$')
        self.re_namespace_scope_templ = re.compile('^template\\s*<\\s*$')
        self.re_namespace = re.compile('^\\n?namespace\\s+\\w+\\s*{\\s*\\n?$')

    def process(self, line):
        if False:
            print('Hello World!')
        if self.reading_copyright:
            if not self.re_copyright_end.match(line):
                self.copyright += line
                return
            self.reading_copyright = 0
        if not self.header_was_written and self.re_copyright_start.match(line):
            self.copyright = line
            self.reading_copyright = 1
            return
        if not self.header_was_written:
            if self.re_header_name_comment.match(line):
                self.header_was_written = 1
                match = self.re_header_name_comment.match(line)
                self.output.write('\n%s\n// *Preprocessed* version of the main "%s" header\n// -- DO NOT modify by hand!\n\n' % (self.copyright, match.group(1)))
            return
        if self.re_junk.match(line):
            return
        if self.inside_c_comment or self.re_c_comment_start.match(line):
            self.inside_c_comment = not self.re_c_comment_end.match(line)
            return
        if self.re_templ_decl.match(line) and self.re_typedef.match(self.prev_line) or (not self.re_empty_line.match(line) and self.re_closing_curly_brace.match(self.prev_line)) or (not self.re_empty_line.match(self.prev_line) and (self.re_namespace_scope_templ.match(line) or (self.re_namespace.match(line) and (not self.re_namespace.match(self.prev_line))))):
            line = '\n%s' % line
        if self.re_empty_line.match(line):
            if self.re_empty_line.match(self.prev_line) or not self.header_was_written:
                return
            if self.re_nsl.match(self.prev_line):
                return
        line = self.re_comma.sub('\\1, ', line)
        line = self.re_assign.sub('\\1 \\2 \\3', line)
        line = self.re_marked_empty_comment.sub('\\n', line)
        line = self.re_type_const.sub('\\2 \\1', line)
        line = self.re_templ_args.sub(handle_args, line)
        line = self.re_inline_templ_args.sub(handle_inline_args, line)
        line = self.re_simple_list.sub(handle_simple_list, line)
        line = self.re_static_const.sub(handle_static, line)
        line = self.re_typedefs.sub(handle_typedefs, line)
        line = self.re_fix_angle_brackets.sub(fix_angle_brackets, line)
        self.output.write(line)
        self.prev_line = line

def main(src, dest):
    if False:
        return 10
    p = pretty(os.path.basename(dest))
    for line in fileinput.input(src):
        p.process(line)
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])