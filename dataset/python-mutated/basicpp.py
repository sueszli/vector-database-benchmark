from __future__ import print_function
import argparse
import re
import sys

def debug(*args, **kwargs):
    if False:
        while True:
            i = 10
    print(*args, file=sys.stderr, **kwargs)

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Preprocess Basic code.')
    parser.add_argument('infiles', type=str, nargs='+', help='the Basic files to preprocess')
    parser.add_argument('--mode', choices=['cbm', 'qbasic'], default='cbm')
    parser.add_argument('--sub-mode', choices=['noui', 'ui'], default='noui')
    parser.add_argument('--keep-rems', action='store_true', default=False, help='The type of REMs to keep (0 (none) -> 4 (all)')
    parser.add_argument('--keep-blank-lines', action='store_true', default=False, help='Keep blank lines from the original file')
    parser.add_argument('--keep-indent', action='store_true', default=False, help='Keep line identing')
    parser.add_argument('--skip-misc-fixups', action='store_true', default=False, help='Skip miscellaneous fixup/shrink fixups')
    parser.add_argument('--skip-combine-lines', action='store_true', default=False, help='Do not combine lines using the ":" separator')
    args = parser.parse_args()
    args.full_mode = '%s-%s' % (args.mode, args.sub_mode)
    if args.keep_rems and (not args.skip_combine_lines):
        debug('Option --keep-rems implies --skip-combine-lines ')
        args.skip_combine_lines = True
    if args.mode == 'qbasic' and (not args.skip_misc_fixups):
        debug("Mode 'qbasic' implies --skip-misc-fixups")
        args.skip_misc_fixups = True
    return args

def resolve_includes(orig_lines, args):
    if False:
        while True:
            i = 10
    included = {}
    lines = orig_lines[:]
    position = 0
    while position < len(lines):
        line = lines[position]
        m = re.match("^(?:#([^ ]*) )? *REM \\$INCLUDE: '([^'\\n]*)' *$", line)
        if m:
            mode = m.group(1)
            f = m.group(2)
            if mode and mode != args.mode and (mode != args.full_mode):
                position += 1
            elif f not in included:
                ilines = [l.rstrip() for l in open(f).readlines()]
                if args.keep_rems:
                    lines.append("REM vvv BEGIN '%s' vvv" % f)
                lines[position:position + 1] = ilines
                if args.keep_rems:
                    lines.append("REM ^^^ END '%s' ^^^" % f)
            else:
                debug('Ignoring already included file: %s' % f)
        else:
            position += 1
    return lines

def resolve_mode(orig_lines, args):
    if False:
        for i in range(10):
            print('nop')
    lines = []
    for line in orig_lines:
        m = re.match('^ *#([^ \\n]*) *([^\\n]*)$', line)
        if m:
            if m.group(1) == args.mode:
                lines.append(m.group(2))
            elif m.group(1) == args.full_mode:
                lines.append(m.group(2))
            continue
        lines.append(line)
    return lines

def drop_blank_lines(orig_lines):
    if False:
        print('Hello World!')
    lines = []
    for line in orig_lines:
        if re.match('^\\W*$', line):
            continue
        lines.append(line)
    return lines

def drop_rems(orig_lines):
    if False:
        while True:
            i = 10
    lines = []
    for line in orig_lines:
        if re.match('^ *REM', line):
            continue
        m = re.match('^(.*): *REM .*$', line)
        if m:
            lines.append(m.group(1))
        else:
            lines.append(line)
    return lines

def remove_indent(orig_lines):
    if False:
        return 10
    lines = []
    for line in orig_lines:
        m = re.match('^ *([^ \\n].*)$', line)
        lines.append(m.group(1))
    return lines

def misc_fixups(orig_lines):
    if False:
        i = 10
        return i + 15
    text = '\n'.join(orig_lines)
    text = re.sub('\\bTHEN GOTO\\b', 'THEN', text)
    text = re.sub('\\bIF ', 'IF', text)
    text = re.sub('\\bPRINT *', 'PRINT', text)
    text = re.sub('\\bDIM ', 'DIM', text)
    text = re.sub('\\OPEN ', 'OPEN', text)
    text = re.sub('\\bGET ', 'GET', text)
    text = re.sub('\\bPOKE ', 'POKE', text)
    text = re.sub('\\bCLOSE ', 'CLOSE', text)
    text = re.sub('\\bFOR ', 'FOR', text)
    text = re.sub(' TO ', 'TO', text)
    text = re.sub('\\bNEXT ', 'NEXT', text)
    text = re.sub(' *GOTO *', 'GOTO', text)
    text = re.sub(' *GOSUB *', 'GOSUB', text)
    text = re.sub(' *THEN *', 'THEN', text)
    text = re.sub('(?<!ST) *AND *', 'AND', text)
    text = re.sub('([^A-Z]) *OR *', '\\g<1>OR', text)
    return text.split('\n')

def finalize(lines, args):
    if False:
        for i in range(10):
            print('nop')
    labels_lines = {}
    lines_labels = {}
    call_index = {}
    cur_sub = None
    src_lines = lines
    lines = []
    lnum = 1
    for line in src_lines:
        m = re.match('^ *([^ :\\n]*): *$', line)
        if m:
            label = m.groups(1)[0]
            labels_lines[label] = lnum
            lines_labels[lnum] = label
            continue
        if re.match('.*CALL  *([^ :\\n]*) *:', line):
            raise Exception('CALL is not the last thing on line %s' % lnum)
        m = re.match('(.*)CALL  *([^ :\\n]*) *$', line)
        if m:
            prefix = m.groups(1)[0]
            sub = m.groups(1)[1]
            if not call_index.has_key(sub):
                call_index[sub] = 0
            call_index[sub] += 1
            label = sub + '_' + str(call_index[sub])
            if args.mode == 'cbm':
                lines.append('%s %sQ=%s:GOSUBPUSH_Q:GOTO%s' % (lnum, prefix, call_index[sub], sub))
            else:
                lines.append('%s %sX=X+1:X%%(X)=%s:GOTO %s' % (lnum, prefix, call_index[sub], sub))
            lnum += 1
            labels_lines[label] = lnum
            lines_labels[lnum] = label
            continue
        lines.append('%s %s' % (lnum, line))
        lnum += 1
    src_lines = lines
    lines = []
    lnum = 1
    for line in src_lines:
        m = re.match('^([0-9][0-9]*)  *SUB  *([^ \\n]*) *$', line)
        if m:
            lnum = int(m.groups(1)[0]) + 1
            label = m.groups(1)[1]
            cur_sub = label
            labels_lines[label] = lnum
            lines_labels[lnum] = label
            continue
        m = re.match('^([0-9][0-9]*)  *END SUB *$', line)
        if m:
            if cur_sub == None:
                raise Exception('END SUB found without preceeding SUB')
            lnum = int(m.groups(1)[0])
            index = call_index[cur_sub]
            ret_labels = [cur_sub + '_' + str(i) for i in range(1, index + 1)]
            if args.mode == 'cbm':
                line = '%s GOSUBPOP_Q:ONQGOTO%s' % (lnum, ','.join(ret_labels))
            else:
                line = '%s X=X-1:ON X%%(X+1) GOTO %s' % (lnum, ','.join(ret_labels))
            cur_sub = None
        lines.append(line)

    def update_labels_lines(text, a, b):
        if False:
            return 10
        stext = ''
        while stext != text:
            stext = text
            text = re.sub('(THEN *)%s\\b' % a, '\\g<1>%s' % b, stext)
            if args.mode == 'cbm':
                text = re.sub('ON *([^:\\n]*) *GOTO *([^:\\n]*)\\b%s\\b' % a, 'ON\\g<1>GOTO\\g<2>%s' % b, text)
                text = re.sub('ON *([^:\\n]*) *GOSUB *([^:\\n]*)\\b%s\\b' % a, 'ON\\g<1>GOSUB\\g<2>%s' % b, text)
            else:
                text = re.sub('(ON [^:\\n]* *GOTO *[^:\\n]*)\\b%s\\b' % a, '\\g<1>%s' % b, text)
                text = re.sub('(ON [^:\\n]* *GOSUB *[^:\\n]*)\\b%s\\b' % a, '\\g<1>%s' % b, text)
            text = re.sub('(GOSUB *)%s\\b' % a, '\\g<1>%s' % b, text)
            text = re.sub('(GOTO *)%s\\b' % a, '\\g<1>%s' % b, text)
        return text
    src_lines = lines
    text = '\n'.join(lines)
    for (label, lnum) in labels_lines.items():
        text = update_labels_lines(text, label, lnum)
    lines = text.split('\n')
    if not args.skip_combine_lines:
        renumber = {}
        src_lines = lines
        lines = []
        pos = 0
        acc_line = ''

        def renum(line):
            if False:
                while True:
                    i = 10
            lnum = len(lines) + 1
            renumber[old_num] = lnum
            return '%s %s' % (lnum, line)
        while pos < len(src_lines):
            line = src_lines[pos]
            m = re.match('^([0-9]*) (.*)$', line)
            old_num = int(m.group(1))
            line = m.group(2)
            if acc_line == '':
                acc_line = renum(line)
            elif old_num in lines_labels or re.match('^ *FOR\\b.*', line):
                lines.append(acc_line)
                acc_line = renum(line)
            elif re.match('.*(?:GOTO|THEN|RETURN).*', acc_line):
                lines.append(acc_line)
                acc_line = renum(line)
            elif len(acc_line) + 1 + len(line) < 88:
                acc_line = acc_line + ':' + line
                if re.match('.*(?:GOTO|THEN|RETURN).*', line):
                    lines.append(acc_line)
                    acc_line = ''
            else:
                lines.append(acc_line)
                acc_line = renum(line)
            pos += 1
        if acc_line != '':
            lines.append(acc_line)
        src_lines = lines
        text = '\n'.join(lines)
        for a in sorted(renumber.keys()):
            b = renumber[a]
            text = update_labels_lines(text, a, b)
        lines = text.split('\n')
    if args.full_mode == 'qbasic-noui':
        lines = ['$CONSOLE', '$SCREENHIDE', '_DEST _CONSOLE'] + lines
    return lines
if __name__ == '__main__':
    args = parse_args()
    debug('Preprocessing basic files: ' + ', '.join(args.infiles))
    lines = [l.rstrip() for f in args.infiles for l in open(f).readlines()]
    debug('Original lines: %s' % len(lines))
    lines = resolve_includes(lines, args)
    debug('Lines after includes: %s' % len(lines))
    lines = resolve_mode(lines, args)
    debug('Lines after resolving mode specific lines: %s' % len(lines))
    if not args.keep_blank_lines:
        lines = drop_blank_lines(lines)
        debug('Lines after dropping blank lines: %s' % len(lines))
    if not args.keep_rems:
        lines = drop_rems(lines)
        debug('Lines after dropping REMs: %s' % len(lines))
    if not args.keep_indent:
        lines = remove_indent(lines)
    if not args.skip_misc_fixups:
        lines = misc_fixups(lines)
    lines = finalize(lines, args)
    debug('Lines after finalizing: %s' % len(lines))
    print('\n'.join(lines))