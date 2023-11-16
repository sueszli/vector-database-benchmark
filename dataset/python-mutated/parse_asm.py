import re
import codecs
from builtins import range
from miasm.core.utils import force_str
from miasm.expression.expression import ExprId, ExprInt, ExprOp, LocKey
import miasm.core.asmblock as asmblock
from miasm.core.cpu import instruction, base_expr
from miasm.core.asm_ast import AstInt, AstId, AstOp
declarator = {'byte': 8, 'word': 16, 'dword': 32, 'qword': 64, 'long': 32}
size2pck = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}
EMPTY_RE = re.compile('\\s*$')
COMMENT_RE = re.compile('\\s*;\\S*')
LOCAL_LABEL_RE = re.compile('\\s*(\\.L\\S+)\\s*:')
DIRECTIVE_START_RE = re.compile('\\s*\\.')
DIRECTIVE_RE = re.compile('\\s*\\.(\\S+)')
LABEL_RE = re.compile('\\s*(\\S+)\\s*:')
FORGET_LABEL_RE = re.compile('\\s*\\.LF[BE]\\d\\s*:')

class Directive(object):
    """Stand for Directive"""
    pass

class DirectiveAlign(Directive):
    """Stand for alignment representation"""

    def __init__(self, alignment=1):
        if False:
            return 10
        self.alignment = alignment

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Alignment %s' % self.alignment

class DirectiveSplit(Directive):
    """Stand for alignment representation"""
    pass

class DirectiveDontSplit(Directive):
    """Stand for alignment representation"""
    pass
STATE_NO_BLOC = 0
STATE_IN_BLOC = 1

def asm_ast_to_expr_with_size(arg, loc_db, size):
    if False:
        while True:
            i = 10
    if isinstance(arg, AstId):
        return ExprId(force_str(arg.name), size)
    if isinstance(arg, AstOp):
        args = [asm_ast_to_expr_with_size(tmp, loc_db, size) for tmp in arg.args]
        return ExprOp(arg.op, *args)
    if isinstance(arg, AstInt):
        return ExprInt(arg.value, size)
    return None

def parse_txt(mnemo, attrib, txt, loc_db):
    if False:
        i = 10
        return i + 15
    'Parse an assembly listing. Returns an AsmCfg instance\n\n    @mnemo: architecture used\n    @attrib: architecture attribute\n    @txt: assembly listing\n    @loc_db: the LocationDB instance used to handle labels of the listing\n\n    '
    C_NEXT = asmblock.AsmConstraint.c_next
    C_TO = asmblock.AsmConstraint.c_to
    lines = []
    for line in txt.split('\n'):
        if EMPTY_RE.match(line):
            continue
        if COMMENT_RE.match(line):
            continue
        if FORGET_LABEL_RE.match(line):
            continue
        match_re = LABEL_RE.match(line)
        if match_re:
            label_name = match_re.group(1)
            label = loc_db.get_or_create_name_location(label_name)
            lines.append(label)
            continue
        if DIRECTIVE_START_RE.match(line):
            match_re = DIRECTIVE_RE.match(line)
            directive = match_re.group(1)
            if directive in ['text', 'data', 'bss']:
                continue
            if directive in ['string', 'ascii']:
                line = line.replace('\\n', '\n').replace('\\r', '\r')
                raw = line[line.find('"') + 1:line.rfind('"')]
                raw = codecs.escape_decode(raw)[0]
                if directive == 'string':
                    raw += b'\x00'
                lines.append(asmblock.AsmRaw(raw))
                continue
            if directive == 'ustring':
                line = line.replace('\\n', '\n').replace('\\r', '\r')
                raw = line[line.find('"') + 1:line.rfind('"')] + '\x00'
                raw = codecs.escape_decode(raw)[0]
                out = b''
                for i in range(len(raw)):
                    out += raw[i:i + 1] + b'\x00'
                lines.append(asmblock.AsmRaw(out))
                continue
            if directive in declarator:
                data_raw = line[match_re.end():].split(' ', 1)[1]
                data_raw = data_raw.split(',')
                size = declarator[directive]
                expr_list = []
                for element in data_raw:
                    element = element.strip()
                    element_parsed = base_expr.parseString(element)[0]
                    element_expr = asm_ast_to_expr_with_size(element_parsed, loc_db, size)
                    expr_list.append(element_expr)
                raw_data = asmblock.AsmRaw(expr_list)
                raw_data.element_size = size
                lines.append(raw_data)
                continue
            if directive == 'comm':
                continue
            if directive == 'split':
                lines.append(DirectiveSplit())
                continue
            if directive == 'dontsplit':
                lines.append(DirectiveDontSplit())
                continue
            if directive == 'align':
                align_value = int(line[match_re.end():], 0)
                lines.append(DirectiveAlign(align_value))
                continue
            if directive in ['file', 'intel_syntax', 'globl', 'local', 'type', 'size', 'align', 'ident', 'section']:
                continue
            if directive[0:4] == 'cfi_':
                continue
            raise ValueError('unknown directive %s' % directive)
        match_re = LABEL_RE.match(line)
        if match_re:
            label_name = match_re.group(1)
            label = loc_db.get_or_create_name_location(label_name)
            lines.append(label)
            continue
        if ';' in line:
            line = line[:line.find(';')]
        line = line.strip(' ').strip('\t')
        instr = mnemo.fromstring(line, loc_db, attrib)
        lines.append(instr)
    asmblock.log_asmblock.info('___pre asm oki___')
    cur_block = None
    state = STATE_NO_BLOC
    i = 0
    asmcfg = asmblock.AsmCFG(loc_db)
    block_to_nlink = None
    delayslot = 0
    while i < len(lines):
        if delayslot:
            delayslot -= 1
            if delayslot == 0:
                state = STATE_NO_BLOC
        line = lines[i]
        if state == STATE_NO_BLOC:
            if isinstance(line, DirectiveDontSplit):
                block_to_nlink = cur_block
                i += 1
                continue
            elif isinstance(line, DirectiveSplit):
                block_to_nlink = None
                i += 1
                continue
            elif not isinstance(line, LocKey):
                loc = loc_db.add_location()
                cur_block = asmblock.AsmBlock(loc_db, loc, alignment=mnemo.alignment)
            else:
                cur_block = asmblock.AsmBlock(loc_db, line, alignment=mnemo.alignment)
                i += 1
            asmcfg.add_block(cur_block)
            state = STATE_IN_BLOC
            if block_to_nlink:
                block_to_nlink.addto(asmblock.AsmConstraint(cur_block.loc_key, C_NEXT))
            block_to_nlink = None
            continue
        elif state == STATE_IN_BLOC:
            if isinstance(line, DirectiveSplit):
                state = STATE_NO_BLOC
                block_to_nlink = None
            elif isinstance(line, DirectiveDontSplit):
                state = STATE_NO_BLOC
                block_to_nlink = cur_block
            elif isinstance(line, DirectiveAlign):
                cur_block.alignment = line.alignment
            elif isinstance(line, asmblock.AsmRaw):
                cur_block.addline(line)
                block_to_nlink = cur_block
            elif isinstance(line, LocKey):
                if block_to_nlink:
                    cur_block.addto(asmblock.AsmConstraint(line, C_NEXT))
                    block_to_nlink = None
                state = STATE_NO_BLOC
                continue
            elif isinstance(line, instruction):
                cur_block.addline(line)
                block_to_nlink = cur_block
                if not line.breakflow():
                    i += 1
                    continue
                if delayslot:
                    raise RuntimeError('Cannot have breakflow in delayslot')
                if line.dstflow():
                    for dst in line.getdstflow(loc_db):
                        if not isinstance(dst, ExprId):
                            continue
                        if dst in mnemo.regs.all_regs_ids:
                            continue
                        cur_block.addto(asmblock.AsmConstraint(dst.name, C_TO))
                if not line.splitflow():
                    block_to_nlink = None
                delayslot = line.delayslot + 1
            else:
                raise RuntimeError('unknown class %s' % line.__class__)
        i += 1
    for block in asmcfg.blocks:
        block.fix_constraints()
        asmblock.log_asmblock.info(block)
    return asmcfg