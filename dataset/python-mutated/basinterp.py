import sys
import math
import random

class BasicInterpreter:

    def __init__(self, prog):
        if False:
            for i in range(10):
                print('nop')
        self.prog = prog
        self.functions = {'SIN': lambda z: math.sin(self.eval(z)), 'COS': lambda z: math.cos(self.eval(z)), 'TAN': lambda z: math.tan(self.eval(z)), 'ATN': lambda z: math.atan(self.eval(z)), 'EXP': lambda z: math.exp(self.eval(z)), 'ABS': lambda z: abs(self.eval(z)), 'LOG': lambda z: math.log(self.eval(z)), 'SQR': lambda z: math.sqrt(self.eval(z)), 'INT': lambda z: int(self.eval(z)), 'RND': lambda z: random.random()}

    def collect_data(self):
        if False:
            return 10
        self.data = []
        for lineno in self.stat:
            if self.prog[lineno][0] == 'DATA':
                self.data = self.data + self.prog[lineno][1]
        self.dc = 0

    def check_end(self):
        if False:
            for i in range(10):
                print('nop')
        has_end = 0
        for lineno in self.stat:
            if self.prog[lineno][0] == 'END' and (not has_end):
                has_end = lineno
        if not has_end:
            print('NO END INSTRUCTION')
            self.error = 1
            return
        if has_end != lineno:
            print('END IS NOT LAST')
            self.error = 1

    def check_loops(self):
        if False:
            while True:
                i = 10
        for pc in range(len(self.stat)):
            lineno = self.stat[pc]
            if self.prog[lineno][0] == 'FOR':
                forinst = self.prog[lineno]
                loopvar = forinst[1]
                for i in range(pc + 1, len(self.stat)):
                    if self.prog[self.stat[i]][0] == 'NEXT':
                        nextvar = self.prog[self.stat[i]][1]
                        if nextvar != loopvar:
                            continue
                        self.loopend[pc] = i
                        break
                else:
                    print('FOR WITHOUT NEXT AT LINE %s' % self.stat[pc])
                    self.error = 1

    def eval(self, expr):
        if False:
            print('Hello World!')
        etype = expr[0]
        if etype == 'NUM':
            return expr[1]
        elif etype == 'GROUP':
            return self.eval(expr[1])
        elif etype == 'UNARY':
            if expr[1] == '-':
                return -self.eval(expr[2])
        elif etype == 'BINOP':
            if expr[1] == '+':
                return self.eval(expr[2]) + self.eval(expr[3])
            elif expr[1] == '-':
                return self.eval(expr[2]) - self.eval(expr[3])
            elif expr[1] == '*':
                return self.eval(expr[2]) * self.eval(expr[3])
            elif expr[1] == '/':
                return float(self.eval(expr[2])) / self.eval(expr[3])
            elif expr[1] == '^':
                return abs(self.eval(expr[2])) ** self.eval(expr[3])
        elif etype == 'VAR':
            (var, dim1, dim2) = expr[1]
            if not dim1 and (not dim2):
                if var in self.vars:
                    return self.vars[var]
                else:
                    print('UNDEFINED VARIABLE %s AT LINE %s' % (var, self.stat[self.pc]))
                    raise RuntimeError
            if dim1 and (not dim2):
                if var in self.functions:
                    return self.functions[var](dim1)
                elif var in self.lists:
                    dim1val = self.eval(dim1)
                    if dim1val < 1 or dim1val > len(self.lists[var]):
                        print('LIST INDEX OUT OF BOUNDS AT LINE %s' % self.stat[self.pc])
                        raise RuntimeError
                    return self.lists[var][dim1val - 1]
            if dim1 and dim2:
                if var in self.tables:
                    dim1val = self.eval(dim1)
                    dim2val = self.eval(dim2)
                    if dim1val < 1 or dim1val > len(self.tables[var]) or dim2val < 1 or (dim2val > len(self.tables[var][0])):
                        print('TABLE INDEX OUT OUT BOUNDS AT LINE %s' % self.stat[self.pc])
                        raise RuntimeError
                    return self.tables[var][dim1val - 1][dim2val - 1]
            print('UNDEFINED VARIABLE %s AT LINE %s' % (var, self.stat[self.pc]))
            raise RuntimeError

    def releval(self, expr):
        if False:
            while True:
                i = 10
        etype = expr[1]
        lhs = self.eval(expr[2])
        rhs = self.eval(expr[3])
        if etype == '<':
            if lhs < rhs:
                return 1
            else:
                return 0
        elif etype == '<=':
            if lhs <= rhs:
                return 1
            else:
                return 0
        elif etype == '>':
            if lhs > rhs:
                return 1
            else:
                return 0
        elif etype == '>=':
            if lhs >= rhs:
                return 1
            else:
                return 0
        elif etype == '=':
            if lhs == rhs:
                return 1
            else:
                return 0
        elif etype == '<>':
            if lhs != rhs:
                return 1
            else:
                return 0

    def assign(self, target, value):
        if False:
            for i in range(10):
                print('nop')
        (var, dim1, dim2) = target
        if not dim1 and (not dim2):
            self.vars[var] = self.eval(value)
        elif dim1 and (not dim2):
            dim1val = self.eval(dim1)
            if not var in self.lists:
                self.lists[var] = [0] * 10
            if dim1val > len(self.lists[var]):
                print('DIMENSION TOO LARGE AT LINE %s' % self.stat[self.pc])
                raise RuntimeError
            self.lists[var][dim1val - 1] = self.eval(value)
        elif dim1 and dim2:
            dim1val = self.eval(dim1)
            dim2val = self.eval(dim2)
            if not var in self.tables:
                temp = [0] * 10
                v = []
                for i in range(10):
                    v.append(temp[:])
                self.tables[var] = v
            if dim1val > len(self.tables[var]) or dim2val > len(self.tables[var][0]):
                print('DIMENSION TOO LARGE AT LINE %s' % self.stat[self.pc])
                raise RuntimeError
            self.tables[var][dim1val - 1][dim2val - 1] = self.eval(value)

    def goto(self, linenum):
        if False:
            print('Hello World!')
        if not linenum in self.prog:
            print('UNDEFINED LINE NUMBER %d AT LINE %d' % (linenum, self.stat[self.pc]))
            raise RuntimeError
        self.pc = self.stat.index(linenum)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.vars = {}
        self.lists = {}
        self.tables = {}
        self.loops = []
        self.loopend = {}
        self.gosub = None
        self.error = 0
        self.stat = list(self.prog)
        self.stat.sort()
        self.pc = 0
        self.collect_data()
        self.check_end()
        self.check_loops()
        if self.error:
            raise RuntimeError
        while 1:
            line = self.stat[self.pc]
            instr = self.prog[line]
            op = instr[0]
            if op == 'END' or op == 'STOP':
                break
            elif op == 'GOTO':
                newline = instr[1]
                self.goto(newline)
                continue
            elif op == 'PRINT':
                plist = instr[1]
                out = ''
                for (label, val) in plist:
                    if out:
                        out += ' ' * (15 - len(out) % 15)
                    out += label
                    if val:
                        if label:
                            out += ' '
                        eval = self.eval(val)
                        out += str(eval)
                sys.stdout.write(out)
                end = instr[2]
                if not (end == ',' or end == ';'):
                    sys.stdout.write('\n')
                if end == ',':
                    sys.stdout.write(' ' * (15 - len(out) % 15))
                if end == ';':
                    sys.stdout.write(' ' * (3 - len(out) % 3))
            elif op == 'LET':
                target = instr[1]
                value = instr[2]
                self.assign(target, value)
            elif op == 'READ':
                for target in instr[1]:
                    if self.dc < len(self.data):
                        value = ('NUM', self.data[self.dc])
                        self.assign(target, value)
                        self.dc += 1
                    else:
                        return
            elif op == 'IF':
                relop = instr[1]
                newline = instr[2]
                if self.releval(relop):
                    self.goto(newline)
                    continue
            elif op == 'FOR':
                loopvar = instr[1]
                initval = instr[2]
                finval = instr[3]
                stepval = instr[4]
                if not self.loops or self.loops[-1][0] != self.pc:
                    newvalue = initval
                    self.assign((loopvar, None, None), initval)
                    if not stepval:
                        stepval = ('NUM', 1)
                    stepval = self.eval(stepval)
                    self.loops.append((self.pc, stepval))
                else:
                    stepval = ('NUM', self.loops[-1][1])
                    newvalue = ('BINOP', '+', ('VAR', (loopvar, None, None)), stepval)
                if self.loops[-1][1] < 0:
                    relop = '>='
                else:
                    relop = '<='
                if not self.releval(('RELOP', relop, newvalue, finval)):
                    self.pc = self.loopend[self.pc]
                    self.loops.pop()
                else:
                    self.assign((loopvar, None, None), newvalue)
            elif op == 'NEXT':
                if not self.loops:
                    print('NEXT WITHOUT FOR AT LINE %s' % line)
                    return
                nextvar = instr[1]
                self.pc = self.loops[-1][0]
                loopinst = self.prog[self.stat[self.pc]]
                forvar = loopinst[1]
                if nextvar != forvar:
                    print("NEXT DOESN'T MATCH FOR AT LINE %s" % line)
                    return
                continue
            elif op == 'GOSUB':
                newline = instr[1]
                if self.gosub:
                    print('ALREADY IN A SUBROUTINE AT LINE %s' % line)
                    return
                self.gosub = self.stat[self.pc]
                self.goto(newline)
                continue
            elif op == 'RETURN':
                if not self.gosub:
                    print('RETURN WITHOUT A GOSUB AT LINE %s' % line)
                    return
                self.goto(self.gosub)
                self.gosub = None
            elif op == 'FUNC':
                fname = instr[1]
                pname = instr[2]
                expr = instr[3]

                def eval_func(pvalue, name=pname, self=self, expr=expr):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assign((pname, None, None), pvalue)
                    return self.eval(expr)
                self.functions[fname] = eval_func
            elif op == 'DIM':
                for (vname, x, y) in instr[1]:
                    if y == 0:
                        self.lists[vname] = [0] * x
                    else:
                        temp = [0] * y
                        v = []
                        for i in range(x):
                            v.append(temp[:])
                        self.tables[vname] = v
            self.pc += 1

    def expr_str(self, expr):
        if False:
            return 10
        etype = expr[0]
        if etype == 'NUM':
            return str(expr[1])
        elif etype == 'GROUP':
            return '(%s)' % self.expr_str(expr[1])
        elif etype == 'UNARY':
            if expr[1] == '-':
                return '-' + str(expr[2])
        elif etype == 'BINOP':
            return '%s %s %s' % (self.expr_str(expr[2]), expr[1], self.expr_str(expr[3]))
        elif etype == 'VAR':
            return self.var_str(expr[1])

    def relexpr_str(self, expr):
        if False:
            i = 10
            return i + 15
        return '%s %s %s' % (self.expr_str(expr[2]), expr[1], self.expr_str(expr[3]))

    def var_str(self, var):
        if False:
            print('Hello World!')
        (varname, dim1, dim2) = var
        if not dim1 and (not dim2):
            return varname
        if dim1 and (not dim2):
            return '%s(%s)' % (varname, self.expr_str(dim1))
        return '%s(%s,%s)' % (varname, self.expr_str(dim1), self.expr_str(dim2))

    def list(self):
        if False:
            print('Hello World!')
        stat = list(self.prog)
        stat.sort()
        for line in stat:
            instr = self.prog[line]
            op = instr[0]
            if op in ['END', 'STOP', 'RETURN']:
                print('%s %s' % (line, op))
                continue
            elif op == 'REM':
                print('%s %s' % (line, instr[1]))
            elif op == 'PRINT':
                _out = '%s %s ' % (line, op)
                first = 1
                for p in instr[1]:
                    if not first:
                        _out += ', '
                    if p[0] and p[1]:
                        _out += '"%s"%s' % (p[0], self.expr_str(p[1]))
                    elif p[1]:
                        _out += self.expr_str(p[1])
                    else:
                        _out += '"%s"' % (p[0],)
                    first = 0
                if instr[2]:
                    _out += instr[2]
                print(_out)
            elif op == 'LET':
                print('%s LET %s = %s' % (line, self.var_str(instr[1]), self.expr_str(instr[2])))
            elif op == 'READ':
                _out = '%s READ ' % line
                first = 1
                for r in instr[1]:
                    if not first:
                        _out += ','
                    _out += self.var_str(r)
                    first = 0
                print(_out)
            elif op == 'IF':
                print('%s IF %s THEN %d' % (line, self.relexpr_str(instr[1]), instr[2]))
            elif op == 'GOTO' or op == 'GOSUB':
                print('%s %s %s' % (line, op, instr[1]))
            elif op == 'FOR':
                _out = '%s FOR %s = %s TO %s' % (line, instr[1], self.expr_str(instr[2]), self.expr_str(instr[3]))
                if instr[4]:
                    _out += ' STEP %s' % self.expr_str(instr[4])
                print(_out)
            elif op == 'NEXT':
                print('%s NEXT %s' % (line, instr[1]))
            elif op == 'FUNC':
                print('%s DEF %s(%s) = %s' % (line, instr[1], instr[2], self.expr_str(instr[3])))
            elif op == 'DIM':
                _out = '%s DIM ' % line
                first = 1
                for (vname, x, y) in instr[1]:
                    if not first:
                        _out += ','
                    first = 0
                    if y == 0:
                        _out += '%s(%d)' % (vname, x)
                    else:
                        _out += '%s(%d,%d)' % (vname, x, y)
                print(_out)
            elif op == 'DATA':
                _out = '%s DATA ' % line
                first = 1
                for v in instr[1]:
                    if not first:
                        _out += ','
                    first = 0
                    _out += v
                print(_out)

    def new(self):
        if False:
            print('Hello World!')
        self.prog = {}

    def add_statements(self, prog):
        if False:
            i = 10
            return i + 15
        for (line, stat) in prog.items():
            self.prog[line] = stat

    def del_line(self, lineno):
        if False:
            while True:
                i = 10
        try:
            del self.prog[lineno]
        except KeyError:
            pass