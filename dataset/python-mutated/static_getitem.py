from numba.core import errors, ir, types
from numba.core.rewrites import register_rewrite, Rewrite

@register_rewrite('before-inference')
class RewriteConstGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$constXX)`
    where `$constXX` is a known constant as
    `static_getitem(value=arr, index=<constant value>)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            for i in range(10):
                print('nop')
        self.getitems = getitems = {}
        self.block = block
        for expr in block.find_exprs(op='getitem'):
            if expr.op == 'getitem':
                try:
                    const = func_ir.infer_constant(expr.index)
                except errors.ConstantInferenceError:
                    continue
                getitems[expr] = const
        return len(getitems) > 0

    def apply(self):
        if False:
            while True:
                i = 10
        '\n        Rewrite all matching getitems as static_getitems.\n        '
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if isinstance(inst, ir.Assign):
                expr = inst.value
                if expr in self.getitems:
                    const = self.getitems[expr]
                    new_expr = ir.Expr.static_getitem(value=expr.value, index=const, index_var=expr.index, loc=expr.loc)
                    inst = ir.Assign(value=new_expr, target=inst.target, loc=inst.loc)
            new_block.append(inst)
        return new_block

@register_rewrite('after-inference')
class RewriteStringLiteralGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$XX)`
    where `$XX` is a StringLiteral value as
    `static_getitem(value=arr, index=<literal value>)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            print('Hello World!')
        '\n        Detect all getitem expressions and find which ones have\n        string literal indexes\n        '
        self.getitems = getitems = {}
        self.block = block
        self.calltypes = calltypes
        for expr in block.find_exprs(op='getitem'):
            if expr.op == 'getitem':
                index_ty = typemap[expr.index.name]
                if isinstance(index_ty, types.StringLiteral):
                    getitems[expr] = (expr.index, index_ty.literal_value)
        return len(getitems) > 0

    def apply(self):
        if False:
            print('Hello World!')
        '\n        Rewrite all matching getitems as static_getitems where the index\n        is the literal value of the string.\n        '
        new_block = ir.Block(self.block.scope, self.block.loc)
        for inst in self.block.body:
            if isinstance(inst, ir.Assign):
                expr = inst.value
                if expr in self.getitems:
                    (const, lit_val) = self.getitems[expr]
                    new_expr = ir.Expr.static_getitem(value=expr.value, index=lit_val, index_var=expr.index, loc=expr.loc)
                    self.calltypes[new_expr] = self.calltypes[expr]
                    inst = ir.Assign(value=new_expr, target=inst.target, loc=inst.loc)
            new_block.append(inst)
        return new_block

@register_rewrite('after-inference')
class RewriteStringLiteralSetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `setitem(value=arr, index=$XX, value=)`
    where `$XX` is a StringLiteral value as
    `static_setitem(value=arr, index=<literal value>, value=)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            print('Hello World!')
        '\n        Detect all setitem expressions and find which ones have\n        string literal indexes\n        '
        self.setitems = setitems = {}
        self.block = block
        self.calltypes = calltypes
        for inst in block.find_insts(ir.SetItem):
            index_ty = typemap[inst.index.name]
            if isinstance(index_ty, types.StringLiteral):
                setitems[inst] = (inst.index, index_ty.literal_value)
        return len(setitems) > 0

    def apply(self):
        if False:
            i = 10
            return i + 15
        '\n        Rewrite all matching setitems as static_setitems where the index\n        is the literal value of the string.\n        '
        new_block = ir.Block(self.block.scope, self.block.loc)
        for inst in self.block.body:
            if isinstance(inst, ir.SetItem):
                if inst in self.setitems:
                    (const, lit_val) = self.setitems[inst]
                    new_inst = ir.StaticSetItem(target=inst.target, index=lit_val, index_var=inst.index, value=inst.value, loc=inst.loc)
                    self.calltypes[new_inst] = self.calltypes[inst]
                    inst = new_inst
            new_block.append(inst)
        return new_block

@register_rewrite('before-inference')
class RewriteConstSetitems(Rewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            for i in range(10):
                print('nop')
        self.setitems = setitems = {}
        self.block = block
        for inst in block.find_insts(ir.SetItem):
            try:
                const = func_ir.infer_constant(inst.index)
            except errors.ConstantInferenceError:
                continue
            setitems[inst] = const
        return len(setitems) > 0

    def apply(self):
        if False:
            return 10
        '\n        Rewrite all matching setitems as static_setitems.\n        '
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.setitems:
                const = self.setitems[inst]
                new_inst = ir.StaticSetItem(inst.target, const, inst.index, inst.value, inst.loc)
                new_block.append(new_inst)
            else:
                new_block.append(inst)
        return new_block