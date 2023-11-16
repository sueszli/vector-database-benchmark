""" This simple Python script can be run to generate
ztriangle_code_*.h, ztriangle_table.*, and ztriangle_*.cxx, which
are a poor man's form of generated code to cover the explosion of
different rendering options while scanning out triangles.

Each different combination of options is compiled to a different
inner-loop triangle scan function.  The code in
tinyGraphicsStateGuardian.cxx will select the appropriate function
pointer at draw time. """
NumSegments = 4
Options = [['zon', 'zoff'], ['cstore', 'cblend', 'cgeneral', 'coff', 'csstore', 'csblend'], ['anone', 'aless', 'amore'], ['znone', 'zless'], ['tnearest', 'tmipmap', 'tgeneral']]
OptionsCount = reduce(lambda a, b: a * b, map(lambda o: len(o), Options))
ExtraOptions = [['white', 'flat', 'smooth'], ['untextured', 'textured', 'perspective', 'multitex2', 'multitex3']]
ExtraOptionsMat = []
for i in range(len(ExtraOptions[0])):
    for j in range(len(ExtraOptions[1])):
        ExtraOptionsMat.append([i, j])
FullOptions = Options + ExtraOptions
CodeTable = {'zon': '#define STORE_Z(zpix, z) (zpix) = (z)', 'zoff': '#define STORE_Z(zpix, z)', 'cstore': '#define STORE_PIX(pix, rgb, r, g, b, a) (pix) = (rgb)', 'cblend': '#define STORE_PIX(pix, rgb, r, g, b, a) (pix) = PIXEL_BLEND_RGB(pix, r, g, b, a)', 'cgeneral': '#define STORE_PIX(pix, rgb, r, g, b, a) zb->store_pix_func(zb, pix, r, g, b, a)', 'coff': '#define STORE_PIX(pix, rgb, r, g, b, a)', 'csstore': '#define STORE_PIX(pix, rgb, r, g, b, a) (pix) = SRGBA_TO_PIXEL(r, g, b, a)', 'csblend': '#define STORE_PIX(pix, rgb, r, g, b, a) (pix) = PIXEL_BLEND_SRGB(pix, r, g, b, a)', 'anone': '#define ACMP(zb, a) 1', 'aless': '#define ACMP(zb, a) (((int)(a)) < (zb)->reference_alpha)', 'amore': '#define ACMP(zb, a) (((int)(a)) > (zb)->reference_alpha)', 'znone': '#define ZCMP(zpix, z) 1', 'zless': '#define ZCMP(zpix, z) ((ZPOINT)(zpix) < (ZPOINT)(z))', 'tnearest': '#define CALC_MIPMAP_LEVEL(mipmap_level, mipmap_dx, dsdx, dtdx)\n#define ZB_LOOKUP_TEXTURE(texture_def, s, t, level, level_dx) ZB_LOOKUP_TEXTURE_NEAREST(texture_def, s, t)', 'tmipmap': '#define CALC_MIPMAP_LEVEL(mipmap_level, mipmap_dx, dsdx, dtdx) DO_CALC_MIPMAP_LEVEL(mipmap_level, mipmap_dx, dsdx, dtdx)\n#define INTERP_MIPMAP\n#define ZB_LOOKUP_TEXTURE(texture_def, s, t, level, level_dx) ZB_LOOKUP_TEXTURE_MIPMAP_NEAREST(texture_def, s, t, level)', 'tgeneral': '#define CALC_MIPMAP_LEVEL(mipmap_level, mipmap_dx, dsdx, dtdx) DO_CALC_MIPMAP_LEVEL(mipmap_level, mipmap_dx, dsdx, dtdx)\n#define INTERP_MIPMAP\n#define ZB_LOOKUP_TEXTURE(texture_def, s, t, level, level_dx) ((level == 0) ? (texture_def)->tex_magfilter_func(texture_def, s, t, level, level_dx) : (texture_def)->tex_minfilter_func(texture_def, s, t, level, level_dx))'}
ZTriangleStub = '\n/* This file is generated code--do not edit.  See ztriangle.py. */\n#include <stdlib.h>\n#include <stdio.h>\n#include "pandabase.h"\n#include "zbuffer.h"\n\n/* Pick up all of the generated code references to ztriangle_two.h,\n   which ultimately calls ztriangle.h, many, many times. */\n\n#include "ztriangle_table.h"\n#include "ztriangle_code_%s.h"\n'
ops = [0] * len(Options)

class DoneException:
    pass
code = None
codeSeg = None
fnameDict = {}
fnameList = None

def incrementOptions(ops, i=-1):
    if False:
        for i in range(10):
            print('nop')
    if i < -len(ops):
        raise DoneException
    if ops[i] + 1 < len(Options[i]):
        ops[i] += 1
        return
    ops[i] = 0
    incrementOptions(ops, i - 1)

def getFname(ops):
    if False:
        for i in range(10):
            print('nop')
    keywordList = []
    for i in range(len(ops)):
        keyword = FullOptions[i][ops[i]]
        keywordList.append(keyword)
    if keywordList[-1].startswith('multitex'):
        keywordList[-2] = 'smooth'
    fname = 'FB_triangle_%s' % '_'.join(keywordList)
    return fname

def getFref(ops):
    if False:
        for i in range(10):
            print('nop')
    fname = getFname(ops)
    (codeSeg, i) = fnameDict[fname]
    fref = 'ztriangle_code_%s[%s]' % (codeSeg, i)
    return fref

def closeCode():
    if False:
        i = 10
        return i + 15
    ' Close the previously-opened code file. '
    if code:
        (print >> code, '')
        (print >> code, 'ZB_fillTriangleFunc ztriangle_code_%s[%s] = {' % (codeSeg, len(fnameList)))
        for fname in fnameList:
            (print >> code, '  %s,' % fname)
        (print >> code, '};')
        code.close()

def openCode(count):
    if False:
        while True:
            i = 10
    ' Open the code file appropriate to the current segment.  We\n    write out the generated code into a series of smaller files,\n    instead of one mammoth file, just to make it easier on the\n    compiler. '
    global code, codeSeg, fnameList
    seg = int(NumSegments * count / OptionsCount) + 1
    if codeSeg != seg:
        closeCode()
        codeSeg = seg
        fnameList = []
        code = open('ztriangle_code_%s.h' % codeSeg, 'wb')
        (print >> code, '/* This file is generated code--do not edit.  See ztriangle.py. */')
        (print >> code, '')
        zt = open('ztriangle_%s.cxx' % codeSeg, 'wb')
        (print >> zt, ZTriangleStub % codeSeg)
count = 0
try:
    while True:
        openCode(count)
        for i in range(len(ops)):
            keyword = Options[i][ops[i]]
            (print >> code, CodeTable[keyword])
        fname = getFname(ops)
        (print >> code, '#define FNAME(name) %s_ ## name' % fname)
        (print >> code, '#include "ztriangle_two.h"')
        (print >> code, '')
        for eops in ExtraOptionsMat:
            fops = ops + eops
            fname = getFname(fops)
            fnameDict[fname] = (codeSeg, len(fnameList))
            fnameList.append(fname)
        count += 1
        incrementOptions(ops)
        assert count < OptionsCount
except DoneException:
    pass
assert count == OptionsCount
closeCode()
table_decl = open('ztriangle_table.h', 'wb')
(print >> table_decl, '/* This file is generated code--do not edit.  See ztriangle.py. */')
(print >> table_decl, '')
table_def = open('ztriangle_table.cxx', 'wb')
(print >> table_def, '/* This file is generated code--do not edit.  See ztriangle.py. */')
(print >> table_def, '')
(print >> table_def, '#include "pandabase.h"')
(print >> table_def, '#include "zbuffer.h"')
(print >> table_def, '#include "ztriangle_table.h"')
(print >> table_def, '')
for i in range(NumSegments):
    (print >> table_def, 'extern ZB_fillTriangleFunc ztriangle_code_%s[];' % (i + 1))
(print >> table_def, '')

def writeTableEntry(ops):
    if False:
        i = 10
        return i + 15
    indent = '  ' * (len(ops) + 1)
    i = len(ops)
    numOps = len(FullOptions[i])
    if i + 1 == len(FullOptions):
        for j in range(numOps - 1):
            (print >> table_def, indent + getFref(ops + [j]) + ',')
        (print >> table_def, indent + getFref(ops + [numOps - 1]))
    else:
        for j in range(numOps - 1):
            (print >> table_def, indent + '{')
            writeTableEntry(ops + [j])
            (print >> table_def, indent + '},')
        (print >> table_def, indent + '{')
        writeTableEntry(ops + [numOps - 1])
        (print >> table_def, indent + '}')
arraySizeList = []
for opList in FullOptions:
    arraySizeList.append('[%s]' % len(opList))
arraySize = ''.join(arraySizeList)
(print >> table_def, 'const ZB_fillTriangleFunc fill_tri_funcs%s = {' % arraySize)
(print >> table_decl, 'extern const ZB_fillTriangleFunc fill_tri_funcs%s;' % arraySize)
writeTableEntry([])
(print >> table_def, '};')