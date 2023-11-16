import libxml2
import libxslt
import re
import math
pixelsPerInch = 96.0
unitHash = {'in': pixelsPerInch, 'cm': pixelsPerInch / 2.54, 'mm': pixelsPerInch / 25.4, 'pc': pixelsPerInch / 72.0 * 12, 'pt': pixelsPerInch / 72.0, 'px': 1}

def adjustColumnWidths(ctx, nodeset):
    if False:
        return 10
    try:
        pctxt = libxslt.xpathParserContext(_obj=ctx)
        ctxt = pctxt.context()
        tctxt = ctxt.transformContext()
    except:
        pass
    varString = lookupVariable(tctxt, 'nominal.table.width', None)
    if varString is None:
        nominalWidth = 6 * pixelsPerInch
    else:
        nominalWidth = convertLength(varString)
    tableWidth = lookupVariable(tctxt, 'table.width', '100%')
    foStylesheet = tctxt.variableLookup('stylesheet.result.type', None) == 'fo'
    relTotal = 0
    relParts = []
    absTotal = 0
    absParts = []
    colgroup = libxml2.xmlNode(_obj=nodeset[0])
    if foStylesheet:
        colChildren = colgroup
    else:
        colChildren = colgroup.children
    col = colChildren
    while col is not None:
        if foStylesheet:
            width = col.prop('column-width')
        else:
            width = col.prop('width')
        if width is None:
            width = '1*'
        relPart = 0.0
        absPart = 0.0
        starPos = width.find('*')
        if starPos >= 0:
            (relPart, absPart) = width.split('*', 2)
            relPart = float(relPart)
            relTotal = relTotal + float(relPart)
        else:
            absPart = width
        pixels = convertLength(absPart)
        absTotal = absTotal + pixels
        relParts.append(relPart)
        absParts.append(pixels)
        col = col.__next__
    widths = []
    if relTotal == 0:
        for absPart in absParts:
            if foStylesheet:
                inches = absPart / pixelsPerInch
                widths.append('%4.2fin' % inches)
            else:
                widths.append('%d' % absPart)
    elif absTotal == 0:
        for relPart in relParts:
            rel = relPart / relTotal * 100
            widths.append(rel)
        widths = correctRoundingError(widths)
    else:
        pixelWidth = nominalWidth
        if '%' not in tableWidth:
            pixelWidth = convertLength(tableWidth)
        if pixelWidth <= absTotal:
            print('Table is wider than table width')
        else:
            pixelWidth = pixelWidth - absTotal
        absTotal = 0
        for count in range(len(relParts)):
            rel = relParts[count] / relTotal * pixelWidth
            relParts[count] = rel + absParts[count]
            absTotal = absTotal + rel + absParts[count]
        if '%' not in tableWidth:
            for count in range(len(relParts)):
                if foStylesheet:
                    pixels = relParts[count]
                    inches = pixels / pixelsPerInch
                    widths.append('%4.2fin' % inches)
                else:
                    widths.append(relParts[count])
        else:
            for count in range(len(relParts)):
                rel = relParts[count] / absTotal * 100
                widths.append(rel)
            widths = correctRoundingError(widths)
    count = 0
    col = colChildren
    while col is not None:
        if foStylesheet:
            col.setProp('column-width', widths[count])
        else:
            col.setProp('width', widths[count])
        count = count + 1
        col = col.__next__
    return nodeset

def convertLength(length):
    if False:
        return 10
    global pixelsPerInch
    global unitHash
    m = re.search('([+-]?[\\d.]+)(\\S+)', length)
    if m is not None and m.lastindex > 1:
        unit = pixelsPerInch
        if m.group(2) in unitHash:
            unit = unitHash[m.group(2)]
        else:
            print('Unrecognized length: ' + m.group(2))
        pixels = unit * float(m.group(1))
    else:
        pixels = 0
    return pixels

def correctRoundingError(floatWidths):
    if False:
        print('Hello World!')
    totalWidth = 0
    widths = []
    for width in floatWidths:
        width = math.floor(width)
        widths.append(width)
        totalWidth = totalWidth + math.floor(width)
    totalError = 100 - totalWidth
    columnError = totalError / len(widths)
    error = 0
    for count in range(len(widths)):
        width = widths[count]
        error = error + columnError
        if error >= 1.0:
            adj = math.floor(error)
            error = error - adj
            widths[count] = '%d%%' % (width + adj)
        else:
            widths[count] = '%d%%' % width
    return widths

def lookupVariable(tctxt, varName, default):
    if False:
        i = 10
        return i + 15
    varString = tctxt.variableLookup(varName, None)
    if varString is None:
        return default
    if isinstance(varString, list):
        varString = varString[0]
    if not isinstance(varString, str):
        varString = varString.content
    return varString