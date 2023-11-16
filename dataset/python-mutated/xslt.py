import sys
import libxml2
import libxslt
from docbook import adjustColumnWidths
usage = 'Usage: %s xmlfile.xml xslfile.xsl [outputfile] [param1=val [param2=val]...]' % sys.argv[0]
xmlfile = None
xslfile = None
outfile = '-'
params = {}
try:
    xmlfile = sys.argv[1]
    xslfile = sys.argv[2]
except IndexError:
    print(usage)
    sys.exit(1)

def quote(astring):
    if False:
        i = 10
        return i + 15
    if astring.find("'") < 0:
        return "'" + astring + "'"
    else:
        return '"' + astring + '"'
try:
    outfile = sys.argv[3]
    if outfile.find('=') > 0:
        (name, value) = outfile.split('=', 2)
        params[name] = quote(value)
        outfile = None
    count = 4
    while sys.argv[count]:
        try:
            (name, value) = sys.argv[count].split('=', 2)
            if name in params:
                print("Warning: '%s' re-specified; replacing value" % name)
            params[name] = quote(value)
        except ValueError:
            print("Invalid parameter specification: '" + sys.argv[count] + "'")
            print(usage)
            sys.exit(1)
        count = count + 1
except IndexError:
    pass
libxml2.lineNumbersDefault(1)
libxml2.substituteEntitiesDefault(1)
libxslt.registerExtModuleFunction('adjustColumnWidths', 'http://nwalsh.com/xslt/ext/xsltproc/python/Table', adjustColumnWidths)
styledoc = libxml2.parseFile(xslfile)
style = libxslt.parseStylesheetDoc(styledoc)
doc = libxml2.parseFile(xmlfile)
result = style.applyStylesheet(doc, params)
if outfile:
    style.saveResultToFilename(outfile, result, 0)
else:
    print(result)
style.freeStylesheet()
doc.freeDoc()
result.freeDoc()