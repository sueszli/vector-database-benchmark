import re
from .style import ListLevelProperties
from .text import ListStyle, ListLevelStyleNumber, ListLevelStyleBullet
from polyglot.builtins import unicode_type
'\nCreate a <text:list-style> element from a string or array.\n\nList styles require a lot of code to create one level at a time.\nThese routines take a string and delimiter, or a list of\nstrings, and creates a <text:list-style> element for you.\nEach item in the string (or array) represents a list level\n * style for levels 1-10.</p>\n *\n * <p>If an item contains <code>1</code>, <code>I</code>,\n * <code>i</code>, <code>A</code>, or <code>a</code>, then it is presumed\n * to be a numbering style; otherwise it is a bulleted style.</p>\n'
_MAX_LIST_LEVEL = 10
SHOW_ALL_LEVELS = True
SHOW_ONE_LEVEL = False

def styleFromString(name, specifiers, delim, spacing, showAllLevels):
    if False:
        for i in range(10):
            print('nop')
    specArray = specifiers.split(delim)
    return styleFromList(name, specArray, spacing, showAllLevels)

def styleFromList(styleName, specArray, spacing, showAllLevels):
    if False:
        for i in range(10):
            print('nop')
    bullet = ''
    numPrefix = ''
    numSuffix = ''
    cssLengthNum = 0
    cssLengthUnits = ''
    numbered = False
    displayLevels = 0
    listStyle = ListStyle(name=styleName)
    numFormatPattern = re.compile('([1IiAa])')
    cssLengthPattern = re.compile('([^a-z]+)\\s*([a-z]+)?')
    m = cssLengthPattern.search(spacing)
    if m is not None:
        cssLengthNum = float(m.group(1))
        if m.lastindex == 2:
            cssLengthUnits = m.group(2)
    i = 0
    while i < len(specArray):
        specification = specArray[i]
        m = numFormatPattern.search(specification)
        if m is not None:
            numPrefix = specification[0:m.start(1)]
            numSuffix = specification[m.end(1):]
            bullet = ''
            numbered = True
            if showAllLevels:
                displayLevels = i + 1
            else:
                displayLevels = 1
        else:
            bullet = specification
            numPrefix = ''
            numSuffix = ''
            displayLevels = 1
            numbered = False
        if numbered:
            lls = ListLevelStyleNumber(level=i + 1)
            if numPrefix != '':
                lls.setAttribute('numprefix', numPrefix)
            if numSuffix != '':
                lls.setAttribute('numsuffix', numSuffix)
            lls.setAttribute('displaylevels', displayLevels)
        else:
            lls = ListLevelStyleBullet(level=i + 1, bulletchar=bullet[0])
        llp = ListLevelProperties()
        llp.setAttribute('spacebefore', unicode_type(cssLengthNum * (i + 1)) + cssLengthUnits)
        llp.setAttribute('minlabelwidth', unicode_type(cssLengthNum) + cssLengthUnits)
        lls.addElement(llp)
        listStyle.addElement(lls)
        i += 1
    return listStyle