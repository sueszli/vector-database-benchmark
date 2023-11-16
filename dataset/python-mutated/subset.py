import getopt
import os
import struct
import subprocess
import sys
import fontforge

def log_namelist(name, unicode):
    if False:
        return 10
    if name and isinstance(unicode, int):
        print(f'0x{unicode:04X}', fontforge.nameeFromUnicode(unicode), file=name)

def select_with_refs(font, unicode, newfont, pe=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    newfont.selection.select(('more', 'unicode'), unicode)
    log_namelist(name, unicode)
    if pe:
        print(f'SelectMore({unicode})', file=pe)
    try:
        for ref in font[unicode].references:
            newfont.selection.select(('more',), ref[0])
            log_namelist(name, ref[0])
            if pe:
                print(f'SelectMore("{ref[0]}")', file=pe)
    except Exception:
        print(f'Resolving references on u+{unicode:04x} failed')

def subset_font_raw(font_in, font_out, unicodes, opts):
    if False:
        return 10
    if '--namelist' in opts:
        name_fn = f'{font_out}.name'
        name = open(name_fn, 'w')
    else:
        name = None
    if '--script' in opts:
        pe_fn = '/tmp/script.pe'
        pe = open(pe_fn, 'w')
    else:
        pe = None
    font = fontforge.open(font_in)
    if pe:
        print(f'Open("{font_in}")', file=pe)
        extract_vert_to_script(font_in, pe)
    for i in unicodes:
        select_with_refs(font, i, font, pe, name)
    addl_glyphs = []
    if '--nmr' in opts:
        addl_glyphs.append('nonmarkingreturn')
    if '--null' in opts:
        addl_glyphs.append('.null')
    if '--nd' in opts:
        addl_glyphs.append('.notdef')
    for glyph in addl_glyphs:
        font.selection.select(('more',), glyph)
        if name:
            print(f'0x{fontforge.unicodeFromName(glyph):0.4X}', glyph, file=name)
        if pe:
            print(f'SelectMore("{glyph}")', file=pe)
    flags = ()
    if '--opentype-features' in opts:
        flags += ('opentype',)
    if '--simplify' in opts:
        font.simplify()
        font.round()
        flags += ('omit-instructions',)
    if '--strip_names' in opts:
        font.sfnt_names = ()
    if '--new' in opts:
        font.copy()
        new = fontforge.font()
        new.encoding = font.encoding
        new.em = font.em
        new.layers['Fore'].is_quadratic = font.layers['Fore'].is_quadratic
        for i in unicodes:
            select_with_refs(font, i, new, pe, name)
        new.paste()
        font.selection.select('space')
        font.copy()
        new.selection.select('space')
        new.paste()
        new.sfnt_names = font.sfnt_names
        font = new
    else:
        font.selection.invert()
        print('SelectInvert()', file=pe)
        font.cut()
        print('Clear()', file=pe)
    if '--move-display' in opts:
        print('Moving display glyphs into Unicode ranges...')
        font.familyname += ' Display'
        font.fullname += ' Display'
        font.fontname += 'Display'
        font.appendSFNTName('English (US)', 'Family', font.familyname)
        font.appendSFNTName('English (US)', 16, font.familyname)
        font.appendSFNTName('English (US)', 17, 'Display')
        font.appendSFNTName('English (US)', 'Fullname', font.fullname)
        for glname in unicodes:
            font.selection.none()
            if isinstance(glname, str):
                if glname.endswith('.display'):
                    font.selection.select(glname)
                    font.copy()
                    font.selection.none()
                    newgl = glname.replace('.display', '')
                    font.selection.select(newgl)
                    font.paste()
                font.selection.select(glname)
                font.cut()
    if name:
        print('Writing NameList', end='')
        name.close()
    if pe:
        print(f'Generate("{font_out}")', file=pe)
        pe.close()
        subprocess.call(['fontforge', '-script', pe_fn])
    else:
        font.generate(font_out, flags=flags)
    font.close()
    if '--roundtrip' in opts:
        font2 = fontforge.open(font_out)
        font2.generate(font_out, flags=flags)

def subset_font(font_in, font_out, unicodes, opts):
    if False:
        i = 10
        return i + 15
    font_out_raw = font_out
    if not font_out_raw.endswith('.ttf'):
        font_out_raw += '.ttf'
    subset_font_raw(font_in, font_out_raw, unicodes, opts)
    if font_out != font_out_raw:
        os.rename(font_out_raw, font_out)

def getsubset(subset, font_in):
    if False:
        while True:
            i = 10
    subsets = subset.split('+')
    quotes = [8211, 8212, 8216, 8217, 8218, 8220, 8221, 8222, 8226, 8249, 8250]
    latin = [*range(32, 127), *range(160, 256), 8364, 338, 339, 59, 183, 305, 710, 730, 732, 8308, 8725, 8260, 57599, 61437, 61440]
    result = quotes
    if 'menu' in subsets:
        font = fontforge.open(font_in)
        result = [*map(ord, font.familyname), 32]
    if 'latin' in subsets:
        result += latin
    if 'latin-ext' in subsets:
        result += [*range(256, 880), *range(7424, 7840), *range(7922, 7936), *range(8304, 8400), *range(11360, 11392), *range(42752, 43008)]
    if 'vietnamese' in subsets:
        result += [192, 193, 194, 195, 200, 201, 202, 204, 205, 210, 211, 212, 213, 217, 218, 221, 224, 225, 226, 227, 232, 233, 234, 236, 237, 242, 243, 244, 245, 249, 250, 253, 258, 259, 272, 273, 296, 297, 360, 361, 416, 417, 431, 432, 8363, *range(7840, 7930)]
    if 'greek' in subsets:
        result += [*range(880, 1024)]
    if 'greek-ext' in subsets:
        result += [*range(880, 1024), *range(7936, 8192)]
    if 'cyrillic' in subsets:
        result += [*range(1024, 1120), 1168, 1169, 1200, 1201, 8470]
    if 'cyrillic-ext' in subsets:
        result += [*range(1024, 1328), 8372, 8470, *range(11744, 11776), *range(42560, 42656)]
    if 'arabic' in subsets:
        result += [13, 32, 1569, 1575, 1581, 1583, 1585, 1587, 1589, 1591, 1593, 1603, 1604, 1605, 1607, 1608, 1609, 1600, 1646, 1647, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1780, 1781, 1782, 1726, 1746, 1705, 1711, 1722, 1642, 1567, 1548, 1563, 1643, 1644, 1645, 1611, 1613, 1614, 1615, 1612, 1616, 1617, 1618, 1619, 1620, 1621, 1648, 1622, 1557, 1670, 1571, 1573, 1570, 1649, 1576, 1662, 1578, 1579, 1657, 1577, 1580, 1582, 1584, 1672, 1586, 1681, 1688, 1588, 1590, 1592, 1594, 1601, 1602, 1606, 1749, 1728, 1572, 1610, 1740, 1747, 1574, 1730, 1729, 1731, 1776, 1777, 1778, 1779, 1785, 1783, 1784, 64611, 1650, 1651, 1653, 1654, 1655, 1656, 1658, 1659, 1660, 1661, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1671, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1682, 1683, 1684, 1685, 1686, 1687, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1701, 1702, 1703, 1704, 1706, 1707, 1708, 1709, 1710, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1723, 1724, 1725, 1727, 1732, 1733, 1741, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1759, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1770, 1771, 1773, 1787, 1788, 1789, 1790, 1536, 1537, 1538, 1539, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1623, 1624, 1774, 1775, 1791, 1547, 1566, 1625, 1626, 1627, 1628, 1629, 1630, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1700, 1734, 1735, 1736, 1737, 1738, 1739, 1743, 1742, 1744, 1745, 1748, 1786, 1757, 1758, 1760, 1769, 1549, 64830, 64831, 9676, 1595, 1596, 1597, 1598, 1599, 1568, 1652, 1652, 1772]
    if 'dejavu-ext' in subsets:
        font = fontforge.open(font_in)
        for glyph in font.glyphs():
            if glyph.glyphname.endswith('.display'):
                result.append(glyph.glyphname)
    return result

class Sfnt:

    def __init__(self, data):
        if False:
            while True:
                i = 10
        (_, numTables, _, _, _) = struct.unpack('>IHHHH', data[:12])
        self.tables = {}
        for i in range(numTables):
            (tag, _, offset, length) = struct.unpack('>4sIII', data[12 + 16 * i:28 + 16 * i])
            self.tables[tag] = data[offset:offset + length]

    def hhea(self):
        if False:
            while True:
                i = 10
        r = {}
        d = self.tables['hhea']
        (r['Ascender'], r['Descender'], r['LineGap']) = struct.unpack('>hhh', d[4:10])
        return r

    def os2(self):
        if False:
            print('Hello World!')
        r = {}
        d = self.tables['OS/2']
        (r['fsSelection'],) = struct.unpack('>H', d[62:64])
        (r['sTypoAscender'], r['sTypoDescender'], r['sTypoLineGap']) = struct.unpack('>hhh', d[68:74])
        (r['usWinAscender'], r['usWinDescender']) = struct.unpack('>HH', d[74:78])
        return r

def set_os2(pe, name, val):
    if False:
        while True:
            i = 10
    print(f'SetOS2Value("{name}", {val:d})', file=pe)

def set_os2_vert(pe, name, val):
    if False:
        for i in range(10):
            print('nop')
    set_os2(pe, name + 'IsOffset', 0)
    set_os2(pe, name, val)

def extract_vert_to_script(font_in, pe):
    if False:
        for i in range(10):
            print('nop')
    with open(font_in, 'rb') as in_file:
        data = in_file.read()
    sfnt = Sfnt(data)
    hhea = sfnt.hhea()
    os2 = sfnt.os2()
    set_os2_vert(pe, 'WinAscent', os2['usWinAscender'])
    set_os2_vert(pe, 'WinDescent', os2['usWinDescender'])
    set_os2_vert(pe, 'TypoAscent', os2['sTypoAscender'])
    set_os2_vert(pe, 'TypoDescent', os2['sTypoDescender'])
    set_os2_vert(pe, 'HHeadAscent', hhea['Ascender'])
    set_os2_vert(pe, 'HHeadDescent', hhea['Descender'])

def main(argv):
    if False:
        print('Hello World!')
    (optlist, args) = getopt.gnu_getopt(argv, '', ['string=', 'strip_names', 'opentype-features', 'simplify', 'new', 'script', 'nmr', 'roundtrip', 'subset=', 'namelist', 'null', 'nd', 'move-display'])
    (font_in, font_out) = args
    opts = dict(optlist)
    if '--string' in opts:
        subset = map(ord, opts['--string'])
    else:
        subset = getsubset(opts.get('--subset', 'latin'), font_in)
    subset_font(font_in, font_out, subset, opts)
if __name__ == '__main__':
    main(sys.argv[1:])