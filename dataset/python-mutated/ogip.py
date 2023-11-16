"""
Handles units in `Office of Guest Investigator Programs (OGIP)
FITS files
<https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_93_001/>`__.
"""
import copy
import keyword
import math
import warnings
from fractions import Fraction
from astropy.utils import parsing
from . import core, generic, utils

class OGIP(generic.Generic):
    """
    Support the units in `Office of Guest Investigator Programs (OGIP)
    FITS files
    <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_93_001/>`__.
    """
    _tokens = ('DIVISION', 'OPEN_PAREN', 'CLOSE_PAREN', 'WHITESPACE', 'STARSTAR', 'STAR', 'SIGN', 'UFLOAT', 'LIT10', 'UINT', 'UNKNOWN', 'UNIT')

    @staticmethod
    def _generate_unit_names():
        if False:
            i = 10
            return i + 15
        from astropy import units as u
        names = {}
        deprecated_names = set()
        bases = ['A', 'C', 'cd', 'eV', 'F', 'g', 'H', 'Hz', 'J', 'Jy', 'K', 'lm', 'lx', 'm', 'mol', 'N', 'ohm', 'Pa', 'pc', 'rad', 's', 'S', 'sr', 'T', 'V', 'W', 'Wb']
        deprecated_bases = []
        prefixes = ['y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', 'c', 'd', '', 'da', 'h', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
        for base in bases + deprecated_bases:
            for prefix in prefixes:
                key = prefix + base
                if keyword.iskeyword(key):
                    continue
                names[key] = getattr(u, key)
        for base in deprecated_bases:
            for prefix in prefixes:
                deprecated_names.add(prefix + base)
        simple_units = ['angstrom', 'arcmin', 'arcsec', 'AU', 'barn', 'bin', 'byte', 'chan', 'count', 'day', 'deg', 'erg', 'G', 'h', 'lyr', 'mag', 'min', 'photon', 'pixel', 'voxel', 'yr']
        for unit in simple_units:
            names[unit] = getattr(u, unit)
        Crab = u.def_unit(['Crab'], prefixes=False, doc='Crab (X-ray flux)')
        mCrab = u.Unit(10 ** (-3) * Crab)
        names['Crab'] = Crab
        names['mCrab'] = mCrab
        deprecated_units = ['Crab', 'mCrab']
        for unit in deprecated_units:
            deprecated_names.add(unit)
        functions = ['log', 'ln', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
        for name in functions:
            names[name] = name
        return (names, deprecated_names, functions)

    @classmethod
    def _make_lexer(cls):
        if False:
            for i in range(10):
                print('nop')
        tokens = cls._tokens
        t_DIVISION = '/'
        t_OPEN_PAREN = '\\('
        t_CLOSE_PAREN = '\\)'
        t_WHITESPACE = '[ \t]+'
        t_STARSTAR = '\\*\\*'
        t_STAR = '\\*'

        def t_UFLOAT(t):
            if False:
                print('Hello World!')
            '(((\\d+\\.?\\d*)|(\\.\\d+))([eE][+-]?\\d+))|(((\\d+\\.\\d*)|(\\.\\d+))([eE][+-]?\\d+)?)'
            t.value = float(t.value)
            return t

        def t_UINT(t):
            if False:
                i = 10
                return i + 15
            '\\d+'
            t.value = int(t.value)
            return t

        def t_SIGN(t):
            if False:
                i = 10
                return i + 15
            '[+-](?=\\d)'
            t.value = float(t.value + '1')
            return t

        def t_X(t):
            if False:
                print('Hello World!')
            '[x×]'
            return t

        def t_LIT10(t):
            if False:
                i = 10
                return i + 15
            '10'
            return 10

        def t_UNKNOWN(t):
            if False:
                while True:
                    i = 10
            '[Uu][Nn][Kk][Nn][Oo][Ww][Nn]'
            return None

        def t_UNIT(t):
            if False:
                for i in range(10):
                    print('nop')
            '[a-zA-Z][a-zA-Z_]*'
            t.value = cls._get_unit(t)
            return t
        t_ignore = ''

        def t_error(t):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError(f'Invalid character at col {t.lexpos}')
        return parsing.lex(lextab='ogip_lextab', package='astropy/units')

    @classmethod
    def _make_parser(cls):
        if False:
            print('Hello World!')
        '\n        The grammar here is based on the description in the\n        `Specification of Physical Units within OGIP FITS files\n        <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_93_001/>`__,\n        which is not terribly precise.  The exact grammar is here is\n        based on the YACC grammar in the `unity library\n        <https://bitbucket.org/nxg/unity/>`_.\n        '
        tokens = cls._tokens

        def p_main(p):
            if False:
                i = 10
                return i + 15
            '\n            main : UNKNOWN\n                 | complete_expression\n                 | scale_factor complete_expression\n                 | scale_factor WHITESPACE complete_expression\n            '
            if len(p) == 4:
                p[0] = p[1] * p[3]
            elif len(p) == 3:
                p[0] = p[1] * p[2]
            else:
                p[0] = p[1]

        def p_complete_expression(p):
            if False:
                for i in range(10):
                    print('nop')
            '\n            complete_expression : product_of_units\n            '
            p[0] = p[1]

        def p_product_of_units(p):
            if False:
                return 10
            '\n            product_of_units : unit_expression\n                             | division unit_expression\n                             | product_of_units product unit_expression\n                             | product_of_units division unit_expression\n            '
            if len(p) == 4:
                if p[2] == 'DIVISION':
                    p[0] = p[1] / p[3]
                else:
                    p[0] = p[1] * p[3]
            elif len(p) == 3:
                p[0] = p[2] ** (-1)
            else:
                p[0] = p[1]

        def p_unit_expression(p):
            if False:
                while True:
                    i = 10
            '\n            unit_expression : unit\n                            | UNIT OPEN_PAREN complete_expression CLOSE_PAREN\n                            | OPEN_PAREN complete_expression CLOSE_PAREN\n                            | UNIT OPEN_PAREN complete_expression CLOSE_PAREN power numeric_power\n                            | OPEN_PAREN complete_expression CLOSE_PAREN power numeric_power\n            '
            p1_str = str(p[1])
            if p1_str in cls._functions and p1_str != 'sqrt':
                raise ValueError(f"The function '{p[1]}' is valid in OGIP, but not understood by astropy.units.")
            if len(p) == 7:
                if p1_str == 'sqrt':
                    p[0] = p[1] * p[3] ** (0.5 * p[6])
                else:
                    p[0] = p[1] * p[3] ** p[6]
            elif len(p) == 6:
                p[0] = p[2] ** p[5]
            elif len(p) == 5:
                if p1_str == 'sqrt':
                    p[0] = p[3] ** 0.5
                else:
                    p[0] = p[1] * p[3]
            elif len(p) == 4:
                p[0] = p[2]
            else:
                p[0] = p[1]

        def p_scale_factor(p):
            if False:
                while True:
                    i = 10
            '\n            scale_factor : LIT10 power numeric_power\n                         | LIT10\n                         | signed_float\n                         | signed_float power numeric_power\n                         | signed_int power numeric_power\n            '
            if len(p) == 4:
                p[0] = 10 ** p[3]
            else:
                p[0] = p[1]
            if math.log10(p[0]) % 1.0 != 0.0:
                from astropy.units.core import UnitsWarning
                warnings.warn(f"'{p[0]}' scale should be a power of 10 in OGIP format", UnitsWarning)

        def p_division(p):
            if False:
                for i in range(10):
                    print('nop')
            '\n            division : DIVISION\n                     | WHITESPACE DIVISION\n                     | WHITESPACE DIVISION WHITESPACE\n                     | DIVISION WHITESPACE\n            '
            p[0] = 'DIVISION'

        def p_product(p):
            if False:
                return 10
            '\n            product : WHITESPACE\n                    | STAR\n                    | WHITESPACE STAR\n                    | WHITESPACE STAR WHITESPACE\n                    | STAR WHITESPACE\n            '
            p[0] = 'PRODUCT'

        def p_power(p):
            if False:
                return 10
            '\n            power : STARSTAR\n            '
            p[0] = 'POWER'

        def p_unit(p):
            if False:
                i = 10
                return i + 15
            '\n            unit : UNIT\n                 | UNIT power numeric_power\n            '
            if len(p) == 4:
                p[0] = p[1] ** p[3]
            else:
                p[0] = p[1]

        def p_numeric_power(p):
            if False:
                while True:
                    i = 10
            '\n            numeric_power : UINT\n                          | signed_float\n                          | OPEN_PAREN signed_int CLOSE_PAREN\n                          | OPEN_PAREN signed_float CLOSE_PAREN\n                          | OPEN_PAREN signed_float division UINT CLOSE_PAREN\n            '
            if len(p) == 6:
                p[0] = Fraction(int(p[2]), int(p[4]))
            elif len(p) == 4:
                p[0] = p[2]
            else:
                p[0] = p[1]

        def p_sign(p):
            if False:
                while True:
                    i = 10
            '\n            sign : SIGN\n                 |\n            '
            if len(p) == 2:
                p[0] = p[1]
            else:
                p[0] = 1.0

        def p_signed_int(p):
            if False:
                print('Hello World!')
            '\n            signed_int : SIGN UINT\n            '
            p[0] = p[1] * p[2]

        def p_signed_float(p):
            if False:
                for i in range(10):
                    print('nop')
            '\n            signed_float : sign UINT\n                         | sign UFLOAT\n            '
            p[0] = p[1] * p[2]

        def p_error(p):
            if False:
                print('Hello World!')
            raise ValueError()
        return parsing.yacc(tabmodule='ogip_parsetab', package='astropy/units')

    @classmethod
    def _validate_unit(cls, unit, detailed_exception=True):
        if False:
            print('Hello World!')
        if unit not in cls._units:
            if detailed_exception:
                raise ValueError(f"Unit '{unit}' not supported by the OGIP standard. " + utils.did_you_mean_units(unit, cls._units, cls._deprecated_units, cls._to_decomposed_alternative))
            else:
                raise ValueError()
        if unit in cls._deprecated_units:
            utils.unit_deprecation_warning(unit, cls._units[unit], 'OGIP', cls._to_decomposed_alternative)

    @classmethod
    def _parse_unit(cls, unit, detailed_exception=True):
        if False:
            return 10
        cls._validate_unit(unit, detailed_exception=detailed_exception)
        return cls._units[unit]

    @classmethod
    def parse(cls, s, debug=False):
        if False:
            while True:
                i = 10
        s = s.strip()
        try:
            return cls._parse_unit(s, detailed_exception=False)
        except ValueError:
            try:
                return core.Unit(cls._parser.parse(s, lexer=cls._lexer, debug=debug))
            except ValueError as e:
                if str(e):
                    raise
                else:
                    raise ValueError(f"Syntax error parsing unit '{s}'")

    @classmethod
    def _get_unit_name(cls, unit):
        if False:
            i = 10
            return i + 15
        name = super()._get_unit_name(unit)
        cls._validate_unit(name)
        return name

    @classmethod
    def _format_superscript(cls, number):
        if False:
            for i in range(10):
                print('nop')
        return f'**({number})' if '/' in number else f'**{number}'

    @classmethod
    def to_string(cls, unit, fraction='inline'):
        if False:
            i = 10
            return i + 15
        unit = utils.decompose_to_known_units(unit, cls._get_unit_name)
        if isinstance(unit, core.CompositeUnit):
            if math.log10(unit.scale) % 1.0 != 0.0:
                warnings.warn(f"'{unit.scale}' scale should be a power of 10 in OGIP format", core.UnitsWarning)
        return super().to_string(unit, fraction=fraction)

    @classmethod
    def _to_decomposed_alternative(cls, unit):
        if False:
            while True:
                i = 10
        unit = utils.decompose_to_known_units(unit, cls._get_unit_name)
        if isinstance(unit, core.CompositeUnit):
            if math.log10(unit.scale) % 1.0 != 0.0:
                scale = unit.scale
                unit = copy.copy(unit)
                unit._scale = 1.0
                return f'{generic._to_string(cls, unit)} (with data multiplied by {scale})'
        return super().to_string(unit)