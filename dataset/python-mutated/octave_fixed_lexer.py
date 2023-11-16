import re
from string import whitespace
from pygments.lexer import RegexLexer, bygroups, include, using, this
from pygments.token import *

class OctaveFixedLexer(RegexLexer):
    """
    For GNU Octave source code.
    Contributed by Ken Schutte <kschutte@csail.mit.edu>.
    Fixed for " used by Octave instead of '
    """
    name = 'OctaveFixed'
    aliases = ['octave_fixed']
    filenames = ['*.m']
    mimetypes = ['text/octave']
    elfun = ['sin', 'sind', 'sinh', 'asin', 'asind', 'asinh', 'cos', 'cosd', 'cosh', 'acos', 'acosd', 'acosh', 'tan', 'tand', 'tanh', 'atan', 'atand', 'atan2', 'atanh', 'sec', 'secd', 'sech', 'asec', 'asecd', 'asech', 'csc', 'cscd', 'csch', 'acsc', 'acscd', 'acsch', 'cot', 'cotd', 'coth', 'acot', 'acotd', 'acoth', 'hypot', 'exp', 'expm1', 'log', 'log1p', 'log10', 'log2', 'pow2', 'realpow', 'reallog', 'realsqrt', 'sqrt', 'nthroot', 'nextpow2', 'abs', 'angle', 'complex', 'conj', 'imag', 'real', 'unwrap', 'isreal', 'cplxpair', 'fix', 'floor', 'ceil', 'round', 'mod', 'rem', 'sign']
    specfun = ['airy', 'besselj', 'bessely', 'besselh', 'besseli', 'besselk', 'beta', 'betainc', 'betaln', 'ellipj', 'ellipke', 'erf', 'erfc', 'erfcx', 'erfinv', 'expint', 'gamma', 'gammainc', 'gammaln', 'psi', 'legendre', 'cross', 'dot', 'factor', 'isprime', 'primes', 'gcd', 'lcm', 'rat', 'rats', 'perms', 'nchoosek', 'factorial', 'cart2sph', 'cart2pol', 'pol2cart', 'sph2cart', 'hsv2rgb', 'rgb2hsv']
    elmat = ['zeros', 'ones', 'eye', 'repmat', 'rand', 'randn', 'linspace', 'logspace', 'freqspace', 'meshgrid', 'accumarray', 'size', 'length', 'ndims', 'numel', 'disp', 'isempty', 'isequal', 'isequalwithequalnans', 'cat', 'reshape', 'diag', 'blkdiag', 'tril', 'triu', 'fliplr', 'flipud', 'flipdim', 'rot90', 'find', 'end', 'sub2ind', 'ind2sub', 'bsxfun', 'ndgrid', 'permute', 'ipermute', 'shiftdim', 'circshift', 'squeeze', 'isscalar', 'isvector', 'ans', 'eps', 'realmax', 'realmin', 'pi', 'i', 'inf', 'nan', 'isnan', 'isinf', 'isfinite', 'j', 'why', 'compan', 'gallery', 'hadamard', 'hankel', 'hilb', 'invhilb', 'magic', 'pascal', 'rosser', 'toeplitz', 'vander', 'wilkinson']
    tokens = {'root': [('^!.*', String.Other), ('%.*$', Comment), ('^\\s*function', Keyword, 'deffunc'), ('(break|case|catch|classdef|continue|else|elseif|end|enumerated|events|for|function|global|if|methods|otherwise|parfor|persistent|properties|return|spmd|switch|try|while)\\b', Keyword), ('(' + '|'.join(elfun + specfun + elmat) + ')\\b', Name.Builtin), ('-|==|~=|<|>|<=|>=|&&|&|~|\\|\\|?', Operator), ('\\.\\*|\\*|\\+|\\.\\^|\\.\\\\|\\.\\/|\\/|\\\\', Operator), ('\\[|\\]|\\(|\\)|\\{|\\}|:|@|\\.|,', Punctuation), ('=|:|;', Punctuation), ("(?<=[\\w\\)\\]])\\'", Operator), ('(?<![\\w\\)\\]])\\"', String, 'string'), ('[a-zA-Z_][a-zA-Z0-9_]*', Name), ('.', Text)], 'string': [('[^\\"]*\\"', String, '#pop')], 'deffunc': [('(\\s*)(?:(.+)(\\s*)(=)(\\s*))?(.+)(\\()(.*)(\\))(\\s*)', bygroups(Text.Whitespace, Text, Text.Whitespace, Punctuation, Text.Whitespace, Name.Function, Punctuation, Text, Punctuation, Text.Whitespace), '#pop')]}

    def analyse_text(text):
        if False:
            while True:
                i = 10
        if re.match('^\\s*%', text, re.M):
            return 0.9
        elif re.match('^!\\w+', text, re.M):
            return 0.9
        return 0.1