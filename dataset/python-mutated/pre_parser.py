import io
import re
from tokenize import COMMENT, NAME, OP, TokenError, TokenInfo, tokenize, untokenize
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from vyper.compiler.settings import OptimizationLevel, Settings
from vyper.evm.opcodes import EVM_VERSIONS
from vyper.exceptions import StructureException, SyntaxException, VersionException
from vyper.typing import ModificationOffsets, ParserPosition

def validate_version_pragma(version_str: str, start: ParserPosition) -> None:
    if False:
        return 10
    '\n    Validates a version pragma directive against the current compiler version.\n    '
    from vyper import __version__
    if len(version_str) == 0:
        raise VersionException('Version specification cannot be empty', start)
    if re.match('[v0-9]', version_str):
        version_str = '==' + version_str
    version_str = re.sub('^\\^', '~=', version_str)
    try:
        spec = SpecifierSet(version_str)
    except InvalidSpecifier:
        raise VersionException(f'Version specification "{version_str}" is not a valid PEP440 specifier', start)
    if not spec.contains(__version__, prereleases=True):
        raise VersionException(f'Version specification "{version_str}" is not compatible with compiler version "{__version__}"', start)
VYPER_CLASS_TYPES = {'enum', 'event', 'interface', 'struct'}
VYPER_EXPRESSION_TYPES = {'log'}

def pre_parse(code: str) -> tuple[Settings, ModificationOffsets, str]:
    if False:
        print('Hello World!')
    '\n    Re-formats a vyper source string into a python source string and performs\n    some validation.  More specifically,\n\n    * Translates "interface", "struct", "enum, and "event" keywords into python "class" keyword\n    * Validates "@version" pragma against current compiler version\n    * Prevents direct use of python "class" keyword\n    * Prevents use of python semi-colon statement separator\n\n    Also returns a mapping of detected interface and struct names to their\n    respective vyper class types ("interface" or "struct").\n\n    Parameters\n    ----------\n    code : str\n        The vyper source code to be re-formatted.\n\n    Returns\n    -------\n    dict\n        Mapping of offsets where source was modified.\n    str\n        Reformatted python source string.\n    '
    result = []
    modification_offsets: ModificationOffsets = {}
    settings = Settings()
    try:
        code_bytes = code.encode('utf-8')
        token_list = list(tokenize(io.BytesIO(code_bytes).readline))
        for i in range(len(token_list)):
            token = token_list[i]
            toks = [token]
            typ = token.type
            string = token.string
            start = token.start
            end = token.end
            line = token.line
            if typ == COMMENT:
                contents = string[1:].strip()
                if contents.startswith('@version'):
                    if settings.compiler_version is not None:
                        raise StructureException('compiler version specified twice!', start)
                    compiler_version = contents.removeprefix('@version ').strip()
                    validate_version_pragma(compiler_version, start)
                    settings.compiler_version = compiler_version
                if contents.startswith('pragma '):
                    pragma = contents.removeprefix('pragma ').strip()
                    if pragma.startswith('version '):
                        if settings.compiler_version is not None:
                            raise StructureException('pragma version specified twice!', start)
                        compiler_version = pragma.removeprefix('version ').strip()
                        validate_version_pragma(compiler_version, start)
                        settings.compiler_version = compiler_version
                    elif pragma.startswith('optimize '):
                        if settings.optimize is not None:
                            raise StructureException('pragma optimize specified twice!', start)
                        try:
                            mode = pragma.removeprefix('optimize').strip()
                            settings.optimize = OptimizationLevel.from_string(mode)
                        except ValueError:
                            raise StructureException(f'Invalid optimization mode `{mode}`', start)
                    elif pragma.startswith('evm-version '):
                        if settings.evm_version is not None:
                            raise StructureException('pragma evm-version specified twice!', start)
                        evm_version = pragma.removeprefix('evm-version').strip()
                        if evm_version not in EVM_VERSIONS:
                            raise StructureException('Invalid evm version: `{evm_version}`', start)
                        settings.evm_version = evm_version
                    else:
                        raise StructureException(f'Unknown pragma `{pragma.split()[0]}`')
            if typ == NAME and string in ('class', 'yield'):
                raise SyntaxException(f'The `{string}` keyword is not allowed. ', code, start[0], start[1])
            if typ == NAME:
                if string in VYPER_CLASS_TYPES and start[1] == 0:
                    toks = [TokenInfo(NAME, 'class', start, end, line)]
                    modification_offsets[start] = f'{string.capitalize()}Def'
                elif string in VYPER_EXPRESSION_TYPES:
                    toks = [TokenInfo(NAME, 'yield', start, end, line)]
                    modification_offsets[start] = string.capitalize()
            if (typ, string) == (OP, ';'):
                raise SyntaxException('Semi-colon statements not allowed', code, start[0], start[1])
            result.extend(toks)
    except TokenError as e:
        raise SyntaxException(e.args[0], code, e.args[1][0], e.args[1][1]) from e
    return (settings, modification_offsets, untokenize(result).decode('utf-8'))