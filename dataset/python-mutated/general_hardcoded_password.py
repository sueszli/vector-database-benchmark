import ast
import re
import bandit
from bandit.core import issue
from bandit.core import test_properties as test
RE_WORDS = '(pas+wo?r?d|pass(phrase)?|pwd|token|secrete?)'
RE_CANDIDATES = re.compile('(^{0}$|_{0}_|^{0}_|_{0}$)'.format(RE_WORDS), re.IGNORECASE)

def _report(value):
    if False:
        i = 10
        return i + 15
    return bandit.Issue(severity=bandit.LOW, confidence=bandit.MEDIUM, cwe=issue.Cwe.HARD_CODED_PASSWORD, text=f"Possible hardcoded password: '{value}'")

@test.checks('Str')
@test.test_id('B105')
def hardcoded_password_string(context):
    if False:
        i = 10
        return i + 15
    '**B105: Test for use of hard-coded password strings**\n\n    The use of hard-coded passwords increases the possibility of password\n    guessing tremendously. This plugin test looks for all string literals and\n    checks the following conditions:\n\n    - assigned to a variable that looks like a password\n    - assigned to a dict key that looks like a password\n    - assigned to a class attribute that looks like a password\n    - used in a comparison with a variable that looks like a password\n\n    Variables are considered to look like a password if they have match any one\n    of:\n\n    - "password"\n    - "pass"\n    - "passwd"\n    - "pwd"\n    - "secret"\n    - "token"\n    - "secrete"\n\n    Note: this can be noisy and may generate false positives.\n\n    **Config Options:**\n\n    None\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: Possible hardcoded password \'(root)\'\n           Severity: Low   Confidence: Low\n           CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)\n           Location: ./examples/hardcoded-passwords.py:5\n        4 def someFunction2(password):\n        5     if password == "root":\n        6         print("OK, logged in")\n\n    .. seealso::\n\n        - https://www.owasp.org/index.php/Use_of_hard-coded_password\n        - https://cwe.mitre.org/data/definitions/259.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    '
    node = context.node
    if isinstance(node._bandit_parent, ast.Assign):
        for targ in node._bandit_parent.targets:
            if isinstance(targ, ast.Name) and RE_CANDIDATES.search(targ.id):
                return _report(node.s)
            elif isinstance(targ, ast.Attribute) and RE_CANDIDATES.search(targ.attr):
                return _report(node.s)
    elif isinstance(node._bandit_parent, ast.Subscript) and RE_CANDIDATES.search(node.s):
        assign = node._bandit_parent._bandit_parent
        if isinstance(assign, ast.Assign) and isinstance(assign.value, ast.Str):
            return _report(assign.value.s)
    elif isinstance(node._bandit_parent, ast.Index) and RE_CANDIDATES.search(node.s):
        assign = node._bandit_parent._bandit_parent._bandit_parent
        if isinstance(assign, ast.Assign) and isinstance(assign.value, ast.Str):
            return _report(assign.value.s)
    elif isinstance(node._bandit_parent, ast.Compare):
        comp = node._bandit_parent
        if isinstance(comp.left, ast.Name):
            if RE_CANDIDATES.search(comp.left.id):
                if isinstance(comp.comparators[0], ast.Str):
                    return _report(comp.comparators[0].s)
        elif isinstance(comp.left, ast.Attribute):
            if RE_CANDIDATES.search(comp.left.attr):
                if isinstance(comp.comparators[0], ast.Str):
                    return _report(comp.comparators[0].s)

@test.checks('Call')
@test.test_id('B106')
def hardcoded_password_funcarg(context):
    if False:
        while True:
            i = 10
    '**B106: Test for use of hard-coded password function arguments**\n\n    The use of hard-coded passwords increases the possibility of password\n    guessing tremendously. This plugin test looks for all function calls being\n    passed a keyword argument that is a string literal. It checks that the\n    assigned local variable does not look like a password.\n\n    Variables are considered to look like a password if they have match any one\n    of:\n\n    - "password"\n    - "pass"\n    - "passwd"\n    - "pwd"\n    - "secret"\n    - "token"\n    - "secrete"\n\n    Note: this can be noisy and may generate false positives.\n\n    **Config Options:**\n\n    None\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded\n        password: \'blerg\'\n           Severity: Low   Confidence: Medium\n           CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)\n           Location: ./examples/hardcoded-passwords.py:16\n        15\n        16    doLogin(password="blerg")\n\n    .. seealso::\n\n        - https://www.owasp.org/index.php/Use_of_hard-coded_password\n        - https://cwe.mitre.org/data/definitions/259.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    '
    for kw in context.node.keywords:
        if isinstance(kw.value, ast.Str) and RE_CANDIDATES.search(kw.arg):
            return _report(kw.value.s)

@test.checks('FunctionDef')
@test.test_id('B107')
def hardcoded_password_default(context):
    if False:
        i = 10
        return i + 15
    '**B107: Test for use of hard-coded password argument defaults**\n\n    The use of hard-coded passwords increases the possibility of password\n    guessing tremendously. This plugin test looks for all function definitions\n    that specify a default string literal for some argument. It checks that\n    the argument does not look like a password.\n\n    Variables are considered to look like a password if they have match any one\n    of:\n\n    - "password"\n    - "pass"\n    - "passwd"\n    - "pwd"\n    - "secret"\n    - "token"\n    - "secrete"\n\n    Note: this can be noisy and may generate false positives.\n\n    **Config Options:**\n\n    None\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: [B107:hardcoded_password_default] Possible hardcoded\n        password: \'Admin\'\n           Severity: Low   Confidence: Medium\n           CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)\n           Location: ./examples/hardcoded-passwords.py:1\n\n        1    def someFunction(user, password="Admin"):\n        2      print("Hi " + user)\n\n    .. seealso::\n\n        - https://www.owasp.org/index.php/Use_of_hard-coded_password\n        - https://cwe.mitre.org/data/definitions/259.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    '
    defs = [None] * (len(context.node.args.args) - len(context.node.args.defaults))
    defs.extend(context.node.args.defaults)
    for (key, val) in zip(context.node.args.args, defs):
        if isinstance(key, (ast.Name, ast.arg)):
            if isinstance(val, ast.Str) and RE_CANDIDATES.search(key.arg):
                return _report(val.s)