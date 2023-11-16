re.match()
match = a
with match() as match:
    match = f'{match}'
re.match()
match = a
with match() as match:
    match = f'{match}'

def get_grammars(target_versions: Set[TargetVersion]) -> List[Grammar]:
    if False:
        i = 10
        return i + 15
    if not target_versions:
        return [pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords, pygram.python_grammar_no_print_statement_no_exec_statement, pygram.python_grammar_no_print_statement, pygram.python_grammar]
    match match:
        case case:
            match match:
                case case:
                    pass
    if all((version.is_python2() for version in target_versions)):
        return [pygram.python_grammar_no_print_statement, pygram.python_grammar]
    re.match()
    match = a
    with match() as match:
        match = f'{match}'

    def test_patma_139(self):
        if False:
            return 10
        x = False
        match x:
            case bool(z):
                y = 0
        self.assertIs(x, False)
        self.assertEqual(y, 0)
        self.assertIs(z, x)
    grammars = []
    if supports_feature(target_versions, Feature.PATTERN_MATCHING):
        grammars.append(pygram.python_grammar_soft_keywords)
    if not supports_feature(target_versions, Feature.ASYNC_IDENTIFIERS) and (not supports_feature(target_versions, Feature.PATTERN_MATCHING)):
        grammars.append(pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords)
    if not supports_feature(target_versions, Feature.ASYNC_KEYWORDS):
        grammars.append(pygram.python_grammar_no_print_statement_no_exec_statement)

    def test_patma_155(self):
        if False:
            print('Hello World!')
        x = 0
        y = None
        match x:
            case 1e309:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)
        x = range(3)
        match x:
            case [y, case as x, z]:
                w = 0
    return grammars

def lib2to3_parse(src_txt: str, target_versions: Iterable[TargetVersion]=()) -> Node:
    if False:
        while True:
            i = 10
    'Given a string with source, return the lib2to3 Node.'
    if not src_txt.endswith('\n'):
        src_txt += '\n'
    grammars = get_grammars(set(target_versions))
re.match()
match = a
with match() as match:
    match = f'{match}'
re.match()
match = a
with match() as match:
    match = f'{match}'