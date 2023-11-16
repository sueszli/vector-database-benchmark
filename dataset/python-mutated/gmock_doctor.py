"""Converts compiler's errors in code using Google Mock to plain English."""
__author__ = 'wan@google.com (Zhanyong Wan)'
import re
import sys
_VERSION = '1.0.3'
_EMAIL = 'googlemock@googlegroups.com'
_COMMON_GMOCK_SYMBOLS = ['_', 'A', 'AddressSatisfies', 'AllOf', 'An', 'AnyOf', 'ContainerEq', 'Contains', 'ContainsRegex', 'DoubleEq', 'ElementsAre', 'ElementsAreArray', 'EndsWith', 'Eq', 'Field', 'FloatEq', 'Ge', 'Gt', 'HasSubstr', 'IsInitializedProto', 'Le', 'Lt', 'MatcherCast', 'Matches', 'MatchesRegex', 'NanSensitiveDoubleEq', 'NanSensitiveFloatEq', 'Ne', 'Not', 'NotNull', 'Pointee', 'Property', 'Ref', 'ResultOf', 'SafeMatcherCast', 'StartsWith', 'StrCaseEq', 'StrCaseNe', 'StrEq', 'StrNe', 'Truly', 'TypedEq', 'Value', 'Assign', 'ByRef', 'DeleteArg', 'DoAll', 'DoDefault', 'IgnoreResult', 'Invoke', 'InvokeArgument', 'InvokeWithoutArgs', 'Return', 'ReturnNew', 'ReturnNull', 'ReturnRef', 'SaveArg', 'SetArgReferee', 'SetArgPointee', 'SetArgumentPointee', 'SetArrayArgument', 'SetErrnoAndReturn', 'Throw', 'WithArg', 'WithArgs', 'WithoutArgs', 'AnyNumber', 'AtLeast', 'AtMost', 'Between', 'Exactly', 'InSequence', 'Sequence', 'DefaultValue', 'Mock']
_GCC_FILE_LINE_RE = '(?P<file>.*):(?P<line>\\d+):(\\d+:)?\\s+'
_CLANG_FILE_LINE_RE = '(?P<file>.*):(?P<line>\\d+):(?P<column>\\d+):\\s+'
_CLANG_NON_GMOCK_FILE_LINE_RE = '(?P<file>.*[/\\\\^](?!gmock-)[^/\\\\]+):(?P<line>\\d+):(?P<column>\\d+):\\s+'

def _FindAllMatches(regex, s):
    if False:
        i = 10
        return i + 15
    'Generates all matches of regex in string s.'
    r = re.compile(regex)
    return r.finditer(s)

def _GenericDiagnoser(short_name, long_name, diagnoses, msg):
    if False:
        print('Hello World!')
    "Diagnoses the given disease by pattern matching.\n\n  Can provide different diagnoses for different patterns.\n\n  Args:\n    short_name: Short name of the disease.\n    long_name:  Long name of the disease.\n    diagnoses:  A list of pairs (regex, pattern for formatting the diagnosis\n                for matching regex).\n    msg:        Compiler's error messages.\n  Yields:\n    Tuples of the form\n      (short name of disease, long name of disease, diagnosis).\n  "
    for (regex, diagnosis) in diagnoses:
        if re.search(regex, msg):
            diagnosis = '%(file)s:%(line)s:' + diagnosis
            for m in _FindAllMatches(regex, msg):
                yield (short_name, long_name, diagnosis % m.groupdict())

def _NeedToReturnReferenceDiagnoser(msg):
    if False:
        while True:
            i = 10
    'Diagnoses the NRR disease, given the error messages by the compiler.'
    gcc_regex = "In member function \\'testing::internal::ReturnAction<R>.*\\n" + _GCC_FILE_LINE_RE + 'instantiated from here\\n.*gmock-actions\\.h.*error: creating array with negative size'
    clang_regex = 'error:.*array.*negative.*\\r?\\n(.*\\n)*?' + _CLANG_NON_GMOCK_FILE_LINE_RE + "note: in instantiation of function template specialization \\'testing::internal::ReturnAction<(?P<type>.*)>::operator Action<.*>\\' requested here"
    clang11_re = 'use_ReturnRef_instead_of_Return_to_return_a_reference.*(.*\\n)*?' + _CLANG_NON_GMOCK_FILE_LINE_RE
    diagnosis = '\nYou are using a Return() action in a function that returns a reference to\n%(type)s.  Please use ReturnRef() instead.'
    return _GenericDiagnoser('NRR', 'Need to Return Reference', [(clang_regex, diagnosis), (clang11_re, diagnosis % {'type': 'a type'}), (gcc_regex, diagnosis % {'type': 'a type'})], msg)

def _NeedToReturnSomethingDiagnoser(msg):
    if False:
        i = 10
        return i + 15
    'Diagnoses the NRS disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + '(instantiated from here\\n.*gmock.*actions\\.h.*error: void value not ignored)|(error: control reaches end of non-void function)'
    clang_regex1 = _CLANG_FILE_LINE_RE + "error: cannot initialize return object of type \\'Result\\' \\(aka \\'(?P<return_type>.*)\\'\\) with an rvalue of type \\'void\\'"
    clang_regex2 = _CLANG_FILE_LINE_RE + "error: cannot initialize return object of type \\'(?P<return_type>.*)\\' with an rvalue of type \\'void\\'"
    diagnosis = '\nYou are using an action that returns void, but it needs to return\n%(return_type)s.  Please tell it *what* to return.  Perhaps you can use\nthe pattern DoAll(some_action, Return(some_value))?'
    return _GenericDiagnoser('NRS', 'Need to Return Something', [(gcc_regex, diagnosis % {'return_type': '*something*'}), (clang_regex1, diagnosis), (clang_regex2, diagnosis)], msg)

def _NeedToReturnNothingDiagnoser(msg):
    if False:
        while True:
            i = 10
    'Diagnoses the NRN disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "instantiated from here\\n.*gmock-actions\\.h.*error: instantiation of \\'testing::internal::ReturnAction<R>::Impl<F>::value_\\' as type \\'void\\'"
    clang_regex1 = "error: field has incomplete type \\'Result\\' \\(aka \\'void\\'\\)(\\r)?\\n(.*\\n)*?" + _CLANG_NON_GMOCK_FILE_LINE_RE + "note: in instantiation of function template specialization \\'testing::internal::ReturnAction<(?P<return_type>.*)>::operator Action<void \\(.*\\)>\\' requested here"
    clang_regex2 = "error: field has incomplete type \\'Result\\' \\(aka \\'void\\'\\)(\\r)?\\n(.*\\n)*?" + _CLANG_NON_GMOCK_FILE_LINE_RE + "note: in instantiation of function template specialization \\'testing::internal::DoBothAction<.*>::operator Action<(?P<return_type>.*) \\(.*\\)>\\' requested here"
    diagnosis = '\nYou are using an action that returns %(return_type)s, but it needs to return\nvoid.  Please use a void-returning action instead.\n\nAll actions but the last in DoAll(...) must return void.  Perhaps you need\nto re-arrange the order of actions in a DoAll(), if you are using one?'
    return _GenericDiagnoser('NRN', 'Need to Return Nothing', [(gcc_regex, diagnosis % {'return_type': '*something*'}), (clang_regex1, diagnosis), (clang_regex2, diagnosis)], msg)

def _IncompleteByReferenceArgumentDiagnoser(msg):
    if False:
        print('Hello World!')
    'Diagnoses the IBRA disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "instantiated from here\\n.*gtest-printers\\.h.*error: invalid application of \\'sizeof\\' to incomplete type \\'(?P<type>.*)\\'"
    clang_regex = ".*gtest-printers\\.h.*error: invalid application of \\'sizeof\\' to an incomplete type \\'(?P<type>.*)( const)?\\'\\r?\\n(.*\\n)*?" + _CLANG_NON_GMOCK_FILE_LINE_RE + "note: in instantiation of member function \\'testing::internal2::TypeWithoutFormatter<.*>::PrintValue\\' requested here"
    diagnosis = '\nIn order to mock this function, Google Mock needs to see the definition\nof type "%(type)s" - declaration alone is not enough.  Either #include\nthe header that defines it, or change the argument to be passed\nby pointer.'
    return _GenericDiagnoser('IBRA', 'Incomplete By-Reference Argument Type', [(gcc_regex, diagnosis), (clang_regex, diagnosis)], msg)

def _OverloadedFunctionMatcherDiagnoser(msg):
    if False:
        for i in range(10):
            print('nop')
    'Diagnoses the OFM disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "error: no matching function for call to \\'Truly\\(<unresolved overloaded function type>\\)"
    clang_regex = _CLANG_FILE_LINE_RE + "error: no matching function for call to \\'Truly"
    diagnosis = '\nThe argument you gave to Truly() is an overloaded function.  Please tell\nyour compiler which overloaded version you want to use.\n\nFor example, if you want to use the version whose signature is\n  bool Foo(int n);\nyou should write\n  Truly(static_cast<bool (*)(int n)>(Foo))'
    return _GenericDiagnoser('OFM', 'Overloaded Function Matcher', [(gcc_regex, diagnosis), (clang_regex, diagnosis)], msg)

def _OverloadedFunctionActionDiagnoser(msg):
    if False:
        print('Hello World!')
    'Diagnoses the OFA disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "error: no matching function for call to \\'Invoke\\(<unresolved overloaded function type>"
    clang_regex = _CLANG_FILE_LINE_RE + "error: no matching function for call to \\'Invoke\\'\\r?\\n(.*\\n)*?.*\\bgmock-generated-actions\\.h:\\d+:\\d+:\\s+note: candidate template ignored:\\s+couldn\\'t infer template argument \\'FunctionImpl\\'"
    diagnosis = '\nFunction you are passing to Invoke is overloaded.  Please tell your compiler\nwhich overloaded version you want to use.\n\nFor example, if you want to use the version whose signature is\n  bool MyFunction(int n, double x);\nyou should write something like\n  Invoke(static_cast<bool (*)(int n, double x)>(MyFunction))'
    return _GenericDiagnoser('OFA', 'Overloaded Function Action', [(gcc_regex, diagnosis), (clang_regex, diagnosis)], msg)

def _OverloadedMethodActionDiagnoser(msg):
    if False:
        i = 10
        return i + 15
    'Diagnoses the OMA disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "error: no matching function for call to \\'Invoke\\(.+, <unresolved overloaded function type>\\)"
    clang_regex = _CLANG_FILE_LINE_RE + "error: no matching function for call to \\'Invoke\\'\\r?\\n(.*\\n)*?.*\\bgmock-generated-actions\\.h:\\d+:\\d+: note: candidate function template not viable: requires .*, but 2 (arguments )?were provided"
    diagnosis = '\nThe second argument you gave to Invoke() is an overloaded method.  Please\ntell your compiler which overloaded version you want to use.\n\nFor example, if you want to use the version whose signature is\n  class Foo {\n    ...\n    bool Bar(int n, double x);\n  };\nyou should write something like\n  Invoke(foo, static_cast<bool (Foo::*)(int n, double x)>(&Foo::Bar))'
    return _GenericDiagnoser('OMA', 'Overloaded Method Action', [(gcc_regex, diagnosis), (clang_regex, diagnosis)], msg)

def _MockObjectPointerDiagnoser(msg):
    if False:
        for i in range(10):
            print('nop')
    'Diagnoses the MOP disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "error: request for member \\'gmock_(?P<method>.+)\\' in \\'(?P<mock_object>.+)\\', which is of non-class type \\'(.*::)*(?P<class_name>.+)\\*\\'"
    clang_regex = _CLANG_FILE_LINE_RE + "error: member reference type \\'(?P<class_name>.*?) *\\' is a pointer; (did you mean|maybe you meant) to use \\'->\\'\\?"
    diagnosis = "\nThe first argument to ON_CALL() and EXPECT_CALL() must be a mock *object*,\nnot a *pointer* to it.  Please write '*(%(mock_object)s)' instead of\n'%(mock_object)s' as your first argument.\n\nFor example, given the mock class:\n\n  class %(class_name)s : public ... {\n    ...\n    MOCK_METHOD0(%(method)s, ...);\n  };\n\nand the following mock instance:\n\n  %(class_name)s* mock_ptr = ...\n\nyou should use the EXPECT_CALL like this:\n\n  EXPECT_CALL(*mock_ptr, %(method)s(...));"
    return _GenericDiagnoser('MOP', 'Mock Object Pointer', [(gcc_regex, diagnosis), (clang_regex, diagnosis % {'mock_object': 'mock_object', 'method': 'method', 'class_name': '%(class_name)s'})], msg)

def _NeedToUseSymbolDiagnoser(msg):
    if False:
        i = 10
        return i + 15
    'Diagnoses the NUS disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "error: \\'(?P<symbol>.+)\\' (was not declared in this scope|has not been declared)"
    clang_regex = _CLANG_FILE_LINE_RE + "error: (use of undeclared identifier|unknown type name|no template named) \\'(?P<symbol>[^\\']+)\\'"
    diagnosis = "\n'%(symbol)s' is defined by Google Mock in the testing namespace.\nDid you forget to write\n  using testing::%(symbol)s;\n?"
    for m in list(_FindAllMatches(gcc_regex, msg)) + list(_FindAllMatches(clang_regex, msg)):
        symbol = m.groupdict()['symbol']
        if symbol in _COMMON_GMOCK_SYMBOLS:
            yield ('NUS', 'Need to Use Symbol', diagnosis % m.groupdict())

def _NeedToUseReturnNullDiagnoser(msg):
    if False:
        while True:
            i = 10
    'Diagnoses the NRNULL disease, given the error messages by the compiler.'
    gcc_regex = "instantiated from 'testing::internal::ReturnAction<R>::operator testing::Action<Func>\\(\\) const.*\n" + _GCC_FILE_LINE_RE + "instantiated from here\\n.*error: no matching function for call to \\'ImplicitCast_\\((:?long )?int&\\)"
    clang_regex = "\\bgmock-actions.h:.* error: no matching function for call to \\'ImplicitCast_\\'\\r?\\n(.*\\n)*?" + _CLANG_NON_GMOCK_FILE_LINE_RE + "note: in instantiation of function template specialization \\'testing::internal::ReturnAction<(int|long)>::operator Action<(?P<type>.*)\\(\\)>\\' requested here"
    diagnosis = "\nYou are probably calling Return(NULL) and the compiler isn't sure how to turn\nNULL into %(type)s. Use ReturnNull() instead.\nNote: the line number may be off; please fix all instances of Return(NULL)."
    return _GenericDiagnoser('NRNULL', 'Need to use ReturnNull', [(clang_regex, diagnosis), (gcc_regex, diagnosis % {'type': 'the right type'})], msg)

def _TypeInTemplatedBaseDiagnoser(msg):
    if False:
        print('Hello World!')
    'Diagnoses the TTB disease, given the error messages by the compiler.'
    gcc_4_3_1_regex_type_in_retval = "In member function \\'int .*\\n" + _GCC_FILE_LINE_RE + 'error: a function call cannot appear in a constant-expression'
    gcc_4_4_0_regex_type_in_retval = 'error: a function call cannot appear in a constant-expression' + _GCC_FILE_LINE_RE + 'error: template argument 1 is invalid\\n'
    gcc_regex_type_of_sole_param = _GCC_FILE_LINE_RE + "error: \\'(?P<type>.+)\\' was not declared in this scope\\n.*error: template argument 1 is invalid\\n"
    gcc_regex_type_of_a_param = "error: expected `;\\' before \\'::\\' token\\n" + _GCC_FILE_LINE_RE + "error: \\'(?P<type>.+)\\' was not declared in this scope\\n.*error: template argument 1 is invalid\\n.*error: \\'.+\\' was not declared in this scope"
    clang_regex_type_of_retval_or_sole_param = _CLANG_FILE_LINE_RE + "error: use of undeclared identifier \\'(?P<type>.*)\\'\\n(.*\\n)*?(?P=file):(?P=line):\\d+: error: non-friend class member \\'Result\\' cannot have a qualified name"
    clang_regex_type_of_a_param = _CLANG_FILE_LINE_RE + 'error: C\\+\\+ requires a type specifier for all declarations\\n(.*\\n)*?(?P=file):(?P=line):(?P=column): error: C\\+\\+ requires a type specifier for all declarations'
    clang_regex_unknown_type = _CLANG_FILE_LINE_RE + "error: unknown type name \\'(?P<type>[^\\']+)\\'"
    diagnosis = '\nIn a mock class template, types or typedefs defined in the base class\ntemplate are *not* automatically visible.  This is how C++ works.  Before\nyou can use a type or typedef named %(type)s defined in base class Base<T>, you\nneed to make it visible.  One way to do it is:\n\n  typedef typename Base<T>::%(type)s %(type)s;'
    for diag in _GenericDiagnoser('TTB', 'Type in Template Base', [(gcc_4_3_1_regex_type_in_retval, diagnosis % {'type': 'Foo'}), (gcc_4_4_0_regex_type_in_retval, diagnosis % {'type': 'Foo'}), (gcc_regex_type_of_sole_param, diagnosis), (gcc_regex_type_of_a_param, diagnosis), (clang_regex_type_of_retval_or_sole_param, diagnosis), (clang_regex_type_of_a_param, diagnosis % {'type': 'Foo'})], msg):
        yield diag
    for m in _FindAllMatches(clang_regex_unknown_type, msg):
        type_ = m.groupdict()['type']
        if type_ not in _COMMON_GMOCK_SYMBOLS:
            yield ('TTB', 'Type in Template Base', diagnosis % m.groupdict())

def _WrongMockMethodMacroDiagnoser(msg):
    if False:
        print('Hello World!')
    'Diagnoses the WMM disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + '.*this_method_does_not_take_(?P<wrong_args>\\d+)_argument.*\\n.*\\n.*candidates are.*FunctionMocker<[^>]+A(?P<args>\\d+)\\)>'
    clang_regex = _CLANG_NON_GMOCK_FILE_LINE_RE + 'error:.*array.*negative.*r?\\n(.*\\n)*?(?P=file):(?P=line):(?P=column): error: too few arguments to function call, expected (?P<args>\\d+), have (?P<wrong_args>\\d+)'
    clang11_re = _CLANG_NON_GMOCK_FILE_LINE_RE + '.*this_method_does_not_take_(?P<wrong_args>\\d+)_argument.*'
    diagnosis = '\nYou are using MOCK_METHOD%(wrong_args)s to define a mock method that has\n%(args)s arguments. Use MOCK_METHOD%(args)s (or MOCK_CONST_METHOD%(args)s,\nMOCK_METHOD%(args)s_T, MOCK_CONST_METHOD%(args)s_T as appropriate) instead.'
    return _GenericDiagnoser('WMM', 'Wrong MOCK_METHODn Macro', [(gcc_regex, diagnosis), (clang11_re, diagnosis % {'wrong_args': 'm', 'args': 'n'}), (clang_regex, diagnosis)], msg)

def _WrongParenPositionDiagnoser(msg):
    if False:
        print('Hello World!')
    'Diagnoses the WPP disease, given the error messages by the compiler.'
    gcc_regex = _GCC_FILE_LINE_RE + "error:.*testing::internal::MockSpec<.* has no member named \\'(?P<method>\\w+)\\'"
    clang_regex = _CLANG_NON_GMOCK_FILE_LINE_RE + "error: no member named \\'(?P<method>\\w+)\\' in \\'testing::internal::MockSpec<.*>\\'"
    diagnosis = '\nThe closing parenthesis of ON_CALL or EXPECT_CALL should be *before*\n".%(method)s".  For example, you should write:\n  EXPECT_CALL(my_mock, Foo(_)).%(method)s(...);\ninstead of:\n  EXPECT_CALL(my_mock, Foo(_).%(method)s(...));'
    return _GenericDiagnoser('WPP', 'Wrong Parenthesis Position', [(gcc_regex, diagnosis), (clang_regex, diagnosis)], msg)
_DIAGNOSERS = [_IncompleteByReferenceArgumentDiagnoser, _MockObjectPointerDiagnoser, _NeedToReturnNothingDiagnoser, _NeedToReturnReferenceDiagnoser, _NeedToReturnSomethingDiagnoser, _NeedToUseReturnNullDiagnoser, _NeedToUseSymbolDiagnoser, _OverloadedFunctionActionDiagnoser, _OverloadedFunctionMatcherDiagnoser, _OverloadedMethodActionDiagnoser, _TypeInTemplatedBaseDiagnoser, _WrongMockMethodMacroDiagnoser, _WrongParenPositionDiagnoser]

def Diagnose(msg):
    if False:
        print('Hello World!')
    'Generates all possible diagnoses given the compiler error message.'
    msg = re.sub('\\x1b\\[[^m]*m', '', msg)
    msg = re.sub('(\\xe2\\x80\\x98|\\xe2\\x80\\x99)', "'", msg)
    diagnoses = []
    for diagnoser in _DIAGNOSERS:
        for diag in diagnoser(msg):
            diagnosis = '[%s - %s]\n%s' % diag
            if not diagnosis in diagnoses:
                diagnoses.append(diagnosis)
    return diagnoses

def main():
    if False:
        for i in range(10):
            print('nop')
    print('Google Mock Doctor v%s - diagnoses problems in code using Google Mock.' % _VERSION)
    if sys.stdin.isatty():
        print('Please copy and paste the compiler errors here.  Press c-D when you are done:')
    else:
        print('Waiting for compiler errors on stdin . . .')
    msg = sys.stdin.read().strip()
    diagnoses = Diagnose(msg)
    count = len(diagnoses)
    if not count:
        print("\nYour compiler complained:\n8<------------------------------------------------------------\n%s\n------------------------------------------------------------>8\n\nUh-oh, I'm not smart enough to figure out what the problem is. :-(\nHowever...\nIf you send your source code and the compiler's error messages to\n%s, you can be helped and I can get smarter --\nwin-win for us!" % (msg, _EMAIL))
    else:
        print('------------------------------------------------------------')
        print('Your code appears to have the following')
        if count > 1:
            print('%s diseases:' % (count,))
        else:
            print('disease:')
        i = 0
        for d in diagnoses:
            i += 1
            if count > 1:
                print('\n#%s:' % (i,))
            print(d)
        print("\nHow did I do?  If you think I'm wrong or unhelpful, please send your\nsource code and the compiler's error messages to %s.\nThen you can be helped and I can get smarter -- I promise I won't be upset!" % _EMAIL)
if __name__ == '__main__':
    main()