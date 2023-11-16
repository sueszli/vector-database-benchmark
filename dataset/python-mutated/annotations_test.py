import unittest
import warnings
from apache_beam.utils.annotations import BeamDeprecationWarning
from apache_beam.utils.annotations import deprecated

class AnnotationTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        warnings.resetwarnings()

    def test_deprecated_with_since_current_message(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:

            @deprecated(since='v.1', current='multiply', extra_message='Do this')
            def fnc_test_deprecated_with_since_current_message():
                if False:
                    return 10
                return 'lol'
            fnc_test_deprecated_with_since_current_message()
            self.check_annotation(warning=w, warning_type=BeamDeprecationWarning, obj_name='fnc_test_deprecated_with_since_current_message', annotation_type='deprecated', label_check_list=[('since', True), ('instead', True), ('Do this', True)])

    def test_deprecated_with_since_current(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:

            @deprecated(since='v.1', current='multiply')
            def fnc_test_deprecated_with_since_current():
                if False:
                    while True:
                        i = 10
                return 'lol'
            fnc_test_deprecated_with_since_current()
            self.check_annotation(warning=w, warning_type=BeamDeprecationWarning, obj_name='fnc_test_deprecated_with_since_current', annotation_type='deprecated', label_check_list=[('since', True), ('instead', True)])

    def test_deprecated_without_current(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as w:

            @deprecated(since='v.1')
            def fnc_test_deprecated_without_current():
                if False:
                    while True:
                        i = 10
                return 'lol'
            fnc_test_deprecated_without_current()
            self.check_annotation(warning=w, warning_type=BeamDeprecationWarning, obj_name='fnc_test_deprecated_without_current', annotation_type='deprecated', label_check_list=[('since', True), ('instead', False)])

    def test_deprecated_without_since_should_fail(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaises(TypeError):

                @deprecated()
                def fnc_test_deprecated_without_since_should_fail():
                    if False:
                        for i in range(10):
                            print('nop')
                    return 'lol'
                fnc_test_deprecated_without_since_should_fail()
            assert not w

    def test_deprecated_without_since_custom_should_fail(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaises(TypeError):

                @deprecated(custom_message='Test %since%')
                def fnc_test_deprecated_without_since_custom_should_fail():
                    if False:
                        i = 10
                        return i + 15
                    return 'lol'
                fnc_test_deprecated_without_since_custom_should_fail()
            assert not w

    def test_deprecated_custom_no_replacements(self):
        if False:
            return 10
        'Tests if custom message prints an empty string\n    for each replacement token when only the\n    custom_message and since parameter are given.'
        with warnings.catch_warnings(record=True) as w:
            strSince = 'v1'
            strCustom = 'Replacement:%since%%current%%extra%'

            @deprecated(since=strSince, custom_message=strCustom)
            def fnc_test_experimental_custom_no_replacements():
                if False:
                    i = 10
                    return i + 15
                return 'lol'
            fnc_test_experimental_custom_no_replacements()
            self.check_custom_annotation(warning=w, warning_type=BeamDeprecationWarning, obj_name='fnc_test_experimental_custom_no_                                   replacements', annotation_type='experimental', intended_message=strCustom.replace('%since%', strSince).replace('%current%', '').replace('%extra%', ''))

    def test_enforce_custom_since_deprecated_must_fail(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests since replacement token inclusion on the\n    custom message for the decapreted string. If no\n    since replacement token is given, the annotation must fail'
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaises(TypeError):
                strSince = 'v1'
                strCustom = 'Replacement:'

                @deprecated(since=strSince, custom_message=strCustom)
                def fnc_test_experimental_custom_no_replacements():
                    if False:
                        i = 10
                        return i + 15
                    return 'lol'
                fnc_test_experimental_custom_no_replacements()
            assert not w

    def test_deprecated_with_since_current_message_custom(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as w:
            strSince = 'v.1'
            strCurrent = 'multiply'
            strExtra = 'Do this'
            strCustom = '%name% Will be deprecated from %since%.                   Please use %current% insted. Will %extra%'

            @deprecated(since=strSince, current=strCurrent, extra_message=strExtra, custom_message=strCustom)
            def fnc_test_deprecated_with_since_current_message_custom():
                if False:
                    print('Hello World!')
                return 'lol'
            strName = fnc_test_deprecated_with_since_current_message_custom.__name__
            fnc_test_deprecated_with_since_current_message_custom()
            self.check_custom_annotation(warning=w, warning_type=BeamDeprecationWarning, obj_name='fnc_test_deprecated_with_since_                                   current_message_custom', annotation_type='deprecated', intended_message=strCustom.replace('%name%', strName).replace('%since%', strSince).replace('%current%', strCurrent).replace('%extra%', strExtra))

    def test_deprecated_with_since_current_message_class(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:

            @deprecated(since='v.1', current='multiply', extra_message='Do this')
            class Class_test_deprecated_with_since_current_message(object):
                fooo = 'lol'

                def __init__(self):
                    if False:
                        print('Hello World!')
                    pass

                def foo(self):
                    if False:
                        return 10
                    return 'lol'
            foo = Class_test_deprecated_with_since_current_message()
            strName = Class_test_deprecated_with_since_current_message.__name__
            foo.foo()
            self.check_annotation(warning=w, warning_type=BeamDeprecationWarning, obj_name=strName, annotation_type='deprecated', label_check_list=[('since', True), ('instead', True), ('Do this', True)])

    def check_annotation(self, warning, warning_type, obj_name, annotation_type, label_check_list):
        if False:
            print('Hello World!')
        self.assertTrue(issubclass(warning[-1].category, warning_type))
        self.assertIn(obj_name + ' is ' + annotation_type, str(warning[-1].message))
        for label in label_check_list:
            if label[1] is True:
                self.assertIn(label[0], str(warning[-1].message))
            else:
                self.assertNotIn(label[0], str(warning[-1].message))

    def check_custom_annotation(self, warning, warning_type, obj_name, annotation_type, intended_message):
        if False:
            return 10
        self.assertTrue(issubclass(warning[-1].category, warning_type))
        self.assertIn(intended_message, str(warning[-1].message))
if __name__ == '__main__':
    unittest.main()