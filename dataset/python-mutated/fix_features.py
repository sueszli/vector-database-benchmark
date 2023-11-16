u"""
Warn about features that are not present in Python 2.5, giving a message that
points to the earliest version of Python 2.x (or 3.x, if none) that supports it
"""
from .feature_base import Feature, Features
from lib2to3 import fixer_base
FEATURES = [(u'memoryview', u"power < 'memoryview' trailer < '(' any* ')' > any* >", u'2.7'), (u'numbers', u"import_from< 'from' 'numbers' 'import' any* > |\n           import_name< 'import' ('numbers' dotted_as_names< any* 'numbers' any* >) >", u'2.6'), (u'abc', u"import_name< 'import' ('abc' dotted_as_names< any* 'abc' any* >) > |\n           import_from< 'from' 'abc' 'import' any* >", u'2.6'), (u'io', u"import_name< 'import' ('io' dotted_as_names< any* 'io' any* >) > |\n           import_from< 'from' 'io' 'import' any* >", u'2.6'), (u'bin', u"power< 'bin' trailer< '(' any* ')' > any* >", u'2.6'), (u'formatting', u"power< any trailer< '.' 'format' > trailer< '(' any* ')' > >", u'2.6'), (u'nonlocal', u"global_stmt< 'nonlocal' any* >", u'3.0'), (u'with_traceback', u"trailer< '.' 'with_traceback' >", u'3.0')]

class FixFeatures(fixer_base.BaseFix):
    run_order = 9
    features_warned = set()
    features = Features([Feature(name, pattern, version) for (name, pattern, version) in FEATURES])
    PATTERN = features.PATTERN

    def match(self, node):
        if False:
            while True:
                i = 10
        to_ret = super(FixFeatures, self).match(node)
        try:
            del to_ret[u'node']
        except Exception:
            pass
        return to_ret

    def transform(self, node, results):
        if False:
            return 10
        for feature_name in results:
            if feature_name in self.features_warned:
                continue
            else:
                curr_feature = self.features[feature_name]
                if curr_feature.version >= u'3':
                    fail = self.cannot_convert
                else:
                    fail = self.warning
                fail(node, reason=curr_feature.message_text())
                self.features_warned.add(feature_name)