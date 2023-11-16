"""Test that all Tree's implement iter_search_rules."""
from bzrlib import rules
from bzrlib.tests.per_tree import TestCaseWithTree

class TestIterSearchRules(TestCaseWithTree):

    def make_per_user_searcher(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Make a _RulesSearcher from a string'
        return rules._IniBasedRulesSearcher(text.splitlines(True))

    def make_tree_with_rules(self, text):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        if text is not None:
            self.fail('No method for in-tree rules agreed on yet.')
            text_utf8 = text.encode('utf-8')
            self.build_tree_contents([(rules.RULES_TREE_FILENAME, text_utf8)])
            tree.add(rules.RULES_TREE_FILENAME)
            tree.commit('add rules file')
        result = self._convert_tree(tree)
        result.lock_read()
        self.addCleanup(result.unlock)
        return result

    def test_iter_search_rules_no_tree(self):
        if False:
            for i in range(10):
                print('nop')
        per_user = self.make_per_user_searcher('[name ./a.txt]\nfoo=baz\n[name *.txt]\nfoo=bar\na=True\n')
        tree = self.make_tree_with_rules(None)
        result = list(tree.iter_search_rules(['a.txt', 'dir/a.txt'], _default_searcher=per_user))
        self.assertEqual((('foo', 'baz'),), result[0])
        self.assertEqual((('foo', 'bar'), ('a', 'True')), result[1])

    def _disabled_test_iter_search_rules_just_tree(self):
        if False:
            print('Hello World!')
        per_user = self.make_per_user_searcher('')
        tree = self.make_tree_with_rules('[name ./a.txt]\nfoo=baz\n[name *.txt]\nfoo=bar\na=True\n')
        result = list(tree.iter_search_rules(['a.txt', 'dir/a.txt'], _default_searcher=per_user))
        self.assertEqual((('foo', 'baz'),), result[0])
        self.assertEqual((('foo', 'bar'), ('a', 'True')), result[1])

    def _disabled_test_iter_search_rules_tree_and_per_user(self):
        if False:
            while True:
                i = 10
        per_user = self.make_per_user_searcher('[name ./a.txt]\nfoo=baz\n[name *.txt]\nfoo=bar\na=True\n')
        tree = self.make_tree_with_rules('[name ./a.txt]\nfoo=qwerty\n')
        result = list(tree.iter_search_rules(['a.txt', 'dir/a.txt'], _default_searcher=per_user))
        self.assertEqual((('foo', 'qwerty'),), result[0])
        self.assertEqual((('foo', 'bar'), ('a', 'True')), result[1])