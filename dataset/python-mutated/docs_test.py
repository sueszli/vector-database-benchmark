import os
import subprocess
import sys
import unittest

class DocsIncludedTest(unittest.TestCase):

    def test_doc_import_works(self):
        if False:
            print('Hello World!')
        from pygame.docs.__main__ import has_local_docs, open_docs

    @unittest.skipIf('CI' not in os.environ, 'Docs not required for local builds')
    def test_docs_included(self):
        if False:
            print('Hello World!')
        from pygame.docs.__main__ import has_local_docs
        self.assertTrue(has_local_docs())

    @unittest.skipIf('CI' not in os.environ, 'Docs not required for local builds')
    def test_docs_command(self):
        if False:
            print('Hello World!')
        try:
            subprocess.run([sys.executable, '-m', 'pygame.docs'], timeout=5, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            pass
if __name__ == '__main__':
    unittest.main()