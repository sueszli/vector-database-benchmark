"""
Tests for static parts of Twitter package
"""
import os
import pytest
pytest.importorskip('twython')
from nltk.twitter import Authenticate

@pytest.fixture
def auth():
    if False:
        print('Hello World!')
    return Authenticate()

class TestCredentials:
    """
    Tests that Twitter credentials from a file are handled correctly.
    """

    @classmethod
    def setup_class(self):
        if False:
            i = 10
            return i + 15
        self.subdir = os.path.join(os.path.dirname(__file__), 'files')
        os.environ['TWITTER'] = 'twitter-files'

    def test_environment(self, auth):
        if False:
            return 10
        '\n        Test that environment variable has been read correctly.\n        '
        fn = os.path.basename(auth.creds_subdir)
        assert fn == os.environ['TWITTER']

    @pytest.mark.parametrize('kwargs', [{'subdir': ''}, {'subdir': None}, {'subdir': '/nosuchdir'}, {}, {'creds_file': 'foobar'}, {'creds_file': 'bad_oauth1-1.txt'}, {'creds_file': 'bad_oauth1-2.txt'}, {'creds_file': 'bad_oauth1-3.txt'}])
    def test_scenarios_that_should_raise_errors(self, kwargs, auth):
        if False:
            return 10
        'Various scenarios that should raise errors'
        try:
            auth.load_creds(**kwargs)
        except (OSError, ValueError):
            pass
        except Exception as e:
            pytest.fail('Unexpected exception thrown: %s' % e)
        else:
            pytest.fail('OSError exception not thrown.')

    def test_correct_file(self, auth):
        if False:
            print('Hello World!')
        'Test that a proper file succeeds and is read correctly'
        oauth = auth.load_creds(subdir=self.subdir)
        assert auth.creds_fullpath == os.path.join(self.subdir, auth.creds_file)
        assert auth.creds_file == 'credentials.txt'
        assert oauth['app_key'] == 'a'