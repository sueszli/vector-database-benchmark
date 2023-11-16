import pickle
import pytest
from praw.exceptions import ClientException
from praw.models import Submission
from ... import UnitTest

class TestSubmission(UnitTest):

    @pytest.mark.filterwarnings('error', category=UserWarning)
    def test_additional_fetch_params_warning(self, reddit):
        if False:
            return 10
        with pytest.raises(UserWarning) as excinfo:
            submission = reddit.submission('1234')
            submission._fetched = True
            submission.add_fetch_param('foo', 'bar')
        assert excinfo.value.args[0] == 'This submission has already been fetched, so adding additional fetch parameters will not have any effect.'

    @pytest.mark.filterwarnings('error', category=UserWarning)
    def test_additional_fetch_params_warning__disabled(self, caplog, reddit):
        if False:
            for i in range(10):
                print('nop')
        reddit.config.warn_additional_fetch_params = False
        submission = reddit.submission('1234')
        submission._fetched = True
        submission.additional_fetch_params = True
        assert caplog.records == []

    @pytest.mark.filterwarnings('error', category=UserWarning)
    def test_comment_sort_warning(self, reddit):
        if False:
            i = 10
            return i + 15
        with pytest.raises(UserWarning) as excinfo:
            submission = reddit.submission('1234')
            submission._fetched = True
            submission.comment_sort = 'new'
        assert excinfo.value.args[0] == 'The comments for this submission have already been fetched, so the updated comment_sort will not have any effect.'

    @pytest.mark.filterwarnings('error', category=UserWarning)
    def test_comment_sort_warning__disabled(self, caplog, reddit):
        if False:
            for i in range(10):
                print('nop')
        reddit.config.warn_comment_sort = False
        submission = reddit.submission('1234')
        submission._fetched = True
        submission.comment_sort = 'new'
        assert caplog.records == []

    def test_construct_failure(self, reddit):
        if False:
            return 10
        message = "Exactly one of 'id', 'url', or '_data' must be provided."
        with pytest.raises(TypeError) as excinfo:
            Submission(reddit)
        assert str(excinfo.value) == message
        with pytest.raises(TypeError) as excinfo:
            Submission(reddit, id='dummy', url='dummy')
        assert str(excinfo.value) == message
        with pytest.raises(TypeError) as excinfo:
            Submission(reddit, 'dummy', _data={'id': 'dummy'})
        assert str(excinfo.value) == message
        with pytest.raises(TypeError) as excinfo:
            Submission(reddit, url='dummy', _data={'id': 'dummy'})
        assert str(excinfo.value) == message
        with pytest.raises(TypeError) as excinfo:
            Submission(reddit, 'dummy', 'dummy', {'id': 'dummy'})
        assert str(excinfo.value) == message
        with pytest.raises(ValueError):
            Submission(reddit, '')
        with pytest.raises(ValueError):
            Submission(reddit, url='')

    def test_construct_from_url(self, reddit):
        if False:
            return 10
        assert Submission(reddit, url='http://my.it/2gmzqe') == '2gmzqe'

    def test_equality(self, reddit):
        if False:
            i = 10
            return i + 15
        submission1 = Submission(reddit, _data={'id': 'dummy1', 'n': 1})
        submission2 = Submission(reddit, _data={'id': 'Dummy1', 'n': 2})
        submission3 = Submission(reddit, _data={'id': 'dummy3', 'n': 2})
        assert submission1 == submission1
        assert submission2 == submission2
        assert submission3 == submission3
        assert submission1 == submission2
        assert submission2 != submission3
        assert submission1 != submission3
        assert 'dummy1' == submission1
        assert submission2 == 'dummy1'

    def test_fullname(self, reddit):
        if False:
            i = 10
            return i + 15
        submission = Submission(reddit, _data={'id': 'dummy'})
        assert submission.fullname == 't3_dummy'

    def test_hash(self, reddit):
        if False:
            i = 10
            return i + 15
        submission1 = Submission(reddit, _data={'id': 'dummy1', 'n': 1})
        submission2 = Submission(reddit, _data={'id': 'Dummy1', 'n': 2})
        submission3 = Submission(reddit, _data={'id': 'dummy3', 'n': 2})
        assert hash(submission1) == hash(submission1)
        assert hash(submission2) == hash(submission2)
        assert hash(submission3) == hash(submission3)
        assert hash(submission1) == hash(submission2)
        assert hash(submission2) != hash(submission3)
        assert hash(submission1) != hash(submission3)

    def test_id_from_url(self):
        if False:
            i = 10
            return i + 15
        urls = ['http://my.it/2gmzqe', 'https://redd.it/2gmzqe', 'https://redd.it/2gmzqe/', 'http://reddit.com/comments/2gmzqe', 'https://www.reddit.com/r/redditdev/comments/2gmzqe/praw_https_enabled_praw_testing_needed/', 'https://www.reddit.com/gallery/2gmzqe']
        for url in urls:
            assert Submission.id_from_url(url) == '2gmzqe', url

    def test_id_from_url__invalid_urls(self):
        if False:
            while True:
                i = 10
        urls = ['', '1', '/', 'my.it/2gmzqe', 'http://my.it/_', 'https://redd.it/_/', 'http://reddit.com/comments/_/2gmzqe', 'https://reddit.com/r/wallpapers/', 'https://reddit.com/r/wallpapers', 'https://www.reddit.com/r/test/comments/', 'https://reddit.com/comments/']
        for url in urls:
            with pytest.raises(ClientException):
                Submission.id_from_url(url)

    def test_pickle(self, reddit):
        if False:
            return 10
        submission = Submission(reddit, _data={'id': 'dummy'})
        for level in range(pickle.HIGHEST_PROTOCOL + 1):
            other = pickle.loads(pickle.dumps(submission, protocol=level))
            assert submission == other

    def test_repr(self, reddit):
        if False:
            print('Hello World!')
        submission = Submission(reddit, id='2gmzqe')
        assert repr(submission) == "Submission(id='2gmzqe')"

    def test_shortlink(self, reddit):
        if False:
            i = 10
            return i + 15
        submission = Submission(reddit, _data={'id': 'dummy'})
        assert submission.shortlink == 'https://redd.it/dummy'

    def test_str(self, reddit):
        if False:
            return 10
        submission = Submission(reddit, _data={'id': 'dummy'})
        assert str(submission) == 'dummy'