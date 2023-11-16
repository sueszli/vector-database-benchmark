from datetime import datetime, timezone
from dateutil.parser import ParserError
import github
from . import Framework

class BadAttributes(Framework.TestCase):

    def testBadSimpleAttribute(self):
        if False:
            while True:
                i = 10
        user = self.g.get_user('klmitch')
        self.assertEqual(user.created_at, datetime(2011, 3, 23, 15, 42, 9, tzinfo=timezone.utc))
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            user.name
        self.assertEqual(raisedexp.exception.actual_value, 42)
        self.assertEqual(raisedexp.exception.expected_type, str)
        self.assertEqual(raisedexp.exception.transformation_exception, None)

    def testBadAttributeTransformation(self):
        if False:
            print('Hello World!')
        user = self.g.get_user('klmitch')
        self.assertEqual(user.name, 'Kevin L. Mitchell')
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            user.created_at
        self.assertEqual(raisedexp.exception.actual_value, 'foobar')
        self.assertEqual(raisedexp.exception.expected_type, str)
        self.assertEqual(raisedexp.exception.transformation_exception.__class__, ParserError)
        self.assertEqual(raisedexp.exception.transformation_exception.args, ('Unknown string format: %s', 'foobar'))

    def testBadTransformedAttribute(self):
        if False:
            while True:
                i = 10
        user = self.g.get_user('klmitch')
        self.assertEqual(user.name, 'Kevin L. Mitchell')
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            user.updated_at
        self.assertEqual(raisedexp.exception.actual_value, 42)
        self.assertEqual(raisedexp.exception.expected_type, str)
        self.assertEqual(raisedexp.exception.transformation_exception, None)

    def testBadSimpleAttributeInList(self):
        if False:
            return 10
        hook = self.g.get_hook('activecollab')
        self.assertEqual(hook.name, 'activecollab')
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            hook.events
        self.assertEqual(raisedexp.exception.actual_value, ['push', 42])
        self.assertEqual(raisedexp.exception.expected_type, [str])
        self.assertEqual(raisedexp.exception.transformation_exception, None)

    def testBadAttributeInClassAttribute(self):
        if False:
            return 10
        repo = self.g.get_repo('klmitch/turnstile')
        owner = repo.owner
        self.assertEqual(owner.id, 686398)
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            owner.avatar_url
        self.assertEqual(raisedexp.exception.actual_value, 42)

    def testBadTransformedAttributeInList(self):
        if False:
            i = 10
            return i + 15
        commit = self.g.get_repo('klmitch/turnstile', lazy=True).get_commit('38d9082a898d0822b5ccdfd78f3a536e2efa6c26')
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            commit.files
        self.assertEqual(raisedexp.exception.actual_value, [42])
        self.assertEqual(raisedexp.exception.expected_type, [dict])
        self.assertEqual(raisedexp.exception.transformation_exception, None)

    def testBadTransformedAttributeInDict(self):
        if False:
            return 10
        gist = self.g.get_gist('6437766')
        with self.assertRaises(github.BadAttributeException) as raisedexp:
            gist.files
        self.assertEqual(raisedexp.exception.actual_value, {'test.py': 42})
        self.assertEqual(raisedexp.exception.expected_type, {str: dict})
        self.assertEqual(raisedexp.exception.transformation_exception, None)

    def testIssue195(self):
        if False:
            while True:
                i = 10
        hooks = self.g.get_hooks()
        self.assertListKeyEqual(hooks, lambda h: h.name, ['activecollab', 'acunote', 'agilebench', 'agilezen', 'amazonsns', 'apiary', 'apoio', 'appharbor', 'apropos', 'asana', 'backlog', 'bamboo', 'basecamp', 'bcx', 'blimp', 'boxcar', 'buddycloud', 'bugherd', 'bugly', 'bugzilla', 'campfire', 'cia', 'circleci', 'codeclimate', 'codeportingcsharp2java', 'codeship', 'coffeedocinfo', 'conductor', 'coop', 'copperegg', 'cube', 'depending', 'deployhq', 'devaria', 'docker', 'ducksboard', 'email', 'firebase', 'fisheye', 'flowdock', 'fogbugz', 'freckle', 'friendfeed', 'gemini', 'gemnasium', 'geocommit', 'getlocalization', 'gitlive', 'grmble', 'grouptalent', 'grove', 'habitualist', 'hakiri', 'hall', 'harvest', 'hipchat', 'hostedgraphite', 'hubcap', 'hubci', 'humbug', 'icescrum', 'irc', 'irker', 'ironmq', 'ironworker', 'jabber', 'jaconda', 'jeapie', 'jenkins', 'jenkinsgit', 'jira', 'jqueryplugins', 'kanbanery', 'kickoff', 'leanto', 'lechat', 'lighthouse', 'lingohub', 'loggly', 'mantisbt', 'masterbranch', 'mqttpub', 'nma', 'nodejitsu', 'notifo', 'ontime', 'pachube', 'packagist', 'phraseapp', 'pivotaltracker', 'planbox', 'planio', 'prowl', 'puppetlinter', 'pushalot', 'pushover', 'pythonpackages', 'railsbp', 'railsbrakeman', 'rally', 'rapidpush', 'rationaljazzhub', 'rationalteamconcert', 'rdocinfo', 'readthedocs', 'redmine', 'rubyforge', 'scrumdo', 'shiningpanda', 'sifter', 'simperium', 'slatebox', 'snowyevening', 'socialcast', 'softlayermessaging', 'sourcemint', 'splendidbacon', 'sprintly', 'sqsqueue', 'stackmob', 'statusnet', 'talker', 'targetprocess', 'tddium', 'teamcity', 'tender', 'tenxer', 'testpilot', 'toggl', 'trac', 'trajectory', 'travis', 'trello', 'twilio', 'twitter', 'unfuddle', 'web', 'weblate', 'webtranslateit', 'yammer', 'youtrack', 'zendesk', 'zohoprojects'])
        for hook in hooks:
            if hook.name != 'circleci':
                hook.events
        for hook in hooks:
            if hook.name == 'circleci':
                with self.assertRaises(github.BadAttributeException) as raisedexp:
                    hook.events
        self.assertEqual(raisedexp.exception.actual_value, [['commit_comment', 'create', 'delete', 'download', 'follow', 'fork', 'fork_apply', 'gist', 'gollum', 'issue_comment', 'issues', 'member', 'public', 'pull_request', 'pull_request_review_comment', 'push', 'status', 'team_add', 'watch']])
        self.assertEqual(raisedexp.exception.expected_type, [str])
        self.assertEqual(raisedexp.exception.transformation_exception, None)