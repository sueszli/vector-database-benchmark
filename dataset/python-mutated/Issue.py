from datetime import datetime, timezone
from . import Framework

class Issue(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.repo = self.g.get_user().get_repo('PyGithub')
        self.issue = self.repo.get_issue(28)

    def testAttributes(self):
        if False:
            return 10
        self.assertEqual(self.issue.assignee.login, 'jacquev6')
        self.assertListKeyEqual(self.issue.assignees, lambda a: a.login, ['jacquev6', 'stuglaser'])
        self.assertEqual(self.issue.body, 'Body edited by PyGithub')
        self.assertEqual(self.issue.closed_at, datetime(2012, 5, 26, 14, 59, 33, tzinfo=timezone.utc))
        self.assertEqual(self.issue.closed_by.login, 'jacquev6')
        self.assertEqual(self.issue.comments, 0)
        self.assertEqual(self.issue.comments_url, 'https://github.com/jacquev6/PyGithub/issues/28/comments')
        self.assertEqual(self.issue.created_at, datetime(2012, 5, 19, 10, 38, 23, tzinfo=timezone.utc))
        self.assertEqual(self.issue.events_url, 'https://github.com/jacquev6/PyGithub/issues/28/events')
        self.assertEqual(self.issue.html_url, 'https://github.com/jacquev6/PyGithub/issues/28')
        self.assertEqual(self.issue.id, 4653757)
        self.assertListKeyEqual(self.issue.labels, lambda l: l.name, ['Bug', 'Project management', 'Question'])
        self.assertEqual(self.issue.labels_url, 'https://github.com/jacquev6/PyGithub/issues/28/labels{/name}')
        self.assertEqual(self.issue.milestone.title, 'Version 0.4')
        self.assertEqual(self.issue.number, 28)
        self.assertEqual(self.issue.pull_request.diff_url, None)
        self.assertEqual(self.issue.pull_request.patch_url, None)
        self.assertEqual(self.issue.pull_request.html_url, None)
        self.assertEqual(self.issue.state, 'closed')
        self.assertEqual(self.issue.state_reason, 'completed')
        self.assertEqual(self.issue.title, 'Issue created by PyGithub')
        self.assertEqual(self.issue.updated_at, datetime(2012, 5, 26, 14, 59, 33, tzinfo=timezone.utc))
        self.assertEqual(self.issue.url, 'https://api.github.com/repos/jacquev6/PyGithub/issues/28')
        self.assertFalse(self.issue.locked)
        self.assertIsNone(self.issue.active_lock_reason)
        self.assertEqual(self.issue.user.login, 'jacquev6')
        self.assertEqual(self.issue.repository.name, 'PyGithub')
        self.assertEqual(repr(self.issue), 'Issue(title="Issue created by PyGithub", number=28)')

    def testEditWithoutParameters(self):
        if False:
            i = 10
            return i + 15
        self.issue.edit()

    def testEditWithAllParameters(self):
        if False:
            return 10
        user = self.g.get_user('jacquev6')
        milestone = self.repo.get_milestone(2)
        self.issue.edit('Title edited by PyGithub', 'Body edited by PyGithub', user, 'open', milestone, ['Bug'], ['jacquev6', 'stuglaser'])
        self.assertEqual(self.issue.assignee.login, 'jacquev6')
        self.assertListKeyEqual(self.issue.assignees, lambda a: a.login, ['jacquev6', 'stuglaser'])
        self.assertEqual(self.issue.body, 'Body edited by PyGithub')
        self.assertEqual(self.issue.state, 'open')
        self.assertEqual(self.issue.title, 'Title edited by PyGithub')
        self.assertListKeyEqual(self.issue.labels, lambda l: l.name, ['Bug'])

    def testEditResetMilestone(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.issue.milestone.title, 'Version 0.4')
        self.issue.edit(milestone=None)
        self.assertEqual(self.issue.milestone, None)

    def testEditResetAssignee(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.issue.assignee.login, 'jacquev6')
        self.issue.edit(assignee=None)
        self.assertEqual(self.issue.assignee, None)

    def testEditWithStateReasonNotPlanned(self):
        if False:
            i = 10
            return i + 15
        self.issue.edit(state='closed', state_reason='not_planned')
        self.assertEqual(self.issue.state, 'closed')
        self.assertEqual(self.issue.state_reason, 'not_planned')

    def testEditWithStateReasonReopened(self):
        if False:
            print('Hello World!')
        self.issue.edit(state='open', state_reason='reopened')
        self.assertEqual(self.issue.state, 'open')
        self.assertEqual(self.issue.state_reason, 'reopened')

    def testLock(self):
        if False:
            print('Hello World!')
        self.issue.lock('resolved')

    def testUnlock(self):
        if False:
            for i in range(10):
                print('nop')
        self.issue.unlock()

    def testCreateComment(self):
        if False:
            for i in range(10):
                print('nop')
        comment = self.issue.create_comment('Comment created by PyGithub')
        self.assertEqual(comment.id, 5808311)

    def testGetComments(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListKeyEqual(self.issue.get_comments(), lambda c: c.user.login, ['jacquev6', 'roskakori'])

    def testGetCommentsSince(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertListKeyEqual(self.issue.get_comments(datetime(2012, 5, 26, 13, 59, 33, tzinfo=timezone.utc)), lambda c: c.user.login, ['jacquev6', 'roskakori'])

    def testGetEvents(self):
        if False:
            i = 10
            return i + 15
        self.assertListKeyEqual(self.issue.get_events(), lambda e: e.id, [15819975, 15820048])

    def testGetLabels(self):
        if False:
            return 10
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])

    def testAddAndRemoveAssignees(self):
        if False:
            for i in range(10):
                print('nop')
        user1 = 'jayfk'
        user2 = self.g.get_user('jzelinskie')
        self.assertListKeyEqual(self.issue.assignees, lambda a: a.login, ['jacquev6', 'stuglaser'])
        self.issue.add_to_assignees(user1, user2)
        self.assertListKeyEqual(self.issue.assignees, lambda a: a.login, ['jacquev6', 'stuglaser', 'jayfk', 'jzelinskie'])
        self.issue.remove_from_assignees(user1, user2)
        self.assertListKeyEqual(self.issue.assignees, lambda a: a.login, ['jacquev6', 'stuglaser'])

    def testAddAndRemoveLabels(self):
        if False:
            while True:
                i = 10
        bug = self.repo.get_label('Bug')
        question = self.repo.get_label('Question')
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])
        self.issue.remove_from_labels(bug)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Project management', 'Question'])
        self.issue.remove_from_labels(question)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Project management'])
        self.issue.add_to_labels(bug, question)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])

    def testAddAndRemoveLabelsWithStringArguments(self):
        if False:
            for i in range(10):
                print('nop')
        bug = 'Bug'
        question = 'Question'
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])
        self.issue.remove_from_labels(bug)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Project management', 'Question'])
        self.issue.remove_from_labels(question)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Project management'])
        self.issue.add_to_labels(bug, question)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])

    def testDeleteAndSetLabels(self):
        if False:
            i = 10
            return i + 15
        bug = self.repo.get_label('Bug')
        question = self.repo.get_label('Question')
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])
        self.issue.delete_labels()
        self.assertListKeyEqual(self.issue.get_labels(), None, [])
        self.issue.set_labels(bug, question)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Question'])

    def testDeleteAndSetLabelsWithStringArguments(self):
        if False:
            while True:
                i = 10
        bug = 'Bug'
        question = 'Question'
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Project management', 'Question'])
        self.issue.delete_labels()
        self.assertListKeyEqual(self.issue.get_labels(), None, [])
        self.issue.set_labels(bug, question)
        self.assertListKeyEqual(self.issue.get_labels(), lambda l: l.name, ['Bug', 'Question'])

    def testGetReactions(self):
        if False:
            print('Hello World!')
        reactions = self.issue.get_reactions()
        self.assertEqual(reactions[0].content, '+1')

    def testCreateReaction(self):
        if False:
            for i in range(10):
                print('nop')
        reaction = self.issue.create_reaction('hooray')
        self.assertEqual(reaction.id, 16917472)
        self.assertEqual(reaction.content, 'hooray')

    def testDeleteReaction(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.issue.delete_reaction(85740167))

    def testGetTimeline(self):
        if False:
            i = 10
            return i + 15
        expected_events = {'referenced', 'cross-referenced', 'locked', 'unlocked', 'closed', 'assigned', 'commented', 'subscribed', 'labeled'}
        events = self.issue.get_timeline()
        first = events[0]
        self.assertEqual(15819975, first.id)
        self.assertEqual('MDE1OlN1YnNjcmliZWRFdmVudDE1ODE5OTc1', first.node_id)
        self.assertEqual('https://api.github.com/repos/PyGithub/PyGithub/issues/events/15819975', first.url)
        self.assertEqual('jacquev6', first.actor.login)
        self.assertEqual(327146, first.actor.id)
        self.assertEqual('subscribed', first.event)
        self.assertIsNone(first.commit_id)
        self.assertIsNone(first.commit_url)
        self.assertEqual(repr(first), 'TimelineEvent(id=15819975)')
        for event in events:
            self.assertIn(event.event, expected_events)
            self.assertIsNotNone(event.created_at)
            self.assertIsNotNone(event.actor)
            if event.event == 'cross-referenced':
                self.assertIsNotNone(event.source)
                self.assertEqual(event.source.type, 'issue')
                self.assertEqual(event.source.issue.number, 857)
                self.assertEqual(repr(event.source), 'TimelineEventSource(type="issue")')
            else:
                self.assertIsNotNone(event.id)
                self.assertIsNotNone(event.node_id)
                if event.event == 'commented':
                    self.assertIsNotNone(event.body)
                else:
                    self.assertIsNone(event.source)
                    self.assertIsNotNone(event.actor)