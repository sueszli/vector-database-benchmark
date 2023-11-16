from .common import KARMA, TestForumCommon
from ..models.forum import KarmaError
from odoo.exceptions import UserError, AccessError
from odoo.tools import mute_logger

class TestForum(TestForumCommon):

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_ask(self):
        if False:
            i = 10
            return i + 15
        Post = self.env['forum.post']
        with self.assertRaises(AccessError):
            Post.sudo(self.user_public).create({'name': ' Question ?', 'forum_id': self.forum.id})
        with self.assertRaises(KarmaError):
            Post.sudo(self.user_portal).create({'name': ' Q_0', 'forum_id': self.forum.id, 'tag_ids': [(0, 0, {'name': 'Tag0', 'forum_id': self.forum.id})]})
        self.user_portal.karma = KARMA['tag_create']
        Post.sudo(self.user_portal).create({'name': ' Q0', 'forum_id': self.forum.id, 'tag_ids': [(0, 0, {'name': 'Tag1', 'forum_id': self.forum.id})]})
        self.assertEqual(self.user_portal.karma, KARMA['tag_create'], 'website_forum: wrong karma generation when asking question')
        self.user_portal.karma = KARMA['post']
        Post.sudo(self.user_portal).create({'name': ' Q0', 'forum_id': self.forum.id, 'tag_ids': [(0, 0, {'name': 'Tag42', 'forum_id': self.forum.id})]})
        self.assertEqual(self.user_portal.karma, KARMA['post'] + KARMA['gen_que_new'], 'website_forum: wrong karma generation when asking question')

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_answer(self):
        if False:
            while True:
                i = 10
        Post = self.env['forum.post']
        with self.assertRaises(KarmaError):
            Post.sudo(self.user_employee).create({'name': ' A0', 'forum_id': self.forum.id, 'parent_id': self.post.id})
        self.user_employee.karma = KARMA['ans']
        Post.sudo(self.user_employee).create({'name': ' A0', 'forum_id': self.forum.id, 'parent_id': self.post.id})
        self.assertEqual(self.user_employee.karma, KARMA['ans'], 'website_forum: wrong karma generation when answering question')

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_vote_crash(self):
        if False:
            return 10
        Post = self.env['forum.post']
        self.user_employee.karma = KARMA['ans']
        emp_answer = Post.sudo(self.user_employee).create({'name': 'TestAnswer', 'forum_id': self.forum.id, 'parent_id': self.post.id})
        with self.assertRaises(UserError):
            emp_answer.vote(upvote=True)
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).vote(upvote=True)

    def test_vote(self):
        if False:
            i = 10
            return i + 15
        self.post.create_uid.karma = KARMA['ask']
        self.user_portal.karma = KARMA['upv']
        self.post.sudo(self.user_portal).vote(upvote=True)
        self.assertEqual(self.post.create_uid.karma, KARMA['ask'] + KARMA['gen_que_upv'], 'website_forum: wrong karma generation of upvoted question author')

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_downvote_crash(self):
        if False:
            for i in range(10):
                print('nop')
        Post = self.env['forum.post']
        self.user_employee.karma = KARMA['ans']
        emp_answer = Post.sudo(self.user_employee).create({'name': 'TestAnswer', 'forum_id': self.forum.id, 'parent_id': self.post.id})
        with self.assertRaises(UserError):
            emp_answer.vote(upvote=False)
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).vote(upvote=False)

    def test_downvote(self):
        if False:
            while True:
                i = 10
        self.post.create_uid.karma = 50
        self.user_portal.karma = KARMA['dwv']
        self.post.sudo(self.user_portal).vote(upvote=False)
        self.assertEqual(self.post.create_uid.karma, 50 + KARMA['gen_que_dwv'], 'website_forum: wrong karma generation of downvoted question author')

    def test_comment_crash(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).message_post(body='Should crash', message_type='comment')

    def test_comment(self):
        if False:
            print('Hello World!')
        self.post.sudo(self.user_employee).message_post(body='Test0', message_type='notification')
        self.user_employee.karma = KARMA['com_all']
        self.post.sudo(self.user_employee).message_post(body='Test1', message_type='comment')
        self.assertEqual(len(self.post.message_ids), 4, 'website_forum: wrong behavior of message_post')

    def test_flag_a_post(self):
        if False:
            return 10
        Post = self.env['forum.post']
        self.user_portal.karma = KARMA['ask']
        post = Post.sudo(self.user_portal).create({'name': 'Q0', 'forum_id': self.forum.id})
        with self.assertRaises(KarmaError):
            post.sudo(self.user_portal).flag()
        self.user_portal.karma = KARMA['flag']
        post.state = 'active'
        post.sudo(self.user_portal).flag()
        self.assertEqual(post.state, 'flagged', 'website_forum: wrong state when flagging a post')

    def test_validate_a_post(self):
        if False:
            return 10
        Post = self.env['forum.post']
        self.user_portal.karma = KARMA['ask']
        post = Post.sudo(self.user_portal).create({'name': 'Q0', 'forum_id': self.forum.id})
        with self.assertRaises(KarmaError):
            post.sudo(self.user_portal).validate()
        self.user_portal.karma = KARMA['moderate']
        post.state = 'pending'
        init_karma = post.create_uid.karma
        post.sudo(self.user_portal).validate()
        self.assertEqual(post.state, 'active', 'website_forum: wrong state when validate a post after pending')
        self.assertEqual(post.create_uid.karma, init_karma + KARMA['gen_que_new'], 'website_forum: wrong karma when validate a post after pending')
        self.user_portal.karma = KARMA['moderate']
        post.state = 'flagged'
        post.sudo(self.user_portal).validate()
        self.assertEqual(post.state, 'active', 'website_forum: wrong state when validate a post after flagged')
        self.user_portal.karma = KARMA['moderate']
        post.state = 'offensive'
        init_karma = post.create_uid.karma
        post.sudo(self.user_portal).validate()
        self.assertEqual(post.state, 'active', 'website_forum: wrong state when validate a post after offensive')

    def test_refuse_a_post(self):
        if False:
            while True:
                i = 10
        Post = self.env['forum.post']
        self.user_portal.karma = KARMA['ask']
        post = Post.sudo(self.user_portal).create({'name': 'Q0', 'forum_id': self.forum.id})
        with self.assertRaises(KarmaError):
            post.sudo(self.user_portal).refuse()
        self.user_portal.karma = KARMA['moderate']
        post.state = 'pending'
        init_karma = post.create_uid.karma
        post.sudo(self.user_portal).refuse()
        self.assertEqual(post.moderator_id, self.user_portal, 'website_forum: wrong moderator_id when refusing')
        self.assertEqual(post.create_uid.karma, init_karma, 'website_forum: wrong karma when refusing a post')

    def test_mark_a_post_as_offensive(self):
        if False:
            return 10
        Post = self.env['forum.post']
        self.user_portal.karma = KARMA['ask']
        post = Post.sudo(self.user_portal).create({'name': 'Q0', 'forum_id': self.forum.id})
        with self.assertRaises(KarmaError):
            post.sudo(self.user_portal).mark_as_offensive(12)
        self.user_portal.karma = KARMA['moderate']
        post.state = 'flagged'
        init_karma = post.create_uid.karma
        post.sudo(self.user_portal).mark_as_offensive(12)
        self.assertEqual(post.state, 'offensive', 'website_forum: wrong state when marking a post as offensive')
        self.assertEqual(post.create_uid.karma, init_karma + KARMA['gen_ans_flag'], 'website_forum: wrong karma when marking a post as offensive')

    def test_convert_answer_to_comment_crash(self):
        if False:
            return 10
        Post = self.env['forum.post']
        new_msg = self.post.sudo(self.user_portal).convert_answer_to_comment()
        self.assertEqual(new_msg.id, False, 'website_forum: question to comment conversion failed')
        self.assertEqual(Post.search([('name', '=', 'TestQuestion')])[0].forum_id.name, 'TestForum', 'website_forum: question to comment conversion failed')
        with self.assertRaises(KarmaError):
            self.answer.sudo(self.user_portal).convert_answer_to_comment()

    def test_convert_answer_to_comment(self):
        if False:
            i = 10
            return i + 15
        self.user_portal.karma = KARMA['com_conv_all']
        post_author = self.answer.create_uid.partner_id
        new_msg = self.answer.sudo(self.user_portal).convert_answer_to_comment()
        self.assertEqual(len(new_msg), 1, 'website_forum: wrong answer to comment conversion')
        self.assertEqual(new_msg.author_id, post_author, 'website_forum: wrong answer to comment conversion')
        self.assertIn('I am an anteater', new_msg.body, 'website_forum: wrong answer to comment conversion')

    def test_edit_post_crash(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).write({'name': 'I am not your father.'})

    def test_edit_post(self):
        if False:
            i = 10
            return i + 15
        self.post.create_uid.karma = KARMA['edit_own']
        self.post.write({'name': 'Actually I am your dog.'})
        self.user_portal.karma = KARMA['edit_all']
        self.post.sudo(self.user_portal).write({'name': 'Actually I am your cat.'})

    def test_close_post_crash(self):
        if False:
            print('Hello World!')
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).close(None)

    def test_close_post_own(self):
        if False:
            while True:
                i = 10
        self.post.create_uid.karma = KARMA['close_own']
        self.post.close(None)

    def test_close_post_all(self):
        if False:
            print('Hello World!')
        self.user_portal.karma = KARMA['close_all']
        self.post.sudo(self.user_portal).close(None)

    def test_deactivate_post_crash(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).write({'active': False})

    def test_deactivate_post_own(self):
        if False:
            print('Hello World!')
        self.post.create_uid.karma = KARMA['unlink_own']
        self.post.write({'active': False})

    def test_deactivate_post_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.user_portal.karma = KARMA['unlink_all']
        self.post.sudo(self.user_portal).write({'active': False})

    def test_unlink_post_crash(self):
        if False:
            return 10
        with self.assertRaises(KarmaError):
            self.post.sudo(self.user_portal).unlink()

    def test_unlink_post_own(self):
        if False:
            while True:
                i = 10
        self.post.create_uid.karma = KARMA['unlink_own']
        self.post.unlink()

    def test_unlink_post_all(self):
        if False:
            print('Hello World!')
        self.user_portal.karma = KARMA['unlink_all']
        self.post.sudo(self.user_portal).unlink()