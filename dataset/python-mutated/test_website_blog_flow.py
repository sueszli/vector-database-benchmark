from odoo.addons.website_blog.tests.common import TestWebsiteBlogCommon

class TestWebsiteBlogFlow(TestWebsiteBlogCommon):

    def test_website_blog_followers(self):
        if False:
            while True:
                i = 10
        ' Test the flow of followers and notifications for blogs. Intended\n        flow :\n\n         - people subscribe to a blog\n         - when creating a new post, nobody except the creator follows it\n         - people subscribed to the blog does not receive comments on posts\n         - when published, a notification is sent to all blog followers\n         - if someone subscribe to the post or comment it, it become follower\n           and receive notification for future comments. '
        test_blog = self.env['blog.blog'].sudo(self.user_blogmanager).create({'name': 'New Blog'})
        self.assertIn(self.user_blogmanager.partner_id, test_blog.message_partner_ids, 'website_blog: blog create should be in the blog followers')
        test_blog.message_subscribe([self.user_employee.partner_id.id, self.user_public.partner_id.id])
        test_blog_post = self.env['blog.post'].sudo(self.user_blogmanager).create({'name': 'New Post', 'blog_id': test_blog.id})
        self.assertNotIn(self.user_employee.partner_id, test_blog_post.message_partner_ids, 'website_blog: subscribing to a blog should not subscribe to its posts')
        self.assertNotIn(self.user_public.partner_id, test_blog_post.message_partner_ids, 'website_blog: subscribing to a blog should not subscribe to its posts')
        test_blog_post.write({'website_published': True})
        publish_message = next((m for m in test_blog_post.blog_id.message_ids if m.subtype_id.id == self.ref('website_blog.mt_blog_blog_published')), None)
        self.assertEqual(publish_message.needaction_partner_ids, self.user_employee.partner_id | self.user_public.partner_id, 'website_blog: peuple following a blog should be notified of a published post')
        test_blog_post.sudo().message_post(body='Armande BlogUser Commented', message_type='comment', author_id=self.user_employee.partner_id.id, subtype='mt_comment')
        self.assertIn(self.user_employee.partner_id, test_blog_post.message_partner_ids, 'website_blog: people commenting a post should follow it afterwards')