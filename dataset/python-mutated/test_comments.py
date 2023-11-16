from django.contrib.auth import get_user_model
from django.test import TestCase
from wagtail.models import Comment, Page

class CommentTestingUtils:

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.page = Page.objects.get(title='Welcome to the Wagtail test site!')
        self.revision_1 = self.page.save_revision()
        self.revision_2 = self.page.save_revision()

    def create_comment(self, revision_created):
        if False:
            i = 10
            return i + 15
        return Comment.objects.create(page=self.page, user=get_user_model().objects.first(), text='test', contentpath='title', revision_created=revision_created)

class TestRevisionDeletion(CommentTestingUtils, TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.revision_3 = self.page.save_revision()
        self.old_comment = self.create_comment(self.revision_1)
        self.new_comment = self.create_comment(self.revision_3)

    def test_deleting_old_revision_moves_comment_revision_created_forwards(self):
        if False:
            while True:
                i = 10
        self.revision_1.delete()
        self.old_comment.refresh_from_db()
        self.assertEqual(self.old_comment.revision_created, self.revision_2)

    def test_deleting_most_recent_revision_deletes_created_comments(self):
        if False:
            for i in range(10):
                print('nop')
        self.revision_3.delete()
        with self.assertRaises(Comment.DoesNotExist):
            self.new_comment.refresh_from_db()