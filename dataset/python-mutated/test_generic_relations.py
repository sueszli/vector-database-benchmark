from django.test import TestCase
from rest_framework import serializers
from .models import Bookmark, Note, Tag

class TestGenericRelations(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.bookmark = Bookmark.objects.create(url='https://www.djangoproject.com/')
        Tag.objects.create(tagged_item=self.bookmark, tag='django')
        Tag.objects.create(tagged_item=self.bookmark, tag='python')
        self.note = Note.objects.create(text='Remember the milk')
        Tag.objects.create(tagged_item=self.note, tag='reminder')

    def test_generic_relation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a relationship that spans a GenericRelation field.\n        IE. A reverse generic relationship.\n        '

        class BookmarkSerializer(serializers.ModelSerializer):
            tags = serializers.StringRelatedField(many=True)

            class Meta:
                model = Bookmark
                fields = ('tags', 'url')
        serializer = BookmarkSerializer(self.bookmark)
        expected = {'tags': ['django', 'python'], 'url': 'https://www.djangoproject.com/'}
        assert serializer.data == expected

    def test_generic_fk(self):
        if False:
            while True:
                i = 10
        '\n        Test a relationship that spans a GenericForeignKey field.\n        IE. A forward generic relationship.\n        '

        class TagSerializer(serializers.ModelSerializer):
            tagged_item = serializers.StringRelatedField()

            class Meta:
                model = Tag
                fields = ('tag', 'tagged_item')
        serializer = TagSerializer(Tag.objects.all(), many=True)
        expected = [{'tag': 'django', 'tagged_item': 'Bookmark: https://www.djangoproject.com/'}, {'tag': 'python', 'tagged_item': 'Bookmark: https://www.djangoproject.com/'}, {'tag': 'reminder', 'tagged_item': 'Note: Remember the milk'}]
        assert serializer.data == expected