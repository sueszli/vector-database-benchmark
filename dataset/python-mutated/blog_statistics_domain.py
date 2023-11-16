"""Domain objects relating to blog post statistics."""
from __future__ import annotations
import datetime
from core import utils
from core.constants import constants
from typing import Dict

class BlogPostViewsAggregatedStats:
    """Domain object representing blog post views aggregated stats model."""

    def __init__(self, blog_post_id: str, views_by_hour: Dict[str, Dict[str, int]], views_by_date: Dict[str, Dict[str, int]], views_by_month: Dict[str, Dict[str, int]], created_on: datetime.datetime) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs an BlogPostViewsAggregatedStats domain object.\n\n        Args:\n            blog_post_id: str. ID of the blog post.\n            views_by_hour: Dict. It will consist of a dict of dictionaries where\n                key is the date (YYYY-MM-DD) in the UTC format) and value is\n                dict with keys as hours (in the UTC format) and number of views\n                as values.\n            views_by_date: Dict. It will consist of key-value pairs where key is\n                the month(YYYY-MM) and value is dict with keys as UTC date and\n                values as  number of views on the blog posts on that date.\n            views_by_month: Dict. It will consist of a dict of dictionaries\n                where key is the year (in the UTC format) and value is dict\n                with keys as month number (in the UTC format) and number of\n                views in that month as value.\n            created_on: datetime. The time at which the blog post was created.\n        '
        self.blog_post_id = blog_post_id
        self.views_by_hour = views_by_hour
        self.views_by_date = views_by_date
        self.views_by_month = views_by_month
        self.created_on = created_on

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks whether the blog post id is a valid one.\n\n        Raises:\n            ValidationError. No blog_post_id specified.\n            ValidationError. The blog_post_id is not a string.\n            ValidationError. The blog_post_id is invalid.\n        '
        if not self.blog_post_id:
            raise utils.ValidationError('No blog_post_id specified')
        if not isinstance(self.blog_post_id, str):
            raise utils.ValidationError('Blog Post ID must be a string, but got %r' % self.blog_post_id)
        if len(self.blog_post_id) != constants.BLOG_POST_ID_LENGTH:
            raise utils.ValidationError('Blog ID %s is invalid' % self.blog_post_id)

class BlogPostReadsAggregatedStats:
    """Domain object representing blog post reads aggregated stats model."""

    def __init__(self, blog_post_id: str, reads_by_hour: Dict[str, Dict[str, int]], reads_by_date: Dict[str, Dict[str, int]], reads_by_month: Dict[str, Dict[str, int]], created_on: datetime.datetime) -> None:
        if False:
            print('Hello World!')
        'Constructs an BlogPostReadsAggregatedStats domain object.\n\n        Args:\n            blog_post_id: str. ID of the blog post.\n            reads_by_hour: Dict. It will consist of a dict of dictionaries where\n                key is the date (YYYY-MM-DD) in the UTC format) and value is\n                dict with keys as hours (in the UTC format) and number of reads\n                as values.\n            reads_by_date: Dict. It will consist of key-value pairs where key is\n                the month(YYYY-MM) and value is dict with keys as UTC date and\n                values as number of reads on the blog posts on that date.\n            reads_by_month: Dict. It will consist of a dict of dictionaries\n                where key is the year (in the UTC format) and value is dict\n                with keys as month number (in the UTC format) and number of\n                reads in that month as value.\n            created_on: datetime. The time at which the blog post was created.\n        '
        self.blog_post_id = blog_post_id
        self.reads_by_hour = reads_by_hour
        self.reads_by_date = reads_by_date
        self.reads_by_month = reads_by_month
        self.created_on = created_on

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Checks whether the blog post id is a valid one.\n\n        Raises:\n            ValidationError. No blog_post_id specified.\n            ValidationError. The blog_post_id is not a string.\n            ValidationError. The blog_post_id is invalid.\n        '
        if not self.blog_post_id:
            raise utils.ValidationError('No blog_post_id specified')
        if not isinstance(self.blog_post_id, str):
            raise utils.ValidationError('Blog Post ID must be a string, but got %r' % self.blog_post_id)
        if len(self.blog_post_id) != constants.BLOG_POST_ID_LENGTH:
            raise utils.ValidationError('Blog ID %s is invalid' % self.blog_post_id)

class BlogPostReadingTime:
    """Domain object representing blog post reading time model."""

    def __init__(self, blog_post_id: str, zero_to_one_min: int, one_to_two_min: int, two_to_three_min: int, three_to_four_min: int, four_to_five_min: int, five_to_six_min: int, six_to_seven_min: int, seven_to_eight_min: int, eight_to_nine_min: int, nine_to_ten_min: int, more_than_ten_min: int) -> None:
        if False:
            print('Hello World!')
        'Constructs an BlogPostReadingTime domain object.\n\n        Args:\n            blog_post_id: str. ID of the blog post.\n            zero_to_one_min: int. Number of user taking less than a minute to\n                read the blog post.\n            one_to_two_min: int. Number of users taking one to two minutes to\n                read the blog post.\n            two_to_three_min: int. Number of users taking two to three minutes\n                to read the blog post.\n            three_to_four_min: int. Number of users taking three to four minutes\n                to read the blog post.\n            four_to_five_min: int. Number of users taking four to five minutes\n                to read the blog post.\n            five_to_six_min: int. Number of users taking five to six minutes to\n                read the blog post.\n            six_to_seven_min: int. Number of users taking six to seven minutes\n                to read the blog post.\n            seven_to_eight_min: int. Number of users taking seven to eight\n                minutes to read the blog post.\n            eight_to_nine_min: int. Number of users taking eight to nine minutes\n                to read the blog post.\n            nine_to_ten_min: int. Number of users taking nine to ten minutes to\n                read the blog post.\n            more_than_ten_min: int. Number of users taking more than ten minutes\n                to read the blog post.\n        '
        self.blog_post_id = blog_post_id
        self.zero_to_one_min = zero_to_one_min
        self.one_to_two_min = one_to_two_min
        self.two_to_three_min = two_to_three_min
        self.three_to_four_min = three_to_four_min
        self.four_to_five_min = four_to_five_min
        self.five_to_six_min = five_to_six_min
        self.six_to_seven_min = six_to_seven_min
        self.seven_to_eight_min = seven_to_eight_min
        self.eight_to_nine_min = eight_to_nine_min
        self.nine_to_ten_min = nine_to_ten_min
        self.more_than_ten_min = more_than_ten_min

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the blog post id is a valid one.\n\n        Raises:\n            ValidationError. No blog_post_id specified.\n            ValidationError. The blog_post_id is not a string.\n            ValidationError. The blog_post_id is invalid.\n        '
        if not self.blog_post_id:
            raise utils.ValidationError('No blog_post_id specified')
        if not isinstance(self.blog_post_id, str):
            raise utils.ValidationError('Blog Post ID must be a string, but got %r' % self.blog_post_id)
        if len(self.blog_post_id) != constants.BLOG_POST_ID_LENGTH:
            raise utils.ValidationError('Blog Post ID %s is invalid' % self.blog_post_id)

class AuthorBlogPostViewsAggregatedStats:
    """Domain object representing author blog post views aggregated stats
    model.
    """

    def __init__(self, author_id: str, views_by_hour: Dict[str, Dict[str, int]], views_by_date: Dict[str, Dict[str, int]], views_by_month: Dict[str, Dict[str, int]], created_on: datetime.datetime) -> None:
        if False:
            while True:
                i = 10
        'Constructs an AuthorBlogPostViewsAggregatedStats domain object.\n\n        Args:\n            author_id: str. User ID of the author.\n            views_by_hour: Dict. It will consist of a dict of dictionaries where\n                key is the date (YYYY-MM-DD) in the UTC format) and value is\n                dict with keys as hours (in the UTC format) and number of views\n                as values.\n            views_by_date: Dict. It will consist of key-value pairs where key is\n                the month(YYYY-MM) and value is dict with keys as UTC date and\n                values as  number of views on the blog posts on that date.\n            views_by_month: Dict. It will consist of a dict of dictionaries\n                where key is the year (in the UTC format) and value is dict\n                with keys as month number (in the UTC format) and number of\n                views in that month as value.\n            created_on: datetime. The time at which the author stats model was\n                created.\n        '
        self.author_id = author_id
        self.views_by_hour = views_by_hour
        self.views_by_date = views_by_date
        self.views_by_month = views_by_month
        self.created_on = created_on

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Checks whether the blog post id is a valid one.\n\n        Raises:\n            ValidationError. No author_id specified.\n            ValidationError. The author_id is not a string.\n            ValidationError. The author_id has invalid format.\n        '
        if not self.author_id:
            raise utils.ValidationError('No author_id specified')
        if not isinstance(self.author_id, str):
            raise utils.ValidationError('Author ID must be a string, but got %r' % self.author_id)
        if not utils.is_user_id_valid(self.author_id):
            raise utils.ValidationError('author_id=%r has the wrong format' % self.author_id)

class AuthorBlogPostReadsAggregatedStats:
    """Domain object representing author blog post reads aggregated stats
    model.
    """

    def __init__(self, author_id: str, reads_by_hour: Dict[str, Dict[str, int]], reads_by_date: Dict[str, Dict[str, int]], reads_by_month: Dict[str, Dict[str, int]], created_on: datetime.datetime) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs an AuthorBlogPostReadsAggregatedStats domain object.\n\n        Args:\n            author_id: str. ID of the author.\n            reads_by_hour: Dict. It will consist of a dict of dictionaries where\n                key is the date (YYYY-MM-DD) in the UTC format) and value is\n                dict with keys as hours (in the UTC format) and number of reads\n                as values.\n            reads_by_date: Dict. It will consist of key-value pairs where key is\n                the month(YYYY-MM) and value is dict with keys as UTC date and\n                values as number of reads on the blog posts on that date.\n            reads_by_month: Dict. It will consist of a dict of dictionaries\n                where key is the year (in the UTC format) and value is dict\n                with keys as month number (in the UTC format) and number of\n                reads in that month as value.\n            created_on: datetime. The time at which the author stats model was\n                created.\n        '
        self.author_id = author_id
        self.reads_by_hour = reads_by_hour
        self.reads_by_date = reads_by_date
        self.reads_by_month = reads_by_month
        self.created_on = created_on

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Checks whether the blog post id is a valid one.\n\n        Raises:\n            ValidationError. No author_id specified.\n            ValidationError. The author_id is not a string.\n            ValidationError. The author_id has invalid format.\n        '
        if not self.author_id:
            raise utils.ValidationError('No author_id specified')
        if not isinstance(self.author_id, str):
            raise utils.ValidationError('Author ID must be a string, but got %r' % self.author_id)
        if not utils.is_user_id_valid(self.author_id):
            raise utils.ValidationError('author_id=%r has the wrong format' % self.author_id)

class AuthorBlogPostsReadingTime:
    """Domain object representing author blog post reading time model."""

    def __init__(self, author_id: str, zero_to_one_min: int, one_to_two_min: int, two_to_three_min: int, three_to_four_min: int, four_to_five_min: int, five_to_six_min: int, six_to_seven_min: int, seven_to_eight_min: int, eight_to_nine_min: int, nine_to_ten_min: int, more_than_ten_min: int) -> None:
        if False:
            return 10
        'Constructs an BlogPostReadingTime domain object.\n\n        Args:\n            author_id: str. User ID of the author.\n            zero_to_one_min: int. Number of user taking less than a minute to\n                read the blog posts written by the author.\n            one_to_two_min: int. Number of users taking one to two minutes to\n                read the blog posts written by the author.\n            two_to_three_min: int. Number of users taking two to three minutes\n                to read the blog posts written by the author.\n            three_to_four_min: int. Number of users taking three to four minutes\n                to read the blog posts written by the author.\n            four_to_five_min: int. Number of users taking four to five minutes\n                to read the blog posts written by the author.\n            five_to_six_min: int. Number of users taking five to six minutes to\n                read the blog posts written by the author.\n            six_to_seven_min: int. Number of users taking six to seven minutes\n                to read the blog posts written by the author.\n            seven_to_eight_min: int. Number of users taking seven to eight\n                minutes to read the blog posts written by the author.\n            eight_to_nine_min: int. Number of users taking eight to nine minutes\n                to read the blog posts written by the author.\n            nine_to_ten_min: int. Number of users taking nine to ten minutes to\n                read the blog posts written by the author.\n            more_than_ten_min: int. Number of users taking more than ten minutes\n                to read the blog posts written by the author.\n        '
        self.author_id = author_id
        self.zero_to_one_min = zero_to_one_min
        self.one_to_two_min = one_to_two_min
        self.two_to_three_min = two_to_three_min
        self.three_to_four_min = three_to_four_min
        self.four_to_five_min = four_to_five_min
        self.five_to_six_min = five_to_six_min
        self.six_to_seven_min = six_to_seven_min
        self.seven_to_eight_min = seven_to_eight_min
        self.eight_to_nine_min = eight_to_nine_min
        self.nine_to_ten_min = nine_to_ten_min
        self.more_than_ten_min = more_than_ten_min

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Checks whether the author id is a valid one.\n\n        Raises:\n            ValidationError. No author_id specified.\n            ValidationError. The author_id is not a string.\n            ValidationError. The author_id has invalid format.\n        '
        if not self.author_id:
            raise utils.ValidationError('No author_id specified')
        if not isinstance(self.author_id, str):
            raise utils.ValidationError('Author ID must be a string, but got %r' % self.author_id)
        if not utils.is_user_id_valid(self.author_id):
            raise utils.ValidationError('author_id=%r has the wrong format' % self.author_id)