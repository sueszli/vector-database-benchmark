"""Controllers for validating access."""
from __future__ import annotations
from core import feconf
from core.constants import constants
from core.controllers import acl_decorators
from core.controllers import base
from core.domain import blog_services
from core.domain import classroom_config_services
from core.domain import learner_group_services
from core.domain import user_services
from typing import Dict, TypedDict

class ClassroomAccessValidationHandlerNormalizedRequestDict(TypedDict):
    """Dict representation of ClassroomAccessValidationHandler's
    normalized_request dictionary.
    """
    classroom_url_fragment: str

class ClassroomAccessValidationHandler(base.BaseHandler[Dict[str, str], ClassroomAccessValidationHandlerNormalizedRequestDict]):
    """Validates whether request made to /learn route.
    """
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS = {'GET': {'classroom_url_fragment': {'schema': {'type': 'basestring'}}}}

    @acl_decorators.open_access
    def get(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Retrieves information about a classroom.\n\n        Raises:\n            PageNotFoundException. The classroom cannot be found.\n        '
        assert self.normalized_request is not None
        classroom_url_fragment = self.normalized_request['classroom_url_fragment']
        classroom = classroom_config_services.get_classroom_by_url_fragment(classroom_url_fragment)
        if not classroom:
            raise self.PageNotFoundException

class ManageOwnAccountValidationHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Validates access to preferences page.
    """
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.can_manage_own_account
    def get(self) -> None:
        if False:
            return 10
        'Handles GET requests.'
        pass

class ProfileExistsValidationHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """The world-viewable profile page."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'username': {'schema': {'type': 'basestring'}}}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.open_access
    def get(self, username: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates access to profile page.\n\n        Args:\n            username: str. The username of the user.\n\n        Raises:\n            PageNotFoundException. No user settings found for the given\n                username.\n        '
        user_settings = user_services.get_user_settings_from_username(username)
        if not user_settings:
            raise self.PageNotFoundException

class ReleaseCoordinatorAccessValidationHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Validates access to release coordinator page."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.can_access_release_coordinator_page
    def get(self) -> None:
        if False:
            while True:
                i = 10
        'Handles GET requests.'
        pass

class ViewLearnerGroupPageAccessValidationHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Validates access to view learner group page."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'learner_group_id': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_regex_matched', 'regex_pattern': constants.LEARNER_GROUP_ID_REGEX}]}}}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.can_access_learner_groups
    def get(self, learner_group_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Retrieves information about a learner group.\n\n        Args:\n            learner_group_id: str. The learner group ID.\n\n        Raises:\n            PageNotFoundException. The learner groups are not enabled.\n            PageNotFoundException. The user is not a member of the learner\n                group.\n        '
        assert self.user_id is not None
        if not learner_group_services.is_learner_group_feature_enabled():
            raise self.PageNotFoundException
        is_valid_request = learner_group_services.is_user_learner(self.user_id, learner_group_id)
        if not is_valid_request:
            raise self.PageNotFoundException

class BlogHomePageAccessValidationHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Validates access to blog home page."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.open_access
    def get(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validates access to blog home page.'
        pass

class BlogPostPageAccessValidationHandlerNormalizedRequestDict(TypedDict):
    """Dict representation of BlogPostPageAccessValidationHandler's
    normalized_request dictionary.
    """
    blog_post_url_fragment: str

class BlogPostPageAccessValidationHandler(base.BaseHandler[Dict[str, str], BlogPostPageAccessValidationHandlerNormalizedRequestDict]):
    """Validates whether request made to correct blog post route."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS = {'GET': {'blog_post_url_fragment': {'schema': {'type': 'basestring'}}}}

    @acl_decorators.open_access
    def get(self) -> None:
        if False:
            print('Hello World!')
        'Retrieves information about a blog post.\n\n        Raises:\n            PageNotFoundException. The blog post cannot be found.\n        '
        assert self.normalized_request is not None
        blog_post_url_fragment = self.normalized_request['blog_post_url_fragment']
        blog_post = blog_services.get_blog_post_by_url_fragment(blog_post_url_fragment)
        if not blog_post:
            raise self.PageNotFoundException

class BlogAuthorProfilePageAccessValidationHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Validates access to blog author profile page."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'author_username': {'schema': {'type': 'basestring'}, 'validators': [{'id': 'has_length_at_most', 'max_value': constants.MAX_AUTHOR_NAME_LENGTH}]}}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.open_access
    def get(self, author_username: str) -> None:
        if False:
            return 10
        'Retrieves information about a blog post author.\n\n        Args:\n            author_username: str. The author username.\n\n        Raises:\n            PageNotFoundException. User with given username does not exist.\n            PageNotFoundException. User with given username is not a blog\n                post author.\n        '
        author_settings = user_services.get_user_settings_from_username(author_username)
        if author_settings is None:
            raise self.PageNotFoundException('User with given username does not exist')
        if not user_services.is_user_blog_post_author(author_settings.user_id):
            raise self.PageNotFoundException('User with given username is not a blog post author.')