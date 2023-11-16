"""Error classes for blog model audits."""
from __future__ import annotations
from core import utils
from core.jobs.types import base_validation_errors
from core.platform import models
from typing import Union
MYPY = False
if MYPY:
    from mypy_imports import blog_models
(blog_models,) = models.Registry.import_models([models.Names.BLOG])

class DuplicateBlogTitleError(base_validation_errors.BaseAuditError):
    """Error class for blog posts with duplicate titles."""

    def __init__(self, model: Union[blog_models.BlogPostModel, blog_models.BlogPostSummaryModel]) -> None:
        if False:
            return 10
        message = 'title=%s is not unique' % utils.quoted(model.title)
        super().__init__(message, model)

class DuplicateBlogUrlError(base_validation_errors.BaseAuditError):
    """Error class for blog posts with duplicate urls."""

    def __init__(self, model: Union[blog_models.BlogPostModel, blog_models.BlogPostSummaryModel]) -> None:
        if False:
            return 10
        message = 'url=%s is not unique' % utils.quoted(model.url_fragment)
        super().__init__(message, model)

class InconsistentLastUpdatedTimestampsError(base_validation_errors.BaseAuditError):
    """Error class for models with inconsistent timestamps."""

    def __init__(self, model: Union[blog_models.BlogPostModel, blog_models.BlogPostSummaryModel]) -> None:
        if False:
            for i in range(10):
                print('nop')
        message = 'created_on=%r is later than last_updated=%r' % (model.created_on, model.last_updated)
        super().__init__(message, model)

class InconsistentPublishLastUpdatedTimestampsError(base_validation_errors.BaseAuditError):
    """Error class for models with inconsistent timestamps."""

    def __init__(self, model: Union[blog_models.BlogPostModel, blog_models.BlogPostSummaryModel]) -> None:
        if False:
            for i in range(10):
                print('nop')
        message = 'published_on=%r is later than last_updated=%r' % (model.published_on, model.last_updated)
        super().__init__(message, model)

class ModelMutatedDuringJobErrorForLastUpdated(base_validation_errors.BaseAuditError):
    """Error class for models mutated during a job."""

    def __init__(self, model: Union[blog_models.BlogPostModel, blog_models.BlogPostSummaryModel]) -> None:
        if False:
            print('Hello World!')
        message = "last_updated=%r is later than the audit job's start time" % model.last_updated
        super().__init__(message, model)

class ModelMutatedDuringJobErrorForPublishedOn(base_validation_errors.BaseAuditError):
    """Error class for models mutated during a job."""

    def __init__(self, model: Union[blog_models.BlogPostModel, blog_models.BlogPostSummaryModel]) -> None:
        if False:
            while True:
                i = 10
        message = "published_on=%r is later than the audit job's start time" % model.published_on
        super().__init__(message, model)

class DuplicateBlogAuthorModelError(base_validation_errors.BaseAuditError):
    """Error class for blog author detail models with duplicate author ids."""

    def __init__(self, model: blog_models.BlogAuthorDetailsModel) -> None:
        if False:
            for i in range(10):
                print('nop')
        message = 'author id=%s is not unique' % utils.quoted(model.author_id)
        super().__init__(message, model)