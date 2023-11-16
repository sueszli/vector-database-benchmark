"""Commands for operating on the search status of activities and blog posts."""
from __future__ import annotations
import math
from core import utils
from core.domain import blog_domain
from core.domain import collection_domain
from core.domain import exp_domain
from core.domain import rights_domain
from core.domain import rights_manager
from core.platform import models
from typing import Final, List, Optional, Tuple, TypedDict
MYPY = False
if MYPY:
    from mypy_imports import search_services as platform_search_services
platform_search_services = models.Registry.import_search_services()
SEARCH_INDEX_EXPLORATIONS: Final = 'explorations'
SEARCH_INDEX_COLLECTIONS: Final = 'collections'
SEARCH_INDEX_BLOG_POSTS: Final = 'blog-posts'
_DEFAULT_RANK: Final = 20

class DomainSearchDict(TypedDict):
    """Dictionary representing the search dictionary of a domain object."""
    id: str
    language_code: str
    title: str
    category: str
    tags: List[str]
    objective: str
    rank: int

def index_exploration_summaries(exp_summaries: List[exp_domain.ExplorationSummary]) -> None:
    if False:
        return 10
    'Adds the explorations to the search index.\n\n    Args:\n        exp_summaries: list(ExplorationSummary). List of Exp Summary domain\n            objects to be indexed.\n    '
    platform_search_services.add_documents_to_index([_exp_summary_to_search_dict(exp_summary) for exp_summary in exp_summaries if _should_index_exploration(exp_summary)], SEARCH_INDEX_EXPLORATIONS)

def _exp_summary_to_search_dict(exp_summary: exp_domain.ExplorationSummary) -> DomainSearchDict:
    if False:
        return 10
    'Updates the dict to be returned, whether the given exploration is to\n    be indexed for further queries or not.\n\n    Args:\n        exp_summary: ExplorationSummary. ExplorationSummary domain object.\n\n    Returns:\n        dict. The representation of the given exploration, in a form that can\n        be used by the search index.\n    '
    doc: DomainSearchDict = {'id': exp_summary.id, 'language_code': exp_summary.language_code, 'title': exp_summary.title, 'category': exp_summary.category, 'tags': exp_summary.tags, 'objective': exp_summary.objective, 'rank': get_search_rank_from_exp_summary(exp_summary)}
    return doc

def _should_index_exploration(exp_summary: exp_domain.ExplorationSummary) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether the given exploration should be indexed for future\n    search queries.\n\n    Args:\n        exp_summary: ExplorationSummary. ExplorationSummary domain object.\n\n    Returns:\n        bool. Whether the given exploration should be indexed for future\n        search queries.\n    '
    return not exp_summary.deleted and exp_summary.status != rights_domain.ACTIVITY_STATUS_PRIVATE

def get_search_rank_from_exp_summary(exp_summary: exp_domain.ExplorationSummary) -> int:
    if False:
        while True:
            i = 10
    "Returns an integer determining the document's rank in search.\n\n    Featured explorations get a ranking bump, and so do explorations that\n    have been more recently updated. Good ratings will increase the ranking\n    and bad ones will lower it.\n\n    Args:\n        exp_summary: ExplorationSummary. ExplorationSummary domain object.\n\n    Returns:\n        int. Document's rank in search.\n    "
    rating_weightings = {'1': -5, '2': -2, '3': 2, '4': 5, '5': 10}
    rank = _DEFAULT_RANK
    if exp_summary.ratings:
        for rating_value in exp_summary.ratings.keys():
            rank += exp_summary.ratings[rating_value] * rating_weightings[rating_value]
    return max(rank, 0)

def index_collection_summaries(collection_summaries: List[collection_domain.CollectionSummary]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Adds the collections to the search index.\n\n    Args:\n        collection_summaries: list(CollectionSummary). List of collection\n            summary domain objects to be indexed.\n    '
    platform_search_services.add_documents_to_index([_collection_summary_to_search_dict(collection_summary) for collection_summary in collection_summaries if _should_index_collection(collection_summary)], SEARCH_INDEX_COLLECTIONS)

def _collection_summary_to_search_dict(collection_summary: collection_domain.CollectionSummary) -> DomainSearchDict:
    if False:
        while True:
            i = 10
    'Converts a collection domain object to a search dict.\n\n    Args:\n        collection_summary: CollectionSummary. The collection\n            summary object to be converted.\n\n    Returns:\n        dict. The search dict of the collection domain object.\n    '
    doc: DomainSearchDict = {'id': collection_summary.id, 'title': collection_summary.title, 'category': collection_summary.category, 'objective': collection_summary.objective, 'language_code': collection_summary.language_code, 'tags': collection_summary.tags, 'rank': _DEFAULT_RANK}
    return doc

def _should_index_collection(collection: collection_domain.CollectionSummary) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if a particular collection should be indexed.\n\n    Args:\n        collection: CollectionSummary. CollectionSummary domain object.\n\n    Returns:\n        bool. Whether a particular collection should be indexed.\n    '
    rights = rights_manager.get_collection_rights(collection.id)
    return rights.status != rights_domain.ACTIVITY_STATUS_PRIVATE

def search_explorations(query: str, categories: List[str], language_codes: List[str], size: int, offset: Optional[int]=None) -> Tuple[List[str], Optional[int]]:
    if False:
        while True:
            i = 10
    "Searches through the available explorations.\n\n    Args:\n        query: str. The query string to search for.\n        categories: list(str). The list of categories to query for. If it is\n            empty, no category filter is applied to the results. If it is not\n            empty, then a result is considered valid if it matches at least one\n            of these categories.\n        language_codes: list(str). The list of language codes to query for. If\n            it is empty, no language code filter is applied to the results. If\n            it is not empty, then a result is considered valid if it matches at\n            least one of these language codes.\n        size: int. The maximum number of results to return.\n        offset: int or None. A marker that is used to get the next page of\n            results. If there are more documents that match the query than\n            'size', this function will return an offset to get the next page.\n\n    Returns:\n        tuple. A 2-tuple consisting of:\n            - list(str). A list of exploration ids that match the query.\n            - int or None. An offset if there are more matching explorations to\n              fetch, None otherwise. If an offset is returned, it will be a\n              web-safe string that can be used in URLs.\n    "
    (result_ids, result_offset) = platform_search_services.search(query, SEARCH_INDEX_EXPLORATIONS, categories, language_codes, offset=offset, size=size)
    return (result_ids, result_offset)

def delete_explorations_from_search_index(exploration_ids: List[str]) -> None:
    if False:
        while True:
            i = 10
    'Deletes the documents corresponding to these exploration_ids from the\n    search index.\n\n    Args:\n        exploration_ids: list(str). A list of exploration ids whose\n            documents are to be deleted from the search index.\n    '
    platform_search_services.delete_documents_from_index(exploration_ids, SEARCH_INDEX_EXPLORATIONS)

def clear_exploration_search_index() -> None:
    if False:
        i = 10
        return i + 15
    'WARNING: This runs in-request, and may therefore fail if there are too\n    many entries in the index.\n    '
    platform_search_services.clear_index(SEARCH_INDEX_EXPLORATIONS)

def search_collections(query: str, categories: List[str], language_codes: List[str], size: int, offset: Optional[int]=None) -> Tuple[List[str], Optional[int]]:
    if False:
        for i in range(10):
            print('nop')
    "Searches through the available collections.\n\n    Args:\n        query: str. The query string to search for.\n        categories: list(str). The list of categories to query for. If it is\n            empty, no category filter is applied to the results. If it is not\n            empty, then a result is considered valid if it matches at least one\n            of these categories.\n        language_codes: list(str). The list of language codes to query for. If\n            it is empty, no language code filter is applied to the results. If\n            it is not empty, then a result is considered valid if it matches at\n            least one of these language codes.\n        size: int. The maximum number of results to return.\n        offset: int|None. An offset, used to get the next page of results.\n            If there are more documents that match the query than 'size', this\n            function will return an offset to get the next page.\n\n    Returns:\n        2-tuple of (collection_ids, offset). Where:\n            - A list of collection ids that match the query.\n            - An offset if there are more matching collections to fetch, None\n              otherwise. If an offset is returned, it will be a web-safe string\n              that can be used in URLs.\n    "
    (result_ids, result_offset) = platform_search_services.search(query, SEARCH_INDEX_COLLECTIONS, categories, language_codes, offset=offset, size=size)
    return (result_ids, result_offset)

def delete_collections_from_search_index(collection_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Removes the given collections from the search index.\n\n    Args:\n        collection_ids: list(str). List of IDs of the collections to be removed\n            from the search index.\n    '
    platform_search_services.delete_documents_from_index(collection_ids, SEARCH_INDEX_COLLECTIONS)

def clear_collection_search_index() -> None:
    if False:
        while True:
            i = 10
    'Clears the search index.\n\n    WARNING: This runs in-request, and may therefore fail if there are too\n    many entries in the index.\n    '
    platform_search_services.clear_index(SEARCH_INDEX_COLLECTIONS)

class BlogPostSummaryDomainSearchDict(TypedDict):
    """Dictionary representing the search dictionary of a blog post summary
    domain object.
    """
    id: str
    title: str
    tags: List[str]
    rank: int

def index_blog_post_summaries(blog_post_summaries: List[blog_domain.BlogPostSummary]) -> None:
    if False:
        print('Hello World!')
    'Adds the blog post summaries to the search index.\n\n    Args:\n        blog_post_summaries: list(BlogPostSummary). List of BlogPostSummary\n            domain objects to be indexed.\n    '
    docs_to_index = [_blog_post_summary_to_search_dict(blog_post_summary) for blog_post_summary in blog_post_summaries]
    platform_search_services.add_documents_to_index([doc for doc in docs_to_index if doc], SEARCH_INDEX_BLOG_POSTS)

def _blog_post_summary_to_search_dict(blog_post_summary: blog_domain.BlogPostSummary) -> Optional[BlogPostSummaryDomainSearchDict]:
    if False:
        return 10
    'Updates the dict to be returned, whether the given blog post summary is\n    to be indexed for further queries or not.\n\n    Args:\n        blog_post_summary: BlogPostSummary. BlogPostSummary domain object.\n\n    Returns:\n        dict. The representation of the given blog post summary, in a form that\n        can be used by the search index.\n    '
    if not blog_post_summary.deleted and blog_post_summary.published_on is not None:
        doc: BlogPostSummaryDomainSearchDict = {'id': blog_post_summary.id, 'title': blog_post_summary.title, 'tags': blog_post_summary.tags, 'rank': math.floor(utils.get_time_in_millisecs(blog_post_summary.published_on))}
        return doc
    return None

def search_blog_post_summaries(query: str, tags: List[str], size: int, offset: Optional[int]=None) -> Tuple[List[str], Optional[int]]:
    if False:
        i = 10
        return i + 15
    "Searches through the available blog post summaries.\n\n    Args:\n        query: str. The query string to search for.\n        tags: list(str). The list of tags to query for. If it is\n            empty, no tags filter is applied to the results. If it is not\n            empty, then a result is considered valid if it matches at least one\n            of these tags.\n        size: int. The maximum number of results to return.\n        offset: int or None. A marker that is used to get the next page of\n            results. If there are more documents that match the query than\n            'size', this function will return an offset to get the next page.\n\n    Returns:\n        tuple. A 2-tuple consisting of:\n            - list(str). A list of blog post ids that match the query.\n            - int or None. An offset if there are more matching blog post\n              summaries to fetch, None otherwise. If an offset is returned, it\n              will be a web-safe string that can be used in URLs.\n    "
    (result_ids, result_offset) = platform_search_services.blog_post_summaries_search(query, tags, offset=offset, size=size)
    return (result_ids, result_offset)

def delete_blog_post_summary_from_search_index(blog_post_id: str) -> None:
    if False:
        while True:
            i = 10
    'Deletes the documents corresponding to the blog_id from the\n    search index.\n\n    Args:\n        blog_post_id: str. Blog post id whose document are to be deleted from\n            the search index.\n    '
    platform_search_services.delete_documents_from_index([blog_post_id], SEARCH_INDEX_BLOG_POSTS)

def clear_blog_post_summaries_search_index() -> None:
    if False:
        print('Hello World!')
    'Clears the blog post search index.\n\n    WARNING: This runs in-request, and may therefore fail if there are too\n    many entries in the index.\n    '
    platform_search_services.clear_index(SEARCH_INDEX_BLOG_POSTS)