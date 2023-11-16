import os
from posixpath import basename
from typing import Any, List, Set
from urllib.parse import urlparse

from typing_extensions import override

from .common.spiders import BaseDocumentationSpider


def get_images_dir(images_path: str) -> str:
    # Get index html file as start url and convert it to file uri
    dir_path = os.path.dirname(os.path.realpath(__file__))
    target_path = os.path.join(dir_path, os.path.join(*[os.pardir] * 4), images_path)
    return os.path.realpath(target_path)


class UnusedImagesLinterSpider(BaseDocumentationSpider):
    images_path = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.static_images: Set[str] = set()
        self.images_static_dir: str = get_images_dir(self.images_path)

    @override
    def _is_external_url(self, url: str) -> bool:
        is_external = url.startswith("http") and self.start_urls[0] not in url
        if self._has_extension(url) and f"localhost:9981/{self.images_path}" in url:
            self.static_images.add(basename(urlparse(url).path))
        return is_external or self._has_extension(url)

    def closed(self, *args: Any, **kwargs: Any) -> None:
        unused_images = set(os.listdir(self.images_static_dir)) - self.static_images
        if unused_images:
            exception_message = (
                "The following images are not used in documentation and can be removed: {}"
            )
            unused_images_relatedpath = [
                os.path.join(self.images_path, img) for img in unused_images
            ]
            self.logger.error(exception_message.format(", ".join(unused_images_relatedpath)))


class HelpDocumentationSpider(UnusedImagesLinterSpider):
    name = "help_documentation_crawler"
    start_urls = ["http://localhost:9981/help/"]
    deny_domains: List[str] = []
    deny = ["/policies/privacy"]
    images_path = "static/images/help"


class APIDocumentationSpider(UnusedImagesLinterSpider):
    name = "api_documentation_crawler"
    start_urls = ["http://localhost:9981/api"]
    deny_domains: List[str] = []
    images_path = "static/images/api"


class PorticoDocumentationSpider(BaseDocumentationSpider):
    @override
    def _is_external_url(self, url: str) -> bool:
        return (
            not url.startswith("http://localhost:9981")
            or url.startswith(("http://localhost:9981/help/", "http://localhost:9981/api"))
            or self._has_extension(url)
        )

    name = "portico_documentation_crawler"
    start_urls = [
        "http://localhost:9981/hello/",
        "http://localhost:9981/history/",
        "http://localhost:9981/plans/",
        "http://localhost:9981/team/",
        "http://localhost:9981/apps/",
        "http://localhost:9981/integrations/",
        "http://localhost:9981/policies/terms",
        "http://localhost:9981/policies/privacy",
        "http://localhost:9981/features/",
        "http://localhost:9981/why-zulip/",
        "http://localhost:9981/for/open-source/",
        "http://localhost:9981/for/business/",
        "http://localhost:9981/for/communities/",
        "http://localhost:9981/for/research/",
        "http://localhost:9981/security/",
    ]
    deny_domains: List[str] = []
