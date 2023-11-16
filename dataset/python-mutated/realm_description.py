from zerver.lib.cache import cache_with_key, realm_rendered_description_cache_key, realm_text_description_cache_key
from zerver.lib.html_to_text import html_to_text
from zerver.lib.markdown import markdown_convert
from zerver.models import Realm

@cache_with_key(realm_rendered_description_cache_key, timeout=3600 * 24 * 7)
def get_realm_rendered_description(realm: Realm) -> str:
    if False:
        i = 10
        return i + 15
    realm_description_raw = realm.description or 'The coolest place in the universe.'
    return markdown_convert(realm_description_raw, message_realm=realm, no_previews=True).rendered_content

@cache_with_key(realm_text_description_cache_key, timeout=3600 * 24 * 7)
def get_realm_text_description(realm: Realm) -> str:
    if False:
        while True:
            i = 10
    html_description = get_realm_rendered_description(realm)
    return html_to_text(html_description, {'p': ' | ', 'li': ' * '})