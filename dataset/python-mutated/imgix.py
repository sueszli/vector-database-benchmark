import hashlib
from pylons import app_globals as g
import requests
from r2.lib.configparse import ConfigValue
from r2.lib.providers.image_resizing import ImageResizingProvider, NotLargeEnough
from r2.lib.utils import UrlParser, query_string

class ImgixImageResizingProvider(ImageResizingProvider):
    """A provider that uses imgix to create on-the-fly resizings."""
    config = {ConfigValue.str: ['imgix_domain']}

    def resize_image(self, image, width=None, censor_nsfw=False, max_ratio=None):
        if False:
            return 10
        url = UrlParser(image['url'])
        url.hostname = g.imgix_domain
        url.scheme = 'https'
        if max_ratio:
            url.update_query(fit='crop')
            url.update_query(crop='faces,entropy')
            url.update_query(arh=max_ratio)
        if width:
            if width > image['width']:
                raise NotLargeEnough()
            url.update_query(w=width)
        if censor_nsfw:
            url.update_query(blur=600)
            url.update_query(px=32)
        if g.imgix_signing:
            url = self._sign_url(url, g.secrets['imgix_signing_token'])
        return url.unparse()

    def _sign_url(self, url, token):
        if False:
            print('Hello World!')
        "Sign a url for imgix's secured sources.\n\n        Based very heavily on the example code in the docs:\n            http://www.imgix.com/docs/tutorials/securing-images\n\n        Arguments:\n\n        * url -- a UrlParser instance of the url to sign.  This object may be\n                 modified by the function, so make a copy beforehand if that is\n                 a concern.\n        * token -- a string token provided by imgix for request signing\n\n        Returns a UrlParser instance with signing parameters.\n        "
        signvalue = token + url.path
        if url.query_dict:
            signvalue += query_string(url.query_dict)
        signature = hashlib.md5(signvalue).hexdigest()
        url.update_query(s=signature)
        return url

    def purge_url(self, url):
        if False:
            i = 10
            return i + 15
        'Purge an image (by url) from imgix.\n\n        Reference: http://www.imgix.com/docs/tutorials/purging-images\n\n        Note that as mentioned in the imgix docs, in order to remove\n        an image, this function should be used *after* already\n        removing the image from our source, or imgix will just re-fetch\n        and replace the image with a new copy even after purging.\n        '
        requests.post('https://api.imgix.com/v2/image/purger', auth=(g.secrets['imgix_api_key'], ''), data={'url': url})