from typing import TYPE_CHECKING
import saml2.metadata
from twisted.web.resource import Resource
from twisted.web.server import Request
if TYPE_CHECKING:
    from synapse.server import HomeServer

class SAML2MetadataResource(Resource):
    """A Twisted web resource which renders the SAML metadata"""
    isLeaf = 1

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        Resource.__init__(self)
        self.sp_config = hs.config.saml2.saml2_sp_config

    def render_GET(self, request: Request) -> bytes:
        if False:
            return 10
        metadata_xml = saml2.metadata.create_metadata_string(configfile=None, config=self.sp_config)
        request.setHeader(b'Content-Type', b'text/xml; charset=utf-8')
        return metadata_xml