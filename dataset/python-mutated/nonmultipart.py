"""Base class for MIME type messages that are not multipart."""
__all__ = ['MIMENonMultipart']
from email import errors
from email.mime.base import MIMEBase

class MIMENonMultipart(MIMEBase):
    """Base class for MIME non-multipart type messages."""

    def attach(self, payload):
        if False:
            i = 10
            return i + 15
        raise errors.MultipartConversionError('Cannot attach additional subparts to non-multipart/*')