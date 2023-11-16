"""A wrapper over a list of QSslErrors."""
from typing import Sequence, Optional
from qutebrowser.qt.network import QSslError, QNetworkReply
from qutebrowser.utils import usertypes, utils, debug, jinja, urlutils

class CertificateErrorWrapper(usertypes.AbstractCertificateErrorWrapper):
    """A wrapper over a list of QSslErrors."""

    def __init__(self, reply: QNetworkReply, errors: Sequence[QSslError]) -> None:
        if False:
            return 10
        super().__init__()
        self._reply = reply
        self._errors = tuple(errors)
        try:
            self._host_tpl: Optional[urlutils.HostTupleType] = urlutils.host_tuple(reply.url())
        except ValueError:
            self._host_tpl = None

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return '\n'.join((err.errorString() for err in self._errors))

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return utils.get_repr(self, errors=[debug.qenum_key(QSslError, err.error()) for err in self._errors], string=str(self))

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash((self._host_tpl, self._errors))

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, CertificateErrorWrapper):
            return NotImplemented
        return self._errors == other._errors and self._host_tpl == other._host_tpl

    def is_overridable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def defer(self) -> None:
        if False:
            print('Hello World!')
        raise usertypes.UndeferrableError('Never deferrable')

    def accept_certificate(self) -> None:
        if False:
            print('Hello World!')
        super().accept_certificate()
        self._reply.ignoreSslErrors()

    def html(self):
        if False:
            i = 10
            return i + 15
        if len(self._errors) == 1:
            return super().html()
        template = jinja.environment.from_string('\n            <ul>\n            {% for err in errors %}\n                <li>{{err.errorString()}}</li>\n            {% endfor %}\n            </ul>\n        '.strip())
        return template.render(errors=self._errors)