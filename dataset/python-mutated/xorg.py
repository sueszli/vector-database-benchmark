from typing import Optional
import spack.package_base
import spack.util.url

class XorgPackage(spack.package_base.PackageBase):
    """Mixin that takes care of setting url and mirrors for x.org
    packages."""
    xorg_mirror_path: Optional[str] = None
    base_mirrors = ['https://www.x.org/archive/individual/', 'https://mirrors.ircam.fr/pub/x.org/individual/', 'https://mirror.transip.net/xorg/individual/', 'ftp://ftp.freedesktop.org/pub/xorg/individual/', 'http://xorg.mirrors.pair.com/individual/']

    @property
    def urls(self):
        if False:
            i = 10
            return i + 15
        self._ensure_xorg_mirror_path_is_set_or_raise()
        return [spack.util.url.join(m, self.xorg_mirror_path, resolve_href=True) for m in self.base_mirrors]

    def _ensure_xorg_mirror_path_is_set_or_raise(self):
        if False:
            print('Hello World!')
        if self.xorg_mirror_path is None:
            cls_name = type(self).__name__
            msg = '{0} must define a `xorg_mirror_path` attribute [none defined]'
            raise AttributeError(msg.format(cls_name))