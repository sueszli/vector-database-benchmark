from typing import Optional
import spack.package_base
import spack.util.url

class SourcewarePackage(spack.package_base.PackageBase):
    """Mixin that takes care of setting url and mirrors for Sourceware.org
    packages."""
    sourceware_mirror_path: Optional[str] = None
    base_mirrors = ['https://sourceware.org/pub/', 'https://mirrors.kernel.org/sourceware/', 'https://ftp.gwdg.de/pub/linux/sources.redhat.com/']

    @property
    def urls(self):
        if False:
            while True:
                i = 10
        self._ensure_sourceware_mirror_path_is_set_or_raise()
        return [spack.util.url.join(m, self.sourceware_mirror_path, resolve_href=True) for m in self.base_mirrors]

    def _ensure_sourceware_mirror_path_is_set_or_raise(self):
        if False:
            i = 10
            return i + 15
        if self.sourceware_mirror_path is None:
            cls_name = type(self).__name__
            msg = '{0} must define a `sourceware_mirror_path` attribute [none defined]'
            raise AttributeError(msg.format(cls_name))