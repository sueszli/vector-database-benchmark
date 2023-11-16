import os
import tempfile
from pathlib import Path
import pytest
from pip._vendor.packaging.requirements import Requirement
from pip._internal.exceptions import InstallationError
from pip._internal.req.constructors import install_req_from_line, install_req_from_req_string
from pip._internal.req.req_install import InstallRequirement

class TestInstallRequirementBuildDirectory:

    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_tmp_build_directory(self) -> None:
        if False:
            i = 10
            return i + 15
        requirement = InstallRequirement(None, None)
        tmp_dir = tempfile.mkdtemp('-build', 'pip-')
        tmp_build_dir = requirement.ensure_build_location(tmp_dir, autodelete=False, parallel_builds=False)
        assert os.path.dirname(tmp_build_dir) == os.path.realpath(os.path.dirname(tmp_dir))
        if os.path.realpath(tmp_dir) != os.path.abspath(tmp_dir):
            assert os.path.dirname(tmp_build_dir) != os.path.dirname(tmp_dir)
        else:
            assert os.path.dirname(tmp_build_dir) == os.path.dirname(tmp_dir)
        os.rmdir(tmp_dir)
        assert not os.path.exists(tmp_dir)

    def test_forward_slash_results_in_a_link(self, tmpdir: Path) -> None:
        if False:
            i = 10
            return i + 15
        install_dir = tmpdir / 'foo' / 'bar'
        setup_py_path = install_dir / 'setup.py'
        os.makedirs(str(install_dir))
        with open(setup_py_path, 'w') as f:
            f.write('')
        requirement = install_req_from_line(install_dir.as_posix())
        assert requirement.link is not None

class TestInstallRequirementFrom:

    def test_install_req_from_string_invalid_requirement(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Requirement strings that cannot be parsed by\n        packaging.requirements.Requirement raise an InstallationError.\n        '
        with pytest.raises(InstallationError) as excinfo:
            install_req_from_req_string('http:/this/is/invalid')
        assert str(excinfo.value) == "Invalid requirement: 'http:/this/is/invalid'"

    def test_install_req_from_string_without_comes_from(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test to make sure that install_req_from_string succeeds\n        when called with URL (PEP 508) but without comes_from.\n        '
        wheel_url = 'https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl'
        install_str = 'torch@ ' + wheel_url
        install_req = install_req_from_req_string(install_str)
        assert isinstance(install_req, InstallRequirement)
        assert install_req.link is not None
        assert install_req.link.url == wheel_url
        assert install_req.req is not None
        assert install_req.req.url == wheel_url
        assert install_req.comes_from is None
        assert install_req.is_wheel

    def test_install_req_from_string_with_comes_from_without_link(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test to make sure that install_req_from_string succeeds\n        when called with URL (PEP 508) and comes_from\n        does not have a link.\n        '
        wheel_url = 'https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl'
        install_str = 'torch@ ' + wheel_url
        comes_from = InstallRequirement(Requirement('numpy>=1.15.0'), comes_from=None)
        install_req = install_req_from_req_string(install_str, comes_from=comes_from)
        assert isinstance(install_req, InstallRequirement)
        assert isinstance(install_req.comes_from, InstallRequirement)
        assert install_req.comes_from.link is None
        assert install_req.link is not None
        assert install_req.link.url == wheel_url
        assert install_req.req is not None
        assert install_req.req.url == wheel_url
        assert install_req.is_wheel