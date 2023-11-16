"""Tests Spack's ability to substitute a different version into a URL."""
import os
import pytest
import spack.url

@pytest.mark.parametrize('base_url,version,expected', [('http://www.mr511.de/software/libelf-0.8.13.tar.gz', '0.8.13', 'http://www.mr511.de/software/libelf-0.8.13.tar.gz'), ('http://www.prevanders.net/libdwarf-20130729.tar.gz', '8.12', 'http://www.prevanders.net/libdwarf-8.12.tar.gz'), ('https://github.com/hpc/mpileaks/releases/download/v1.0/mpileaks-1.0.tar.gz', '2.1.3', 'https://github.com/hpc/mpileaks/releases/download/v2.1.3/mpileaks-2.1.3.tar.gz'), ('https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.0.tar.bz2', '2.2.0', 'https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.2.0.tar.bz2'), ('https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.0.tar.bz2', '2.2', 'https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.2.tar.bz2'), ('file://{0}/turbolinux702.tar.gz'.format(os.getcwd()), '703', 'file://{0}/turbolinux703.tar.gz'.format(os.getcwd())), ('https://github.com/losalamos/CLAMR/blob/packages/PowerParser_v2.0.7.tgz?raw=true', '2.0.7', 'https://github.com/losalamos/CLAMR/blob/packages/PowerParser_v2.0.7.tgz?raw=true'), ('https://github.com/losalamos/CLAMR/blob/packages/PowerParser_v2.0.7.tgz?raw=true', '4.7', 'https://github.com/losalamos/CLAMR/blob/packages/PowerParser_v4.7.tgz?raw=true'), ('http://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz', '1.2.3', 'http://math.lbl.gov/voro++/download/dir/voro++-1.2.3.tar.gz')])
def test_url_substitution(base_url, version, expected):
    if False:
        i = 10
        return i + 15
    computed = spack.url.substitute_version(base_url, version)
    assert computed == expected