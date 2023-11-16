"""
Functional tests for PyGObject.
"""
import pytest
from PyInstaller.utils.tests import importorskip, parametrize
gi_repositories = [('Gst', '1.0'), ('GLib', '2.0'), ('GModule', '2.0'), ('GObject', '2.0'), ('GdkPixbuf', '2.0'), ('Gio', '2.0'), ('Clutter', '1.0'), ('GtkClutter', '1.0'), ('Champlain', '0.12'), ('GtkChamplain', '0.12')]
gi_repository_names = [x[0] for x in gi_repositories]
gi_repositories_skipped_if_unimportable = [pytest.param(gi_repository_name, gi_repository_version, marks=importorskip('gi.repository.' + gi_repository_name)) for (gi_repository_name, gi_repository_version) in gi_repositories]

@importorskip('gi.repository')
@parametrize(('repository_name', 'version'), gi_repositories_skipped_if_unimportable, ids=gi_repository_names)
def test_gi_repository(pyi_builder, repository_name, version):
    if False:
        while True:
            i = 10
    "\n    Test the importability of the `gi.repository` subpackage with the passed name installed with PyGObject. For example,\n    `GLib`, corresponds to the `gi.repository.GLib` subpackage. Version '1.0' are for PyGObject >=1.0,\n    '2.0' for PyGObject >= 2.0. Some other libraries have strange version (e.g., Champlain).\n    "
    pyi_builder.test_source("\n        import gi\n        gi.require_version('{repository_name}', '{version}')\n        from gi.repository import {repository_name}\n        print({repository_name})\n        ".format(repository_name=repository_name, version=version))