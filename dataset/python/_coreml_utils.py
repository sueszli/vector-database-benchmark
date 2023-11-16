# -*- coding: utf-8 -*-
# Copyright © 2017 Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _


def _mlmodel_short_description(model_type):
    """
    Returns a string to be used in an Core ML model's description metadata.
    """
    from turicreate import __version__

    return "%s created by Turi Create (version %s)" % (
        model_type.capitalize(),
        __version__,
    )


def _get_tc_version_info():
    """
    Return metadata related to the package to be added to the CoreML model
    """
    from turicreate import __version__
    import platform

    return {
        "com.github.apple.turicreate.version": __version__,
        "com.github.apple.os.platform": platform.platform(),
    }


def _get_model_metadata(model_class, metadata=None, version=None):
    """
    Returns user-defined metadata, making sure information all models should
    have is also available, as a dictionary
    """
    info = _get_tc_version_info()
    info["type"] = model_class
    if version is not None:
        info["version"] = str(version)
    if metadata is not None:
        info.update(metadata)
    return info


def _set_model_metadata(mlmodel, model_class, metadata, version=None):
    """
    Sets user-defined metadata, making sure information all models should have
    is also available
    """
    info = _get_model_metadata(model_class, metadata, version)
    mlmodel.user_defined_metadata.update(info)
