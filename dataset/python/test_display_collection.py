# -*- coding: utf-8 -*-
# Copyright (c) 2020 Ansible Project
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

from __future__ import annotations

import pytest

from ansible.cli.galaxy import _display_collection
from ansible.galaxy.dependency_resolution.dataclasses import Requirement


@pytest.fixture
def collection_object():
    def _cobj(fqcn='sandwiches.ham'):
        return Requirement(fqcn, '1.5.0', None, 'galaxy', None)
    return _cobj


def test_display_collection(capsys, collection_object):
    _display_collection(collection_object())
    out, err = capsys.readouterr()

    assert out == 'sandwiches.ham 1.5.0  \n'


def test_display_collections_small_max_widths(capsys, collection_object):
    _display_collection(collection_object(), 1, 1)
    out, err = capsys.readouterr()

    assert out == 'sandwiches.ham 1.5.0  \n'


def test_display_collections_large_max_widths(capsys, collection_object):
    _display_collection(collection_object(), 20, 20)
    out, err = capsys.readouterr()

    assert out == 'sandwiches.ham       1.5.0               \n'


def test_display_collection_small_minimum_widths(capsys, collection_object):
    _display_collection(collection_object('a.b'), min_cwidth=0, min_vwidth=0)
    out, err = capsys.readouterr()

    assert out == 'a.b        1.5.0  \n'
