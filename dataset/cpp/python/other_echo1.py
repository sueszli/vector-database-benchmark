#!/usr/bin/python
from __future__ import annotations

from ansible_collections.testns.testcoll.plugins.module_utils.echo_impl import do_echo


def main():
    do_echo()


if __name__ == '__main__':
    main()
