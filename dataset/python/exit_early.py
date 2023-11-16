# Copyright 2016 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--silent', action='store_true')
    args = parser.parse_args()
    if not args.silent:
        sys.stdout.write('%d\n' % (os.getpid(), ))
        sys.stdout.flush()
    max_time = time.time() + 2
    while True:
        time.sleep(0.1)
        target = time.time() + 0.1
        while True:
            now = time.time()
            if now >= max_time:
                return
            if now >= target:
                break


if __name__ == '__main__':
    main()
