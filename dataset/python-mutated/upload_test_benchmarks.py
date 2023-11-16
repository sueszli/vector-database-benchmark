"""Command to upload benchmark test results to a cloud datastore.

This uploader script is typically run periodically as a cron job.  It locates,
in a specified data directory, files that contain benchmark test results.  The
results are written by the "run_and_gather_logs.py" script using the JSON-format
serialization of the "TestResults" protobuf message (core/util/test_log.proto).

For each file, the uploader reads the "TestResults" data, transforms it into
the schema used in the datastore (see below), and upload it to the datastore.
After processing a file, the uploader moves it to a specified archive directory
for safe-keeping.

The uploader uses file-level exclusive locking (non-blocking flock) which allows
multiple instances of this script to run concurrently if desired, splitting the
task among them, each one processing and archiving different files.

The "TestResults" object contains test metadata and multiple benchmark entries.
The datastore schema splits this information into two Kinds (like tables), one
holding the test metadata in a single "Test" Entity (like rows), and one holding
each related benchmark entry in a separate "Entry" Entity.  Datastore create a
unique ID (retrieval key) for each Entity, and this ID is always returned along
with the data when an Entity is fetched.

* Test:
  - test:   unique name of this test (string)
  - start:  start time of this test run (datetime)
  - info:   JSON-encoded test metadata (string, not indexed)

* Entry:
  - test:   unique name of this test (string)
  - entry:  unique name of this benchmark entry within this test (string)
  - start:  start time of this test run (datetime)
  - timing: average time (usec) per iteration of this test/entry run (float)
  - info:   JSON-encoded entry metadata (string, not indexed)

A few composite indexes are created (upload_test_benchmarks_index.yaml) for fast
retrieval of benchmark data and reduced I/O to the client without adding a lot
of indexing and storage burden:

* Test: (test, start) is indexed to fetch recent start times for a given test.

* Entry: (test, entry, start, timing) is indexed to use projection and only
fetch the recent (start, timing) data for a given test/entry benchmark.

Example retrieval GQL statements:

* Get the recent start times for a given test:
  SELECT start FROM Test WHERE test = <test-name> AND
    start >= <recent-datetime> LIMIT <count>

* Get the recent timings for a given benchmark:
  SELECT start, timing FROM Entry WHERE test = <test-name> AND
    entry = <entry-name> AND start >= <recent-datetime> LIMIT <count>

* Get all test names uniquified (e.g. display a list of available tests):
  SELECT DISTINCT ON (test) test FROM Test

* For a given test (from the list above), get all its entry names.  The list of
  entry names can be extracted from the test "info" metadata for a given test
  name and start time (e.g. pick the latest start time for that test).
  SELECT * FROM Test WHERE test = <test-name> AND start = <latest-datetime>
"""
import argparse
import datetime
import fcntl
import json
import os
import shutil
from google.cloud import datastore

def is_real_file(dirpath, fname):
    if False:
        for i in range(10):
            print('nop')
    fpath = os.path.join(dirpath, fname)
    return os.path.isfile(fpath) and (not os.path.islink(fpath))

def get_mtime(dirpath, fname):
    if False:
        print('Hello World!')
    fpath = os.path.join(dirpath, fname)
    return os.stat(fpath).st_mtime

def list_files_by_mtime(dirpath):
    if False:
        print('Hello World!')
    'Return a list of files in the directory, sorted in increasing "mtime".\n\n  Return a list of files in the given directory, sorted from older to newer file\n  according to their modification times.  Only return actual files, skipping\n  directories, symbolic links, pipes, etc.\n\n  Args:\n    dirpath: directory pathname\n\n  Returns:\n    A list of file names relative to the given directory path.\n  '
    files = [f for f in os.listdir(dirpath) if is_real_file(dirpath, f)]
    return sorted(files, key=lambda f: get_mtime(dirpath, f))

def lock(fd):
    if False:
        while True:
            i = 10
    fcntl.flock(fd, fcntl.LOCK_EX)

def unlock(fd):
    if False:
        for i in range(10):
            print('nop')
    fcntl.flock(fd, fcntl.LOCK_UN)

def trylock(fd):
    if False:
        return 10
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except Exception:
        return False

def upload_benchmark_data(client, data):
    if False:
        while True:
            i = 10
    'Parse benchmark data and use the client to upload it to the datastore.\n\n  Parse the given benchmark data from the serialized JSON-format used to write\n  the test results file.  Create the different datastore Entities from that data\n  and upload them to the datastore in a batch using the client connection.\n\n  Args:\n    client: datastore client connection\n    data: JSON-encoded benchmark data\n  '
    test_result = json.loads(data)
    test_name = str(test_result['name'])
    start_time = datetime.datetime.utcfromtimestamp(float(test_result['startTime']))
    batch = []
    t_key = client.key('Test')
    t_val = datastore.Entity(t_key, exclude_from_indexes=['info'])
    t_val.update({'test': test_name, 'start': start_time, 'info': str(data)})
    batch.append(t_val)
    for ent in test_result['entries'].get('entry', []):
        ent_name = str(ent['name'])
        e_key = client.key('Entry')
        e_val = datastore.Entity(e_key, exclude_from_indexes=['info'])
        e_val.update({'test': test_name, 'start': start_time, 'entry': ent_name, 'timing': ent['wallTime'], 'info': str(json.dumps(ent))})
        batch.append(e_val)
    client.put_multi(batch)

def upload_benchmark_files(opts):
    if False:
        print('Hello World!')
    'Find benchmark files, process them, and upload their data to the datastore.\n\n  Locate benchmark files in the data directory, process them, and upload their\n  data to the datastore.  After processing each file, move it to the archive\n  directory for safe-keeping.  Each file is locked for processing, which allows\n  multiple uploader instances to run concurrently if needed, each one handling\n  different benchmark files, skipping those already locked by another.\n\n  Args:\n    opts: command line options object\n\n  Note: To use locking, the file is first opened, then its descriptor is used to\n  lock and read it.  The lock is released when the file is closed.  Do not open\n  that same file a 2nd time while the lock is already held, because when that\n  2nd file descriptor is closed, the lock will be released prematurely.\n  '
    client = datastore.Client()
    for fname in list_files_by_mtime(opts.datadir):
        fpath = os.path.join(opts.datadir, fname)
        try:
            with open(fpath, 'r') as fd:
                if trylock(fd):
                    upload_benchmark_data(client, fd.read())
                    shutil.move(fpath, os.path.join(opts.archivedir, fname))
        except Exception as e:
            print("Cannot process '%s', skipping. Error: %s" % (fpath, e))

def parse_cmd_line():
    if False:
        i = 10
        return i + 15
    'Parse command line options.\n\n  Returns:\n    The parsed arguments object.\n  '
    desc = 'Upload benchmark results to datastore.'
    opts = [('-a', '--archivedir', str, None, True, 'Directory where benchmark files are archived.'), ('-d', '--datadir', str, None, True, 'Directory of benchmark files to upload.')]
    parser = argparse.ArgumentParser(description=desc)
    for opt in opts:
        parser.add_argument(opt[0], opt[1], type=opt[2], default=opt[3], required=opt[4], help=opt[5])
    return parser.parse_args()

def main():
    if False:
        i = 10
        return i + 15
    options = parse_cmd_line()
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        raise ValueError('GOOGLE_APPLICATION_CREDENTIALS env. var. is not set.')
    upload_benchmark_files(options)
if __name__ == '__main__':
    main()