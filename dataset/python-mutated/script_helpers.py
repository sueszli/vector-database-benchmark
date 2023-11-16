"""
Helper functions for Rerun scripts.

These helper functions can be used to wire up common Rerun features to your script CLi arguments.

Example
-------
```python
import argparse
import rerun as rr

parser = argparse.ArgumentParser()
rr.script_add_args(parser)
args = parser.parse_args()
rr.script_setup(args, "rerun_example_application")
# … Run your logging code here …
rr.script_teardown(args)
```

"""
from __future__ import annotations
from argparse import ArgumentParser, Namespace
import rerun as rr
from rerun.recording_stream import RecordingStream

def script_add_args(parser: ArgumentParser) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Add common Rerun script arguments to `parser`.\n\n    Parameters\n    ----------\n    parser : ArgumentParser\n        The parser to add arguments to.\n\n    '
    parser.add_argument('--headless', action='store_true', help="Don't show GUI")
    parser.add_argument('--connect', dest='connect', action='store_true', help='Connect to an external viewer')
    parser.add_argument('--serve', dest='serve', action='store_true', help='Serve a web viewer (WARNING: experimental feature)')
    parser.add_argument('--addr', type=str, default=None, help='Connect to this ip:port')
    parser.add_argument('--save', type=str, default=None, help='Save data to a .rrd file at this path')

def script_setup(args: Namespace, application_id: str) -> RecordingStream:
    if False:
        for i in range(10):
            print('nop')
    '\n    Run common Rerun script setup actions. Connect to the viewer if necessary.\n\n    Parameters\n    ----------\n    args : Namespace\n        The parsed arguments from `parser.parse_args()`.\n    application_id : str\n        The application ID to use for the viewer.\n\n    '
    rr.init(application_id=application_id, default_enabled=True, strict=True)
    rec: RecordingStream = rr.get_global_data_recording()
    if args.serve:
        rec.serve()
    elif args.connect:
        rec.connect(args.addr)
    elif args.save is not None:
        rec.save(args.save)
    elif not args.headless:
        rec.spawn()
    return rec

def script_teardown(args: Namespace) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Run common post-actions. Sleep if serving the web viewer.\n\n    Parameters\n    ----------\n    args : Namespace\n        The parsed arguments from `parser.parse_args()`.\n\n    '
    if args.serve:
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('Ctrl-C received. Exiting.')