"""Create views from nmap and passive databases."""
import argparse
from functools import reduce
from multiprocessing import Pool, cpu_count
from typing import Generator, List, Optional
from ivre.active.data import merge_host_docs
from ivre.activecli import displayfunction_json
from ivre.db import DB, DBView, db
from ivre.types import Record
from ivre.view import nmap_to_view, passive_to_view, prepare_record, to_view, to_view_parallel

def merge_and_output(records: List[Record]) -> None:
    if False:
        while True:
            i = 10
    result = reduce(lambda r1, r2: merge_host_docs(r1, r2, auto_tags=False, openports_attribute=False), records)
    w_output(prepare_record(result, w_datadb))

def worker_initializer(dburl: Optional[str], no_merge: bool) -> None:
    if False:
        while True:
            i = 10
    global w_datadb, w_outdb, w_output
    w_outdb = db.view if dburl is None else DBView.from_url(dburl)
    if no_merge:
        w_output = w_outdb.store_host
    else:
        w_output = w_outdb.store_or_merge_host
    try:
        w_datadb = w_outdb.globaldb.data
    except AttributeError:
        w_datadb = None
    w_outdb.start_store_hosts()

def worker_destroyer(_: None) -> None:
    if False:
        i = 10
        return i + 15
    w_outdb.stop_store_hosts()

def main() -> None:
    if False:
        print('Hello World!')
    default_processes = max(1, cpu_count())
    parser = argparse.ArgumentParser(description=__doc__, parents=[DB().argparser])
    if db.nmap is None:
        fltnmap = None
    else:
        fltnmap = db.nmap.flt_empty
    if db.passive is None:
        fltpass = None
    else:
        fltpass = db.passive.flt_empty
    _from: List[Generator[Record, None, None]] = []
    parser.add_argument('--view-category', metavar='CATEGORY', help='Choose a different category than the default')
    parser.add_argument('--test', '-t', action='store_true', help='Give results in standard output instead of inserting them in database.')
    parser.add_argument('--verbose', '-v', action='store_true', help='For test output, print out formatted results.')
    parser.add_argument('--no-merge', action='store_true', help='Do **not** merge with existing results for same host and source.')
    parser.add_argument('--to-db', metavar='DB_URL', help='Store data to the provided URL instead of the default DB for view.')
    parser.add_argument('--processes', metavar='COUNT', type=int, help=f'The number of processes to use to build the records. Default on this system is {default_processes}.', default=default_processes)
    subparsers = parser.add_subparsers(dest='view_source', help="Accepted values are 'nmap' and 'passive'. None or 'all' will do both")
    if db.nmap is not None:
        subparsers.add_parser('nmap', parents=[db.nmap.argparser])
    if db.passive is not None:
        subparsers.add_parser('passive', parents=[db.passive.argparser])
    subparsers.add_parser('all')
    args = parser.parse_args()
    view_category = args.view_category
    if not args.view_source:
        args.view_source = 'all'
    if args.view_source == 'all':
        _from = []
        if db.nmap is not None:
            fltnmap = db.nmap.parse_args(args, flt=fltnmap)
            _from.append(nmap_to_view(fltnmap, category=view_category))
        if db.passive is not None:
            fltpass = db.passive.parse_args(args, flt=fltpass)
            _from.append(passive_to_view(fltpass, category=view_category))
    elif args.view_source == 'nmap':
        if db.nmap is None:
            parser.error('Cannot use "nmap" (no Nmap database exists)')
        fltnmap = db.nmap.parse_args(args, fltnmap)
        _from = [nmap_to_view(fltnmap, category=view_category)]
    elif args.view_source == 'passive':
        if db.passive is None:
            parser.error('Cannot use "passive" (no Passive database exists)')
        fltpass = db.passive.parse_args(args, fltpass)
        _from = [passive_to_view(fltpass, category=view_category)]
    if args.test:
        args.processes = 1
    outdb = db.view if args.to_db is None else DBView.from_url(args.to_db)
    if args.processes > 1:
        nprocs = max(args.processes - 1, 1)
        with Pool(nprocs, initializer=worker_initializer, initargs=(args.to_db, args.no_merge)) as pool:
            for _ in pool.imap(merge_and_output, to_view_parallel(_from)):
                pass
            for _ in pool.imap(worker_destroyer, [None] * nprocs):
                pass
    else:
        if args.test:

            def output(host: Record) -> None:
                if False:
                    i = 10
                    return i + 15
                return displayfunction_json([host], outdb)
        elif args.no_merge:
            output = outdb.store_host
        else:
            output = outdb.store_or_merge_host
        try:
            datadb = outdb.globaldb.data
        except AttributeError:
            datadb = None
        outdb.start_store_hosts()
        for record in to_view(_from, datadb):
            output(record)
        outdb.stop_store_hosts()