"""
Try to regenerate the permacache items devoted to listings after a
storage failure in Cassandra
"""
'\ncat > mr_permacache <<HERE\n#!/bin/sh\ncd ~/reddit/r2\npaster run staging.ini ./mr_permacache.py -c "\\$1"\nHERE\nchmod u+x mr_permacache\n\nLINKDBHOST=prec01\nCOMMENTDBHOST=db02s1\nVOTEDBHOST=db03s1\nSAVEHIDEDBHOST=db01s1\n\n## links\ntime psql -F"\t" -A -t -d newreddit -U ri -h $LINKDBHOST      -c "\\copy (select t.thing_id, \'thing\', \'link\',\n                        t.ups, t.downs, t.deleted, t.spam, extract(epoch from t.date)\n                   from reddit_thing_link t) to \'reddit_thing_link.dump\'"\ntime psql -F"\t" -A -t -d newreddit -U ri -h $LINKDBHOST      -c "\\copy (select d.thing_id, \'data\', \'link\',\n                        d.key, d.value\n                   from reddit_data_link d\n                  where d.key = \'author_id\' or d.key = \'sr_id\') to \'reddit_data_link.dump\'"\npv reddit_data_link.dump reddit_thing_link.dump | sort -T. -S200m | ./mr_permacache "join_links()" > links.joined\npv links.joined | ./mr_permacache "link_listings()" | sort -T. -S200m > links.listings\n\n## comments\npsql -F"\t" -A -t -d newreddit -U ri -h $COMMENTDBHOST      -c "\\copy (select t.thing_id, \'thing\', \'comment\',\n                        t.ups, t.downs, t.deleted, t.spam, extract(epoch from t.date)\n                   from reddit_thing_comment t) to \'reddit_thing_comment.dump\'"\npsql -F"\t" -A -t -d newreddit -U ri -h $COMMENTDBHOST      -c "\\copy (select d.thing_id, \'data\', \'comment\',\n                        d.key, d.value\n                   from reddit_data_comment d\n                  where d.key = \'author_id\') to \'reddit_data_comment.dump\'"\ncat reddit_data_comment.dump reddit_thing_comment.dump | sort -T. -S200m | ./mr_permacache "join_comments()" > comments.joined\ncat links.joined | ./mr_permacache "comment_listings()" | sort -T. -S200m > comments.listings\n\n## linkvotes\npsql -F"\t" -A -t -d newreddit -U ri -h $VOTEDBHOST      -c "\\copy (select r.rel_id, \'vote_account_link\',\n                        r.thing1_id, r.thing2_id, r.name, extract(epoch from r.date)\n                   from reddit_rel_vote_account_link r) to \'reddit_linkvote.dump\'"\npv reddit_linkvote.dump | ./mr_permacache "linkvote_listings()" | sort -T. -S200m > linkvotes.listings\n\n#savehide\npsql -F"\t" -A -t -d newreddit -U ri -h $SAVEHIDEDBHOST      -c "\\copy (select r.rel_id, \'savehide\',\n                        r.thing1_id, r.thing2_id, r.name, extract(epoch from r.date)\n                   from reddit_rel_savehide r) to \'reddit_savehide.dump\'"\npv reddit_savehide.dump | ./mr_permacache "savehide_listings()" | sort -T. -S200m > savehide.listings\n\n## load them up\n# the individual .listings files are sorted so even if it\'s not sorted\n# overall we don\'t need to re-sort them\nmkdir listings\npv *.listings | ./mr_permacache "top1k_writefiles(\'listings\')"\n./mr_permacache "write_permacache_from_dir(\'$PWD/listings\')"\n\n'
import os, os.path, errno
import sys
import itertools
from hashlib import md5
from r2.lib import mr_tools
from r2.lib.mr_tools import dataspec_m_thing, dataspec_m_rel, join_things
from dateutil.parser import parse as parse_timestamp
from r2.models import *
from r2.lib.db.sorts import epoch_seconds, score, controversy, _hot
from r2.lib.utils import fetch_things2, in_chunks, progress, UniqueIterator, tup
from r2.lib import comment_tree
from r2.lib.db import queries
from r2.lib.jsontemplates import make_fullname

def join_links():
    if False:
        return 10
    join_things(('author_id', 'sr_id'))

def link_listings():
    if False:
        i = 10
        return i + 15

    @dataspec_m_thing(('author_id', int), ('sr_id', int))
    def process(link):
        if False:
            while True:
                i = 10
        assert link.thing_type == 'link'
        author_id = link.author_id
        timestamp = link.timestamp
        fname = make_fullname(Link, link.thing_id)
        yield ('user-submitted-%d' % author_id, timestamp, fname)
        if not link.spam:
            sr_id = link.sr_id
            (ups, downs) = (link.ups, link.downs)
            yield ('sr-hot-all-%d' % sr_id, _hot(ups, downs, timestamp), timestamp, fname)
            yield ('sr-new-all-%d' % sr_id, timestamp, fname)
            yield ('sr-top-all-%d' % sr_id, score(ups, downs), timestamp, fname)
            yield ('sr-controversial-all-%d' % sr_id, controversy(ups, downs), timestamp, fname)
            for time in ('1 year', '1 month', '1 week', '1 day', '1 hour'):
                if timestamp > epoch_seconds(timeago(time)):
                    tkey = time.split(' ')[1]
                    yield ('sr-top-%s-%d' % (tkey, sr_id), score(ups, downs), timestamp, fname)
                    yield ('sr-controversial-%s-%d' % (tkey, sr_id), controversy(ups, downs), timestamp, fname)
    mr_tools.mr_map(process)

def join_comments():
    if False:
        i = 10
        return i + 15
    join_things(('author_id',))

def comment_listings():
    if False:
        for i in range(10):
            print('nop')

    @dataspec_m_thing(('author_id', int))
    def process(comment):
        if False:
            for i in range(10):
                print('nop')
        assert comment.thing_type == 'comment'
        yield ('user-commented-%d' % comment.author_id, comment.timestamp, make_fullname(Comment, comment.thing_id))
    mr_tools.mr_map(process)

def rel_listings(names, thing2_cls=Link):
    if False:
        while True:
            i = 10

    @dataspec_m_rel()
    def process(rel):
        if False:
            i = 10
            return i + 15
        if rel.name in names:
            yield ('%s-%s' % (names[rel.name], rel.thing1_id), rel.timestamp, make_fullname(thing2_cls, rel.thing2_id))
    mr_tools.mr_map(process)

def linkvote_listings():
    if False:
        print('Hello World!')
    rel_listings({'1': 'liked', '-1': 'disliked'})

def savehide_listings():
    if False:
        while True:
            i = 10
    rel_listings({'save': 'saved', 'hide': 'hidden'})

def insert_to_query(q, items):
    if False:
        return 10
    q._insert_tuples(items)

def store_keys(key, maxes):
    if False:
        return 10
    userrel_fns = dict(liked=queries.get_liked, disliked=queries.get_disliked, saved=queries.get_saved, hidden=queries.get_hidden)
    if key.startswith('user-'):
        (acc_str, keytype, account_id) = key.split('-')
        account_id = int(account_id)
        fn = queries.get_submitted if keytype == 'submitted' else queries.get_comments
        q = fn(Account._byID(account_id), 'new', 'all')
        insert_to_query(q, [(fname, float(timestamp)) for (timestamp, fname) in maxes])
    elif key.startswith('sr-'):
        (sr_str, sort, time, sr_id) = key.split('-')
        sr_id = int(sr_id)
        if sort == 'controversy':
            sort = 'controversial'
        q = queries.get_links(Subreddit._byID(sr_id), sort, time)
        insert_to_query(q, [tuple([item[-1]] + map(float, item[:-1])) for item in maxes])
    elif key.split('-')[0] in userrel_fns:
        (key_type, account_id) = key.split('-')
        account_id = int(account_id)
        fn = userrel_fns[key_type]
        q = fn(Account._byID(account_id))
        insert_to_query(q, [tuple([item[-1]] + map(float, item[:-1])) for item in maxes])

def top1k_writefiles(dirname):
    if False:
        for i in range(10):
            print('nop')
    'Divide up the top 1k of each key into its own file to make\n       restarting after a failure much easier. Pairs with\n       write_permacache_from_dir'

    def hashdir(name, levels=[3]):
        if False:
            print('Hello World!')
        h = md5(name).hexdigest()
        last = 0
        dirs = []
        for l in levels:
            dirs.append(h[last:last + l])
            last += l
        return os.path.join(*dirs)

    def post(key, maxes):
        if False:
            return 10
        hd = os.path.join(dirname, hashdir(key))
        try:
            os.makedirs(hd)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        filename = os.path.join(hd, key)
        with open(filename, 'w') as f:
            for item in maxes:
                f.write('%s\t' % key)
                f.write('\t'.join(item))
                f.write('\n')
    mr_tools.mr_reduce_max_per_key(lambda x: map(float, x[:-1]), num=1000, post=post)

def top1k_writepermacache(fd=sys.stdin):
    if False:
        i = 10
        return i + 15
    mr_tools.mr_reduce_max_per_key(lambda x: map(float, x[:-1]), num=1000, post=store_keys, fd=fd)

def write_permacache_from_dir(dirname):
    if False:
        print('Hello World!')
    allfiles = []
    for (root, dirs, files) in os.walk(dirname):
        for f in files:
            allfiles.append(os.path.join(root, f))
    for fname in progress(allfiles, persec=True):
        try:
            write_permacache_from_file(fname)
            os.unlink(fname)
        except:
            mr_tools.status('failed on %r' % fname)
            raise
    mr_tools.status('Removing empty directories')
    for (root, dirs, files) in os.walk(dirname, topdown=False):
        for d in dirs:
            dname = os.path.join(root, d)
            try:
                os.rmdir(dname)
            except OSError as e:
                if e.errno == errno.ENOTEMPTY:
                    mr_tools.status('%s not empty' % (dname,))
                else:
                    raise

def write_permacache_from_file(fname):
    if False:
        return 10
    with open(fname) as fd:
        top1k_writepermacache(fd=fd)