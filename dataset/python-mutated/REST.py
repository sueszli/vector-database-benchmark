"""Provides code to access the REST-style KEGG online API.

This module aims to make the KEGG online REST-style API easier to use. See:
https://www.kegg.jp/kegg/rest/keggapi.html

The KEGG REST-style API provides simple access to a range of KEGG databases.
This works using simple URLs (which this module will construct for you),
with any errors indicated via HTTP error levels.

The functionality is somewhat similar to Biopython's Bio.TogoWS and Bio.Entrez
modules.

Currently KEGG does not provide any usage guidelines (unlike the NCBI whose
requirements are reasonably clear). To avoid risking overloading the service,
Biopython will only allow three calls per second.

References:
Kanehisa, M. and Goto, S.; KEGG: Kyoto Encyclopedia of Genes and Genomes.
Nucleic Acids Res. 28, 29-34 (2000).

"""
import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous

@function_with_previous
def _q(op, arg1, arg2=None, arg3=None):
    if False:
        return 10
    delay = 0.333333333
    current = time.time()
    wait = _q.previous + delay - current
    if wait > 0:
        time.sleep(wait)
        _q.previous = current + wait
    else:
        _q.previous = current
    URL = 'https://rest.kegg.jp/%s'
    if arg2 and arg3:
        args = f'{op}/{arg1}/{arg2}/{arg3}'
    elif arg2:
        args = f'{op}/{arg1}/{arg2}'
    else:
        args = f'{op}/{arg1}'
    resp = urlopen(URL % args)
    if 'image' == arg2:
        return resp
    handle = io.TextIOWrapper(resp, encoding='UTF-8')
    handle.url = resp.url
    return handle
_q.previous = 0

def kegg_info(database):
    if False:
        i = 10
        return i + 15
    "KEGG info - Displays the current statistics of a given database.\n\n    db - database or organism (string)\n\n    The argument db can be a KEGG database name (e.g. 'pathway' or its\n    official abbreviation, 'path'), or a KEGG organism code or T number\n    (e.g. 'hsa' or 'T01001' for human).\n\n    A valid list of organism codes and their T numbers can be obtained\n    via kegg_info('organism') or https://rest.kegg.jp/list/organism\n\n    "
    return _q('info', database)

def kegg_list(database, org=None):
    if False:
        return 10
    'KEGG list - Entry list for database, or specified database entries.\n\n    db - database or organism (string)\n    org - optional organism (string), see below.\n\n    For the pathway and module databases the optional organism can be\n    used to restrict the results.\n\n    '
    if database in ('pathway', 'module') and org:
        resp = _q('list', database, org)
    elif isinstance(database, str) and database and org:
        raise ValueError('Invalid database arg for kegg list request.')
    else:
        if isinstance(database, list):
            if len(database) > 100:
                raise ValueError('Maximum number of databases is 100 for kegg list query')
            database = '+'.join(database)
        resp = _q('list', database)
    return resp

def kegg_find(database, query, option=None):
    if False:
        print('Hello World!')
    "KEGG find - Data search.\n\n    Finds entries with matching query keywords or other query data in\n    a given database.\n\n    db - database or organism (string)\n    query - search terms (string)\n    option - search option (string), see below.\n\n    For the compound and drug database, set option to the string 'formula',\n    'exact_mass' or 'mol_weight' to search on that field only. The\n    chemical formula search is a partial match irrespective of the order\n    of atoms given. The exact mass (or molecular weight) is checked by\n    rounding off to the same decimal place as the query data. A range of\n    values may also be specified with the minus(-) sign.\n\n    "
    if database in ['compound', 'drug'] and option in ['formula', 'exact_mass', 'mol_weight']:
        resp = _q('find', database, query, option)
    elif option:
        raise ValueError('Invalid option arg for kegg find request.')
    else:
        if isinstance(query, list):
            query = '+'.join(query)
        resp = _q('find', database, query)
    return resp

def kegg_get(dbentries, option=None):
    if False:
        for i in range(10):
            print('nop')
    'KEGG get - Data retrieval.\n\n    dbentries - Identifiers (single string, or list of strings), see below.\n    option - One of "aaseq", "ntseq", "mol", "kcf", "image", "kgml" (string)\n\n    The input is limited up to 10 entries.\n    The input is limited to one pathway entry with the image or kgml option.\n    The input is limited to one compound/glycan/drug entry with the image option.\n\n    Returns a handle.\n    '
    if isinstance(dbentries, list) and len(dbentries) <= 10:
        dbentries = '+'.join(dbentries)
    elif isinstance(dbentries, list) and len(dbentries) > 10:
        raise ValueError('Maximum number of dbentries is 10 for kegg get query')
    if option in ['aaseq', 'ntseq', 'mol', 'kcf', 'image', 'kgml', 'json']:
        resp = _q('get', dbentries, option)
    elif option:
        raise ValueError('Invalid option arg for kegg get request.')
    else:
        resp = _q('get', dbentries)
    return resp

def kegg_conv(target_db, source_db, option=None):
    if False:
        for i in range(10):
            print('nop')
    'KEGG conv - convert KEGG identifiers to/from outside identifiers.\n\n    Arguments:\n     - target_db - Target database\n     - source_db_or_dbentries - source database or database entries\n     - option - Can be "turtle" or "n-triple" (string).\n\n    '
    if option and option not in ['turtle', 'n-triple']:
        raise ValueError('Invalid option arg for kegg conv request.')
    if isinstance(source_db, list):
        source_db = '+'.join(source_db)
    if target_db in ['ncbi-gi', 'ncbi-geneid', 'uniprot'] or source_db in ['ncbi-gi', 'ncbi-geneid', 'uniprot'] or (target_db in ['drug', 'compound', 'glycan'] and source_db in ['pubchem', 'glycan']) or (target_db in ['pubchem', 'glycan'] and source_db in ['drug', 'compound', 'glycan']):
        if option:
            resp = _q('conv', target_db, source_db, option)
        else:
            resp = _q('conv', target_db, source_db)
        return resp
    else:
        raise ValueError('Bad argument target_db or source_db for kegg conv request.')

def kegg_link(target_db, source_db, option=None):
    if False:
        i = 10
        return i + 15
    'KEGG link - find related entries by using database cross-references.\n\n    target_db - Target database\n    source_db_or_dbentries - source database\n    option - Can be "turtle" or "n-triple" (string).\n    '
    if option and option not in ['turtle', 'n-triple']:
        raise ValueError('Invalid option arg for kegg conv request.')
    if isinstance(source_db, list):
        source_db = '+'.join(source_db)
    if option:
        resp = _q('link', target_db, source_db, option)
    else:
        resp = _q('link', target_db, source_db)
    return resp