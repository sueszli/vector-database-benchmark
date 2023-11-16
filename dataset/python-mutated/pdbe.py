"""
 PDBe (Protein Data Bank in Europe)
"""
from json import loads
from flask_babel import gettext
about = {'website': 'https://www.ebi.ac.uk/pdbe', 'wikidata_id': 'Q55823905', 'official_api_documentation': 'https://www.ebi.ac.uk/pdbe/api/doc/search.html', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['science']
hide_obsolete = False
pdb_unpublished_codes = ['HPUB', 'HOLD', 'PROC', 'WAIT', 'AUTH', 'AUCO', 'REPL', 'POLC', 'REFI', 'TRSF', 'WDRN']
pdbe_solr_url = 'https://www.ebi.ac.uk/pdbe/search/pdb/select?'
pdbe_entry_url = 'https://www.ebi.ac.uk/pdbe/entry/pdb/{pdb_id}'
pdbe_preview_url = 'https://www.ebi.ac.uk/pdbe/static/entry/{pdb_id}_deposited_chain_front_image-200x200.png'

def request(query, params):
    if False:
        print('Hello World!')
    params['url'] = pdbe_solr_url
    params['method'] = 'POST'
    params['data'] = {'q': query, 'wt': 'json'}
    return params

def construct_body(result):
    if False:
        return 10
    title = result['title']
    content = '{title} - {authors} {journal} ({volume}) {page} ({year})'
    try:
        if result['journal']:
            content = content.format(title=result['citation_title'], authors=result['entry_author_list'][0], journal=result['journal'], volume=result['journal_volume'], page=result['journal_page'], year=result['citation_year'])
        else:
            content = content.format(title=result['citation_title'], authors=result['entry_author_list'][0], journal='', volume='', page='', year=result['release_year'])
        img_src = pdbe_preview_url.format(pdb_id=result['pdb_id'])
    except KeyError:
        content = None
        img_src = None
    try:
        img_src = pdbe_preview_url.format(pdb_id=result['pdb_id'])
    except KeyError:
        img_src = None
    return [title, content, img_src]

def response(resp):
    if False:
        for i in range(10):
            print('nop')
    results = []
    json = loads(resp.text)['response']['docs']
    for result in json:
        if result['status'] in pdb_unpublished_codes:
            continue
        if hide_obsolete:
            continue
        if result['status'] == 'OBS':
            title = gettext('{title} (OBSOLETE)').format(title=result['title'])
            try:
                superseded_url = pdbe_entry_url.format(pdb_id=result['superseded_by'])
            except:
                continue
            msg_superseded = gettext('This entry has been superseded by')
            content = '{msg_superseded}: {url} ({pdb_id})'.format(msg_superseded=msg_superseded, url=superseded_url, pdb_id=result['superseded_by'])
            img_src = None
        else:
            (title, content, img_src) = construct_body(result)
        results.append({'url': pdbe_entry_url.format(pdb_id=result['pdb_id']), 'title': title, 'content': content, 'img_src': img_src})
    return results