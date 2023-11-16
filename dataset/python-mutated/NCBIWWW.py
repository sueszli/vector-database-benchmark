"""Code to invoke the NCBI BLAST server over the internet.

This module provides code to work with the WWW version of BLAST
provided by the NCBI. https://blast.ncbi.nlm.nih.gov/

Variables:

    - email        Set the Blast email parameter (default is None).
    - tool         Set the Blast tool parameter (default is ``biopython``).

"""
import warnings
from io import StringIO
import time
from urllib.parse import urlencode
from urllib.request import build_opener, install_opener
from urllib.request import urlopen
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler
from urllib.request import Request
from Bio import BiopythonWarning
from Bio._utils import function_with_previous
email = None
tool = 'biopython'
NCBI_BLAST_URL = 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'

@function_with_previous
def qblast(program, database, sequence, url_base=NCBI_BLAST_URL, auto_format=None, composition_based_statistics=None, db_genetic_code=None, endpoints=None, entrez_query='(none)', expect=10.0, filter=None, gapcosts=None, genetic_code=None, hitlist_size=50, i_thresh=None, layout=None, lcase_mask=None, matrix_name=None, nucl_penalty=None, nucl_reward=None, other_advanced=None, perc_ident=None, phi_pattern=None, query_file=None, query_believe_defline=None, query_from=None, query_to=None, searchsp_eff=None, service=None, threshold=None, ungapped_alignment=None, word_size=None, short_query=None, alignments=500, alignment_view=None, descriptions=500, entrez_links_new_window=None, expect_low=None, expect_high=None, format_entrez_query=None, format_object=None, format_type='XML', ncbi_gi=None, results_file=None, show_overview=None, megablast=None, template_type=None, template_length=None, username='blast', password=None):
    if False:
        while True:
            i = 10
    'BLAST search using NCBI\'s QBLAST server or a cloud service provider.\n\n    Supports all parameters of the old qblast API for Put and Get.\n\n    Please note that NCBI uses the new Common URL API for BLAST searches\n    on the internet (http://ncbi.github.io/blast-cloud/dev/api.html). Thus,\n    some of the parameters used by this function are not (or are no longer)\n    officially supported by NCBI. Although they are still functioning, this\n    may change in the future.\n\n    The Common URL API (http://ncbi.github.io/blast-cloud/dev/api.html) allows\n    doing BLAST searches on cloud servers. To use this feature, please set\n    ``url_base=\'http://host.my.cloud.service.provider.com/cgi-bin/blast.cgi\'``\n    and ``format_object=\'Alignment\'``. For more details, please see\n    https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=CloudBlast\n\n    Some useful parameters:\n\n     - program        blastn, blastp, blastx, tblastn, or tblastx (lower case)\n     - database       Which database to search against (e.g. "nr").\n     - sequence       The sequence to search.\n     - ncbi_gi        TRUE/FALSE whether to give \'gi\' identifier.\n     - descriptions   Number of descriptions to show.  Def 500.\n     - alignments     Number of alignments to show.  Def 500.\n     - expect         An expect value cutoff.  Def 10.0.\n     - matrix_name    Specify an alt. matrix (PAM30, PAM70, BLOSUM80, BLOSUM45).\n     - filter         "none" turns off filtering.  Default no filtering\n     - format_type    "HTML", "Text", "ASN.1", or "XML".  Def. "XML".\n     - entrez_query   Entrez query to limit Blast search\n     - hitlist_size   Number of hits to return. Default 50\n     - megablast      TRUE/FALSE whether to use MEga BLAST algorithm (blastn only)\n     - short_query    TRUE/FALSE whether to adjust the search parameters for a\n                      short query sequence. Note that this will override\n                      manually set parameters like word size and e value. Turns\n                      off when sequence length is > 30 residues. Default: None.\n     - service        plain, psi, phi, rpsblast, megablast (lower case)\n\n    This function does no checking of the validity of the parameters\n    and passes the values to the server as is.  More help is available at:\n    https://ncbi.github.io/blast-cloud/dev/api.html\n\n    '
    programs = ['blastn', 'blastp', 'blastx', 'tblastn', 'tblastx']
    if program not in programs:
        raise ValueError(f"Program specified is {program}. Expected one of {', '.join(programs)}")
    if short_query and program == 'blastn':
        short_query = None
        if len(sequence) < 31:
            expect = 1000
            word_size = 7
            nucl_reward = 1
            filter = None
            lcase_mask = None
            warnings.warn('"SHORT_QUERY_ADJUST" is incorrectly implemented (by NCBI) for blastn. We bypass the problem by manually adjusting the search parameters. Thus, results may slightly differ from web page searches.', BiopythonWarning)
    parameters = {'AUTO_FORMAT': auto_format, 'COMPOSITION_BASED_STATISTICS': composition_based_statistics, 'DATABASE': database, 'DB_GENETIC_CODE': db_genetic_code, 'ENDPOINTS': endpoints, 'ENTREZ_QUERY': entrez_query, 'EXPECT': expect, 'FILTER': filter, 'GAPCOSTS': gapcosts, 'GENETIC_CODE': genetic_code, 'HITLIST_SIZE': hitlist_size, 'I_THRESH': i_thresh, 'LAYOUT': layout, 'LCASE_MASK': lcase_mask, 'MEGABLAST': megablast, 'MATRIX_NAME': matrix_name, 'NUCL_PENALTY': nucl_penalty, 'NUCL_REWARD': nucl_reward, 'OTHER_ADVANCED': other_advanced, 'PERC_IDENT': perc_ident, 'PHI_PATTERN': phi_pattern, 'PROGRAM': program, 'QUERY': sequence, 'QUERY_FILE': query_file, 'QUERY_BELIEVE_DEFLINE': query_believe_defline, 'QUERY_FROM': query_from, 'QUERY_TO': query_to, 'SEARCHSP_EFF': searchsp_eff, 'SERVICE': service, 'SHORT_QUERY_ADJUST': short_query, 'TEMPLATE_TYPE': template_type, 'TEMPLATE_LENGTH': template_length, 'THRESHOLD': threshold, 'UNGAPPED_ALIGNMENT': ungapped_alignment, 'WORD_SIZE': word_size, 'CMD': 'Put'}
    if password is not None:
        password_mgr = HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url_base, username, password)
        handler = HTTPBasicAuthHandler(password_mgr)
        opener = build_opener(handler)
        install_opener(opener)
    if url_base == NCBI_BLAST_URL:
        parameters.update({'email': email, 'tool': tool})
    parameters = {key: value for (key, value) in parameters.items() if value is not None}
    message = urlencode(parameters).encode()
    request = Request(url_base, message, {'User-Agent': 'BiopythonClient'})
    handle = urlopen(request)
    (rid, rtoe) = _parse_qblast_ref_page(handle)
    parameters = {'ALIGNMENTS': alignments, 'ALIGNMENT_VIEW': alignment_view, 'DESCRIPTIONS': descriptions, 'ENTREZ_LINKS_NEW_WINDOW': entrez_links_new_window, 'EXPECT_LOW': expect_low, 'EXPECT_HIGH': expect_high, 'FORMAT_ENTREZ_QUERY': format_entrez_query, 'FORMAT_OBJECT': format_object, 'FORMAT_TYPE': format_type, 'NCBI_GI': ncbi_gi, 'RID': rid, 'RESULTS_FILE': results_file, 'SERVICE': service, 'SHOW_OVERVIEW': show_overview, 'CMD': 'Get'}
    parameters = {key: value for (key, value) in parameters.items() if value is not None}
    message = urlencode(parameters).encode()
    delay = 20
    while True:
        current = time.time()
        wait = qblast.previous + delay - current
        if wait > 0:
            time.sleep(wait)
            qblast.previous = current + wait
        else:
            qblast.previous = current
        if delay < 60 and url_base == NCBI_BLAST_URL:
            delay = 60
        request = Request(url_base, message, {'User-Agent': 'BiopythonClient'})
        handle = urlopen(request)
        results = handle.read().decode()
        if results == '\n\n':
            continue
        if 'Status=' not in results:
            break
        i = results.index('Status=')
        j = results.index('\n', i)
        status = results[i + len('Status='):j].strip()
        if status.upper() == 'READY':
            break
    return StringIO(results)
qblast.previous = 0

def _parse_qblast_ref_page(handle):
    if False:
        return 10
    "Extract a tuple of RID, RTOE from the 'please wait' page (PRIVATE).\n\n    The NCBI FAQ pages use TOE for 'Time of Execution', so RTOE is probably\n    'Request Time of Execution' and RID would be 'Request Identifier'.\n    "
    s = handle.read().decode()
    i = s.find('RID =')
    if i == -1:
        rid = None
    else:
        j = s.find('\n', i)
        rid = s[i + len('RID ='):j].strip()
    i = s.find('RTOE =')
    if i == -1:
        rtoe = None
    else:
        j = s.find('\n', i)
        rtoe = s[i + len('RTOE ='):j].strip()
    if not rid and (not rtoe):
        i = s.find('<div class="error msInf">')
        if i != -1:
            msg = s[i + len('<div class="error msInf">'):].strip()
            msg = msg.split('</div>', 1)[0].split('\n', 1)[0].strip()
            if msg:
                raise ValueError(f'Error message from NCBI: {msg}')
        i = s.find('<p class="error">')
        if i != -1:
            msg = s[i + len('<p class="error">'):].strip()
            msg = msg.split('</p>', 1)[0].split('\n', 1)[0].strip()
            if msg:
                raise ValueError(f'Error message from NCBI: {msg}')
        i = s.find('Message ID#')
        if i != -1:
            msg = s[i:].split('<', 1)[0].split('\n', 1)[0].strip()
            raise ValueError(f'Error message from NCBI: {msg}')
        raise ValueError("No RID and no RTOE found in the 'please wait' page, there was probably an error in your request but we could not extract a helpful error message.")
    elif not rid:
        raise ValueError(f"No RID found in the 'please wait' page. (although RTOE = {rtoe!r})")
    elif not rtoe:
        raise ValueError(f"No RTOE found in the 'please wait' page. (although RID = {rid!r})")
    try:
        return (rid, int(rtoe))
    except ValueError:
        raise ValueError(f"A non-integer RTOE found in the 'please wait' page, {rtoe!r}") from None