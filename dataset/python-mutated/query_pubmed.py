"""Query PubMed and print MEDLINE format results."""
import sys
import getopt
from Bio import Entrez

def print_usage():
    if False:
        return 10
    'Print a help message.'
    print("query_pubmed.py [-h] [-c] [-d delay] query\n\nThis script sends a query to PubMed (via the NCBI Entrez webservice*)\nand prints the MEDLINE formatted results to the screen.\n\nArguments:\n    -h           Print out this help message.\n    -c           Count the hits, and don't print them out.\n\n* http://www.ncbi.nlm.nih.gov/Entrez/\n")
if __name__ == '__main__':
    try:
        (optlist, args) = getopt.getopt(sys.argv[1:], 'hcd:')
    except getopt.error as x:
        print(x)
        sys.exit(0)
    if len(args) != 1:
        print_usage()
        sys.exit(0)
    query = args[0]
    show_help = False
    count_only = False
    for (opt, arg) in optlist:
        if opt == '-h':
            show_help = True
        elif opt == '-c':
            count_only = True
        elif opt == '-d':
            sys.stderr.write('The delay parameter is now ignored\n')
    if show_help:
        print_usage()
        sys.exit(0)
    print('Doing a PubMed search for %r...' % query)
    if count_only:
        handle = Entrez.esearch(db='pubmed', term=query)
    else:
        handle = Entrez.esearch(db='pubmed', term=query, usehistory='Y')
    search_results = Entrez.read(handle)
    ids = search_results['IdList']
    count = len(ids)
    print(f'Found {count:d} citations')
    if count_only:
        sys.exit(0)
    webenv = search_results['WebEnv']
    query_key = search_results['QueryKey']
    batch_size = 3
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        fetch_handle = Entrez.efetch(db='pubmed', rettype='medline', retmode='text', retstart=start, retmax=batch_size, webenv=webenv, query_key=query_key)
        data = fetch_handle.read()
        fetch_handle.close()
        sys.stdout.write(data)
        sys.stdout.flush()