"""
Common functions for working with deb packages
"""

def combine_comments(comments):
    if False:
        while True:
            i = 10
    '\n    Given a list of comments, or a comment submitted as a string, return a\n    single line of text containing all of the comments.\n    '
    if isinstance(comments, list):
        comments = [c if isinstance(c, str) else str(c) for c in comments]
    elif not isinstance(comments, str):
        comments = [str(comments)]
    else:
        comments = [comments]
    return ' '.join(comments).strip()

def strip_uri(repo):
    if False:
        i = 10
        return i + 15
    '\n    Remove the trailing slash from the URI in a repo definition\n    '
    splits = repo.split()
    for (idx, val) in enumerate(splits):
        if any((val.startswith(x) for x in ('http://', 'https://', 'ftp://'))):
            splits[idx] = val.rstrip('/')
    return ' '.join(splits)