import os
import sys
from releaseutils import JIRA, JIRAError, get_jira_name, Github, get_github_name, contributors_file_name, is_valid_author, capitalize_author, yesOrNoPrompt
JIRA_API_BASE = os.environ.get('JIRA_API_BASE', 'https://issues.apache.org/jira')
JIRA_USERNAME = os.environ.get('JIRA_USERNAME', None)
JIRA_PASSWORD = os.environ.get('JIRA_PASSWORD', None)
GITHUB_OAUTH_KEY = os.environ.get('GITHUB_OAUTH_KEY', os.environ.get('GITHUB_API_TOKEN', None))
if not JIRA_USERNAME or not JIRA_PASSWORD:
    sys.exit('Both JIRA_USERNAME and JIRA_PASSWORD must be set')
if not GITHUB_OAUTH_KEY:
    sys.exit('GITHUB_OAUTH_KEY must be set')
if not os.path.isfile(contributors_file_name):
    print('Contributors file %s does not exist!' % contributors_file_name)
    print('Have you run ./generate-contributors.py yet?')
    sys.exit(1)
contributors_file = open(contributors_file_name, 'r')
warnings = []
INTERACTIVE_MODE = True
if len(sys.argv) > 1:
    options = set(sys.argv[1:])
    if '--non-interactive' in options:
        INTERACTIVE_MODE = False
if INTERACTIVE_MODE:
    print('Running in interactive mode. To disable this, provide the --non-interactive flag.')
jira_options = {'server': JIRA_API_BASE}
jira_client = JIRA(options=jira_options, basic_auth=(JIRA_USERNAME, JIRA_PASSWORD))
github_client = Github(GITHUB_OAUTH_KEY)
known_translations = {}
known_translations_file_name = 'known_translations'
known_translations_file = open(known_translations_file_name, 'r')
for line in known_translations_file:
    if line.startswith('#'):
        continue
    [old_name, new_name] = line.strip('\n').split(' - ')
    known_translations[old_name] = new_name
known_translations_file.close()
known_translations_file = open(known_translations_file_name, 'a')
NOT_FOUND = 'Not found'

def generate_candidates(author, issues):
    if False:
        print('Hello World!')
    candidates = []
    github_name = get_github_name(author, github_client)
    if github_name:
        candidates.append((github_name, 'Full name of GitHub user %s' % author))
    else:
        candidates.append((NOT_FOUND, 'No full name found for GitHub user %s' % author))
    jira_name = get_jira_name(author, jira_client)
    if jira_name:
        candidates.append((jira_name, 'Full name of JIRA user %s' % author))
    else:
        candidates.append((NOT_FOUND, 'No full name found for JIRA user %s' % author))
    for issue in issues:
        try:
            jira_issue = jira_client.issue(issue)
        except JIRAError as e:
            if e.status_code == 404:
                warnings.append('Issue %s not found!' % issue)
                continue
            raise e
        jira_assignee = jira_issue.fields.assignee
        if jira_assignee:
            user_name = jira_assignee.name
            display_name = jira_assignee.displayName
            if display_name:
                candidates.append((display_name, 'Full name of %s assignee %s' % (issue, user_name)))
            else:
                candidates.append((NOT_FOUND, 'No full name found for %s assignee %s' % (issue, user_name)))
        else:
            candidates.append((NOT_FOUND, 'No assignee found for %s' % issue))
    for (i, (candidate, source)) in enumerate(candidates):
        candidate = candidate.strip()
        candidates[i] = (candidate, source)
    return candidates
print('\n========================== Translating contributor list ==========================')
lines = contributors_file.readlines()
contributions = []
for (i, line) in enumerate(lines):
    temp_author = line.strip(' * ').split(' -- ')[0].strip()
    print('Processing author %s (%d/%d)' % (temp_author, i + 1, len(lines)))
    if not temp_author:
        error_msg = '    ERROR: Expected the following format " * <author> -- <contributions>"\n'
        error_msg += '    ERROR: Actual = %s' % line
        print(error_msg)
        warnings.append(error_msg)
        contributions.append(line)
        continue
    author = temp_author.split('/')[0]
    if author in known_translations:
        line = line.replace(temp_author, known_translations[author])
    elif not is_valid_author(author):
        new_author = author
        issues = temp_author.split('/')[1:]
        candidates = generate_candidates(author, issues)
        candidate_names = []
        bad_prompts = []
        good_prompts = []
        for (candidate, source) in candidates:
            if candidate == NOT_FOUND:
                bad_prompts.append('    [X] %s' % source)
            else:
                index = len(candidate_names)
                candidate_names.append(candidate)
                good_prompts.append('    [%d] %s - %s' % (index, candidate, source))
        raw_index = len(candidate_names)
        custom_index = len(candidate_names) + 1
        for p in bad_prompts:
            print(p)
        if bad_prompts:
            print('    ---')
        for p in good_prompts:
            print(p)
        if INTERACTIVE_MODE:
            print('    [%d] %s - Raw GitHub username' % (raw_index, author))
            print('    [%d] Custom' % custom_index)
            response = input('    Your choice: ')
            last_index = custom_index
            while not response.isdigit() or int(response) > last_index:
                response = input('    Please enter an integer between 0 and %d: ' % last_index)
            response = int(response)
            if response == custom_index:
                new_author = input('    Please type a custom name for this author: ')
            elif response != raw_index:
                new_author = candidate_names[response]
        else:
            valid_candidate_names = [name for (name, _) in candidates if is_valid_author(name) and name != NOT_FOUND]
            if valid_candidate_names:
                new_author = valid_candidate_names[0]
        if is_valid_author(new_author):
            new_author = capitalize_author(new_author)
        else:
            warnings.append('Unable to find a valid name %s for author %s' % (author, temp_author))
        print('    * Replacing %s with %s' % (author, new_author))
        if INTERACTIVE_MODE and author not in known_translations and yesOrNoPrompt('    Add mapping %s -> %s to known translations file?' % (author, new_author)):
            known_translations_file.write('%s - %s\n' % (author, new_author))
            known_translations_file.flush()
        line = line.replace(temp_author, author)
    contributions.append(line)
print('==================================================================================\n')
contributors_file.close()
known_translations_file.close()
contributions.sort()
all_authors = set()
new_contributors_file_name = contributors_file_name + '.final'
new_contributors_file = open(new_contributors_file_name, 'w')
for line in contributions:
    author = line.strip(' * ').split(' -- ')[0]
    if author in all_authors:
        warnings.append('Detected duplicate author name %s. Please merge these manually.' % author)
    all_authors.add(author)
    new_contributors_file.write(line)
new_contributors_file.close()
print('Translated contributors list successfully written to %s!' % new_contributors_file_name)
if warnings:
    print('\n========== Warnings encountered while translating the contributor list ===========')
    for w in warnings:
        print(w)
    print('Please manually correct these in the final contributors list at %s.' % new_contributors_file_name)
    print('==================================================================================\n')