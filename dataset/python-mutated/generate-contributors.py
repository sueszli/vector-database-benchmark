import os
import re
import sys
from releaseutils import tag_exists, get_commits, yesOrNoPrompt, get_date, is_valid_author, capitalize_author, JIRA, find_components, translate_issue_type, translate_component, CORE_COMPONENT, contributors_file_name, nice_join
JIRA_API_BASE = os.environ.get('JIRA_API_BASE', 'https://issues.apache.org/jira')
RELEASE_TAG = os.environ.get('RELEASE_TAG', 'v1.2.0-rc2')
PREVIOUS_RELEASE_TAG = os.environ.get('PREVIOUS_RELEASE_TAG', 'v1.1.0')
while not tag_exists(RELEASE_TAG):
    RELEASE_TAG = input('Please provide a valid release tag: ')
while not tag_exists(PREVIOUS_RELEASE_TAG):
    print('Please specify the previous release tag.')
    PREVIOUS_RELEASE_TAG = input('For instance, if you are releasing v1.2.0, you should specify v1.1.0: ')
print('Gathering new commits between tags %s and %s' % (PREVIOUS_RELEASE_TAG, RELEASE_TAG))
release_commits = get_commits(RELEASE_TAG)
previous_release_commits = get_commits(PREVIOUS_RELEASE_TAG)
previous_release_hashes = set()
previous_release_prs = set()
for old_commit in previous_release_commits:
    previous_release_hashes.add(old_commit.get_hash())
    if old_commit.get_pr_number():
        previous_release_prs.add(old_commit.get_pr_number())
new_commits = []
for this_commit in release_commits:
    this_hash = this_commit.get_hash()
    this_pr_number = this_commit.get_pr_number()
    if this_hash in previous_release_hashes:
        continue
    if this_pr_number and this_pr_number in previous_release_prs:
        continue
    new_commits.append(this_commit)
if not new_commits:
    sys.exit('There are no new commits between %s and %s!' % (PREVIOUS_RELEASE_TAG, RELEASE_TAG))
print('\n==================================================================================')
print('JIRA server: %s' % JIRA_API_BASE)
print('Release tag: %s' % RELEASE_TAG)
print('Previous release tag: %s' % PREVIOUS_RELEASE_TAG)
print('Number of commits in this range: %s' % len(new_commits))
print('')

def print_indented(_list):
    if False:
        while True:
            i = 10
    for x in _list:
        print('  %s' % x)
if yesOrNoPrompt('Show all commits?'):
    print_indented(new_commits)
print('==================================================================================\n')
if not yesOrNoPrompt('Does this look correct?'):
    sys.exit('Ok, exiting')
releases = []
maintenance = []
reverts = []
nojiras = []
filtered_commits = []

def is_release(commit_title):
    if False:
        while True:
            i = 10
    return '[release]' in commit_title.lower() or 'preparing spark release' in commit_title.lower() or 'preparing development version' in commit_title.lower() or ('CHANGES.txt' in commit_title)

def is_maintenance(commit_title):
    if False:
        print('Hello World!')
    return 'maintenance' in commit_title.lower() or 'manually close' in commit_title.lower()

def has_no_jira(commit_title):
    if False:
        for i in range(10):
            print('nop')
    return not re.findall('SPARK-[0-9]+', commit_title.upper())

def is_revert(commit_title):
    if False:
        i = 10
        return i + 15
    return 'revert' in commit_title.lower()

def is_docs(commit_title):
    if False:
        i = 10
        return i + 15
    return re.findall('docs*', commit_title.lower()) or 'programming guide' in commit_title.lower()
for c in new_commits:
    t = c.get_title()
    if not t:
        continue
    elif is_release(t):
        releases.append(c)
    elif is_maintenance(t):
        maintenance.append(c)
    elif is_revert(t):
        reverts.append(c)
    elif is_docs(t):
        filtered_commits.append(c)
    elif has_no_jira(t):
        nojiras.append(c)
    else:
        filtered_commits.append(c)
if releases or maintenance or reverts or nojiras:
    print('\n==================================================================================')
    if releases:
        print('Found %d release commits' % len(releases))
    if maintenance:
        print('Found %d maintenance commits' % len(maintenance))
    if reverts:
        print('Found %d revert commits' % len(reverts))
    if nojiras:
        print('Found %d commits with no JIRA' % len(nojiras))
    print('* Warning: these commits will be ignored.\n')
    if yesOrNoPrompt('Show ignored commits?'):
        if releases:
            print('Release (%d)' % len(releases))
            print_indented(releases)
        if maintenance:
            print('Maintenance (%d)' % len(maintenance))
            print_indented(maintenance)
        if reverts:
            print('Revert (%d)' % len(reverts))
            print_indented(reverts)
        if nojiras:
            print('No JIRA (%d)' % len(nojiras))
            print_indented(nojiras)
    print('==================== Warning: the above commits will be ignored ==================\n')
prompt_msg = '%d commits left to process after filtering. Ok to proceed?' % len(filtered_commits)
if not yesOrNoPrompt(prompt_msg):
    sys.exit('Ok, exiting.')
warnings = []
invalid_authors = {}
author_info = {}
jira_options = {'server': JIRA_API_BASE}
jira_client = JIRA(options=jira_options)
print('\n=========================== Compiling contributor list ===========================')
for commit in filtered_commits:
    _hash = commit.get_hash()
    title = commit.get_title()
    issues = re.findall('SPARK-[0-9]+', title.upper())
    author = commit.get_author()
    date = get_date(_hash)
    if is_valid_author(author):
        author = capitalize_author(author)
    else:
        if author not in invalid_authors:
            invalid_authors[author] = set()
        for issue in issues:
            invalid_authors[author].add(issue)
    commit_components = find_components(title, _hash)

    def populate(issue_type, components):
        if False:
            for i in range(10):
                print('nop')
        components = components or [CORE_COMPONENT]
        if author not in author_info:
            author_info[author] = {}
        if issue_type not in author_info[author]:
            author_info[author][issue_type] = set()
        for component in components:
            author_info[author][issue_type].add(component)
    for issue in issues:
        try:
            jira_issue = jira_client.issue(issue)
            jira_type = jira_issue.fields.issuetype.name
            jira_type = translate_issue_type(jira_type, issue, warnings)
            jira_components = [translate_component(c.name, _hash, warnings) for c in jira_issue.fields.components]
            all_components = set(jira_components + commit_components)
            populate(jira_type, all_components)
        except Exception as e:
            print('Unexpected error:', e)
    if is_docs(title) and (not issues):
        populate('documentation', commit_components)
    print('  Processed commit %s authored by %s on %s' % (_hash, author, date))
print('==================================================================================\n')
contributors_file = open(contributors_file_name, 'w')
authors = list(author_info.keys())
authors.sort()
for author in authors:
    contribution = ''
    components = set()
    issue_types = set()
    for (issue_type, comps) in author_info[author].items():
        components.update(comps)
        issue_types.add(issue_type)
    if len(components) == 1:
        contribution = '%s in %s' % (nice_join(issue_types), next(iter(components)))
    else:
        contributions = ['%s in %s' % (issue_type, nice_join(comps)) for (issue_type, comps) in author_info[author].items()]
        contribution = '; '.join(contributions)
    assert contribution
    contribution = contribution[0].capitalize() + contribution[1:]
    if author in invalid_authors and invalid_authors[author]:
        author = author + '/' + '/'.join(invalid_authors[author])
    line = author
    contributors_file.write(line + '\n')
contributors_file.close()
print('Contributors list is successfully written to %s!' % contributors_file_name)
if invalid_authors:
    warnings.append('Found the following invalid authors:')
    for a in invalid_authors:
        warnings.append('\t%s' % a)
    warnings.append("Please run './translate-contributors.py' to translate them.")
if warnings:
    print('\n============ Warnings encountered while creating the contributor list ============')
    for w in warnings:
        print(w)
    print('Please correct these in the final contributors list at %s.' % contributors_file_name)
    print('==================================================================================\n')