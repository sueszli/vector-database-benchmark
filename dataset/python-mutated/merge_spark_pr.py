import json
import os
import re
import subprocess
import sys
import traceback
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import HTTPError
try:
    import jira.client
    JIRA_IMPORTED = True
except ImportError:
    JIRA_IMPORTED = False
SPARK_HOME = os.environ.get('SPARK_HOME', os.getcwd())
PR_REMOTE_NAME = os.environ.get('PR_REMOTE_NAME', 'apache-github')
PUSH_REMOTE_NAME = os.environ.get('PUSH_REMOTE_NAME', 'apache')
JIRA_USERNAME = os.environ.get('JIRA_USERNAME', '')
JIRA_PASSWORD = os.environ.get('JIRA_PASSWORD', '')
JIRA_ACCESS_TOKEN = os.environ.get('JIRA_ACCESS_TOKEN')
GITHUB_OAUTH_KEY = os.environ.get('GITHUB_OAUTH_KEY')
GITHUB_BASE = 'https://github.com/apache/spark/pull'
GITHUB_API_BASE = 'https://api.github.com/repos/apache/spark'
JIRA_BASE = 'https://issues.apache.org/jira/browse'
JIRA_API_BASE = 'https://issues.apache.org/jira'
BRANCH_PREFIX = 'PR_TOOL'

def get_json(url):
    if False:
        print('Hello World!')
    try:
        request = Request(url)
        if GITHUB_OAUTH_KEY:
            request.add_header('Authorization', 'token %s' % GITHUB_OAUTH_KEY)
        return json.load(urlopen(request))
    except HTTPError as e:
        if 'X-RateLimit-Remaining' in e.headers and e.headers['X-RateLimit-Remaining'] == '0':
            print('Exceeded the GitHub API rate limit; see the instructions in ' + 'dev/merge_spark_pr.py to configure an OAuth token for making authenticated ' + 'GitHub requests.')
        elif e.code == 401:
            print('GITHUB_OAUTH_KEY is invalid or expired. Please regenerate a new one with ' + "at least the 'public_repo' scope on https://github.com/settings/tokens and " + 'update your local settings before you try again.')
        else:
            print('Unable to fetch URL, exiting: %s' % url)
        sys.exit(-1)

def fail(msg):
    if False:
        for i in range(10):
            print('nop')
    print(msg)
    clean_up()
    sys.exit(-1)

def run_cmd(cmd):
    if False:
        while True:
            i = 10
    print(cmd)
    if isinstance(cmd, list):
        return subprocess.check_output(cmd).decode('utf-8')
    else:
        return subprocess.check_output(cmd.split(' ')).decode('utf-8')

def continue_maybe(prompt):
    if False:
        print('Hello World!')
    result = input('\n%s (y/n): ' % prompt)
    if result.lower() != 'y':
        fail('Okay, exiting')

def clean_up():
    if False:
        return 10
    if 'original_head' in globals():
        print('Restoring head pointer to %s' % original_head)
        run_cmd('git checkout %s' % original_head)
        branches = run_cmd('git branch').replace(' ', '').split('\n')
        for branch in list(filter(lambda x: x.startswith(BRANCH_PREFIX), branches)):
            print('Deleting local branch %s' % branch)
            run_cmd('git branch -D %s' % branch)

def merge_pr(pr_num, target_ref, title, body, pr_repo_desc):
    if False:
        while True:
            i = 10
    pr_branch_name = '%s_MERGE_PR_%s' % (BRANCH_PREFIX, pr_num)
    target_branch_name = '%s_MERGE_PR_%s_%s' % (BRANCH_PREFIX, pr_num, target_ref.upper())
    run_cmd('git fetch %s pull/%s/head:%s' % (PR_REMOTE_NAME, pr_num, pr_branch_name))
    run_cmd('git fetch %s %s:%s' % (PUSH_REMOTE_NAME, target_ref, target_branch_name))
    run_cmd('git checkout %s' % target_branch_name)
    had_conflicts = False
    try:
        run_cmd(['git', 'merge', pr_branch_name, '--squash'])
    except Exception as e:
        msg = 'Error merging: %s\nWould you like to manually fix-up this merge?' % e
        continue_maybe(msg)
        msg = "Okay, please fix any conflicts and 'git add' conflicting files... Finished?"
        continue_maybe(msg)
        had_conflicts = True
    commit_authors = run_cmd(['git', 'log', 'HEAD..%s' % pr_branch_name, '--pretty=format:%an <%ae>', '--reverse']).split('\n')
    distinct_authors = sorted(list(dict.fromkeys(commit_authors)), key=lambda x: commit_authors.count(x), reverse=True)
    primary_author = input('Enter primary author in the format of "name <email>" [%s]: ' % distinct_authors[0])
    if primary_author == '':
        primary_author = distinct_authors[0]
    else:
        distinct_authors = list(filter(lambda x: x != primary_author, distinct_authors))
        distinct_authors.insert(0, primary_author)
    merge_message_flags = []
    merge_message_flags += ['-m', title]
    if body is not None:
        merge_message_flags += ['-m', body.replace('@', '')]
    committer_name = run_cmd('git config --get user.name').strip()
    committer_email = run_cmd('git config --get user.email').strip()
    if had_conflicts:
        message = 'This patch had conflicts when merged, resolved by\nCommitter: %s <%s>' % (committer_name, committer_email)
        merge_message_flags += ['-m', message]
    merge_message_flags += ['-m', 'Closes #%s from %s.' % (pr_num, pr_repo_desc)]
    authors = 'Authored-by:' if len(distinct_authors) == 1 else 'Lead-authored-by:'
    authors += ' %s' % distinct_authors.pop(0)
    if len(distinct_authors) > 0:
        authors += '\n' + '\n'.join(['Co-authored-by: %s' % a for a in distinct_authors])
    authors += '\n' + 'Signed-off-by: %s <%s>' % (committer_name, committer_email)
    merge_message_flags += ['-m', authors]
    run_cmd(['git', 'commit', '--author="%s"' % primary_author] + merge_message_flags)
    continue_maybe('Merge complete (local ref %s). Push to %s?' % (target_branch_name, PUSH_REMOTE_NAME))
    try:
        run_cmd('git push %s %s:%s' % (PUSH_REMOTE_NAME, target_branch_name, target_ref))
    except Exception as e:
        clean_up()
        fail('Exception while pushing: %s' % e)
    merge_hash = run_cmd('git rev-parse %s' % target_branch_name)[:8]
    clean_up()
    print('Pull request #%s merged!' % pr_num)
    print('Merge hash: %s' % merge_hash)
    return merge_hash

def cherry_pick(pr_num, merge_hash, default_branch):
    if False:
        for i in range(10):
            print('nop')
    pick_ref = input('Enter a branch name [%s]: ' % default_branch)
    if pick_ref == '':
        pick_ref = default_branch
    pick_branch_name = '%s_PICK_PR_%s_%s' % (BRANCH_PREFIX, pr_num, pick_ref.upper())
    run_cmd('git fetch %s %s:%s' % (PUSH_REMOTE_NAME, pick_ref, pick_branch_name))
    run_cmd('git checkout %s' % pick_branch_name)
    try:
        run_cmd('git cherry-pick -sx %s' % merge_hash)
    except Exception as e:
        msg = 'Error cherry-picking: %s\nWould you like to manually fix-up this merge?' % e
        continue_maybe(msg)
        msg = 'Okay, please fix any conflicts and finish the cherry-pick. Finished?'
        continue_maybe(msg)
    continue_maybe('Pick complete (local ref %s). Push to %s?' % (pick_branch_name, PUSH_REMOTE_NAME))
    try:
        run_cmd('git push %s %s:%s' % (PUSH_REMOTE_NAME, pick_branch_name, pick_ref))
    except Exception as e:
        clean_up()
        fail('Exception while pushing: %s' % e)
    pick_hash = run_cmd('git rev-parse %s' % pick_branch_name)[:8]
    clean_up()
    print('Pull request #%s picked into %s!' % (pr_num, pick_ref))
    print('Pick hash: %s' % pick_hash)
    return pick_ref

def print_jira_issue_summary(issue):
    if False:
        for i in range(10):
            print('nop')
    summary = issue.fields.summary
    assignee = issue.fields.assignee
    if assignee is not None:
        assignee = assignee.displayName
    status = issue.fields.status.name
    print('=== JIRA %s ===' % issue.key)
    print('summary\t\t%s\nassignee\t%s\nstatus\t\t%s\nurl\t\t%s/%s\n' % (summary, assignee, status, JIRA_BASE, issue.key))

def get_jira_issue(prompt, default_jira_id=''):
    if False:
        i = 10
        return i + 15
    jira_id = input('%s [%s]: ' % (prompt, default_jira_id))
    if jira_id == '':
        jira_id = default_jira_id
        if jira_id == '':
            print('JIRA ID not found, skipping.')
            return None
    try:
        issue = asf_jira.issue(jira_id)
        print_jira_issue_summary(issue)
        status = issue.fields.status.name
        if status == 'Resolved' or status == 'Closed':
            print("JIRA issue %s already has status '%s'" % (jira_id, status))
            return None
        if input('Check if the JIRA information is as expected (y/n): ').lower() != 'n':
            return issue
        else:
            return get_jira_issue('Enter the revised JIRA ID again or leave blank to skip')
    except Exception as e:
        print('ASF JIRA could not find %s: %s' % (jira_id, e))
        return get_jira_issue('Enter the revised JIRA ID again or leave blank to skip')

def resolve_jira_issue(merge_branches, comment, default_jira_id=''):
    if False:
        while True:
            i = 10
    issue = get_jira_issue('Enter a JIRA id', default_jira_id)
    if issue is None:
        return
    if issue.fields.assignee is None:
        choose_jira_assignee(issue)
    versions = asf_jira.project_versions('SPARK')
    versions = [x for x in versions if not x.raw['released'] and (not x.raw['archived']) and re.match('\\d+\\.\\d+\\.\\d+', x.name)]
    versions = sorted(versions, key=lambda x: x.name, reverse=True)
    default_fix_versions = []
    for b in merge_branches:
        if b == 'master':
            default_fix_versions.append(versions[0].name)
        else:
            found = False
            found_versions = []
            for v in versions:
                if v.name.startswith(b.replace('branch-', '')):
                    found_versions.append(v.name)
                    found = True
            if found:
                default_fix_versions.append(found_versions[-1])
            else:
                print('Target version for %s is not found on JIRA, it may be archived or not created. Skipping it.' % b)
    for v in default_fix_versions:
        (major, minor, patch) = v.split('.')
        if patch == '0':
            previous = '%s.%s.%s' % (major, int(minor) - 1, 0)
            if previous in default_fix_versions:
                default_fix_versions = list(filter(lambda x: x != v, default_fix_versions))
    default_fix_versions = ','.join(default_fix_versions)
    available_versions = set(list(map(lambda v: v.name, versions)))
    while True:
        try:
            fix_versions = input('Enter comma-separated fix version(s) [%s]: ' % default_fix_versions)
            if fix_versions == '':
                fix_versions = default_fix_versions
            fix_versions = fix_versions.replace(' ', '').split(',')
            if set(fix_versions).issubset(available_versions):
                break
            else:
                print('Specified version(s) [%s] not found in the available versions, try again (or leave blank and fix manually).' % ', '.join(fix_versions))
        except KeyboardInterrupt:
            raise
        except BaseException:
            traceback.print_exc()
            print('Error setting fix version(s), try again (or leave blank and fix manually)')

    def get_version_json(version_str):
        if False:
            return 10
        return list(filter(lambda v: v.name == version_str, versions))[0].raw
    jira_fix_versions = list(map(lambda v: get_version_json(v), fix_versions))
    resolve = list(filter(lambda a: a['name'] == 'Resolve Issue', asf_jira.transitions(issue.key)))[0]
    resolution = list(filter(lambda r: r.raw['name'] == 'Fixed', asf_jira.resolutions()))[0]
    asf_jira.transition_issue(issue.key, resolve['id'], fixVersions=jira_fix_versions, comment=comment, resolution={'id': resolution.raw['id']})
    try:
        print_jira_issue_summary(asf_jira.issue(issue.key))
    except Exception:
        print('Unable to fetch JIRA issue %s after resolving' % issue.key)
    print('Successfully resolved %s with fixVersions=%s!' % (issue.key, fix_versions))

def choose_jira_assignee(issue):
    if False:
        while True:
            i = 10
    '\n    Prompt the user to choose who to assign the issue to in jira, given a list of candidates,\n    including the original reporter and all commentators\n    '
    while True:
        try:
            reporter = issue.fields.reporter
            commentators = list(map(lambda x: x.author, issue.fields.comment.comments))
            candidates = set(commentators)
            candidates.add(reporter)
            candidates = list(candidates)
            print('JIRA is unassigned, choose assignee')
            for (idx, author) in enumerate(candidates):
                if author.key == 'apachespark':
                    continue
                annotations = ['Reporter'] if author == reporter else []
                if author in commentators:
                    annotations.append('Commentator')
                print('[%d] %s (%s)' % (idx, author.displayName, ','.join(annotations)))
            raw_assignee = input('Enter number of user, or userid, to assign to (blank to leave unassigned):')
            if raw_assignee == '':
                return None
            else:
                try:
                    id = int(raw_assignee)
                    assignee = candidates[id]
                except BaseException:
                    assignee = asf_jira.user(raw_assignee)
                try:
                    assign_issue(issue.key, assignee.name)
                except Exception as e:
                    if e.__class__.__name__ == 'JIRAError' and "'%s' cannot be assigned" % assignee.name in getattr(e, 'response').text:
                        continue_maybe("User '%s' cannot be assigned, add to contributors role and try again?" % assignee.name)
                        grant_contributor_role(assignee.name)
                        assign_issue(issue.key, assignee.name)
                    else:
                        raise e
                return assignee
        except KeyboardInterrupt:
            raise
        except BaseException:
            traceback.print_exc()
            print('Error assigning JIRA, try again (or leave blank and fix manually)')

def grant_contributor_role(user: str):
    if False:
        i = 10
        return i + 15
    role = asf_jira.project_role('SPARK', 10010)
    role.add_user(user)
    print("Successfully added user '%s' to contributors role" % user)

def assign_issue(issue: int, assignee: str) -> bool:
    if False:
        return 10
    "\n    Assign an issue to a user, which is a shorthand for jira.client.JIRA.assign_issue.\n    The original one has an issue that it will search users again and only choose the assignee\n    from 20 candidates. If it's unmatched, it picks the head blindly. In our case, the assignee\n    is already resolved.\n    "
    url = getattr(asf_jira, '_get_latest_url')(f'issue/{issue}/assignee')
    payload = {'name': assignee}
    getattr(asf_jira, '_session').put(url, data=json.dumps(payload))
    return True

def resolve_jira_issues(title, merge_branches, comment):
    if False:
        return 10
    jira_ids = re.findall('SPARK-[0-9]{4,5}', title)
    if len(jira_ids) == 0:
        resolve_jira_issue(merge_branches, comment)
    for jira_id in jira_ids:
        resolve_jira_issue(merge_branches, comment, jira_id)

def standardize_jira_ref(text):
    if False:
        return 10
    '\n    Standardize the [SPARK-XXXXX] [MODULE] prefix\n    Converts "[SPARK-XXX][mllib] Issue", "[MLLib] SPARK-XXX. Issue" or "SPARK XXX [MLLIB]: Issue" to\n    "[SPARK-XXX][MLLIB] Issue"\n\n    >>> standardize_jira_ref(\n    ...     "[SPARK-5821] [SQL] ParquetRelation2 CTAS should check if delete is successful")\n    \'[SPARK-5821][SQL] ParquetRelation2 CTAS should check if delete is successful\'\n    >>> standardize_jira_ref(\n    ...     "[SPARK-4123][Project Infra][WIP]: Show new dependencies added in pull requests")\n    \'[SPARK-4123][PROJECT INFRA][WIP] Show new dependencies added in pull requests\'\n    >>> standardize_jira_ref("[MLlib] Spark  5954: Top by key")\n    \'[SPARK-5954][MLLIB] Top by key\'\n    >>> standardize_jira_ref("[SPARK-979] a LRU scheduler for load balancing in TaskSchedulerImpl")\n    \'[SPARK-979] a LRU scheduler for load balancing in TaskSchedulerImpl\'\n    >>> standardize_jira_ref(\n    ...     "SPARK-1094 Support MiMa for reporting binary compatibility across versions.")\n    \'[SPARK-1094] Support MiMa for reporting binary compatibility across versions.\'\n    >>> standardize_jira_ref("[WIP]  [SPARK-1146] Vagrant support for Spark")\n    \'[SPARK-1146][WIP] Vagrant support for Spark\'\n    >>> standardize_jira_ref(\n    ...     "SPARK-1032. If Yarn app fails before registering, app master stays aroun...")\n    \'[SPARK-1032] If Yarn app fails before registering, app master stays aroun...\'\n    >>> standardize_jira_ref(\n    ...     "[SPARK-6250][SPARK-6146][SPARK-5911][SQL] Types are now reserved words in DDL parser.")\n    \'[SPARK-6250][SPARK-6146][SPARK-5911][SQL] Types are now reserved words in DDL parser.\'\n    >>> standardize_jira_ref("Additional information for users building from source code")\n    \'Additional information for users building from source code\'\n    '
    jira_refs = []
    components = []
    if re.search('^\\[SPARK-[0-9]{3,6}\\](\\[[A-Z0-9_\\s,]+\\] )+\\S+', text):
        return text
    pattern = re.compile('(SPARK[-\\s]*[0-9]{3,6})+', re.IGNORECASE)
    for ref in pattern.findall(text):
        jira_refs.append('[' + re.sub('\\s+', '-', ref.upper()) + ']')
        text = text.replace(ref, '')
    pattern = re.compile('(\\[[\\w\\s,.-]+\\])', re.IGNORECASE)
    for component in pattern.findall(text):
        components.append(component.upper())
        text = text.replace(component, '')
    pattern = re.compile('^\\W+(.*)', re.IGNORECASE)
    if pattern.search(text) is not None:
        text = pattern.search(text).groups()[0]
    clean_text = ''.join(jira_refs).strip() + ''.join(components).strip() + ' ' + text.strip()
    clean_text = re.sub('\\s+', ' ', clean_text.strip())
    return clean_text

def get_current_ref():
    if False:
        return 10
    ref = run_cmd('git rev-parse --abbrev-ref HEAD').strip()
    if ref == 'HEAD':
        return run_cmd('git rev-parse HEAD').strip()
    else:
        return ref

def initialize_jira():
    if False:
        return 10
    global asf_jira
    jira_server = {'server': JIRA_API_BASE}
    if not JIRA_IMPORTED:
        print("ERROR finding jira library. Run 'pip3 install jira' to install.")
        continue_maybe('Continue without jira?')
    elif JIRA_ACCESS_TOKEN:
        client = jira.client.JIRA(jira_server, token_auth=JIRA_ACCESS_TOKEN)
        try:
            client.current_user()
            asf_jira = client
        except Exception as e:
            if e.__class__.__name__ == 'JIRAError' and getattr(e, 'status_code', None) == 401:
                msg = "ASF JIRA could not authenticate with the invalid or expired token '%s'" % JIRA_ACCESS_TOKEN
                fail(msg)
            else:
                raise e
    elif JIRA_USERNAME and JIRA_PASSWORD:
        print('You can use JIRA_ACCESS_TOKEN instead of JIRA_USERNAME/JIRA_PASSWORD.')
        print('Visit https://issues.apache.org/jira/secure/ViewProfile.jspa ')
        print("and click 'Personal Access Tokens' menu to manage your own tokens.")
        asf_jira = jira.client.JIRA(jira_server, basic_auth=(JIRA_USERNAME, JIRA_PASSWORD))
    else:
        print('Neither JIRA_ACCESS_TOKEN nor JIRA_USERNAME/JIRA_PASSWORD are set.')
        continue_maybe('Continue without jira?')

def main():
    if False:
        for i in range(10):
            print('nop')
    initialize_jira()
    global original_head
    os.chdir(SPARK_HOME)
    original_head = get_current_ref()
    branches = get_json('%s/branches' % GITHUB_API_BASE)
    branch_names = list(filter(lambda x: x.startswith('branch-'), [x['name'] for x in branches]))
    branch_names = sorted(branch_names, reverse=True)
    branch_iter = iter(branch_names)
    pr_num = input('Which pull request would you like to merge? (e.g. 34): ')
    pr = get_json('%s/pulls/%s' % (GITHUB_API_BASE, pr_num))
    pr_events = get_json('%s/issues/%s/events' % (GITHUB_API_BASE, pr_num))
    url = pr['url']
    if '[WIP]' in pr['title']:
        msg = 'The PR title has `[WIP]`:\n%s\nContinue?' % pr['title']
        continue_maybe(msg)
    modified_title = standardize_jira_ref(pr['title']).rstrip('.')
    if modified_title != pr['title']:
        print("I've re-written the title as follows to match the standard format:")
        print('Original: %s' % pr['title'])
        print('Modified: %s' % modified_title)
        result = input('Would you like to use the modified title? (y/n): ')
        if result.lower() == 'y':
            title = modified_title
            print('Using modified title:')
        else:
            title = pr['title']
            print('Using original title:')
        print(title)
    else:
        title = pr['title']
    body = pr['body']
    if body is None:
        body = ''
    modified_body = re.sub(re.compile('<!--[^>]*-->\\n?', re.DOTALL), '', body).lstrip()
    if modified_body != body:
        print('=' * 80)
        print(modified_body)
        print('=' * 80)
        print("I've removed the comments from PR template like the above:")
        result = input('Would you like to use the modified body? (y/n): ')
        if result.lower() == 'y':
            body = modified_body
            print('Using modified body:')
        else:
            print('Using original body:')
        print('=' * 80)
        print(body)
        print('=' * 80)
    target_ref = pr['base']['ref']
    user_login = pr['user']['login']
    base_ref = pr['head']['ref']
    pr_repo_desc = '%s/%s' % (user_login, base_ref)
    merge_commits = [e for e in pr_events if e['event'] == 'closed' and e['commit_id'] is not None]
    if merge_commits and pr['state'] == 'closed':
        merge_commits = sorted(merge_commits, key=lambda x: x['created_at'])
        merge_hash = merge_commits[-1]['commit_id']
        message = get_json('%s/commits/%s' % (GITHUB_API_BASE, merge_hash))['commit']['message']
        print('Pull request %s has already been merged, assuming you want to backport' % pr_num)
        commit_is_downloaded = run_cmd(['git', 'rev-parse', '--quiet', '--verify', '%s^{commit}' % merge_hash]).strip() != ''
        if not commit_is_downloaded:
            fail("Couldn't find any merge commit for #%s, you may need to update HEAD." % pr_num)
        print('Found commit %s:\n%s' % (merge_hash, message))
        cherry_pick(pr_num, merge_hash, next(branch_iter, branch_names[0]))
        sys.exit(0)
    if not bool(pr['mergeable']):
        msg = 'Pull request %s is not mergeable in its current form.\n' % pr_num + 'Continue? (experts only!)'
        continue_maybe(msg)
    print('\n=== Pull Request #%s ===' % pr_num)
    print('title\t%s\nsource\t%s\ntarget\t%s\nurl\t%s' % (title, pr_repo_desc, target_ref, url))
    continue_maybe('Proceed with merging pull request #%s?' % pr_num)
    merged_refs = [target_ref]
    merge_hash = merge_pr(pr_num, target_ref, title, body, pr_repo_desc)
    pick_prompt = 'Would you like to pick %s into another branch?' % merge_hash
    while input('\n%s (y/n): ' % pick_prompt).lower() == 'y':
        merged_refs = merged_refs + [cherry_pick(pr_num, merge_hash, next(branch_iter, branch_names[0]))]
    if asf_jira is not None:
        continue_maybe('Would you like to update an associated JIRA?')
        jira_comment = 'Issue resolved by pull request %s\n[%s/%s]' % (pr_num, GITHUB_BASE, pr_num)
        resolve_jira_issues(title, merged_refs, jira_comment)
    else:
        print('Exiting without trying to close the associated JIRA.')
if __name__ == '__main__':
    import doctest
    (failure_count, test_count) = doctest.testmod()
    if failure_count:
        sys.exit(-1)
    try:
        main()
    except BaseException:
        clean_up()
        raise