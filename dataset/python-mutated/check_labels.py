"""Check whether a PR has required labels."""
import sys
from typing import Any
from github_utils import gh_delete_comment, gh_post_pr_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from label_utils import has_required_labels, is_label_err_comment, LABEL_ERR_MSG
from trymerge import GitHubPR

def delete_all_label_err_comments(pr: 'GitHubPR') -> None:
    if False:
        i = 10
        return i + 15
    for comment in pr.get_comments():
        if is_label_err_comment(comment):
            gh_delete_comment(pr.org, pr.project, comment.database_id)

def add_label_err_comment(pr: 'GitHubPR') -> None:
    if False:
        print('Hello World!')
    if not any((is_label_err_comment(comment) for comment in pr.get_comments())):
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, LABEL_ERR_MSG)

def parse_args() -> Any:
    if False:
        while True:
            i = 10
    from argparse import ArgumentParser
    parser = ArgumentParser('Check PR labels')
    parser.add_argument('pr_num', type=int)
    return parser.parse_args()

def main() -> None:
    if False:
        return 10
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    (org, project) = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)
    try:
        if not has_required_labels(pr):
            print(LABEL_ERR_MSG)
            add_label_err_comment(pr)
        else:
            delete_all_label_err_comments(pr)
    except Exception as e:
        pass
    sys.exit(0)
if __name__ == '__main__':
    main()