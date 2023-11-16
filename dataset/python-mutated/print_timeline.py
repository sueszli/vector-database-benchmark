"""Prints releases' timeline in RST format."""
import shlex
import subprocess
entry = "- {date}:\n  `{ver} <https://pypi.org/project/psutil/{ver}/#files>`__ -\n  `what's new <https://github.com/giampaolo/psutil/blob/master/HISTORY.rst#{nodotver}>`__ -\n  `diff <https://github.com/giampaolo/psutil/compare/{prevtag}...{tag}#files_bucket>`__"

def sh(cmd):
    if False:
        return 10
    return subprocess.check_output(shlex.split(cmd), universal_newlines=True).strip()

def get_tag_date(tag):
    if False:
        while True:
            i = 10
    out = sh('git log -1 --format=%ai {}'.format(tag))
    return out.split(' ')[0]

def main():
    if False:
        while True:
            i = 10
    releases = []
    out = sh('git tag')
    for line in out.split('\n'):
        tag = line.split(' ')[0]
        ver = tag.replace('release-', '')
        nodotver = ver.replace('.', '')
        date = get_tag_date(tag)
        releases.append((tag, ver, nodotver, date))
    releases.sort(reverse=True)
    for (i, rel) in enumerate(releases):
        (tag, ver, nodotver, date) = rel
        try:
            prevtag = releases[i + 1][0]
        except IndexError:
            prevtag = sh('git rev-list --max-parents=0 HEAD')
        print(entry.format(**locals()))
if __name__ == '__main__':
    main()