"""
Post-release script to publish artifacts created from Azure Pipelines.

This currently includes:

* Moving snaps from the beta channel to the stable channel
* Publishing the Windows installer in a GitHub release

Setup:
 - Install the snapcraft command line tool and log in to a privileged account.
   - https://snapcraft.io/docs/installing-snapcraft
   - Use the command `snapcraft login` to log in.

Run:

python tools/finish_release.py --css <URL of code signing server>

Testing:

This script can be safely run between releases. When this is done, the script
should execute successfully until the final step when it tries to set draft
equal to false on the GitHub release. This step should fail because a published
release with that name already exists.

"""
import argparse
import getpass
import glob
import os.path
import re
import subprocess
import sys
import tempfile
from zipfile import ZipFile
from azure.devops.connection import Connection
import requests
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PLUGIN_SNAPS = [os.path.basename(path) for path in glob.glob(os.path.join(REPO_ROOT, 'certbot-dns-*')) if not path.endswith('certbot-dns-cloudxns')]
ALL_SNAPS = ['certbot'] + PLUGIN_SNAPS
SNAP_ARCH_COUNT = 3

def parse_args(args):
    if False:
        for i in range(10):
            print('nop')
    'Parse command line arguments.\n\n    :param args: command line arguments with the program name removed. This is\n        usually taken from sys.argv[1:].\n    :type args: `list` of `str`\n\n    :returns: parsed arguments\n    :rtype: argparse.Namespace\n\n    '
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--css', type=str, required=True, help='hostname of code signing server')
    return parser.parse_args(args)

def publish_windows(css):
    if False:
        return 10
    'SSH into CSS and trigger downloading Azure Pipeline assets, sign, and upload to Github\n\n    :param str css: CSS host name\n\n    '
    username = input('CSS username (usually EFF username): ')
    host = css
    command = 'ssh -t {}@{} bash /opt/certbot-misc/css/venv.sh'.format(username, host)
    print('SSH into CSS to trigger signing and uploading of Windows installer...')
    subprocess.run(command, check=True, universal_newlines=True, shell=True)

def assert_logged_into_snapcraft():
    if False:
        i = 10
        return i + 15
    "Confirms that snapcraft is logged in to an account.\n\n    :raises SystemExit: if the command snapcraft is unavailable or it\n        isn't logged into an account\n\n    "
    cmd = 'snapcraft whoami'.split()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, universal_newlines=True)
    except (subprocess.CalledProcessError, OSError):
        print('Please make sure that the command line tool snapcraft is')
        print('installed and that you have logged in to an account by running')
        print("'snapcraft login'.")
        sys.exit(1)

def get_snap_revisions(snap, channel, version):
    if False:
        print('Hello World!')
    'Finds the revisions for the snap and version in the given channel.\n\n    If you call this function without being logged in with snapcraft, it\n    will hang with no output.\n\n    :param str snap: the name of the snap on the snap store\n    :param str channel: snap channel to pull revisions from\n    :param str version: snap version number, e.g. 1.7.0\n\n    :returns: list of revision numbers\n    :rtype: `list` of `str`\n\n    :raises subprocess.CalledProcessError: if the snapcraft command\n        fails\n\n    :raises AssertionError: if the expected snaps are not found\n\n    '
    print('Getting revision numbers for', snap, version)
    cmd = ['snapcraft', 'status', snap]
    process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    pattern = f'^\\s+{channel}\\s+{version}\\s+(\\d+)\\s*'
    revisions = re.findall(pattern, process.stdout, re.MULTILINE)
    assert len(revisions) == SNAP_ARCH_COUNT, f'Unexpected number of snaps found for {channel} {snap} {version} (expected {SNAP_ARCH_COUNT}, found {len(revisions)})'
    return revisions

def promote_snaps(snaps, source_channel, version, progressive_percentage=None):
    if False:
        print('Hello World!')
    "Promotes the given snaps from source_channel to the stable channel.\n\n    If the snaps have already been released to the stable channel, this\n    function will try to release them again which has no effect.\n\n    :param snaps: snap package names to be promoted\n    :type snaps: `list` of `str`\n    :param str source_channel: snap channel to promote from\n    :param str version: the version number that should be found in the\n        candidate channel, e.g. 1.7.0\n    :param progressive_percentage: specifies the percentage of a progressive\n        deployment\n    :type progressive_percentage: int or None\n\n    :raises SystemExit: if the command snapcraft is unavailable or it\n        isn't logged into an account\n\n    :raises subprocess.CalledProcessError: if a snapcraft command fails\n        for another reason\n\n    "
    assert_logged_into_snapcraft()
    for snap in snaps:
        revisions = get_snap_revisions(snap, source_channel, version)
        print('Releasing', snap, 'snaps to the stable channel')
        for revision in revisions:
            cmd = ['snapcraft', 'release', snap, revision, 'stable']
            if progressive_percentage:
                cmd.extend(f'--progressive {progressive_percentage}'.split())
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True)
            except subprocess.CalledProcessError as e:
                print('The command', f"'{' '.join(cmd)}'", 'failed.')
                print('The output printed to stdout was:')
                print(e.stdout)
                raise

def fetch_version_number(major_version=None):
    if False:
        print('Hello World!')
    'Retrieve version number for release from Azure Pipelines\n\n    :param major_version: only consider releases for the specified major\n        version\n    :type major_version: str or None\n\n    :returns: version number\n\n    '
    organization_url = 'https://dev.azure.com/certbot'
    connection = Connection(base_url=organization_url)
    build_client = connection.clients.get_build_client()
    builds = build_client.get_builds('certbot', definitions='3')
    for build in builds:
        version = build_client.get_build('certbot', build.id).source_branch.split('v')[1]
        if major_version is None or version.split('.')[0] == major_version:
            return version
    raise ValueError('Release not found on Azure Pipelines!')

def main(args):
    if False:
        return 10
    parsed_args = parse_args(args)
    css = parsed_args.css
    version = fetch_version_number()
    promote_snaps(ALL_SNAPS, 'beta', version)
    publish_windows(css)
if __name__ == '__main__':
    main(sys.argv[1:])