"""Help include git hash in tensorflow bazel build.

This creates symlinks from the internal git repository directory so
that the build system can see changes in the version state. We also
remember what branch git was on so when the branch changes we can
detect that the ref file is no longer correct (so we can suggest users
run ./configure again).

NOTE: this script is only used in opensource.

"""
import argparse
from builtins import bytes
import json
import os
import shutil
import subprocess

def parse_branch_ref(filename):
    if False:
        for i in range(10):
            print('nop')
    'Given a filename of a .git/HEAD file return ref path.\n\n  In particular, if git is in detached head state, this will\n  return None. If git is in attached head, it will return\n  the branch reference. E.g. if on \'master\', the HEAD will\n  contain \'ref: refs/heads/master\' so \'refs/heads/master\'\n  will be returned.\n\n  Example: parse_branch_ref(".git/HEAD")\n  Args:\n    filename: file to treat as a git HEAD file\n  Returns:\n    None if detached head, otherwise ref subpath\n  Raises:\n    RuntimeError: if the HEAD file is unparseable.\n  '
    data = open(filename).read().strip()
    items = data.split(' ')
    if len(items) == 1:
        return None
    elif len(items) == 2 and items[0] == 'ref:':
        return items[1].strip()
    else:
        raise RuntimeError('Git directory has unparseable HEAD')

def configure(src_base_path, gen_path, debug=False):
    if False:
        for i in range(10):
            print('nop')
    'Configure `src_base_path` to embed git hashes if available.'
    git_path = os.path.join(src_base_path, '.git')
    if os.path.exists(gen_path):
        if os.path.isdir(gen_path):
            try:
                shutil.rmtree(gen_path)
            except OSError:
                raise RuntimeError('Cannot delete directory %s due to permission error, inspect and remove manually' % gen_path)
        else:
            raise RuntimeError('Cannot delete non-directory %s, inspect ', 'and remove manually' % gen_path)
    os.makedirs(gen_path)
    if not os.path.isdir(gen_path):
        raise RuntimeError('gen_git_source.py: Failed to create dir')
    spec = {}
    link_map = {'head': None, 'branch_ref': None}
    if not os.path.isdir(git_path):
        spec['git'] = False
        open(os.path.join(gen_path, 'head'), 'w').write('')
        open(os.path.join(gen_path, 'branch_ref'), 'w').write('')
    else:
        spec['git'] = True
        spec['path'] = src_base_path
        git_head_path = os.path.join(git_path, 'HEAD')
        spec['branch'] = parse_branch_ref(git_head_path)
        link_map['head'] = git_head_path
        if spec['branch'] is not None:
            link_map['branch_ref'] = os.path.join(git_path, *os.path.split(spec['branch']))
    for (target, src) in link_map.items():
        if src is None:
            open(os.path.join(gen_path, target), 'w').write('')
        elif not os.path.exists(src):
            open(os.path.join(gen_path, target), 'w').write('')
            spec['git'] = False
        else:
            try:
                if hasattr(os, 'symlink'):
                    os.symlink(src, os.path.join(gen_path, target))
                else:
                    shutil.copy2(src, os.path.join(gen_path, target))
            except OSError:
                shutil.copy2(src, os.path.join(gen_path, target))
    json.dump(spec, open(os.path.join(gen_path, 'spec.json'), 'w'), indent=2)
    if debug:
        print('gen_git_source.py: list %s' % gen_path)
        print('gen_git_source.py: %s' + repr(os.listdir(gen_path)))
        print('gen_git_source.py: spec is %r' % spec)

def get_git_version(git_base_path, git_tag_override):
    if False:
        print('Hello World!')
    "Get the git version from the repository.\n\n  This function runs `git describe ...` in the path given as `git_base_path`.\n  This will return a string of the form:\n  <base-tag>-<number of commits since tag>-<shortened sha hash>\n\n  For example, 'v0.10.0-1585-gbb717a6' means v0.10.0 was the last tag when\n  compiled. 1585 commits are after that commit tag, and we can get back to this\n  version by running `git checkout gbb717a6`.\n\n  Args:\n    git_base_path: where the .git directory is located\n    git_tag_override: Override the value for the git tag. This is useful for\n      releases where we want to build the release before the git tag is\n      created.\n  Returns:\n    A bytestring representing the git version\n  "
    unknown_label = b'unknown'
    try:
        val = bytes(subprocess.check_output(['git', str('--git-dir=%s/.git' % git_base_path), str('--work-tree=%s' % git_base_path), 'describe', '--long', '--tags']).strip())
        version_separator = b'-'
        if git_tag_override and val:
            split_val = val.split(version_separator)
            if len(split_val) < 3:
                raise Exception("Expected git version in format 'TAG-COMMITS AFTER TAG-HASH' but got '%s'" % val)
            abbrev_commit = split_val[-1]
            val = version_separator.join([bytes(git_tag_override, 'utf-8'), b'0', abbrev_commit])
        return val if val else unknown_label
    except (subprocess.CalledProcessError, OSError):
        return unknown_label

def write_version_info(filename, git_version):
    if False:
        print('Hello World!')
    'Write a c file that defines the version functions.\n\n  Args:\n    filename: filename to write to.\n    git_version: the result of a git describe.\n  '
    if b'"' in git_version or b'\\' in git_version:
        git_version = b'git_version_is_invalid'
    contents = '\n/*  Generated by gen_git_source.py  */\n\n#ifndef TENSORFLOW_CORE_UTIL_VERSION_INFO_H_\n#define TENSORFLOW_CORE_UTIL_VERSION_INFO_H_\n\n#define STRINGIFY(x) #x\n#define TOSTRING(x) STRINGIFY(x)\n\n#define TF_GIT_VERSION "%s"\n#ifdef _MSC_VER\n#define TF_COMPILER_VERSION "MSVC " TOSTRING(_MSC_FULL_VER)\n#else\n#define TF_COMPILER_VERSION __VERSION__\n#endif\n#ifdef _GLIBCXX_USE_CXX11_ABI\n#define TF_CXX11_ABI_FLAG _GLIBCXX_USE_CXX11_ABI\n#else\n#define TF_CXX11_ABI_FLAG 0\n#endif\n#define TF_CXX_VERSION __cplusplus\n#ifdef TENSORFLOW_MONOLITHIC_BUILD\n#define TF_MONOLITHIC_BUILD 1\n#else\n#define TF_MONOLITHIC_BUILD 0\n#endif\n\n#endif  // TENSORFLOW_CORE_UTIL_VERSION_INFO_H_\n' % git_version.decode('utf-8')
    open(filename, 'w').write(contents)

def generate(arglist, git_tag_override=None):
    if False:
        while True:
            i = 10
    "Generate version_info.cc as given `destination_file`.\n\n  Args:\n    arglist: should be a sequence that contains\n             spec, head_symlink, ref_symlink, destination_file.\n\n  `destination_file` is the filename where version_info.cc will be written\n\n  `spec` is a filename where the file contains a JSON dictionary\n    'git' bool that is true if the source is in a git repo\n    'path' base path of the source code\n    'branch' the name of the ref specification of the current branch/tag\n\n  `head_symlink` is a filename to HEAD that is cross-referenced against\n    what is contained in the json branch designation.\n\n  `ref_symlink` is unused in this script but passed, because the build\n    system uses that file to detect when commits happen.\n\n    git_tag_override: Override the value for the git tag. This is useful for\n      releases where we want to build the release before the git tag is\n      created.\n\n  Raises:\n    RuntimeError: If ./configure needs to be run, RuntimeError will be raised.\n  "
    (spec, head_symlink, _, dest_file) = arglist
    data = json.load(open(spec))
    git_version = None
    if not data['git']:
        git_version = b'unknown'
    else:
        old_branch = data['branch']
        new_branch = parse_branch_ref(head_symlink)
        if new_branch != old_branch:
            raise RuntimeError("Run ./configure again, branch was '%s' but is now '%s'" % (old_branch, new_branch))
        git_version = get_git_version(data['path'], git_tag_override)
    write_version_info(dest_file, git_version)

def raw_generate(output_file, source_dir, git_tag_override=None):
    if False:
        print('Hello World!')
    'Simple generator used for cmake/make build systems.\n\n  This does not create any symlinks. It requires the build system\n  to build unconditionally.\n\n  Args:\n    output_file: Output filename for the version info cc\n    source_dir: Base path of the source code\n    git_tag_override: Override the value for the git tag. This is useful for\n      releases where we want to build the release before the git tag is\n      created.\n  '
    git_version = get_git_version(source_dir, git_tag_override)
    write_version_info(output_file, git_version)
parser = argparse.ArgumentParser(description='Git hash injection into bazel.\nIf used with --configure <path> will search for git directory and put symlinks\ninto source so that a bazel genrule can call --generate')
parser.add_argument('--debug', type=bool, help='print debugging information about paths', default=False)
parser.add_argument('--configure', type=str, help='Path to configure as a git repo dependency tracking sentinel')
parser.add_argument('--gen_root_path', type=str, help='Root path to place generated git files (created by --configure).')
parser.add_argument('--git_tag_override', type=str, help='Override git tag value in the __git_version__ string. Useful when creating release builds before the release tag is created.')
parser.add_argument('--generate', type=str, help='Generate given spec-file, HEAD-symlink-file, ref-symlink-file', nargs='+')
parser.add_argument('--raw_generate', type=str, help='Generate version_info.cc (simpler version used for cmake/make)')
parser.add_argument('--source_dir', type=str, help='Base path of the source code (used for cmake/make)')
args = parser.parse_args()
if args.configure is not None:
    if args.gen_root_path is None:
        raise RuntimeError('Must pass --gen_root_path arg when running --configure')
    configure(args.configure, args.gen_root_path, debug=args.debug)
elif args.generate is not None:
    generate(args.generate, args.git_tag_override)
elif args.raw_generate is not None:
    source_path = '.'
    if args.source_dir is not None:
        source_path = args.source_dir
    raw_generate(args.raw_generate, source_path, args.git_tag_override)
else:
    raise RuntimeError('--configure or --generate or --raw_generate must be used')