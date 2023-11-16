from __future__ import absolute_import
from StringIO import StringIO
import re
from bzrlib import lazy_import
lazy_import.lazy_import(globals(), '\nfrom bzrlib import (\n    branch as _mod_branch,\n    diff,\n    email_message,\n    errors,\n    gpg,\n    hooks,\n    registry,\n    revision as _mod_revision,\n    rio,\n    testament,\n    timestamp,\n    trace,\n    )\nfrom bzrlib.bundle import (\n    serializer as bundle_serializer,\n    )\n')

class MergeRequestBodyParams(object):
    """Parameter object for the merge_request_body hook."""

    def __init__(self, body, orig_body, directive, to, basename, subject, branch, tree=None):
        if False:
            for i in range(10):
                print('nop')
        self.body = body
        self.orig_body = orig_body
        self.directive = directive
        self.branch = branch
        self.tree = tree
        self.to = to
        self.basename = basename
        self.subject = subject

class MergeDirectiveHooks(hooks.Hooks):
    """Hooks for MergeDirective classes."""

    def __init__(self):
        if False:
            return 10
        hooks.Hooks.__init__(self, 'bzrlib.merge_directive', 'BaseMergeDirective.hooks')
        self.add_hook('merge_request_body', 'Called with a MergeRequestBodyParams when a body is needed for a merge request.  Callbacks must return a body.  If more than one callback is registered, the output of one callback is provided to the next.', (1, 15, 0))

class BaseMergeDirective(object):
    """A request to perform a merge into a branch.

    This is the base class that all merge directive implementations 
    should derive from.

    :cvar multiple_output_files: Whether or not this merge directive 
        stores a set of revisions in more than one file
    """
    hooks = MergeDirectiveHooks()
    multiple_output_files = False

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, source_branch=None, message=None, bundle=None):
        if False:
            print('Hello World!')
        'Constructor.\n\n        :param revision_id: The revision to merge\n        :param testament_sha1: The sha1 of the testament of the revision to\n            merge.\n        :param time: The current POSIX timestamp time\n        :param timezone: The timezone offset\n        :param target_branch: Location of branch to apply the merge to\n        :param patch: The text of a diff or bundle\n        :param source_branch: A public location to merge the revision from\n        :param message: The message to use when committing this merge\n        '
        self.revision_id = revision_id
        self.testament_sha1 = testament_sha1
        self.time = time
        self.timezone = timezone
        self.target_branch = target_branch
        self.patch = patch
        self.source_branch = source_branch
        self.message = message

    def to_lines(self):
        if False:
            print('Hello World!')
        'Serialize as a list of lines\n\n        :return: a list of lines\n        '
        raise NotImplementedError(self.to_lines)

    def to_files(self):
        if False:
            while True:
                i = 10
        'Serialize as a set of files.\n\n        :return: List of tuples with filename and contents as lines\n        '
        raise NotImplementedError(self.to_files)

    def get_raw_bundle(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the bundle for this merge directive.\n\n        :return: bundle text or None if there is no bundle\n        '
        return None

    def _to_lines(self, base_revision=False):
        if False:
            for i in range(10):
                print('nop')
        'Serialize as a list of lines\n\n        :return: a list of lines\n        '
        time_str = timestamp.format_patch_date(self.time, self.timezone)
        stanza = rio.Stanza(revision_id=self.revision_id, timestamp=time_str, target_branch=self.target_branch, testament_sha1=self.testament_sha1)
        for key in ('source_branch', 'message'):
            if self.__dict__[key] is not None:
                stanza.add(key, self.__dict__[key])
        if base_revision:
            stanza.add('base_revision_id', self.base_revision_id)
        lines = ['# ' + self._format_string + '\n']
        lines.extend(rio.to_patch_lines(stanza))
        lines.append('# \n')
        return lines

    def write_to_directory(self, path):
        if False:
            print('Hello World!')
        'Write this merge directive to a series of files in a directory.\n\n        :param path: Filesystem path to write to\n        '
        raise NotImplementedError(self.write_to_directory)

    @classmethod
    def from_objects(klass, repository, revision_id, time, timezone, target_branch, patch_type='bundle', local_target_branch=None, public_branch=None, message=None):
        if False:
            return 10
        "Generate a merge directive from various objects\n\n        :param repository: The repository containing the revision\n        :param revision_id: The revision to merge\n        :param time: The POSIX timestamp of the date the request was issued.\n        :param timezone: The timezone of the request\n        :param target_branch: The url of the branch to merge into\n        :param patch_type: 'bundle', 'diff' or None, depending on the type of\n            patch desired.\n        :param local_target_branch: the submit branch, either itself or a local copy\n        :param public_branch: location of a public branch containing\n            the target revision.\n        :param message: Message to use when committing the merge\n        :return: The merge directive\n\n        The public branch is always used if supplied.  If the patch_type is\n        not 'bundle', the public branch must be supplied, and will be verified.\n\n        If the message is not supplied, the message from revision_id will be\n        used for the commit.\n        "
        t_revision_id = revision_id
        if revision_id == _mod_revision.NULL_REVISION:
            t_revision_id = None
        t = testament.StrictTestament3.from_revision(repository, t_revision_id)
        if local_target_branch is None:
            submit_branch = _mod_branch.Branch.open(target_branch)
        else:
            submit_branch = local_target_branch
        if submit_branch.get_public_branch() is not None:
            target_branch = submit_branch.get_public_branch()
        if patch_type is None:
            patch = None
        else:
            submit_revision_id = submit_branch.last_revision()
            submit_revision_id = _mod_revision.ensure_null(submit_revision_id)
            repository.fetch(submit_branch.repository, submit_revision_id)
            graph = repository.get_graph()
            ancestor_id = graph.find_unique_lca(revision_id, submit_revision_id)
            type_handler = {'bundle': klass._generate_bundle, 'diff': klass._generate_diff, None: lambda x, y, z: None}
            patch = type_handler[patch_type](repository, revision_id, ancestor_id)
        if public_branch is not None and patch_type != 'bundle':
            public_branch_obj = _mod_branch.Branch.open(public_branch)
            if not public_branch_obj.repository.has_revision(revision_id):
                raise errors.PublicBranchOutOfDate(public_branch, revision_id)
        return klass(revision_id, t.as_sha1(), time, timezone, target_branch, patch, patch_type, public_branch, message)

    def get_disk_name(self, branch):
        if False:
            print('Hello World!')
        'Generate a suitable basename for storing this directive on disk\n\n        :param branch: The Branch this merge directive was generated fro\n        :return: A string\n        '
        (revno, revision_id) = branch.last_revision_info()
        if self.revision_id == revision_id:
            revno = [revno]
        else:
            revno = branch.get_revision_id_to_revno_map().get(self.revision_id, ['merge'])
        nick = re.sub('(\\W+)', '-', branch.nick).strip('-')
        return '%s-%s' % (nick, '.'.join((str(n) for n in revno)))

    @staticmethod
    def _generate_diff(repository, revision_id, ancestor_id):
        if False:
            for i in range(10):
                print('nop')
        tree_1 = repository.revision_tree(ancestor_id)
        tree_2 = repository.revision_tree(revision_id)
        s = StringIO()
        diff.show_diff_trees(tree_1, tree_2, s, old_label='', new_label='')
        return s.getvalue()

    @staticmethod
    def _generate_bundle(repository, revision_id, ancestor_id):
        if False:
            print('Hello World!')
        s = StringIO()
        bundle_serializer.write_bundle(repository, revision_id, ancestor_id, s)
        return s.getvalue()

    def to_signed(self, branch):
        if False:
            while True:
                i = 10
        'Serialize as a signed string.\n\n        :param branch: The source branch, to get the signing strategy\n        :return: a string\n        '
        my_gpg = gpg.GPGStrategy(branch.get_config_stack())
        return my_gpg.sign(''.join(self.to_lines()))

    def to_email(self, mail_to, branch, sign=False):
        if False:
            for i in range(10):
                print('nop')
        'Serialize as an email message.\n\n        :param mail_to: The address to mail the message to\n        :param branch: The source branch, to get the signing strategy and\n            source email address\n        :param sign: If True, gpg-sign the email\n        :return: an email message\n        '
        mail_from = branch.get_config_stack().get('email')
        if self.message is not None:
            subject = self.message
        else:
            revision = branch.repository.get_revision(self.revision_id)
            subject = revision.message
        if sign:
            body = self.to_signed(branch)
        else:
            body = ''.join(self.to_lines())
        message = email_message.EmailMessage(mail_from, mail_to, subject, body)
        return message

    def install_revisions(self, target_repo):
        if False:
            i = 10
            return i + 15
        'Install revisions and return the target revision'
        if not target_repo.has_revision(self.revision_id):
            if self.patch_type == 'bundle':
                info = bundle_serializer.read_bundle(StringIO(self.get_raw_bundle()))
                try:
                    info.install_revisions(target_repo, stream_input=False)
                except errors.RevisionNotPresent:
                    try:
                        submit_branch = _mod_branch.Branch.open(self.target_branch)
                    except errors.NotBranchError:
                        raise errors.TargetNotBranch(self.target_branch)
                    missing_revisions = []
                    bundle_revisions = set((r.revision_id for r in info.real_revisions))
                    for revision in info.real_revisions:
                        for parent_id in revision.parent_ids:
                            if parent_id not in bundle_revisions and (not target_repo.has_revision(parent_id)):
                                missing_revisions.append(parent_id)
                    unique_missing = []
                    unique_missing_set = set()
                    for revision in reversed(missing_revisions):
                        if revision in unique_missing_set:
                            continue
                        unique_missing.append(revision)
                        unique_missing_set.add(revision)
                    for missing_revision in unique_missing:
                        target_repo.fetch(submit_branch.repository, missing_revision)
                    info.install_revisions(target_repo, stream_input=False)
            else:
                source_branch = _mod_branch.Branch.open(self.source_branch)
                target_repo.fetch(source_branch.repository, self.revision_id)
        return self.revision_id

    def compose_merge_request(self, mail_client, to, body, branch, tree=None):
        if False:
            i = 10
            return i + 15
        'Compose a request to merge this directive.\n\n        :param mail_client: The mail client to use for composing this request.\n        :param to: The address to compose the request to.\n        :param branch: The Branch that was used to produce this directive.\n        :param tree: The Tree (if any) for the Branch used to produce this\n            directive.\n        '
        basename = self.get_disk_name(branch)
        subject = '[MERGE] '
        if self.message is not None:
            subject += self.message
        else:
            revision = branch.repository.get_revision(self.revision_id)
            subject += revision.get_summary()
        if getattr(mail_client, 'supports_body', False):
            orig_body = body
            for hook in self.hooks['merge_request_body']:
                params = MergeRequestBodyParams(body, orig_body, self, to, basename, subject, branch, tree)
                body = hook(params)
        elif len(self.hooks['merge_request_body']) > 0:
            trace.warning('Cannot run merge_request_body hooks because mail client %s does not support message bodies.', mail_client.__class__.__name__)
        mail_client.compose_merge_request(to, subject, ''.join(self.to_lines()), basename, body)

class MergeDirective(BaseMergeDirective):
    """A request to perform a merge into a branch.

    Designed to be serialized and mailed.  It provides all the information
    needed to perform a merge automatically, by providing at minimum a revision
    bundle or the location of a branch.

    The serialization format is robust against certain common forms of
    deterioration caused by mailing.

    The format is also designed to be patch-compatible.  If the directive
    includes a diff or revision bundle, it should be possible to apply it
    directly using the standard patch program.
    """
    _format_string = 'Bazaar merge directive format 1'

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, patch_type=None, source_branch=None, message=None, bundle=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n        :param revision_id: The revision to merge\n        :param testament_sha1: The sha1 of the testament of the revision to\n            merge.\n        :param time: The current POSIX timestamp time\n        :param timezone: The timezone offset\n        :param target_branch: Location of the branch to apply the merge to\n        :param patch: The text of a diff or bundle\n        :param patch_type: None, "diff" or "bundle", depending on the contents\n            of patch\n        :param source_branch: A public location to merge the revision from\n        :param message: The message to use when committing this merge\n        '
        BaseMergeDirective.__init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch, source_branch, message)
        if patch_type not in (None, 'diff', 'bundle'):
            raise ValueError(patch_type)
        if patch_type != 'bundle' and source_branch is None:
            raise errors.NoMergeSource()
        if patch_type is not None and patch is None:
            raise errors.PatchMissing(patch_type)
        self.patch_type = patch_type

    def clear_payload(self):
        if False:
            print('Hello World!')
        self.patch = None
        self.patch_type = None

    def get_raw_bundle(self):
        if False:
            while True:
                i = 10
        return self.bundle

    def _bundle(self):
        if False:
            print('Hello World!')
        if self.patch_type == 'bundle':
            return self.patch
        else:
            return None
    bundle = property(_bundle)

    @classmethod
    def from_lines(klass, lines):
        if False:
            while True:
                i = 10
        'Deserialize a MergeRequest from an iterable of lines\n\n        :param lines: An iterable of lines\n        :return: a MergeRequest\n        '
        line_iter = iter(lines)
        firstline = ''
        for line in line_iter:
            if line.startswith('# Bazaar merge directive format '):
                return _format_registry.get(line[2:].rstrip())._from_lines(line_iter)
            firstline = firstline or line.strip()
        raise errors.NotAMergeDirective(firstline)

    @classmethod
    def _from_lines(klass, line_iter):
        if False:
            return 10
        stanza = rio.read_patch_stanza(line_iter)
        patch_lines = list(line_iter)
        if len(patch_lines) == 0:
            patch = None
            patch_type = None
        else:
            patch = ''.join(patch_lines)
            try:
                bundle_serializer.read_bundle(StringIO(patch))
            except (errors.NotABundle, errors.BundleNotSupported, errors.BadBundle):
                patch_type = 'diff'
            else:
                patch_type = 'bundle'
        (time, timezone) = timestamp.parse_patch_date(stanza.get('timestamp'))
        kwargs = {}
        for key in ('revision_id', 'testament_sha1', 'target_branch', 'source_branch', 'message'):
            try:
                kwargs[key] = stanza.get(key)
            except KeyError:
                pass
        kwargs['revision_id'] = kwargs['revision_id'].encode('utf-8')
        return MergeDirective(time=time, timezone=timezone, patch_type=patch_type, patch=patch, **kwargs)

    def to_lines(self):
        if False:
            while True:
                i = 10
        lines = self._to_lines()
        if self.patch is not None:
            lines.extend(self.patch.splitlines(True))
        return lines

    @staticmethod
    def _generate_bundle(repository, revision_id, ancestor_id):
        if False:
            return 10
        s = StringIO()
        bundle_serializer.write_bundle(repository, revision_id, ancestor_id, s, '0.9')
        return s.getvalue()

    def get_merge_request(self, repository):
        if False:
            return 10
        'Provide data for performing a merge\n\n        Returns suggested base, suggested target, and patch verification status\n        '
        return (None, self.revision_id, 'inapplicable')

class MergeDirective2(BaseMergeDirective):
    _format_string = 'Bazaar merge directive format 2 (Bazaar 0.90)'

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, source_branch=None, message=None, bundle=None, base_revision_id=None):
        if False:
            while True:
                i = 10
        if source_branch is None and bundle is None:
            raise errors.NoMergeSource()
        BaseMergeDirective.__init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch, source_branch, message)
        self.bundle = bundle
        self.base_revision_id = base_revision_id

    def _patch_type(self):
        if False:
            print('Hello World!')
        if self.bundle is not None:
            return 'bundle'
        elif self.patch is not None:
            return 'diff'
        else:
            return None
    patch_type = property(_patch_type)

    def clear_payload(self):
        if False:
            i = 10
            return i + 15
        self.patch = None
        self.bundle = None

    def get_raw_bundle(self):
        if False:
            print('Hello World!')
        if self.bundle is None:
            return None
        else:
            return self.bundle.decode('base-64')

    @classmethod
    def _from_lines(klass, line_iter):
        if False:
            i = 10
            return i + 15
        stanza = rio.read_patch_stanza(line_iter)
        patch = None
        bundle = None
        try:
            start = line_iter.next()
        except StopIteration:
            pass
        else:
            if start.startswith('# Begin patch'):
                patch_lines = []
                for line in line_iter:
                    if line.startswith('# Begin bundle'):
                        start = line
                        break
                    patch_lines.append(line)
                else:
                    start = None
                patch = ''.join(patch_lines)
            if start is not None:
                if start.startswith('# Begin bundle'):
                    bundle = ''.join(line_iter)
                else:
                    raise errors.IllegalMergeDirectivePayload(start)
        (time, timezone) = timestamp.parse_patch_date(stanza.get('timestamp'))
        kwargs = {}
        for key in ('revision_id', 'testament_sha1', 'target_branch', 'source_branch', 'message', 'base_revision_id'):
            try:
                kwargs[key] = stanza.get(key)
            except KeyError:
                pass
        kwargs['revision_id'] = kwargs['revision_id'].encode('utf-8')
        kwargs['base_revision_id'] = kwargs['base_revision_id'].encode('utf-8')
        return klass(time=time, timezone=timezone, patch=patch, bundle=bundle, **kwargs)

    def to_lines(self):
        if False:
            return 10
        lines = self._to_lines(base_revision=True)
        if self.patch is not None:
            lines.append('# Begin patch\n')
            lines.extend(self.patch.splitlines(True))
        if self.bundle is not None:
            lines.append('# Begin bundle\n')
            lines.extend(self.bundle.splitlines(True))
        return lines

    @classmethod
    def from_objects(klass, repository, revision_id, time, timezone, target_branch, include_patch=True, include_bundle=True, local_target_branch=None, public_branch=None, message=None, base_revision_id=None):
        if False:
            print('Hello World!')
        'Generate a merge directive from various objects\n\n        :param repository: The repository containing the revision\n        :param revision_id: The revision to merge\n        :param time: The POSIX timestamp of the date the request was issued.\n        :param timezone: The timezone of the request\n        :param target_branch: The url of the branch to merge into\n        :param include_patch: If true, include a preview patch\n        :param include_bundle: If true, include a bundle\n        :param local_target_branch: the target branch, either itself or a local copy\n        :param public_branch: location of a public branch containing\n            the target revision.\n        :param message: Message to use when committing the merge\n        :return: The merge directive\n\n        The public branch is always used if supplied.  If no bundle is\n        included, the public branch must be supplied, and will be verified.\n\n        If the message is not supplied, the message from revision_id will be\n        used for the commit.\n        '
        locked = []
        try:
            repository.lock_write()
            locked.append(repository)
            t_revision_id = revision_id
            if revision_id == 'null:':
                t_revision_id = None
            t = testament.StrictTestament3.from_revision(repository, t_revision_id)
            if local_target_branch is None:
                submit_branch = _mod_branch.Branch.open(target_branch)
            else:
                submit_branch = local_target_branch
            submit_branch.lock_read()
            locked.append(submit_branch)
            if submit_branch.get_public_branch() is not None:
                target_branch = submit_branch.get_public_branch()
            submit_revision_id = submit_branch.last_revision()
            submit_revision_id = _mod_revision.ensure_null(submit_revision_id)
            graph = repository.get_graph(submit_branch.repository)
            ancestor_id = graph.find_unique_lca(revision_id, submit_revision_id)
            if base_revision_id is None:
                base_revision_id = ancestor_id
            if (include_patch, include_bundle) != (False, False):
                repository.fetch(submit_branch.repository, submit_revision_id)
            if include_patch:
                patch = klass._generate_diff(repository, revision_id, base_revision_id)
            else:
                patch = None
            if include_bundle:
                bundle = klass._generate_bundle(repository, revision_id, ancestor_id).encode('base-64')
            else:
                bundle = None
            if public_branch is not None and (not include_bundle):
                public_branch_obj = _mod_branch.Branch.open(public_branch)
                public_branch_obj.lock_read()
                locked.append(public_branch_obj)
                if not public_branch_obj.repository.has_revision(revision_id):
                    raise errors.PublicBranchOutOfDate(public_branch, revision_id)
            testament_sha1 = t.as_sha1()
        finally:
            for entry in reversed(locked):
                entry.unlock()
        return klass(revision_id, testament_sha1, time, timezone, target_branch, patch, public_branch, message, bundle, base_revision_id)

    def _verify_patch(self, repository):
        if False:
            return 10
        calculated_patch = self._generate_diff(repository, self.revision_id, self.base_revision_id)
        stored_patch = re.sub('\r\n?', '\n', self.patch)
        calculated_patch = re.sub('\r\n?', '\n', calculated_patch)
        calculated_patch = re.sub(' *\n', '\n', calculated_patch)
        stored_patch = re.sub(' *\n', '\n', stored_patch)
        return calculated_patch == stored_patch

    def get_merge_request(self, repository):
        if False:
            print('Hello World!')
        'Provide data for performing a merge\n\n        Returns suggested base, suggested target, and patch verification status\n        '
        verified = self._maybe_verify(repository)
        return (self.base_revision_id, self.revision_id, verified)

    def _maybe_verify(self, repository):
        if False:
            for i in range(10):
                print('nop')
        if self.patch is not None:
            if self._verify_patch(repository):
                return 'verified'
            else:
                return 'failed'
        else:
            return 'inapplicable'

class MergeDirectiveFormatRegistry(registry.Registry):

    def register(self, directive, format_string=None):
        if False:
            print('Hello World!')
        if format_string is None:
            format_string = directive._format_string
        registry.Registry.register(self, format_string, directive)
_format_registry = MergeDirectiveFormatRegistry()
_format_registry.register(MergeDirective)
_format_registry.register(MergeDirective2)
_format_registry.register(MergeDirective2, 'Bazaar merge directive format 2 (Bazaar 0.19)')