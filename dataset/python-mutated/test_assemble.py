import io
import os
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from unittest import mock
from unittest.mock import patch
from django.core.files.base import ContentFile
from sentry.models.artifactbundle import ArtifactBundle, ArtifactBundleFlatFileIndex, ArtifactBundleIndexingState, DebugIdArtifactBundle, ProjectArtifactBundle, ReleaseArtifactBundle, SourceFileType
from sentry.models.debugfile import ProjectDebugFile
from sentry.models.files.file import File
from sentry.models.files.fileblob import FileBlob
from sentry.models.files.fileblobowner import FileBlobOwner
from sentry.models.releasefile import ReleaseFile, read_artifact_index
from sentry.tasks.assemble import ArtifactBundlePostAssembler, AssembleResult, AssembleTask, ChunkFileState, assemble_artifacts, assemble_dif, assemble_file, get_assemble_status
from sentry.testutils.cases import TestCase
from sentry.testutils.helpers.datetime import freeze_time

class BaseAssembleTest(TestCase):

    def setUp(self):
        if False:
            return 10
        self.organization = self.create_organization(owner=self.user)
        self.team = self.create_team(organization=self.organization)
        self.project = self.create_project(teams=[self.team], organization=self.organization, name='foo')

class AssembleDifTest(BaseAssembleTest):

    def test_wrong_dif(self):
        if False:
            return 10
        content1 = b'foo'
        fileobj1 = ContentFile(content1)
        content2 = b'bar'
        fileobj2 = ContentFile(content2)
        content3 = b'baz'
        fileobj3 = ContentFile(content3)
        total_checksum = sha1(content2 + content1 + content3).hexdigest()
        blob1 = FileBlob.from_file(fileobj1)
        blob3 = FileBlob.from_file(fileobj3)
        blob2 = FileBlob.from_file(fileobj2)
        chunks = [blob2.checksum, blob1.checksum, blob3.checksum]
        assemble_dif(project_id=self.project.id, name='foo.sym', checksum=total_checksum, chunks=chunks)
        (status, _) = get_assemble_status(AssembleTask.DIF, self.project.id, total_checksum)
        assert status == ChunkFileState.ERROR

    def test_dif(self):
        if False:
            for i in range(10):
                print('nop')
        sym_file = self.load_fixture('crash.sym')
        blob1 = FileBlob.from_file(ContentFile(sym_file))
        total_checksum = sha1(sym_file).hexdigest()
        assemble_dif(project_id=self.project.id, name='crash.sym', checksum=total_checksum, chunks=[blob1.checksum])
        (status, _) = get_assemble_status(AssembleTask.DIF, self.project.id, total_checksum)
        assert status == ChunkFileState.OK
        dif = ProjectDebugFile.objects.filter(project_id=self.project.id, checksum=total_checksum).get()
        assert dif.file.headers == {'Content-Type': 'text/x-breakpad'}

    def test_assemble_from_files(self):
        if False:
            i = 10
            return i + 15
        files = []
        file_checksum = sha1()
        for _ in range(8):
            blob = os.urandom(1024 * 1024 * 8)
            hash = sha1(blob).hexdigest()
            file_checksum.update(blob)
            files.append((io.BytesIO(blob), hash))
        FileBlob.from_files(files, organization=self.organization)
        for (reference, checksum) in files:
            file_blob = FileBlob.objects.get(checksum=checksum)
            ref_bytes = reference.getvalue()
            with file_blob.getfile() as f:
                assert f.read(len(ref_bytes)) == ref_bytes
            FileBlobOwner.objects.filter(blob=file_blob, organization_id=self.organization.id).get()
        rv = assemble_file(AssembleTask.DIF, self.project, 'testfile', file_checksum.hexdigest(), [x[1] for x in files], 'dummy.type')
        assert rv is not None
        (f, tmp) = rv
        tmp.close()
        assert f.checksum == file_checksum.hexdigest()
        assert f.type == 'dummy.type'
        for (f, _) in files:
            f.seek(0)
        FileBlob.from_files(files, organization=self.organization)
        rv = assemble_file(AssembleTask.DIF, self.project, 'testfile', file_checksum.hexdigest(), [x[1] for x in files], 'dummy.type')
        assert rv is not None
        (f, tmp) = rv
        tmp.close()
        assert f.checksum == file_checksum.hexdigest()

    def test_assemble_duplicate_blobs(self):
        if False:
            for i in range(10):
                print('nop')
        files = []
        file_checksum = sha1()
        blob = os.urandom(1024 * 1024 * 8)
        hash = sha1(blob).hexdigest()
        for _ in range(8):
            file_checksum.update(blob)
            files.append((io.BytesIO(blob), hash))
        FileBlob.from_files(files, organization=self.organization)
        for (reference, checksum) in files:
            file_blob = FileBlob.objects.get(checksum=checksum)
            ref_bytes = reference.getvalue()
            with file_blob.getfile() as f:
                assert f.read(len(ref_bytes)) == ref_bytes
            FileBlobOwner.objects.filter(blob=file_blob, organization_id=self.organization.id).get()
        rv = assemble_file(AssembleTask.DIF, self.project, 'testfile', file_checksum.hexdigest(), [x[1] for x in files], 'dummy.type')
        assert rv is not None
        (f, tmp) = rv
        tmp.close()
        assert f.checksum == file_checksum.hexdigest()
        assert f.type == 'dummy.type'

    def test_assemble_debug_id_override(self):
        if False:
            i = 10
            return i + 15
        sym_file = self.load_fixture('crash.sym')
        blob1 = FileBlob.from_file(ContentFile(sym_file))
        total_checksum = sha1(sym_file).hexdigest()
        assemble_dif(project_id=self.project.id, name='crash.sym', checksum=total_checksum, chunks=[blob1.checksum], debug_id='67e9247c-814e-392b-a027-dbde6748fcbf-beef')
        (status, _) = get_assemble_status(AssembleTask.DIF, self.project.id, total_checksum)
        assert status == ChunkFileState.OK
        dif = ProjectDebugFile.objects.filter(project_id=self.project.id, checksum=total_checksum).get()
        assert dif.file.headers == {'Content-Type': 'text/x-breakpad'}
        assert dif.debug_id == '67e9247c-814e-392b-a027-dbde6748fcbf-beef'

class AssembleArtifactsTest(BaseAssembleTest):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()

    def test_artifacts_with_debug_ids(self):
        if False:
            print('Hello World!')
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        expected_source_file_types = [SourceFileType.MINIFIED_SOURCE, SourceFileType.SOURCE_MAP]
        expected_debug_ids = ['eb6e60f1-65ff-4f6f-adff-f1bbeded627b']
        for (version, dist, count) in [(None, None, 0), ('1.0', None, 1), (None, 'android', 0), ('1.0', 'android', 1)]:
            assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=version, dist=dist, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
            assert self.release.count_artifacts() == 0
            (status, details) = get_assemble_status(AssembleTask.ARTIFACT_BUNDLE, self.organization.id, total_checksum)
            assert status == ChunkFileState.OK
            assert details is None
            for debug_id in expected_debug_ids:
                debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(organization_id=self.organization.id, debug_id=debug_id).order_by('-debug_id', 'source_file_type')
                assert len(debug_id_artifact_bundles) == 2
                assert debug_id_artifact_bundles[0].artifact_bundle.file.size == len(bundle_file)
                for entry in debug_id_artifact_bundles:
                    assert str(entry.artifact_bundle.bundle_id) == '67429b2f-1d9e-43bb-a626-771a1e37555c'
                for (index, entry) in enumerate(debug_id_artifact_bundles):
                    assert entry.source_file_type == expected_source_file_types[index].value
                release_artifact_bundle = ReleaseArtifactBundle.objects.filter(organization_id=self.organization.id)
                assert len(release_artifact_bundle) == count
                if count == 1:
                    release_artifact_bundle[0].version_name = version
                    release_artifact_bundle[0].dist_name = dist
                project_artifact_bundles = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
                assert len(project_artifact_bundles) == 1
            ArtifactBundle.objects.all().delete()
            DebugIdArtifactBundle.objects.all().delete()
            ReleaseArtifactBundle.objects.all().delete()
            ProjectArtifactBundle.objects.all().delete()
            (status, details) = get_assemble_status(AssembleTask.ARTIFACT_BUNDLE, self.organization.id, total_checksum)
            assert status is None

    @patch('sentry.tasks.assemble.ArtifactBundlePostAssembler.post_assemble')
    def test_assembled_bundle_is_deleted_if_post_assembler_error_occurs(self, post_assemble):
        if False:
            return 10
        post_assemble.side_effect = Exception
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version='1.0', dist='android', checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        files = File.objects.filter()
        assert len(files) == 0

    @patch('sentry.tasks.assemble.ArtifactBundleArchive')
    def test_assembled_bundle_is_deleted_if_archive_is_invalid(self, artifact_bundle_archive):
        if False:
            return 10
        artifact_bundle_archive.side_effect = Exception
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version='1.0', dist='android', checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        files = File.objects.filter()
        assert len(files) == 0

    def test_assembled_bundle_is_deleted_if_checksum_mismatches(self):
        if False:
            for i in range(10):
                print('nop')
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = 'a' * 40
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version='1.0', dist='android', checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        files = File.objects.filter()
        assert len(files) == 0

    def test_upload_artifacts_with_duplicated_debug_ids(self):
        if False:
            print('Hello World!')
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_duplicated_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        expected_debug_ids = ['eb6e60f1-65ff-4f6f-adff-f1bbeded627b']
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version='1.0', dist='android', checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        for debug_id in expected_debug_ids:
            debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(organization_id=self.organization.id, debug_id=debug_id)
            assert len(debug_id_artifact_bundles) == 2

    def test_upload_multiple_artifacts_with_same_bundle_id(self):
        if False:
            i = 10
            return i + 15
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        bundle_id = '67429b2f-1d9e-43bb-a626-771a1e37555c'
        debug_id = 'eb6e60f1-65ff-4f6f-adff-f1bbeded627b'
        for time in ('2023-05-31T10:00:00', '2023-05-31T11:00:00', '2023-05-31T12:00:00'):
            with freeze_time(time):
                assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version='1.0', dist='android', checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        expected_updated_date = datetime.fromisoformat('2023-05-31T12:00:00').replace(tzinfo=timezone.utc)
        artifact_bundles = ArtifactBundle.objects.filter(bundle_id=bundle_id)
        assert len(artifact_bundles) == 1
        assert artifact_bundles[0].date_added == expected_updated_date
        assert artifact_bundles[0].date_last_modified == expected_updated_date
        files = File.objects.filter()
        assert len(files) == 1
        debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(debug_id=debug_id)
        assert len(debug_id_artifact_bundles) == 2
        assert debug_id_artifact_bundles[0].date_added == expected_updated_date
        assert debug_id_artifact_bundles[1].date_added == expected_updated_date
        release_artifact_bundle = ReleaseArtifactBundle.objects.filter(release_name='1.0', dist_name='android')
        assert len(release_artifact_bundle) == 1
        assert release_artifact_bundle[0].date_added == expected_updated_date
        project_artifact_bundle = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
        assert len(project_artifact_bundle) == 1
        assert project_artifact_bundle[0].date_added == expected_updated_date

    def test_upload_multiple_artifacts_with_same_bundle_id_and_no_release_dist_pair(self):
        if False:
            return 10
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        bundle_id = '67429b2f-1d9e-43bb-a626-771a1e37555c'
        debug_id = 'eb6e60f1-65ff-4f6f-adff-f1bbeded627b'
        for i in range(0, 3):
            assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=None, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        artifact_bundles = ArtifactBundle.objects.filter(bundle_id=bundle_id)
        assert len(artifact_bundles) == 1
        files = File.objects.filter()
        assert len(files) == 1
        debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(debug_id=debug_id)
        assert len(debug_id_artifact_bundles) == 2
        project_artifact_bundle = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
        assert len(project_artifact_bundle) == 1

    def test_upload_multiple_artifacts_with_same_bundle_id_and_different_release_dist_pair(self):
        if False:
            while True:
                i = 10
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        bundle_id = '67429b2f-1d9e-43bb-a626-771a1e37555c'
        debug_id = 'eb6e60f1-65ff-4f6f-adff-f1bbeded627b'
        combinations = (('1.0', 'android'), ('2.0', 'android'), ('1.0', 'ios'), ('2.0', 'ios'))
        for (version, dist) in combinations:
            assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=version, dist=dist, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        artifact_bundles = ArtifactBundle.objects.filter(bundle_id=bundle_id)
        assert len(artifact_bundles) == 1
        files = File.objects.filter()
        assert len(files) == 1
        debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(debug_id=debug_id)
        assert len(debug_id_artifact_bundles) == 2
        for (version, dist) in combinations:
            release_artifact_bundle = ReleaseArtifactBundle.objects.filter(release_name=version, dist_name=dist)
            assert len(release_artifact_bundle) == 1
        project_artifact_bundle = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
        assert len(project_artifact_bundle) == 1

    def test_upload_multiple_artifacts_with_first_release_and_second_no_release_and_same_bundle_id(self):
        if False:
            for i in range(10):
                print('nop')
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        bundle_id = '67429b2f-1d9e-43bb-a626-771a1e37555c'
        debug_id = 'eb6e60f1-65ff-4f6f-adff-f1bbeded627b'
        for version in ('1.0', None):
            assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        artifact_bundles = ArtifactBundle.objects.filter(bundle_id=bundle_id)
        assert len(artifact_bundles) == 1
        files = File.objects.filter()
        assert len(files) == 1
        debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(debug_id=debug_id)
        assert len(debug_id_artifact_bundles) == 2
        release_artifact_bundle = ReleaseArtifactBundle.objects.filter(release_name='1.0')
        assert len(release_artifact_bundle) == 1
        project_artifact_bundle = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
        assert len(project_artifact_bundle) == 1

    def test_upload_multiple_artifacts_with_first_no_release_and_second_release_and_same_bundle_id(self):
        if False:
            i = 10
            return i + 15
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        bundle_id = '67429b2f-1d9e-43bb-a626-771a1e37555c'
        debug_id = 'eb6e60f1-65ff-4f6f-adff-f1bbeded627b'
        for version in (None, '1.0'):
            assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        artifact_bundles = ArtifactBundle.objects.filter(bundle_id=bundle_id)
        assert len(artifact_bundles) == 1
        files = File.objects.filter()
        assert len(files) == 1
        debug_id_artifact_bundles = DebugIdArtifactBundle.objects.filter(debug_id=debug_id)
        assert len(debug_id_artifact_bundles) == 2
        release_artifact_bundle = ReleaseArtifactBundle.objects.filter(release_name='1.0')
        assert len(release_artifact_bundle) == 1
        project_artifact_bundle = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
        assert len(project_artifact_bundle) == 1

    def test_upload_multiple_artifacts_with_existing_bundle_id_duplicate(self):
        if False:
            while True:
                i = 10
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        bundle_id = '67429b2f-1d9e-43bb-a626-771a1e37555c'
        ArtifactBundle.objects.create(organization_id=self.organization.id, bundle_id=bundle_id, file=File.objects.create(name='artifact_bundle.zip', type='artifact_bundle'), artifact_count=1)
        ArtifactBundle.objects.create(organization_id=self.organization.id, bundle_id=bundle_id, file=File.objects.create(name='artifact_bundle.zip', type='artifact_bundle'), artifact_count=4)
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=None, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=True)
        artifact_bundles = ArtifactBundle.objects.filter(bundle_id=bundle_id)
        assert len(artifact_bundles) == 1
        files = File.objects.filter()
        assert len(files) == 1
        project_artifact_bundle = ProjectArtifactBundle.objects.filter(project_id=self.project.id)
        assert len(project_artifact_bundle) == 1

    @patch('sentry.tasks.assemble.index_artifact_bundles_for_release')
    def test_bundle_indexing_started_when_over_threshold(self, index_artifact_bundles_for_release):
        if False:
            while True:
                i = 10
        release = '1.0'
        dist = 'android'
        bundle_file_1 = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1_1 = FileBlob.from_file(ContentFile(bundle_file_1))
        total_checksum_1 = sha1(bundle_file_1).hexdigest()
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=release, dist=dist, checksum=total_checksum_1, chunks=[blob1_1.checksum], upload_as_artifact_bundle=True)
        index_artifact_bundles_for_release.assert_not_called()
        bundle_file_2 = self.create_artifact_bundle_zip(fixture_path='artifact_bundle', project=self.project.id)
        blob1_2 = FileBlob.from_file(ContentFile(bundle_file_2))
        total_checksum_2 = sha1(bundle_file_2).hexdigest()
        assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=release, dist=dist, checksum=total_checksum_2, chunks=[blob1_2.checksum], upload_as_artifact_bundle=True)
        bundles = ArtifactBundle.objects.all()
        index_artifact_bundles_for_release.assert_called_with(organization_id=self.organization.id, artifact_bundles=[(bundles[0], None), (bundles[1], mock.ANY)])

    def test_bundle_flat_file_indexing(self):
        if False:
            while True:
                i = 10
        release = '1.0'
        dist = 'android'
        bundle_file_1 = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1_1 = FileBlob.from_file(ContentFile(bundle_file_1))
        total_checksum_1 = sha1(bundle_file_1).hexdigest()
        with self.feature('organizations:sourcemaps-bundle-flat-file-indexing'):
            assemble_artifacts(org_id=self.organization.id, project_ids=[self.project.id], version=release, dist=dist, checksum=total_checksum_1, chunks=[blob1_1.checksum], upload_as_artifact_bundle=True)
        flat_file_index = ArtifactBundleFlatFileIndex.objects.get(project_id=self.project.id, release_name=release, dist_name=dist)
        assert flat_file_index.load_flat_file_index() is not None

    def test_artifacts_without_debug_ids(self):
        if False:
            i = 10
            return i + 15
        bundle_file = self.create_artifact_bundle_zip(org=self.organization.slug, release=self.release.version)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        for min_files in (10, 1):
            with self.options({'processing.release-archive-min-files': min_files}):
                ReleaseFile.objects.filter(release_id=self.release.id).delete()
                assert self.release.count_artifacts() == 0
                assemble_artifacts(org_id=self.organization.id, version=self.release.version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=False)
                assert self.release.count_artifacts() == 2
                (status, details) = get_assemble_status(AssembleTask.RELEASE_BUNDLE, self.organization.id, total_checksum)
                assert status == ChunkFileState.OK
                assert details is None
                if min_files == 1:
                    index = read_artifact_index(self.release, dist=None)
                    assert index is not None
                    archive_ident = index['files']['~/index.js']['archive_ident']
                    releasefile = ReleaseFile.objects.get(release_id=self.release.id, ident=archive_ident)
                    assert releasefile.file.size == len(bundle_file)
                else:
                    release_file = ReleaseFile.objects.get(organization_id=self.organization.id, release_id=self.release.id, name='~/index.js', dist_id=None)
                    assert release_file.file.headers == {'Sourcemap': 'index.js.map'}

    def test_artifacts_invalid_org(self):
        if False:
            return 10
        bundle_file = self.create_artifact_bundle_zip(org='invalid', release=self.release.version)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        assemble_artifacts(org_id=self.organization.id, version=self.release.version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=False)
        (status, details) = get_assemble_status(AssembleTask.RELEASE_BUNDLE, self.organization.id, total_checksum)
        assert status == ChunkFileState.ERROR

    def test_artifacts_invalid_release(self):
        if False:
            for i in range(10):
                print('nop')
        bundle_file = self.create_artifact_bundle_zip(org=self.organization.slug, release='invalid')
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        assemble_artifacts(org_id=self.organization.id, version=self.release.version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=False)
        (status, details) = get_assemble_status(AssembleTask.RELEASE_BUNDLE, self.organization.id, total_checksum)
        assert status == ChunkFileState.ERROR

    def test_artifacts_invalid_zip(self):
        if False:
            return 10
        bundle_file = b''
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        assemble_artifacts(org_id=self.organization.id, version=self.release.version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=False)
        (status, details) = get_assemble_status(AssembleTask.RELEASE_BUNDLE, self.organization.id, total_checksum)
        assert status == ChunkFileState.ERROR

    @patch('sentry.tasks.assemble.update_artifact_index', side_effect=RuntimeError('foo'))
    def test_failing_update(self, _):
        if False:
            for i in range(10):
                print('nop')
        bundle_file = self.create_artifact_bundle_zip(org=self.organization.slug, release=self.release.version)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        with self.options({'processing.save-release-archives': True, 'processing.release-archive-min-files': 1}):
            assemble_artifacts(org_id=self.organization.id, version=self.release.version, checksum=total_checksum, chunks=[blob1.checksum], upload_as_artifact_bundle=False)
            (status, details) = get_assemble_status(AssembleTask.RELEASE_BUNDLE, self.organization.id, total_checksum)
            assert status == ChunkFileState.OK

@freeze_time('2023-05-31T10:00:00')
class ArtifactBundleIndexingTest(TestCase):

    def _create_bundle_and_bind_to_release(self, release, dist, bundle_id, indexing_state, date):
        if False:
            i = 10
            return i + 15
        artifact_bundle = ArtifactBundle.objects.create(organization_id=self.organization.id, bundle_id=bundle_id, file=File.objects.create(name='bundle.zip', type='artifact_bundle'), artifact_count=10, indexing_state=indexing_state, date_uploaded=date, date_added=date, date_last_modified=date)
        ReleaseArtifactBundle.objects.create(organization_id=self.organization.id, release_name=release, dist_name=dist, artifact_bundle=artifact_bundle, date_added=date)
        return artifact_bundle

    def mock_assemble_result(self) -> AssembleResult:
        if False:
            i = 10
            return i + 15
        bundle_file = self.create_artifact_bundle_zip(fixture_path='artifact_bundle_debug_ids', project=self.project.id)
        blob1 = FileBlob.from_file(ContentFile(bundle_file))
        total_checksum = sha1(bundle_file).hexdigest()
        rv = assemble_file(task=AssembleTask.ARTIFACT_BUNDLE, org_or_project=self.organization, name='bundle.zip', checksum=total_checksum, chunks=[blob1.checksum], file_type='artifact.bundle')
        assert rv is not None
        return rv

    @patch('sentry.tasks.assemble.index_artifact_bundles_for_release')
    def test_index_if_needed_with_no_bundles(self, index_artifact_bundles_for_release):
        if False:
            while True:
                i = 10
        release = '1.0'
        dist = 'android'
        with ArtifactBundlePostAssembler(assemble_result=self.mock_assemble_result(), organization=self.organization, release=release, dist=dist, project_ids=[]) as post_assembler:
            post_assembler._index_bundle_if_needed(artifact_bundle=None, release=release, dist=dist, date_snapshot=datetime.now())
        index_artifact_bundles_for_release.assert_not_called()

    @patch('sentry.tasks.assemble.index_artifact_bundles_for_release')
    def test_index_if_needed_with_lower_bundles_than_threshold(self, index_artifact_bundles_for_release):
        if False:
            print('Hello World!')
        release = '1.0'
        dist = 'android'
        self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='2c5b367b-4fef-4db8-849d-b9e79607d630', indexing_state=ArtifactBundleIndexingState.NOT_INDEXED.value, date=datetime.now() - timedelta(hours=1))
        with ArtifactBundlePostAssembler(assemble_result=self.mock_assemble_result(), organization=self.organization, release=release, dist=dist, project_ids=[]) as post_assembler:
            post_assembler._index_bundle_if_needed(artifact_bundle=None, release=release, dist=dist, date_snapshot=datetime.now())
        index_artifact_bundles_for_release.assert_not_called()

    @patch('sentry.tasks.assemble.index_artifact_bundles_for_release')
    def test_index_if_needed_with_higher_bundles_than_threshold(self, index_artifact_bundles_for_release):
        if False:
            for i in range(10):
                print('nop')
        release = '1.0'
        dist = 'android'
        artifact_bundle_1 = self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='2c5b367b-4fef-4db8-849d-b9e79607d630', indexing_state=ArtifactBundleIndexingState.NOT_INDEXED.value, date=datetime.now() - timedelta(hours=2))
        artifact_bundle_2 = self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='0cf678f2-0771-4e2f-8ace-d6cea8493f0d', indexing_state=ArtifactBundleIndexingState.NOT_INDEXED.value, date=datetime.now() - timedelta(hours=1))
        with ArtifactBundlePostAssembler(assemble_result=self.mock_assemble_result(), organization=self.organization, release=release, dist=dist, project_ids=[]) as post_assembler:
            post_assembler._index_bundle_if_needed(artifact_bundle=artifact_bundle_2, release=release, dist=dist, date_snapshot=datetime.now())
        index_artifact_bundles_for_release.assert_called_with(organization_id=self.organization.id, artifact_bundles=[(artifact_bundle_1, None), (artifact_bundle_2, mock.ANY)])

    @patch('sentry.tasks.assemble.index_artifact_bundles_for_release')
    def test_index_if_needed_with_bundles_already_indexed(self, index_artifact_bundles_for_release):
        if False:
            print('Hello World!')
        release = '1.0'
        dist = 'android'
        self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='2c5b367b-4fef-4db8-849d-b9e79607d630', indexing_state=ArtifactBundleIndexingState.WAS_INDEXED.value, date=datetime.now() - timedelta(hours=2))
        self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='0cf678f2-0771-4e2f-8ace-d6cea8493f0d', indexing_state=ArtifactBundleIndexingState.WAS_INDEXED.value, date=datetime.now() - timedelta(hours=1))
        with ArtifactBundlePostAssembler(assemble_result=self.mock_assemble_result(), organization=self.organization, release=release, dist=dist, project_ids=[]) as post_assembler:
            post_assembler._index_bundle_if_needed(artifact_bundle=None, release=release, dist=dist, date_snapshot=datetime.now())
        index_artifact_bundles_for_release.assert_not_called()

    @patch('sentry.tasks.assemble.index_artifact_bundles_for_release')
    def test_index_if_needed_with_newer_bundle_already_stored(self, index_artifact_bundles_for_release):
        if False:
            i = 10
            return i + 15
        release = '1.0'
        dist = 'android'
        artifact_bundle_1 = self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='2c5b367b-4fef-4db8-849d-b9e79607d630', indexing_state=ArtifactBundleIndexingState.NOT_INDEXED.value, date=datetime.now() - timedelta(hours=1))
        artifact_bundle_2 = self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='2c5b367b-4fef-4db8-849d-b9e79607d630', indexing_state=ArtifactBundleIndexingState.NOT_INDEXED.value, date=datetime.now() - timedelta(hours=2))
        self._create_bundle_and_bind_to_release(release=release, dist=dist, bundle_id='0cf678f2-0771-4e2f-8ace-d6cea8493f0d', indexing_state=ArtifactBundleIndexingState.NOT_INDEXED.value, date=datetime.now() + timedelta(hours=1))
        with ArtifactBundlePostAssembler(assemble_result=self.mock_assemble_result(), organization=self.organization, release=release, dist=dist, project_ids=[]) as post_assembler:
            post_assembler._index_bundle_if_needed(artifact_bundle=artifact_bundle_1, release=release, dist=dist, date_snapshot=datetime.now())
        index_artifact_bundles_for_release.assert_called_with(organization_id=self.organization.id, artifact_bundles=[(artifact_bundle_1, mock.ANY), (artifact_bundle_2, None)])