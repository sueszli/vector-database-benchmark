"""Test cases for :module:`artifact_service_client`."""
import contextlib
import io
import threading
import unittest
from urllib.parse import quote
from apache_beam.portability import common_urns
from apache_beam.portability.api import beam_artifact_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.portability import artifact_service
from apache_beam.utils import proto_utils

class InMemoryFileManager(object):

    def __init__(self, contents=()):
        if False:
            print('Hello World!')
        self._contents = dict(contents)

    def get(self, path):
        if False:
            for i in range(10):
                print('nop')
        return self._contents[path]

    def file_reader(self, path):
        if False:
            return 10
        return io.BytesIO(self._contents[path])

    def file_writer(self, name):
        if False:
            i = 10
            return i + 15
        path = 'prefix:' + name

        @contextlib.contextmanager
        def writable():
            if False:
                while True:
                    i = 10
            buffer = io.BytesIO()
            yield buffer
            buffer.seek(0)
            self._contents[path] = buffer.read()
        return (writable(), path)

class ArtifactServiceTest(unittest.TestCase):

    def file_artifact(self, path):
        if False:
            for i in range(10):
                print('nop')
        return beam_runner_api_pb2.ArtifactInformation(type_urn=common_urns.artifact_types.FILE.urn, type_payload=beam_runner_api_pb2.ArtifactFilePayload(path=path).SerializeToString())

    def embedded_artifact(self, data, name=None):
        if False:
            for i in range(10):
                print('nop')
        return beam_runner_api_pb2.ArtifactInformation(type_urn=common_urns.artifact_types.EMBEDDED.urn, type_payload=beam_runner_api_pb2.EmbeddedFilePayload(data=data).SerializeToString(), role_urn=common_urns.artifact_roles.STAGING_TO.urn if name else None, role_payload=beam_runner_api_pb2.ArtifactStagingToRolePayload(staged_name=name).SerializeToString() if name else None)

    def test_file_retrieval(self):
        if False:
            for i in range(10):
                print('nop')
        file_manager = InMemoryFileManager({'path/to/a': b'a', 'path/to/b': b'b' * 37})
        retrieval_service = artifact_service.ArtifactRetrievalService(file_manager.file_reader, chunk_size=10)
        dep_a = self.file_artifact('path/to/a')
        self.assertEqual(retrieval_service.ResolveArtifacts(beam_artifact_api_pb2.ResolveArtifactsRequest(artifacts=[dep_a])), beam_artifact_api_pb2.ResolveArtifactsResponse(replacements=[dep_a]))
        self.assertEqual(list(retrieval_service.GetArtifact(beam_artifact_api_pb2.GetArtifactRequest(artifact=dep_a))), [beam_artifact_api_pb2.GetArtifactResponse(data=b'a')])
        self.assertEqual(list(retrieval_service.GetArtifact(beam_artifact_api_pb2.GetArtifactRequest(artifact=self.file_artifact('path/to/b')))), [beam_artifact_api_pb2.GetArtifactResponse(data=b'b' * 10), beam_artifact_api_pb2.GetArtifactResponse(data=b'b' * 10), beam_artifact_api_pb2.GetArtifactResponse(data=b'b' * 10), beam_artifact_api_pb2.GetArtifactResponse(data=b'b' * 7)])

    def test_embedded_retrieval(self):
        if False:
            while True:
                i = 10
        retrieval_service = artifact_service.ArtifactRetrievalService(None)
        embedded_dep = self.embedded_artifact(b'some_data')
        self.assertEqual(list(retrieval_service.GetArtifact(beam_artifact_api_pb2.GetArtifactRequest(artifact=embedded_dep))), [beam_artifact_api_pb2.GetArtifactResponse(data=b'some_data')])

    def test_url_retrieval(self):
        if False:
            for i in range(10):
                print('nop')
        retrieval_service = artifact_service.ArtifactRetrievalService(None)
        url_dep = beam_runner_api_pb2.ArtifactInformation(type_urn=common_urns.artifact_types.URL.urn, type_payload=beam_runner_api_pb2.ArtifactUrlPayload(url='file:' + quote(__file__)).SerializeToString())
        content = b''.join([r.data for r in retrieval_service.GetArtifact(beam_artifact_api_pb2.GetArtifactRequest(artifact=url_dep))])
        with open(__file__, 'rb') as fin:
            self.assertEqual(content, fin.read())

    def test_push_artifacts(self):
        if False:
            for i in range(10):
                print('nop')
        unresolved = beam_runner_api_pb2.ArtifactInformation(type_urn='unresolved')
        resolved_a = self.embedded_artifact(data=b'a', name='a.txt')
        resolved_b = self.embedded_artifact(data=b'bb', name='b.txt')
        dep_big = self.embedded_artifact(data=b'big ' * 100, name='big.txt')

        class TestArtifacts(object):

            def ResolveArtifacts(self, request):
                if False:
                    return 10
                replacements = []
                for artifact in request.artifacts:
                    if artifact.type_urn == 'unresolved':
                        replacements += [resolved_a, resolved_b]
                    else:
                        replacements.append(artifact)
                return beam_artifact_api_pb2.ResolveArtifactsResponse(replacements=replacements)

            def GetArtifact(self, request):
                if False:
                    for i in range(10):
                        print('nop')
                if request.artifact.type_urn == common_urns.artifact_types.EMBEDDED.urn:
                    content = proto_utils.parse_Bytes(request.artifact.type_payload, beam_runner_api_pb2.EmbeddedFilePayload).data
                    for k in range(0, len(content), 13):
                        yield beam_artifact_api_pb2.GetArtifactResponse(data=content[k:k + 13])
                else:
                    raise NotImplementedError
        file_manager = InMemoryFileManager()
        server = artifact_service.ArtifactStagingService(file_manager.file_writer)
        server.register_job('staging_token', {'env': [unresolved, dep_big]})
        t = threading.Thread(target=lambda : artifact_service.offer_artifacts(server, TestArtifacts(), 'staging_token'))
        t.daemon = True
        t.start()
        resolved_deps = server.resolved_deps('staging_token', timeout=5)['env']
        expected = {'a.txt': b'a', 'b.txt': b'bb', 'big.txt': b'big ' * 100}
        for dep in resolved_deps:
            self.assertEqual(dep.type_urn, common_urns.artifact_types.FILE.urn)
            self.assertEqual(dep.role_urn, common_urns.artifact_roles.STAGING_TO.urn)
            type_payload = proto_utils.parse_Bytes(dep.type_payload, beam_runner_api_pb2.ArtifactFilePayload)
            role_payload = proto_utils.parse_Bytes(dep.role_payload, beam_runner_api_pb2.ArtifactStagingToRolePayload)
            self.assertTrue(type_payload.path.endswith(role_payload.staged_name), type_payload.path)
            self.assertEqual(file_manager.get(type_payload.path), expected.pop(role_payload.staged_name))
        self.assertEqual(expected, {})
if __name__ == '__main__':
    unittest.main()