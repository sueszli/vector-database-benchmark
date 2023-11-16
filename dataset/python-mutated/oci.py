import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPResponse
from typing import NamedTuple, Tuple
from urllib.request import Request
import llnl.util.tty as tty
import spack.binary_distribution
import spack.config
import spack.error
import spack.fetch_strategy
import spack.mirror
import spack.oci.opener
import spack.repo
import spack.spec
import spack.stage
import spack.traverse
import spack.util.crypto
from .image import Digest, ImageReference

class Blob(NamedTuple):
    compressed_digest: Digest
    uncompressed_digest: Digest
    size: int

def create_tarball(spec: spack.spec.Spec, tarfile_path):
    if False:
        print('Hello World!')
    buildinfo = spack.binary_distribution.get_buildinfo_dict(spec)
    return spack.binary_distribution._do_create_tarball(tarfile_path, spec.prefix, buildinfo)

def _log_upload_progress(digest: Digest, size: int, elapsed: float):
    if False:
        while True:
            i = 10
    elapsed = max(elapsed, 0.001)
    tty.info(f'Uploaded {digest} ({elapsed:.2f}s, {size / elapsed / 1024 / 1024:.2f} MB/s)')

def with_query_param(url: str, param: str, value: str) -> str:
    if False:
        while True:
            i = 10
    'Add a query parameter to a URL\n\n    Args:\n        url: The URL to add the parameter to.\n        param: The parameter name.\n        value: The parameter value.\n\n    Returns:\n        The URL with the parameter added.\n    '
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if param in query:
        query[param].append(value)
    else:
        query[param] = [value]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))

def upload_blob(ref: ImageReference, file: str, digest: Digest, force: bool=False, small_file_size: int=0, _urlopen: spack.oci.opener.MaybeOpen=None) -> bool:
    if False:
        while True:
            i = 10
    "Uploads a blob to an OCI registry\n\n    We only do monolithic uploads, even though it's very simple to do chunked.\n    Observed problems with chunked uploads:\n    (1) it's slow, many sequential requests, (2) some registries set an *unknown*\n    max chunk size, and the spec doesn't say how to obtain it\n\n    Args:\n        ref: The image reference.\n        file: The file to upload.\n        digest: The digest of the file.\n        force: Whether to force upload the blob, even if it already exists.\n        small_file_size: For files at most this size, attempt\n            to do a single POST request instead of POST + PUT.\n            Some registries do no support single requests, and others\n            do not specify what size they support in single POST.\n            For now this feature is disabled by default (0KB)\n\n    Returns:\n        True if the blob was uploaded, False if it already existed.\n    "
    _urlopen = _urlopen or spack.oci.opener.urlopen
    if not force and blob_exists(ref, digest, _urlopen):
        return False
    start = time.time()
    with open(file, 'rb') as f:
        file_size = os.fstat(f.fileno()).st_size
        if file_size <= small_file_size:
            request = Request(url=ref.uploads_url(digest), method='POST', data=f, headers={'Content-Type': 'application/octet-stream', 'Content-Length': str(file_size)})
        else:
            request = Request(url=ref.uploads_url(), method='POST', headers={'Content-Length': '0'})
        response = _urlopen(request)
        if response.status == 201:
            _log_upload_progress(digest, file_size, time.time() - start)
            return True
        spack.oci.opener.ensure_status(response, 202)
        assert 'Location' in response.headers
        upload_url = with_query_param(ref.endpoint(response.headers['Location']), 'digest', str(digest))
        f.seek(0)
        response = _urlopen(Request(url=upload_url, method='PUT', data=f, headers={'Content-Type': 'application/octet-stream', 'Content-Length': str(file_size)}))
        spack.oci.opener.ensure_status(response, 201)
    _log_upload_progress(digest, file_size, time.time() - start)
    return True

def upload_manifest(ref: ImageReference, oci_manifest: dict, tag: bool=True, _urlopen: spack.oci.opener.MaybeOpen=None):
    if False:
        i = 10
        return i + 15
    'Uploads a manifest/index to a registry\n\n    Args:\n        ref: The image reference.\n        oci_manifest: The OCI manifest or index.\n        tag: When true, use the tag, otherwise use the digest,\n            this is relevant for multi-arch images, where the\n            tag is an index, referencing the manifests by digest.\n\n    Returns:\n        The digest and size of the uploaded manifest.\n    '
    _urlopen = _urlopen or spack.oci.opener.urlopen
    data = json.dumps(oci_manifest, separators=(',', ':')).encode()
    digest = Digest.from_sha256(hashlib.sha256(data).hexdigest())
    size = len(data)
    if not tag:
        ref = ref.with_digest(digest)
    response = _urlopen(Request(url=ref.manifest_url(), method='PUT', data=data, headers={'Content-Type': oci_manifest['mediaType']}))
    spack.oci.opener.ensure_status(response, 201)
    return (digest, size)

def image_from_mirror(mirror: spack.mirror.Mirror) -> ImageReference:
    if False:
        for i in range(10):
            print('nop')
    'Given an OCI based mirror, extract the URL and image name from it'
    url = mirror.push_url
    if not url.startswith('oci://'):
        raise ValueError(f'Mirror {mirror} is not an OCI mirror')
    return ImageReference.from_string(url[6:])

def blob_exists(ref: ImageReference, digest: Digest, _urlopen: spack.oci.opener.MaybeOpen=None) -> bool:
    if False:
        print('Hello World!')
    'Checks if a blob exists in an OCI registry'
    try:
        _urlopen = _urlopen or spack.oci.opener.urlopen
        response = _urlopen(Request(url=ref.blob_url(digest), method='HEAD'))
        return response.status == 200
    except urllib.error.HTTPError as e:
        if e.getcode() == 404:
            return False
        raise

def copy_missing_layers(src: ImageReference, dst: ImageReference, architecture: str, _urlopen: spack.oci.opener.MaybeOpen=None) -> Tuple[dict, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Copy image layers from src to dst for given architecture.\n\n    Args:\n        src: The source image reference.\n        dst: The destination image reference.\n        architecture: The architecture (when referencing an index)\n\n    Returns:\n        Tuple of manifest and config of the base image.\n    '
    _urlopen = _urlopen or spack.oci.opener.urlopen
    (manifest, config) = get_manifest_and_config(src, architecture, _urlopen=_urlopen)
    digests = [Digest.from_string(layer['digest']) for layer in manifest['layers']]
    missing_digests = [digest for digest in digests if not blob_exists(dst, digest, _urlopen=_urlopen)]
    if not missing_digests:
        return (manifest, config)
    with spack.stage.StageComposite.from_iterable((make_stage(url=src.blob_url(digest), digest=digest, _urlopen=_urlopen) for digest in missing_digests)) as stages:
        stages.fetch()
        stages.check()
        stages.cache_local()
        for (stage, digest) in zip(stages, missing_digests):
            upload_blob(dst, file=stage.save_filename, force=True, digest=digest, _urlopen=_urlopen)
    return (manifest, config)
manifest_content_type = ['application/vnd.oci.image.manifest.v1+json', 'application/vnd.docker.distribution.manifest.v2+json']
index_content_type = ['application/vnd.oci.image.index.v1+json', 'application/vnd.docker.distribution.manifest.list.v2+json']
all_content_type = manifest_content_type + index_content_type

def get_manifest_and_config(ref: ImageReference, architecture='amd64', recurse=3, _urlopen: spack.oci.opener.MaybeOpen=None) -> Tuple[dict, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Recursively fetch manifest and config for a given image reference\n    with a given architecture.\n\n    Args:\n        ref: The image reference.\n        architecture: The architecture (when referencing an index)\n        recurse: How many levels of index to recurse into.\n\n    Returns:\n        A tuple of (manifest, config)'
    _urlopen = _urlopen or spack.oci.opener.urlopen
    response: HTTPResponse = _urlopen(Request(url=ref.manifest_url(), headers={'Accept': ', '.join(all_content_type)}))
    if response.headers['Content-Type'] in index_content_type:
        if recurse == 0:
            raise Exception('Maximum recursion depth reached while fetching OCI manifest')
        index = json.load(response)
        manifest_meta = next((manifest for manifest in index['manifests'] if manifest['platform']['architecture'] == architecture))
        return get_manifest_and_config(ref.with_digest(manifest_meta['digest']), architecture=architecture, recurse=recurse - 1, _urlopen=_urlopen)
    if response.headers['Content-Type'] not in manifest_content_type:
        raise Exception(f"Unknown content type {response.headers['Content-Type']}")
    manifest = json.load(response)
    config_digest = Digest.from_string(manifest['config']['digest'])
    with make_stage(ref.blob_url(config_digest), config_digest, _urlopen=_urlopen) as stage:
        stage.fetch()
        stage.check()
        stage.cache_local()
        with open(stage.save_filename, 'rb') as f:
            config = json.load(f)
    return (manifest, config)
upload_manifest_with_retry = spack.oci.opener.default_retry(upload_manifest)
upload_blob_with_retry = spack.oci.opener.default_retry(upload_blob)
get_manifest_and_config_with_retry = spack.oci.opener.default_retry(get_manifest_and_config)
copy_missing_layers_with_retry = spack.oci.opener.default_retry(copy_missing_layers)

def make_stage(url: str, digest: Digest, keep: bool=False, _urlopen: spack.oci.opener.MaybeOpen=None) -> spack.stage.Stage:
    if False:
        while True:
            i = 10
    _urlopen = _urlopen or spack.oci.opener.urlopen
    fetch_strategy = spack.fetch_strategy.OCIRegistryFetchStrategy(url, checksum=digest.digest, _urlopen=_urlopen)
    return spack.stage.Stage(fetch_strategy, mirror_paths=spack.mirror.OCIImageLayout(digest), name=digest.digest, keep=keep)