"""Methods for SavedModel fingerprinting.

This module contains classes and functions for reading the SavedModel
fingerprint.
"""
from typing import Any
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.util.tf_export import tf_export

@tf_export('saved_model.experimental.Fingerprint', v1=[])
class Fingerprint:
    """The SavedModel fingerprint.

  Each attribute of this class is named after a field name in the
  FingerprintDef proto and contains the value of the respective field in the
  protobuf.

  Attributes:
    saved_model_checksum: A uint64 containing the `saved_model_checksum`.
    graph_def_program_hash: A uint64 containing `graph_def_program_hash`.
    signature_def_hash: A uint64 containing the `signature_def_hash`.
    saved_object_graph_hash: A uint64 containing the `saved_object_graph_hash`.
    checkpoint_hash: A uint64 containing the`checkpoint_hash`.
    version: An int32 containing the producer field of the VersionDef.
  """

    def __init__(self, saved_model_checksum: int=None, graph_def_program_hash: int=None, signature_def_hash: int=None, saved_object_graph_hash: int=None, checkpoint_hash: int=None, version: int=None):
        if False:
            i = 10
            return i + 15
        'Initializes the instance based on values in the SavedModel fingerprint.\n\n    Args:\n      saved_model_checksum: Value of the`saved_model_checksum`.\n      graph_def_program_hash: Value of the `graph_def_program_hash`.\n      signature_def_hash: Value of the `signature_def_hash`.\n      saved_object_graph_hash: Value of the `saved_object_graph_hash`.\n      checkpoint_hash: Value of the `checkpoint_hash`.\n      version: Value of the producer field of the VersionDef.\n    '
        self.saved_model_checksum = saved_model_checksum
        self.graph_def_program_hash = graph_def_program_hash
        self.signature_def_hash = signature_def_hash
        self.saved_object_graph_hash = saved_object_graph_hash
        self.checkpoint_hash = checkpoint_hash
        self.version = version

    @classmethod
    def from_proto(cls, proto: fingerprint_pb2.FingerprintDef) -> 'Fingerprint':
        if False:
            for i in range(10):
                print('nop')
        'Constructs Fingerprint object from protocol buffer message.'
        if isinstance(proto, bytes):
            proto = fingerprint_pb2.FingerprintDef.FromString(proto)
        try:
            return Fingerprint(proto.saved_model_checksum, proto.graph_def_program_hash, proto.signature_def_hash, proto.saved_object_graph_hash, proto.checkpoint_hash, proto.version)
        except AttributeError as e:
            raise ValueError(f'Given proto could not be deserialized as fingerprint.{e}') from None

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, Fingerprint) or isinstance(other, fingerprint_pb2.FingerprintDef):
            try:
                return self.saved_model_checksum == other.saved_model_checksum and self.graph_def_program_hash == other.graph_def_program_hash and (self.signature_def_hash == other.signature_def_hash) and (self.saved_object_graph_hash == other.saved_object_graph_hash) and (self.checkpoint_hash == other.checkpoint_hash)
            except AttributeError:
                pass
        return False

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(['SavedModel Fingerprint', f'  saved_model_checksum: {self.saved_model_checksum}', f'  graph_def_program_hash: {self.graph_def_program_hash}', f'  signature_def_hash: {self.signature_def_hash}', f'  saved_object_graph_hash: {self.saved_object_graph_hash}', f'  checkpoint_hash: {self.checkpoint_hash}'])

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'Fingerprint({self.saved_model_checksum}, {self.graph_def_program_hash}, {self.signature_def_hash}, {self.saved_object_graph_hash}, {self.checkpoint_hash})'

    def singleprint(self) -> fingerprinting_pywrap.Singleprint:
        if False:
            i = 10
            return i + 15
        "Canonical fingerprinting ID for a SavedModel.\n\n    Uniquely identifies a SavedModel based on the regularized fingerprint\n    attributes. (saved_model_checksum is sensitive to immaterial changes and\n    thus non-deterministic.)\n\n    Returns:\n      The string concatenation of `graph_def_program_hash`,\n      `signature_def_hash`, `saved_object_graph_hash`, and `checkpoint_hash`\n      fingerprint attributes (separated by '/').\n\n    Raises:\n      ValueError: If the fingerprint fields cannot be used to construct the\n      singleprint.\n    "
        try:
            return fingerprinting_pywrap.Singleprint(self.graph_def_program_hash, self.signature_def_hash, self.saved_object_graph_hash, self.checkpoint_hash)
        except (TypeError, fingerprinting_pywrap.FingerprintException) as e:
            raise ValueError(f'Encounted invalid fingerprint values when constructing singleprint.graph_def_program_hash: {self.graph_def_program_hash}signature_def_hash: {self.signature_def_hash}saved_object_graph_hash: {self.saved_object_graph_hash}checkpoint_hash: {self.checkpoint_hash}{e}') from None

@tf_export('saved_model.experimental.read_fingerprint', v1=[])
def read_fingerprint(export_dir: str) -> Fingerprint:
    if False:
        print('Hello World!')
    'Reads the fingerprint of a SavedModel in `export_dir`.\n\n  Returns a `tf.saved_model.experimental.Fingerprint` object that contains\n  the values of the SavedModel fingerprint, which is persisted on disk in the\n  `fingerprint.pb` file in the `export_dir`.\n\n  Read more about fingerprints in the SavedModel guide at\n  https://www.tensorflow.org/guide/saved_model.\n\n  Args:\n    export_dir: The directory that contains the SavedModel.\n\n  Returns:\n    A `tf.saved_model.experimental.Fingerprint`.\n\n  Raises:\n    FileNotFoundError: If no or an invalid fingerprint is found.\n  '
    try:
        fingerprint = fingerprinting_pywrap.ReadSavedModelFingerprint(export_dir)
    except fingerprinting_pywrap.FileNotFoundException as e:
        raise FileNotFoundError(f'SavedModel Fingerprint Error: {e}') from None
    except fingerprinting_pywrap.FingerprintException as e:
        raise RuntimeError(f'SavedModel Fingerprint Error: {e}') from None
    return Fingerprint.from_proto(fingerprint_pb2.FingerprintDef().FromString(fingerprint))