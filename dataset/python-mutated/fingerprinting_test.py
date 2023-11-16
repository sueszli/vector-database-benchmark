"""Tests for SavedModel fingerprinting.

These tests verify that fingerprint is written correctly and that APIs for
reading it are correct.
"""
import os
import shutil
from tensorflow.core.config import flags
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.trackable import autotrackable

class FingerprintingTest(test.TestCase):

    def _create_saved_model(self):
        if False:
            for i in range(10):
                print('nop')
        root = autotrackable.AutoTrackable()
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save.save(root, save_dir)
        self.addCleanup(shutil.rmtree, save_dir)
        return save_dir

    def _create_model_with_function(self):
        if False:
            while True:
                i = 10
        root = autotrackable.AutoTrackable()
        root.f = def_function.function(lambda x: 2.0 * x)
        return root

    def _create_model_with_input_signature(self):
        if False:
            print('Hello World!')
        root = autotrackable.AutoTrackable()
        root.f = def_function.function(lambda x: 2.0 * x, input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
        return root

    def _create_model_with_data(self):
        if False:
            return 10
        root = autotrackable.AutoTrackable()
        root.x = constant_op.constant(1.0, dtype=dtypes.float32)
        root.f = def_function.function(lambda x: root.x * x, input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
        return root

    def _read_fingerprint(self, filename):
        if False:
            for i in range(10):
                print('nop')
        fingerprint_def = fingerprint_pb2.FingerprintDef()
        with file_io.FileIO(filename, 'rb') as f:
            fingerprint_def.ParseFromString(f.read())
        return fingerprint_def

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        flags.config().saved_model_fingerprinting.reset(True)

    def test_basic_module(self):
        if False:
            for i in range(10):
                print('nop')
        save_dir = self._create_saved_model()
        files = file_io.list_directory_v2(save_dir)
        self.assertLen(files, 4)
        self.assertIn(constants.FINGERPRINT_FILENAME, files)
        fingerprint_def = self._read_fingerprint(file_io.join(save_dir, constants.FINGERPRINT_FILENAME))
        self.assertGreater(fingerprint_def.saved_model_checksum, 0)
        self.assertEqual(fingerprint_def.graph_def_program_hash, 14830488309055091319)
        self.assertEqual(fingerprint_def.signature_def_hash, 12089566276354592893)
        self.assertEqual(fingerprint_def.saved_object_graph_hash, 0)
        self.assertGreater(fingerprint_def.checkpoint_hash, 0)

    def test_model_saved_with_different_signature_options(self):
        if False:
            print('Hello World!')
        model = self._create_model_with_function()
        sig_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save.save(model, sig_dir, signatures=model.f.get_concrete_function(tensor_spec.TensorSpec(None, dtypes.float32)))
        no_sig_dir = os.path.join(self.get_temp_dir(), 'saved_model2')
        save.save(model, no_sig_dir)
        input_sig_dir = os.path.join(self.get_temp_dir(), 'saved_model3')
        save.save(self._create_model_with_input_signature(), input_sig_dir)
        fingerprint_sig = self._read_fingerprint(file_io.join(sig_dir, constants.FINGERPRINT_FILENAME))
        fingerprint_no_sig = self._read_fingerprint(file_io.join(no_sig_dir, constants.FINGERPRINT_FILENAME))
        fingerprint_input_sig = self._read_fingerprint(file_io.join(input_sig_dir, constants.FINGERPRINT_FILENAME))
        self.assertNotEqual(fingerprint_sig.signature_def_hash, fingerprint_no_sig.signature_def_hash)
        self.assertEqual(fingerprint_sig.graph_def_program_hash, fingerprint_input_sig.graph_def_program_hash)
        self.assertEqual(fingerprint_sig.signature_def_hash, fingerprint_input_sig.signature_def_hash)

    def test_read_fingerprint_api(self):
        if False:
            while True:
                i = 10
        save_dir = self._create_saved_model()
        fingerprint = fingerprinting.read_fingerprint(save_dir)
        fingerprint_def = self._read_fingerprint(file_io.join(save_dir, constants.FINGERPRINT_FILENAME))
        self.assertEqual(fingerprint, fingerprint_def)

    def test_read_fingerprint_file_not_found(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(FileNotFoundError, 'SavedModel Fingerprint Error'):
            fingerprinting.read_fingerprint('foo')

    def test_write_fingerprint(self):
        if False:
            for i in range(10):
                print('nop')
        save_dir = os.path.join(self.get_temp_dir(), 'model_and_fingerprint')
        save.save_and_return_nodes(self._create_model_with_data(), save_dir, experimental_skip_checkpoint=True)
        fingerprint_def = fingerprinting.read_fingerprint(save_dir)
        self.assertGreater(fingerprint_def.saved_model_checksum, 0)
        self.assertEqual(fingerprint_def.graph_def_program_hash, 8947653168630125217)
        self.assertEqual(fingerprint_def.signature_def_hash, 15354238402988963670)
        self.assertEqual(fingerprint_def.checkpoint_hash, 0)

    def test_valid_singleprint(self):
        if False:
            while True:
                i = 10
        save_dir = os.path.join(self.get_temp_dir(), 'singleprint_model')
        save.save(self._create_model_with_data(), save_dir)
        fingerprint = fingerprinting.read_fingerprint(save_dir)
        singleprint = fingerprint.singleprint()
        self.assertRegex(singleprint, '/'.join(['8947653168630125217', '15354238402988963670', '1613952301283913051']))

    def test_invalid_singleprint(self):
        if False:
            while True:
                i = 10
        fingerprint = fingerprinting.Fingerprint()
        with self.assertRaisesRegex(ValueError, 'Encounted invalid fingerprint values'):
            fingerprint.singleprint()

    def test_valid_from_proto(self):
        if False:
            i = 10
            return i + 15
        save_dir = os.path.join(self.get_temp_dir(), 'from_proto_model')
        save.save(self._create_model_with_data(), save_dir)
        fingerprint_def = fingerprint_pb2.FingerprintDef().FromString(fingerprinting_pywrap.ReadSavedModelFingerprint(save_dir))
        fingerprint = fingerprinting.Fingerprint.from_proto(fingerprint_def)
        self.assertEqual(fingerprint, fingerprint_def)

    def test_invalid_from_proto(self):
        if False:
            return 10
        save_dir = os.path.join(self.get_temp_dir(), 'from_proto_model')
        save.save(self._create_model_with_data(), save_dir)
        wrong_def = saved_model_pb2.SavedModel(saved_model_schema_version=1)
        with self.assertRaisesRegex(ValueError, 'Given proto could not be deserialized as'):
            fingerprinting.Fingerprint.from_proto(wrong_def)

    def test_fingerprint_to_proto(self):
        if False:
            while True:
                i = 10
        save_dir = os.path.join(self.get_temp_dir(), 'from_proto_model')
        save.save(self._create_model_with_data(), save_dir)
        fingerprint = fingerprinting.read_fingerprint(save_dir)
        fingerprint_def = fingerprinting_utils.to_proto(fingerprint)
        self.assertEqual(fingerprint, fingerprint_def)
if __name__ == '__main__':
    test.main()