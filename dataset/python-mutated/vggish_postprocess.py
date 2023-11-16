"""Post-process embeddings from VGGish."""
import numpy as np
import vggish_params

class Postprocessor(object):
    """Post-processes VGGish embeddings.

  The initial release of AudioSet included 128-D VGGish embeddings for each
  segment of AudioSet. These released embeddings were produced by applying
  a PCA transformation (technically, a whitening transform is included as well)
  and 8-bit quantization to the raw embedding output from VGGish, in order to
  stay compatible with the YouTube-8M project which provides visual embeddings
  in the same format for a large set of YouTube videos. This class implements
  the same PCA (with whitening) and quantization transformations.
  """

    def __init__(self, pca_params_npz_path):
        if False:
            while True:
                i = 10
        'Constructs a postprocessor.\n\n    Args:\n      pca_params_npz_path: Path to a NumPy-format .npz file that\n        contains the PCA parameters used in postprocessing.\n    '
        params = np.load(pca_params_npz_path)
        self._pca_matrix = params[vggish_params.PCA_EIGEN_VECTORS_NAME]
        self._pca_means = params[vggish_params.PCA_MEANS_NAME].reshape(-1, 1)
        assert self._pca_matrix.shape == (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE), 'Bad PCA matrix shape: %r' % (self._pca_matrix.shape,)
        assert self._pca_means.shape == (vggish_params.EMBEDDING_SIZE, 1), 'Bad PCA means shape: %r' % (self._pca_means.shape,)

    def postprocess(self, embeddings_batch):
        if False:
            while True:
                i = 10
        'Applies postprocessing to a batch of embeddings.\n\n    Args:\n      embeddings_batch: An nparray of shape [batch_size, embedding_size]\n        containing output from the embedding layer of VGGish.\n\n    Returns:\n      An nparray of the same shape as the input but of type uint8,\n      containing the PCA-transformed and quantized version of the input.\n    '
        assert len(embeddings_batch.shape) == 2, 'Expected 2-d batch, got %r' % (embeddings_batch.shape,)
        assert embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE, 'Bad batch shape: %r' % (embeddings_batch.shape,)
        pca_applied = np.dot(self._pca_matrix, embeddings_batch.T - self._pca_means).T
        clipped_embeddings = np.clip(pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL)
        quantized_embeddings = (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL) * (255.0 / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL))
        quantized_embeddings = quantized_embeddings.astype(np.uint8)
        return quantized_embeddings