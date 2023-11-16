from typing import List, Optional, Tuple, Union
from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core import _Augmentation, signed_bin
from nvidia.dali.auto_aug.core._args import forbid_unused_kwargs as _forbid_unused_kwargs
from nvidia.dali.auto_aug.core._utils import get_translations as _get_translations, pretty_select as _pretty_select
from nvidia.dali.data_node import DataNode as _DataNode

def trivial_augment_wide(data: _DataNode, num_magnitude_bins: int=31, shape: Optional[Union[_DataNode, Tuple[int, int]]]=None, fill_value: Optional[int]=128, interp_type: Optional[types.DALIInterpType]=None, max_translate_abs: Optional[int]=None, max_translate_rel: Optional[float]=None, seed: Optional[int]=None, excluded: Optional[List[str]]=None) -> _DataNode:
    if False:
        print('Hello World!')
    '\n    Applies TrivialAugment Wide (https://arxiv.org/abs/2103.10158) augmentation scheme to the\n    provided batch of samples.\n\n    Args\n    ----\n    data : DataNode\n        A batch of samples to be processed. The supported samples are images\n        of `HWC` layout and videos of `FHWC` layout, the supported data type is `uint8`.\n    num_magnitude_bins: int, optional\n        The number of bins to divide the magnitude ranges into.\n    fill_value: int, optional\n        A value to be used as a padding for images/frames transformed with warp_affine ops\n        (translation, shear and rotate). If `None` is specified, the images/frames are padded\n        with the border value repeated (clamped).\n    interp_type: types.DALIInterpType, optional\n        Interpolation method used by the warp_affine ops (translation, shear and rotate).\n        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.\n    max_translate_abs: int or (int, int), optional\n        Only valid when ``shapes`` is not provided. Specifies the maximal shift (in pixels)\n        in the translation augmentation. If a tuple is specified, the first component limits\n        height, the second the width. Defaults to 32, which means the maximal magnitude\n        shifts the image by 32 pixels.\n    max_translate_rel: float or (float, float), optional\n        Only valid when ``shapes`` argument is provided. Specifies the maximal shift as a\n        fraction of image shape in the translation augmentations.\n        If a tuple is specified, the first component limits the height, the second the width.\n        Defaults to 1, which means the maximal magnitude shifts the image entirely out of\n        the canvas.\n    seed: int, optional\n        Seed to be used to randomly sample operations (and to negate magnitudes).\n    excluded: List[str], optional\n        A list of names of the operations to be excluded from the default suite of augmentations.\n        If, instead of just limiting the set of operations, you need to include some custom\n        operations or fine-tuned of the existing ones, you can use the\n        :meth:`~nvidia.dali.auto_aug.trivial_augment.apply_trivial_augment` directly,\n        which accepts a list of augmentations.\n\n    Returns\n    -------\n    DataNode\n        A batch of transformed samples.\n    '
    aug_kwargs = {'fill_value': fill_value, 'interp_type': interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs['shape'] = shape
    augmentations = get_trivial_augment_wide_suite(use_shape=use_shape, max_translate_abs=max_translate_abs, max_translate_rel=max_translate_rel)
    augmentation_names = set((aug.name for aug in augmentations))
    assert len(augmentation_names) == len(augmentations)
    excluded = excluded or []
    for name in excluded:
        if name not in augmentation_names:
            raise Exception(f"The `{name}` was specified in `excluded`, but the TrivialAugmentWide suite does not contain augmentation with this name. The augmentations in the suite are: {', '.join(augmentation_names)}.")
    selected_augments = [aug for aug in augmentations if aug.name not in excluded]
    return apply_trivial_augment(selected_augments, data, num_magnitude_bins=num_magnitude_bins, seed=seed, **aug_kwargs)

def apply_trivial_augment(augmentations: List[_Augmentation], data: _DataNode, num_magnitude_bins: int=31, seed: Optional[int]=None, **kwargs) -> _DataNode:
    if False:
        return 10
    '\n    Applies the list of `augmentations` in TrivialAugment\n    (https://arxiv.org/abs/2103.10158) fashion.\n    Each sample is processed with randomly selected transformation form `augmentations` list.\n    The magnitude bin for every transformation is randomly selected from\n    `[0, num_magnitude_bins - 1]`.\n\n    Args\n    ----\n    augmentations : List[core._Augmentation]\n        List of augmentations to be sampled and applied in TrivialAugment fashion.\n    data : DataNode\n        A batch of samples to be processed.\n    num_magnitude_bins: int, optional\n        The number of bins to divide the magnitude ranges into.\n    seed: int, optional\n        Seed to be used to randomly sample operations (and to negate magnitudes).\n    kwargs:\n        Any extra parameters to be passed when calling `augmentations`.\n        The signature of each augmentation is checked for any extra arguments and if\n        the name of the argument matches one from the `kwargs`, the value is\n        passed as an argument. For example, some augmentations from the default\n        TrivialAugment suite accept ``shapes``, ``fill_value`` and ``interp_type``.\n\n    Returns\n    -------\n    DataNode\n        A batch of transformed samples.\n    '
    if not isinstance(num_magnitude_bins, int) or num_magnitude_bins < 1:
        raise Exception(f'The `num_magnitude_bins` must be a positive integer, got {num_magnitude_bins}.')
    if len(augmentations) == 0:
        raise Exception('The `augmentations` list cannot be empty. Got empty list in `apply_trivial_augment` call.')
    magnitude_bin = fn.random.uniform(values=list(range(num_magnitude_bins)), dtype=types.INT32, seed=seed)
    use_signed_magnitudes = any((aug.randomly_negate for aug in augmentations))
    if use_signed_magnitudes:
        magnitude_bin = signed_bin(magnitude_bin, seed=seed)
    _forbid_unused_kwargs(augmentations, kwargs, 'apply_trivial_augment')
    op_kwargs = dict(data=data, magnitude_bin=magnitude_bin, num_magnitude_bins=num_magnitude_bins, **kwargs)
    op_idx = fn.random.uniform(values=list(range(len(augmentations))), seed=seed, dtype=types.INT32)
    return _pretty_select(augmentations, op_idx, op_kwargs, auto_aug_name='apply_trivial_augment', ref_suite_name='get_trivial_augment_wide_suite')

def get_trivial_augment_wide_suite(use_shape: bool=False, max_translate_abs: Optional[int]=None, max_translate_rel: Optional[float]=None) -> List[_Augmentation]:
    if False:
        i = 10
        return i + 15
    '\n    Creates a list of 14 augmentations referred as wide augmentation space in TrivialAugment paper\n    (https://arxiv.org/abs/2103.10158).\n\n    Args\n    ----\n    use_shape : bool\n        If true, the translation offset is computed as a percentage of the image/frame shape.\n        Useful if the samples processed with the auto augment have different shapes.\n        If false, the offsets range is bounded by a constant (`max_translate_abs`).\n    max_translate_abs: int or (int, int), optional\n        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation\n        augmentations. If a tuple is specified, the first component limits height, the second the\n        width. Defaults to 32.\n    max_translate_rel: float or (float, float), optional\n        Only valid with use_shape=True, specifies the maximal shift as a fraction of image/frame\n        shape in the translation augmentations. If a tuple is specified, the first component limits\n        height, the second the width. Defaults to 1.\n    '
    (default_translate_abs, default_translate_rel) = (32, 1.0)
    translations = _get_translations(use_shape, default_translate_abs, default_translate_rel, max_translate_abs, max_translate_rel)
    return translations + [a.shear_x.augmentation((0, 0.99), True), a.shear_y.augmentation((0, 0.99), True), a.rotate.augmentation((0, 135), True), a.brightness.augmentation((0.01, 0.99), True, a.shift_enhance_range), a.contrast.augmentation((0.01, 0.99), True, a.shift_enhance_range), a.color.augmentation((0.01, 0.99), True, a.shift_enhance_range), a.sharpness.augmentation((0.01, 0.99), True, a.sharpness_kernel), a.posterize.augmentation((8, 2), False, a.poster_mask_uint8), a.solarize.augmentation((256, 0)), a.equalize, a.auto_contrast, a.identity]