from receptivefield.pytorch import PytorchReceptiveField
import numpy as np

def _get_rf(model, sample_pil_img):
    if False:
        return 10

    def model_fn():
        if False:
            i = 10
            return i + 15
        model.eval()
        return model
    input_shape = np.array(sample_pil_img).shape
    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape=input_shape)
    return (rf, rf_params)

def plot_receptive_field(model, sample_pil_img, layout=(2, 2), figsize=(6, 6)):
    if False:
        for i in range(10):
            print('nop')
    (rf, rf_params) = _get_rf(model, sample_pil_img)
    return rf.plot_rf_grids(custom_image=sample_pil_img, figsize=figsize, layout=layout)

def plot_grads_at(model, sample_pil_img, feature_map_index=0, point=(8, 8), figsize=(6, 6)):
    if False:
        i = 10
        return i + 15
    (rf, rf_params) = _get_rf(model, sample_pil_img)
    return rf.plot_gradient_at(fm_id=feature_map_index, point=point, image=None, figsize=figsize)