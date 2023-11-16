from utils_cv.segmentation.plot import plot_image_and_mask, plot_segmentation, plot_mask_stats, plot_confusion_matrix

def test_plot_image_and_mask(seg_im_and_mask):
    if False:
        i = 10
        return i + 15
    plot_image_and_mask(seg_im_and_mask[0], seg_im_and_mask[1])

def test_plot_segmentation(seg_im_and_mask, seg_prediction):
    if False:
        i = 10
        return i + 15
    (mask, scores) = seg_prediction
    plot_segmentation(seg_im_and_mask[0], mask, scores)

def test_plot_mask_stats(tiny_seg_databunch, seg_classes):
    if False:
        return 10
    plot_mask_stats(tiny_seg_databunch, seg_classes)
    plot_mask_stats(tiny_seg_databunch, seg_classes, exclude_classes=['background'])