from .base_mpl_plot import BaseMplPlot
from .. import utils, image_utils
import numpy as np

class ImagePlot(BaseMplPlot):

    def init_stream_plot(self, stream_vis, rows=2, cols=5, img_width=None, img_height=None, img_channels=None, colormap=None, viz_img_scale=None, **stream_vis_args):
        if False:
            return 10
        (stream_vis.rows, stream_vis.cols) = (rows, cols)
        (stream_vis.img_channels, stream_vis.colormap) = (img_channels, colormap)
        (stream_vis.img_width, stream_vis.img_height) = (img_width, img_height)
        stream_vis.viz_img_scale = viz_img_scale
        stream_vis.axs = [[None for _ in range(cols)] for _ in range(rows)]
        stream_vis.ax_imgs = [[None for _ in range(cols)] for _ in range(rows)]

    def clear_plot(self, stream_vis, clear_history):
        if False:
            i = 10
            return i + 15
        for row in range(stream_vis.rows):
            for col in range(stream_vis.cols):
                img = stream_vis.ax_imgs[row][col]
                if img:
                    (x, y) = img.get_size()
                    img.set_data(np.zeros((x, y)))

    def _show_stream_items(self, stream_vis, stream_items):
        if False:
            while True:
                i = 10
        'Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.\n        '
        stream_item = None
        for er in reversed(stream_items):
            if not (er.ended or er.value is None):
                stream_item = er
                break
        if stream_item is None:
            return True
        (row, col, i) = (0, 0, 0)
        dirty = False
        for image_list in stream_item.value:
            images = [image_utils.to_imshow_array(img, stream_vis.img_width, stream_vis.img_height) for img in image_list.images if img is not None]
            img_viz = image_utils.stitch_horizontal(images, width_dim=1)
            if stream_vis.viz_img_scale is not None:
                import skimage.transform
                if isinstance(img_viz, np.ndarray) and np.issubdtype(img_viz.dtype, np.floating):
                    img_viz = img_viz.clip(-1, 1)
                img_viz = skimage.transform.rescale(img_viz, (stream_vis.viz_img_scale, stream_vis.viz_img_scale), mode='reflect', preserve_range=False)
            ax = stream_vis.axs[row][col]
            if ax is None:
                ax = stream_vis.axs[row][col] = self.figure.add_subplot(stream_vis.rows, stream_vis.cols, i + 1)
                ax.set_xticks([])
                ax.set_yticks([])
            cmap = image_list.cmap or ('Greys' if stream_vis.colormap is None and len(img_viz.shape) == 2 else stream_vis.colormap)
            stream_vis.ax_imgs[row][col] = ax.imshow(img_viz, interpolation='none', cmap=cmap, alpha=image_list.alpha)
            dirty = True
            title = image_list.title
            if len(title) > 12:
                title = utils.wrap_string(title) if len(title) > 24 else title
                fontsize = 8
            else:
                fontsize = 12
            ax.set_title(title, fontsize=fontsize)
            col = col + 1
            if col >= stream_vis.cols:
                col = 0
                row = row + 1
                if row >= stream_vis.rows:
                    break
            i += 1
        return not dirty

    def has_legend(self):
        if False:
            print('Hello World!')
        return self.show_legend or False