import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_decomposition(people, pca):
    if False:
        print('Hello World!')
    image_shape = people.images[0].shape
    plt.figure(figsize=(20, 3))
    ax = plt.gca()
    imagebox = OffsetImage(people.images[0], zoom=1.5, cmap='gray')
    ab = AnnotationBbox(imagebox, (0.05, 0.4), pad=0.0, xycoords='data')
    ax.add_artist(ab)
    for i in range(4):
        imagebox = OffsetImage(pca.components_[i].reshape(image_shape), zoom=1.5, cmap='viridis')
        ab = AnnotationBbox(imagebox, (0.3 + 0.2 * i, 0.4), pad=0.0, xycoords='data')
        ax.add_artist(ab)
        if i == 0:
            plt.text(0.18, 0.25, 'x_%d *' % i, fontdict={'fontsize': 50})
        else:
            plt.text(0.15 + 0.2 * i, 0.25, '+ x_%d *' % i, fontdict={'fontsize': 50})
    plt.text(0.95, 0.25, '+ ...', fontdict={'fontsize': 50})
    plt.rc('text', usetex=True)
    plt.text(0.13, 0.3, '\\approx', fontdict={'fontsize': 50})
    plt.axis('off')