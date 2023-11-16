import os
import os.path as osp
import sys
from glob import glob
import imageio

def main():
    if False:
        print('Hello World!')
    assert len(sys.argv) >= 2
    d = sys.argv[1]
    fps = glob(osp.join(d, '*.jpg'))
    fps = sorted(fps, key=lambda x: int(x.split('/')[-1].replace('.jpg', '')))
    imgs = []
    for fp in fps:
        img = imageio.imread(fp)
        imgs.append(img)
    if len(sys.argv) >= 3:
        imageio.mimwrite(sys.argv[2], imgs, fps=24, macro_block_size=None)
    else:
        imageio.mimwrite(osp.basename(d) + '.mp4', imgs, fps=24, macro_block_size=None)
if __name__ == '__main__':
    main()