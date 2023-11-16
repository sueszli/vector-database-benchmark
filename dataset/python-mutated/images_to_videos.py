"""Converts temp directories of images to videos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--view_dirs', type=str, default='', help='Comma-separated list of temp view image directories.')
parser.add_argument('--vid_paths', type=str, default='', help='Comma-separated list of video output paths.')
parser.add_argument('--debug_path', type=str, default='', help='Output path to debug video.')
parser.add_argument('--debug_lhs_view', type=str, default='', help='Output path to debug video.')
parser.add_argument('--debug_rhs_view', type=str, default='', help='Output path to debug video.')

def create_vids(view_dirs, vid_paths, debug_path=None, debug_lhs_view=0, debug_rhs_view=1):
    if False:
        i = 10
        return i + 15
    'Creates one video per view per sequence.'
    for (view_dir, vidpath) in zip(view_dirs, vid_paths):
        encode_vid_cmd = 'mencoder mf://%s/*.png \\\n    -mf fps=29:type=png \\\n    -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell \\\n    -oac copy -o %s' % (view_dir, vidpath)
        os.system(encode_vid_cmd)
    if debug_path:
        lhs = vid_paths[int(debug_lhs_view)]
        rhs = vid_paths[int(debug_rhs_view)]
        os.system("avconv \\\n      -i %s \\\n      -i %s \\\n      -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \\\n      -map [vid] \\\n      -c:v libx264 \\\n      -crf 23 \\\n      -preset veryfast \\\n      %s" % (lhs, rhs, debug_path))

def main():
    if False:
        print('Hello World!')
    (FLAGS, _) = parser.parse_known_args()
    assert FLAGS.view_dirs
    assert FLAGS.vid_paths
    view_dirs = FLAGS.view_dirs.split(',')
    vid_paths = FLAGS.vid_paths.split(',')
    create_vids(view_dirs, vid_paths, FLAGS.debug_path, FLAGS.debug_lhs_view, FLAGS.debug_rhs_view)
    for i in view_dirs:
        shutil.rmtree(i)
if __name__ == '__main__':
    main()