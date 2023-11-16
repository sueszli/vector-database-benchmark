"""A simple streaming movie server example"""
import hug

@hug.get(output=hug.output_format.mp4_video)
def watch():
    if False:
        for i in range(10):
            print('nop')
    'Watch an example movie, streamed directly to you from hug'
    return 'movie.mp4'