"""
Description of the video:
Mimic of Star Wars' opening title. A text with a (false)
perspective effect goes towards the end of space, on a
background made of stars. Slight fading effect on the text.
"""
import numpy as np
from skimage import transform as tf
from moviepy import *
from moviepy.video.tools.drawing import color_gradient
w = 720
h = w * 9 // 16
moviesize = (w, h)
txt = '\n'.join(['A long time ago, in a faraway galaxy,', 'there lived a prince and a princess', 'who had never seen the stars, for they', 'lived deep underground.', '', "Many years before, the prince's", 'grandfather had ventured out to the', 'surface and had been burnt to ashes by', 'solar winds.', '', 'One day, as the princess was coding', 'and the prince was shopping online, a', 'meteor landed just a few megameters', "from the couple's flat."])
txt = 10 * '\n' + txt + 10 * '\n'
clip_txt = TextClip(txt, color='white', align='West', font_size=25, font='Xolonium-Bold', method='label')
txt_speed = 27
fl = lambda gf, t: gf(t)[int(txt_speed * t):int(txt_speed * t) + h, :]
moving_txt = clip_txt.transform(fl, apply_to=['mask'])
grad = color_gradient(moving_txt.size, p1=(0, 2 * h / 3), p2=(0, h / 4), color_1=0.0, color_2=1.0)
gradmask = ImageClip(grad, is_mask=True)
fl = lambda pic: np.minimum(pic, gradmask.img)
moving_txt.mask = moving_txt.mask.image_transform(fl)

def trapzWarp(pic, cx, cy, is_mask=False):
    if False:
        return 10
    'Complicated function (will be latex packaged as a fx).'
    (Y, X) = pic.shape[:2]
    src = np.array([[0, 0], [X, 0], [X, Y], [0, Y]])
    dst = np.array([[cx * X, cy * Y], [(1 - cx) * X, cy * Y], [X, Y], [0, Y]])
    tform = tf.ProjectiveTransform()
    tform.estimate(src, dst)
    im = tf.warp(pic, tform.inverse, output_shape=(Y, X))
    return im if is_mask else (im * 255).astype('uint8')
fl_im = lambda pic: trapzWarp(pic, 0.2, 0.3)
fl_mask = lambda pic: trapzWarp(pic, 0.2, 0.3, is_mask=True)
warped_txt = moving_txt.image_transform(fl_im)
warped_txt.mask = warped_txt.mask.image_transform(fl_mask)
stars = ImageClip('../../videos/stars.jpg')
stars_darkened = stars.image_transform(lambda pic: (0.6 * pic).astype('int16'))
final = CompositeVideoClip([stars_darkened, warped_txt.set_pos(('center', 'bottom'))], size=moviesize)
final.with_duration(8).write_videofile('starworms.avi', fps=5)
'=====================================================================\n\n    CODE FOR THE VIDEO TUTORIAL\n\n  We will now code the video tutorial for this video.\n  When you think about it, it is a code for a video explaining how to\n  make another video using some code (this is so meta!).\n  This code uses the variables of the previous code (it should be placed\n  after that previous code to work).\n\n====================================================================='

def annotate(clip, txt, txt_color='white', bg_color=(0, 0, 255)):
    if False:
        print('Hello World!')
    'Writes a text at the bottom of the clip.'
    txtclip = TextClip(txt, font_size=20, font='Ubuntu-bold', color=txt_color)
    txtclip = txtclip.on_color((clip.w, txtclip.h + 6), color=(0, 0, 255), pos=(6, 'center'))
    cvc = CompositeVideoClip([clip, txtclip.set_pos((0, 'bottom'))])
    return cvc.with_duration(clip.duration)

def resizeCenter(clip):
    if False:
        i = 10
        return i + 15
    return clip.resize(height=h).set_pos('center')

def composeCenter(clip):
    if False:
        while True:
            i = 10
    return CompositeVideoClip([clip.set_pos('center')], size=moviesize)
annotated_clips = [annotate(clip, text) for (clip, text) in [(composeCenter(resizeCenter(stars)).subclip(0, 3), 'This is a public domain picture of stars'), (CompositeVideoClip([stars], moviesize).subclip(0, 3), 'We only keep one part.'), (CompositeVideoClip([stars_darkened], moviesize).subclip(0, 3), 'We darken it a little.'), (composeCenter(resizeCenter(clip_txt)).subclip(0, 3), 'We generate a text image.'), (composeCenter(moving_txt.with_mask(None)).subclip(6, 9), 'We scroll the text by cropping a moving region of it.'), (composeCenter(gradmask.to_RGB()).subclip(0, 2), 'We add this mask to the clip.'), (composeCenter(moving_txt).subclip(6, 9), 'Here is the result'), (composeCenter(warped_txt).subclip(6, 9), 'We now warp this clip in a trapezoid.'), (final.subclip(6, 9), 'We finally superimpose with the stars.')]]
concatenate_videoclips(annotated_clips).write_videofile('tutorial.avi', fps=5)