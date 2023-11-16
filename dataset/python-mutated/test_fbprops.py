from panda3d.core import FrameBufferProperties

def test_fbprops_copy_ctor():
    if False:
        for i in range(10):
            print('nop')
    default = FrameBufferProperties.get_default()
    fbprops = FrameBufferProperties(default)
    assert fbprops == default

def test_fbquality_depth():
    if False:
        for i in range(10):
            print('nop')
    req = FrameBufferProperties()
    req.depth_bits = 1
    fb_d16s8 = FrameBufferProperties()
    fb_d16s8.rgb_color = True
    fb_d16s8.depth_bits = 16
    fb_d16s8.stencil_bits = 8
    fb_d24s8 = FrameBufferProperties()
    fb_d24s8.rgb_color = True
    fb_d24s8.depth_bits = 24
    fb_d24s8.stencil_bits = 8
    fb_d32 = FrameBufferProperties()
    fb_d32.rgb_color = True
    fb_d32.depth_bits = 32
    fb_d32s8 = FrameBufferProperties()
    fb_d32s8.rgb_color = True
    fb_d32s8.depth_bits = 32
    fb_d32s8.stencil_bits = 8
    assert fb_d16s8.get_quality(req) < fb_d24s8.get_quality(req)
    assert fb_d16s8.get_quality(req) < fb_d32.get_quality(req)
    assert fb_d16s8.get_quality(req) < fb_d32s8.get_quality(req)
    assert fb_d32.get_quality(req) > fb_d16s8.get_quality(req)
    assert fb_d32.get_quality(req) > fb_d24s8.get_quality(req)
    assert fb_d32s8.get_quality(req) > fb_d24s8.get_quality(req)
    assert fb_d32s8.get_quality(req) < fb_d32.get_quality(req)

def test_fbquality_rgba64():
    if False:
        for i in range(10):
            print('nop')
    req_color0 = FrameBufferProperties()
    req_color0.color_bits = 0
    req_color1 = FrameBufferProperties()
    req_color1.color_bits = 1
    req_color0_alpha0 = FrameBufferProperties()
    req_color0_alpha0.color_bits = 0
    req_color0_alpha0.alpha_bits = 0
    req_color1_alpha1 = FrameBufferProperties()
    req_color1_alpha1.color_bits = 1
    req_color1_alpha1.alpha_bits = 1
    req_rgb0 = FrameBufferProperties()
    req_rgb0.set_rgba_bits(0, 0, 0, 0)
    req_rgb1 = FrameBufferProperties()
    req_rgb1.set_rgba_bits(1, 1, 1, 0)
    req_rgb0_alpha0 = FrameBufferProperties()
    req_rgb0_alpha0.set_rgba_bits(0, 0, 0, 0)
    req_rgb1_alpha1 = FrameBufferProperties()
    req_rgb1_alpha1.set_rgba_bits(1, 1, 1, 1)
    fb_rgba8 = FrameBufferProperties()
    fb_rgba8.rgb_color = True
    fb_rgba8.set_rgba_bits(8, 8, 8, 8)
    fb_rgba16 = FrameBufferProperties()
    fb_rgba16.rgb_color = True
    fb_rgba16.set_rgba_bits(16, 16, 16, 16)
    assert fb_rgba8.get_quality(req_color0) > fb_rgba16.get_quality(req_color0)
    assert fb_rgba8.get_quality(req_color1) > fb_rgba16.get_quality(req_color1)
    assert fb_rgba8.get_quality(req_color0_alpha0) > fb_rgba16.get_quality(req_color0_alpha0)
    assert fb_rgba8.get_quality(req_color1_alpha1) > fb_rgba16.get_quality(req_color1_alpha1)
    assert fb_rgba8.get_quality(req_rgb0) > fb_rgba16.get_quality(req_rgb0)
    assert fb_rgba8.get_quality(req_rgb1) > fb_rgba16.get_quality(req_rgb1)
    assert fb_rgba8.get_quality(req_rgb0_alpha0) > fb_rgba16.get_quality(req_rgb0_alpha0)
    assert fb_rgba8.get_quality(req_rgb1_alpha1) > fb_rgba16.get_quality(req_rgb1_alpha1)

def test_fbquality_multi_samples():
    if False:
        i = 10
        return i + 15
    fb_0_samples = FrameBufferProperties()
    fb_0_samples.set_rgb_color(1)
    fb_0_samples.set_multisamples(0)
    fb_2_samples = FrameBufferProperties()
    fb_2_samples.set_rgb_color(1)
    fb_2_samples.set_multisamples(2)
    fb_4_samples = FrameBufferProperties()
    fb_4_samples.set_rgb_color(1)
    fb_4_samples.set_multisamples(4)
    fb_8_samples = FrameBufferProperties()
    fb_8_samples.set_rgb_color(1)
    fb_8_samples.set_multisamples(8)
    fb_16_samples = FrameBufferProperties()
    fb_16_samples.set_rgb_color(1)
    fb_16_samples.set_multisamples(16)
    req_0_samples = FrameBufferProperties()
    req_0_samples.set_multisamples(0)
    req_1_samples = FrameBufferProperties()
    req_1_samples.set_multisamples(1)
    req_2_samples = FrameBufferProperties()
    req_2_samples.set_multisamples(2)
    req_4_samples = FrameBufferProperties()
    req_4_samples.set_multisamples(4)
    req_8_samples = FrameBufferProperties()
    req_8_samples.set_multisamples(8)
    req_16_samples = FrameBufferProperties()
    req_16_samples.set_multisamples(16)
    assert fb_2_samples.get_quality(req_4_samples) < fb_2_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_4_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_8_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_16_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_16_samples.get_quality(req_16_samples)
    assert fb_8_samples.get_quality(req_16_samples) < fb_2_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_4_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_8_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_16_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_16_samples.get_quality(req_8_samples)
    assert fb_16_samples.get_quality(req_2_samples) < fb_8_samples.get_quality(req_2_samples)
    assert fb_8_samples.get_quality(req_2_samples) < fb_4_samples.get_quality(req_2_samples)
    assert fb_16_samples.get_quality(req_1_samples) > fb_8_samples.get_quality(req_1_samples)
    assert fb_16_samples.get_quality(req_1_samples) > fb_4_samples.get_quality(req_1_samples)
    assert fb_16_samples.get_quality(req_1_samples) > fb_2_samples.get_quality(req_1_samples)
    assert fb_8_samples.get_quality(req_1_samples) > fb_4_samples.get_quality(req_1_samples)
    assert fb_8_samples.get_quality(req_1_samples) > fb_2_samples.get_quality(req_1_samples)
    assert fb_16_samples.get_quality(req_0_samples) < fb_8_samples.get_quality(req_0_samples)
    assert fb_16_samples.get_quality(req_0_samples) < fb_4_samples.get_quality(req_0_samples)
    assert fb_16_samples.get_quality(req_0_samples) < fb_2_samples.get_quality(req_0_samples)
    assert fb_8_samples.get_quality(req_0_samples) < fb_4_samples.get_quality(req_0_samples)
    assert fb_8_samples.get_quality(req_0_samples) < fb_2_samples.get_quality(req_0_samples)
    assert fb_0_samples.get_quality(req_2_samples) < fb_2_samples.get_quality(req_4_samples)
    assert fb_0_samples.get_quality(req_2_samples) < fb_2_samples.get_quality(req_8_samples)
    assert fb_0_samples.get_quality(req_2_samples) < fb_2_samples.get_quality(req_16_samples)
    assert fb_0_samples.get_quality(req_0_samples) > fb_2_samples.get_quality(req_0_samples)
    assert fb_0_samples.get_quality(req_0_samples) > fb_4_samples.get_quality(req_0_samples)
    assert fb_0_samples.get_quality(req_0_samples) > fb_8_samples.get_quality(req_0_samples)
    assert fb_0_samples.get_quality(req_0_samples) > fb_16_samples.get_quality(req_0_samples)

def test_fbquality_coverage_samples():
    if False:
        for i in range(10):
            print('nop')
    fb_2_samples = FrameBufferProperties()
    fb_2_samples.set_rgb_color(1)
    fb_2_samples.set_coverage_samples(2)
    fb_4_samples = FrameBufferProperties()
    fb_4_samples.set_rgb_color(1)
    fb_4_samples.set_coverage_samples(4)
    fb_8_samples = FrameBufferProperties()
    fb_8_samples.set_rgb_color(1)
    fb_8_samples.set_coverage_samples(8)
    fb_16_samples = FrameBufferProperties()
    fb_16_samples.set_rgb_color(1)
    fb_16_samples.set_coverage_samples(16)
    req_0_samples = FrameBufferProperties()
    req_0_samples.set_coverage_samples(0)
    req_1_samples = FrameBufferProperties()
    req_1_samples.set_coverage_samples(1)
    req_2_samples = FrameBufferProperties()
    req_2_samples.set_coverage_samples(2)
    req_4_samples = FrameBufferProperties()
    req_4_samples.set_coverage_samples(4)
    req_8_samples = FrameBufferProperties()
    req_8_samples.set_coverage_samples(8)
    req_16_samples = FrameBufferProperties()
    req_16_samples.set_coverage_samples(16)
    assert fb_2_samples.get_quality(req_4_samples) < fb_2_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_4_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_8_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_16_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_4_samples) < fb_16_samples.get_quality(req_16_samples)
    assert fb_8_samples.get_quality(req_16_samples) < fb_2_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_4_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_8_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_16_samples.get_quality(req_2_samples)
    assert fb_2_samples.get_quality(req_2_samples) > fb_16_samples.get_quality(req_8_samples)
    assert fb_16_samples.get_quality(req_2_samples) < fb_8_samples.get_quality(req_2_samples)
    assert fb_8_samples.get_quality(req_2_samples) < fb_4_samples.get_quality(req_2_samples)
    assert fb_16_samples.get_quality(req_1_samples) > fb_8_samples.get_quality(req_1_samples)
    assert fb_16_samples.get_quality(req_1_samples) > fb_4_samples.get_quality(req_1_samples)
    assert fb_16_samples.get_quality(req_1_samples) > fb_2_samples.get_quality(req_1_samples)
    assert fb_8_samples.get_quality(req_1_samples) > fb_4_samples.get_quality(req_1_samples)
    assert fb_8_samples.get_quality(req_1_samples) > fb_2_samples.get_quality(req_1_samples)
    assert fb_16_samples.get_quality(req_0_samples) < fb_8_samples.get_quality(req_0_samples)
    assert fb_16_samples.get_quality(req_0_samples) < fb_4_samples.get_quality(req_0_samples)
    assert fb_16_samples.get_quality(req_0_samples) < fb_2_samples.get_quality(req_0_samples)
    assert fb_8_samples.get_quality(req_0_samples) < fb_4_samples.get_quality(req_0_samples)
    assert fb_8_samples.get_quality(req_0_samples) < fb_2_samples.get_quality(req_0_samples)