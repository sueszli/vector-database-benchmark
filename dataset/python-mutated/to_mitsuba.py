import open3d as o3d
import mitsuba as mi

def render_mesh(mesh, mesh_center):
    if False:
        while True:
            i = 10
    scene = mi.load_dict({'type': 'scene', 'integrator': {'type': 'path'}, 'light': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': 1.0}}, 'sensor': {'type': 'perspective', 'focal_length': '50mm', 'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, 5], target=mesh_center, up=[0, 1, 0]), 'thefilm': {'type': 'hdrfilm', 'width': 1024, 'height': 768}, 'thesampler': {'type': 'multijitter', 'sample_count': 64}}, 'themesh': mesh})
    img = mi.render(scene, spp=256)
    return img
mi.set_variant('llvm_ad_rgb')
dataset = o3d.data.MonkeyModel()
mesh = o3d.t.io.read_triangle_mesh(dataset.path)
mesh_center = mesh.get_axis_aligned_bounding_box().get_center()
mesh.material.set_default_properties()
mesh.material.material_name = 'defaultLit'
mesh.material.scalar_properties['metallic'] = 1.0
mesh.material.texture_maps['albedo'] = o3d.t.io.read_image(dataset.path_map['albedo'])
mesh.material.texture_maps['roughness'] = o3d.t.io.read_image(dataset.path_map['roughness'])
mesh.material.texture_maps['metallic'] = o3d.t.io.read_image(dataset.path_map['metallic'])
print('Render mesh with material converted to Mitsuba principled BSDF')
mi_mesh = mesh.to_mitsuba('monkey')
img = render_mesh(mi_mesh, mesh_center.numpy())
mi.Bitmap(img).write('test.exr')
print('Render mesh with normal-mapped prnincipled BSDF')
mesh.material.texture_maps['normal'] = o3d.t.io.read_image(dataset.path_map['normal'])
mi_mesh = mesh.to_mitsuba('monkey')
img = render_mesh(mi_mesh, mesh_center.numpy())
mi.Bitmap(img).write('test2.exr')
print('Rendering mesh with Mitsuba smooth plastic BSDF')
bsdf_smooth_plastic = mi.load_dict({'type': 'plastic', 'diffuse_reflectance': {'type': 'rgb', 'value': [0.1, 0.27, 0.36]}, 'int_ior': 1.9})
mi_mesh = mesh.to_mitsuba('monkey', bsdf=bsdf_smooth_plastic)
img = render_mesh(mi_mesh, mesh_center.numpy())
mi.Bitmap(img).write('test3.exr')
o3d.visualization.draw(mesh)