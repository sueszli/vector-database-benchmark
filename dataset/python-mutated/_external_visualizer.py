import open3d as o3d
__all__ = ['ExternalVisualizer', 'EV']

class ExternalVisualizer:
    """This class allows to send data to an external Visualizer

    Example:
        This example sends a point cloud to the visualizer::

            import open3d as o3d
            import numpy as np
            ev = o3d.visualization.ExternalVisualizer()
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(100,3)))
            ev.set(pcd)

    Args:
        address: The address where the visualizer is running.
            The default is localhost.
        timeout: The timeout for sending data in milliseconds.
    """

    def __init__(self, address='tcp://127.0.0.1:51454', timeout=10000):
        if False:
            while True:
                i = 10
        self.address = address
        self.timeout = timeout

    def set(self, obj=None, path='', time=0, layer='', connection=None):
        if False:
            while True:
                i = 10
        "Send Open3D objects for visualization to the visualizer.\n\n        Example:\n            To quickly send a single object just write::\n                ev.set(point_cloud)\n\n            To place the object at a specific location in the scene tree do::\n                ev.set(point_cloud, path='group/mypoints', time=42, layer='')\n            Note that depending on the visualizer some arguments like time or\n            layer may not be supported and will be ignored.\n\n            To set multiple objects use a list to pass multiple objects::\n                ev.set([point_cloud, mesh, camera])\n            Each entry in the list can be a tuple specifying all or some of the\n            location parameters::\n                ev.set(objs=[(point_cloud,'group/mypoints', 1, 'layer1'),\n                             (mesh, 'group/mymesh'),\n                             camera\n                            ]\n\n        Args:\n            obj: A geometry or camera object or a list of objects. See the\n            example seection for usage instructions.\n\n            path: A path describing a location in the scene tree.\n\n            time: An integer time value associated with the object.\n\n            layer: The layer associated with the object.\n\n            connection: A connection object to use for sending data. This\n                parameter can be used to override the default object.\n        "
        if connection is None:
            connection = o3d.io.rpc.Connection(address=self.address, timeout=self.timeout)
        result = []
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if isinstance(item, (tuple, list)):
                    if len(item) in range(1, 5):
                        result.append(self.set(*item, connection=connection))
                else:
                    result.append(self.set(item, connection=connection))
        elif isinstance(obj, o3d.geometry.PointCloud):
            status = o3d.io.rpc.set_point_cloud(obj, path=path, time=time, layer=layer, connection=connection)
            result.append(status)
        elif isinstance(obj, (o3d.t.geometry.TriangleMesh, o3d.geometry.TriangleMesh)):
            status = o3d.io.rpc.set_triangle_mesh(obj, path=path, time=time, layer=layer, connection=connection)
            result.append(status)
        elif isinstance(obj, o3d.camera.PinholeCameraParameters):
            status = o3d.io.rpc.set_legacy_camera(obj, path=path, time=time, layer=layer, connection=connection)
            result.append(status)
        else:
            raise Exception("Unsupported object type '{}'".format(str(type(obj))))
        return all(result)

    def set_time(self, time):
        if False:
            print('Hello World!')
        'Sets the time in the external visualizer\n\n        Note that this function is a placeholder for future functionality and\n        not yet supported by the receiving visualizer.\n\n        Args:\n            time: The time value\n        '
        connection = o3d.io.rpc.Connection(address=self.address, timeout=self.timeout)
        return o3d.io.rpc.set_time(time, connection)

    def set_active_camera(self, path):
        if False:
            while True:
                i = 10
        'Sets the active camera in the external visualizer\n\n        Note that this function is a placeholder for future functionality and\n        not yet supported by the receiving visualizer.\n\n        Args:\n            path: A path describing a location in the scene tree.\n        '
        connection = o3d.io.rpc.Connection(address=self.address, timeout=self.timeout)
        return o3d.io.rpc.set_active_camera(path, connection)

    def draw(self, geometry=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        "This function has the same functionality as 'set'.\n\n        This function is compatible with the standalone 'draw' function and can\n        be used to redirect calls to the external visualizer. Note that only\n        the geometry argument is supported, all other arguments will be\n        ignored.\n\n        Example:\n            Here we use draw with the default external visualizer::\n                import open3d as o3d\n\n                torus = o3d.geometry.TriangleMesh.create_torus()\n                sphere = o3d.geometry.TriangleMesh.create_sphere()\n\n                draw = o3d.visualization.EV.draw\n                draw([ {'geometry': sphere, 'name': 'sphere'},\n                       {'geometry': torus, 'name': 'torus', 'time': 1} ])\n\n                # now use the standard draw function as comparison\n                draw = o3d.visualization.draw\n                draw([ {'geometry': sphere, 'name': 'sphere'},\n                       {'geometry': torus, 'name': 'torus', 'time': 1} ])\n\n        Args:\n            geometry: The geometry to draw. This can be a geometry object, a\n            list of geometries. To pass additional information along with the\n            geometry we can use a dictionary. Supported keys for the dictionary\n            are 'geometry', 'name', and 'time'.\n        "
        if args or kwargs:
            import warnings
            warnings.warn("ExternalVisualizer.draw() does only support the 'geometry' argument", Warning)

        def add(g):
            if False:
                print('Hello World!')
            if isinstance(g, dict):
                obj = g['geometry']
                path = g.get('name', '')
                time = g.get('time', 0)
                self.set(obj=obj, path=path, time=time)
            else:
                self.set(g)
        if isinstance(geometry, (tuple, list)):
            for g in geometry:
                add(g)
        elif geometry is not None:
            add(geometry)
EV = ExternalVisualizer()