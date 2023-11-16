"""
Mesh refinement for triangular grids.
"""
import numpy as np
from matplotlib import _api
from matplotlib.tri._triangulation import Triangulation
import matplotlib.tri._triinterpolate

class TriRefiner:
    """
    Abstract base class for classes implementing mesh refinement.

    A TriRefiner encapsulates a Triangulation object and provides tools for
    mesh refinement and interpolation.

    Derived classes must implement:

    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
      the optional keyword arguments *kwargs* are defined in each
      TriRefiner concrete implementation, and which returns:

      - a refined triangulation,
      - optionally (depending on *return_tri_index*), for each
        point of the refined triangulation: the index of
        the initial triangulation triangle to which it belongs.

    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:

      - *z* array of field values (to refine) defined at the base
        triangulation nodes,
      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
      - the other optional keyword arguments *kwargs* are defined in
        each TriRefiner concrete implementation;

      and which returns (as a tuple) a refined triangular mesh and the
      interpolated values of the field at the refined triangulation nodes.
    """

    def __init__(self, triangulation):
        if False:
            for i in range(10):
                print('nop')
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation

class UniformTriRefiner(TriRefiner):
    """
    Uniform mesh refinement by recursive subdivisions.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The encapsulated triangulation (to be refined)
    """

    def __init__(self, triangulation):
        if False:
            return 10
        super().__init__(triangulation)

    def refine_triangulation(self, return_tri_index=False, subdiv=3):
        if False:
            while True:
                i = 10
        '\n        Compute a uniformly refined triangulation *refi_triangulation* of\n        the encapsulated :attr:`triangulation`.\n\n        This function refines the encapsulated triangulation by splitting each\n        father triangle into 4 child sub-triangles built on the edges midside\n        nodes, recursing *subdiv* times.  In the end, each triangle is hence\n        divided into ``4**subdiv`` child triangles.\n\n        Parameters\n        ----------\n        return_tri_index : bool, default: False\n            Whether an index table indicating the father triangle index of each\n            point is returned.\n        subdiv : int, default: 3\n            Recursion level for the subdivision.\n            Each triangle is divided into ``4**subdiv`` child triangles;\n            hence, the default results in 64 refined subtriangles for each\n            triangle of the initial triangulation.\n\n        Returns\n        -------\n        refi_triangulation : `~matplotlib.tri.Triangulation`\n            The refined triangulation.\n        found_index : int array\n            Index of the initial triangulation containing triangle, for each\n            point of *refi_triangulation*.\n            Returned only if *return_tri_index* is set to True.\n        '
        refi_triangulation = self._triangulation
        ntri = refi_triangulation.triangles.shape[0]
        ancestors = np.arange(ntri, dtype=np.int32)
        for _ in range(subdiv):
            (refi_triangulation, ancestors) = self._refine_triangulation_once(refi_triangulation, ancestors)
        refi_npts = refi_triangulation.x.shape[0]
        refi_triangles = refi_triangulation.triangles
        if return_tri_index:
            found_index = np.full(refi_npts, -1, dtype=np.int32)
            tri_mask = self._triangulation.mask
            if tri_mask is None:
                found_index[refi_triangles] = np.repeat(ancestors, 3).reshape(-1, 3)
            else:
                ancestor_mask = tri_mask[ancestors]
                found_index[refi_triangles[ancestor_mask, :]] = np.repeat(ancestors[ancestor_mask], 3).reshape(-1, 3)
                found_index[refi_triangles[~ancestor_mask, :]] = np.repeat(ancestors[~ancestor_mask], 3).reshape(-1, 3)
            return (refi_triangulation, found_index)
        else:
            return refi_triangulation

    def refine_field(self, z, triinterpolator=None, subdiv=3):
        if False:
            i = 10
            return i + 15
        '\n        Refine a field defined on the encapsulated triangulation.\n\n        Parameters\n        ----------\n        z : (npoints,) array-like\n            Values of the field to refine, defined at the nodes of the\n            encapsulated triangulation. (``n_points`` is the number of points\n            in the initial triangulation)\n        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional\n            Interpolator used for field interpolation. If not specified,\n            a `~matplotlib.tri.CubicTriInterpolator` will be used.\n        subdiv : int, default: 3\n            Recursion level for the subdivision.\n            Each triangle is divided into ``4**subdiv`` child triangles.\n\n        Returns\n        -------\n        refi_tri : `~matplotlib.tri.Triangulation`\n             The returned refined triangulation.\n        refi_z : 1D array of length: *refi_tri* node count.\n             The returned interpolated field (at *refi_tri* nodes).\n        '
        if triinterpolator is None:
            interp = matplotlib.tri.CubicTriInterpolator(self._triangulation, z)
        else:
            _api.check_isinstance(matplotlib.tri.TriInterpolator, triinterpolator=triinterpolator)
            interp = triinterpolator
        (refi_tri, found_index) = self.refine_triangulation(subdiv=subdiv, return_tri_index=True)
        refi_z = interp._interpolate_multikeys(refi_tri.x, refi_tri.y, tri_index=found_index)[0]
        return (refi_tri, refi_z)

    @staticmethod
    def _refine_triangulation_once(triangulation, ancestors=None):
        if False:
            return 10
        '\n        Refine a `.Triangulation` by splitting each triangle into 4\n        child-masked_triangles built on the edges midside nodes.\n\n        Masked triangles, if present, are also split, but their children\n        returned masked.\n\n        If *ancestors* is not provided, returns only a new triangulation:\n        child_triangulation.\n\n        If the array-like key table *ancestor* is given, it shall be of shape\n        (ntri,) where ntri is the number of *triangulation* masked_triangles.\n        In this case, the function returns\n        (child_triangulation, child_ancestors)\n        child_ancestors is defined so that the 4 child masked_triangles share\n        the same index as their father: child_ancestors.shape = (4 * ntri,).\n        '
        x = triangulation.x
        y = triangulation.y
        neighbors = triangulation.neighbors
        triangles = triangulation.triangles
        npts = np.shape(x)[0]
        ntri = np.shape(triangles)[0]
        if ancestors is not None:
            ancestors = np.asarray(ancestors)
            if np.shape(ancestors) != (ntri,):
                raise ValueError(f'Incompatible shapes provide for triangulation.masked_triangles and ancestors: {np.shape(triangles)} and {np.shape(ancestors)}')
        borders = np.sum(neighbors == -1)
        added_pts = (3 * ntri + borders) // 2
        refi_npts = npts + added_pts
        refi_x = np.zeros(refi_npts)
        refi_y = np.zeros(refi_npts)
        refi_x[:npts] = x
        refi_y[:npts] = y
        edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
        edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
        edge_neighbors = neighbors[edge_elems, edge_apexes]
        mask_masters = edge_elems > edge_neighbors
        masters = edge_elems[mask_masters]
        apex_masters = edge_apexes[mask_masters]
        x_add = (x[triangles[masters, apex_masters]] + x[triangles[masters, (apex_masters + 1) % 3]]) * 0.5
        y_add = (y[triangles[masters, apex_masters]] + y[triangles[masters, (apex_masters + 1) % 3]]) * 0.5
        refi_x[npts:] = x_add
        refi_y[npts:] = y_add
        new_pt_corner = triangles
        new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
        cum_sum = npts
        for imid in range(3):
            mask_st_loc = imid == apex_masters
            n_masters_loc = np.sum(mask_st_loc)
            elem_masters_loc = masters[mask_st_loc]
            new_pt_midside[:, imid][elem_masters_loc] = np.arange(n_masters_loc, dtype=np.int32) + cum_sum
            cum_sum += n_masters_loc
        mask_slaves = np.logical_not(mask_masters)
        slaves = edge_elems[mask_slaves]
        slaves_masters = edge_neighbors[mask_slaves]
        diff_table = np.abs(neighbors[slaves_masters, :] - np.outer(slaves, np.ones(3, dtype=np.int32)))
        slave_masters_apex = np.argmin(diff_table, axis=1)
        slaves_apex = edge_apexes[mask_slaves]
        new_pt_midside[slaves, slaves_apex] = new_pt_midside[slaves_masters, slave_masters_apex]
        child_triangles = np.empty([ntri * 4, 3], dtype=np.int32)
        child_triangles[0::4, :] = np.vstack([new_pt_corner[:, 0], new_pt_midside[:, 0], new_pt_midside[:, 2]]).T
        child_triangles[1::4, :] = np.vstack([new_pt_corner[:, 1], new_pt_midside[:, 1], new_pt_midside[:, 0]]).T
        child_triangles[2::4, :] = np.vstack([new_pt_corner[:, 2], new_pt_midside[:, 2], new_pt_midside[:, 1]]).T
        child_triangles[3::4, :] = np.vstack([new_pt_midside[:, 0], new_pt_midside[:, 1], new_pt_midside[:, 2]]).T
        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
        if triangulation.mask is not None:
            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
        if ancestors is None:
            return child_triangulation
        else:
            return (child_triangulation, np.repeat(ancestors, 4))