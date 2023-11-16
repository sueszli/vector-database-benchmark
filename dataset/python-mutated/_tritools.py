"""
Tools for triangular grids.
"""
import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation

class TriAnalyzer:
    """
    Define basic tools for triangular mesh analysis and improvement.

    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
    tools for mesh analysis and mesh improvement.

    Attributes
    ----------
    scale_factors

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The encapsulated triangulation to analyze.
    """

    def __init__(self, triangulation):
        if False:
            print('Hello World!')
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation

    @property
    def scale_factors(self):
        if False:
            i = 10
            return i + 15
        '\n        Factors to rescale the triangulation into a unit square.\n\n        Returns\n        -------\n        (float, float)\n            Scaling factors (kx, ky) so that the triangulation\n            ``[triangulation.x * kx, triangulation.y * ky]``\n            fits exactly inside a unit square.\n        '
        compressed_triangles = self._triangulation.get_masked_triangles()
        node_used = np.bincount(np.ravel(compressed_triangles), minlength=self._triangulation.x.size) != 0
        return (1 / np.ptp(self._triangulation.x[node_used]), 1 / np.ptp(self._triangulation.y[node_used]))

    def circle_ratios(self, rescale=True):
        if False:
            while True:
                i = 10
        "\n        Return a measure of the triangulation triangles flatness.\n\n        The ratio of the incircle radius over the circumcircle radius is a\n        widely used indicator of a triangle flatness.\n        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral\n        triangles. Circle ratios below 0.01 denote very flat triangles.\n\n        To avoid unduly low values due to a difference of scale between the 2\n        axis, the triangular mesh can first be rescaled to fit inside a unit\n        square with `scale_factors` (Only if *rescale* is True, which is\n        its default value).\n\n        Parameters\n        ----------\n        rescale : bool, default: True\n            If True, internally rescale (based on `scale_factors`), so that the\n            (unmasked) triangles fit exactly inside a unit square mesh.\n\n        Returns\n        -------\n        masked array\n            Ratio of the incircle radius over the circumcircle radius, for\n            each 'rescaled' triangle of the encapsulated triangulation.\n            Values corresponding to masked triangles are masked out.\n\n        "
        if rescale:
            (kx, ky) = self.scale_factors
        else:
            (kx, ky) = (1.0, 1.0)
        pts = np.vstack([self._triangulation.x * kx, self._triangulation.y * ky]).T
        tri_pts = pts[self._triangulation.triangles]
        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
        a = np.hypot(a[:, 0], a[:, 1])
        b = np.hypot(b[:, 0], b[:, 1])
        c = np.hypot(c[:, 0], c[:, 1])
        s = (a + b + c) * 0.5
        prod = s * (a + b - s) * (a + c - s) * (b + c - s)
        bool_flat = prod == 0.0
        if np.any(bool_flat):
            ntri = tri_pts.shape[0]
            circum_radius = np.empty(ntri, dtype=np.float64)
            circum_radius[bool_flat] = np.inf
            abc = a * b * c
            circum_radius[~bool_flat] = abc[~bool_flat] / (4.0 * np.sqrt(prod[~bool_flat]))
        else:
            circum_radius = a * b * c / (4.0 * np.sqrt(prod))
        in_radius = a * b * c / (4.0 * circum_radius * s)
        circle_ratio = in_radius / circum_radius
        mask = self._triangulation.mask
        if mask is None:
            return circle_ratio
        else:
            return np.ma.array(circle_ratio, mask=mask)

    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
        if False:
            i = 10
            return i + 15
        '\n        Eliminate excessively flat border triangles from the triangulation.\n\n        Returns a mask *new_mask* which allows to clean the encapsulated\n        triangulation from its border-located flat triangles\n        (according to their :meth:`circle_ratios`).\n        This mask is meant to be subsequently applied to the triangulation\n        using `.Triangulation.set_mask`.\n        *new_mask* is an extension of the initial triangulation mask\n        in the sense that an initially masked triangle will remain masked.\n\n        The *new_mask* array is computed recursively; at each step flat\n        triangles are removed only if they share a side with the current mesh\n        border. Thus, no new holes in the triangulated domain will be created.\n\n        Parameters\n        ----------\n        min_circle_ratio : float, default: 0.01\n            Border triangles with incircle/circumcircle radii ratio r/R will\n            be removed if r/R < *min_circle_ratio*.\n        rescale : bool, default: True\n            If True, first, internally rescale (based on `scale_factors`) so\n            that the (unmasked) triangles fit exactly inside a unit square\n            mesh.  This rescaling accounts for the difference of scale which\n            might exist between the 2 axis.\n\n        Returns\n        -------\n        array of bool\n            Mask to apply to encapsulated triangulation.\n            All the initially masked triangles remain masked in the\n            *new_mask*.\n\n        Notes\n        -----\n        The rationale behind this function is that a Delaunay\n        triangulation - of an unstructured set of points - sometimes contains\n        almost flat triangles at its border, leading to artifacts in plots\n        (especially for high-resolution contouring).\n        Masked with computed *new_mask*, the encapsulated\n        triangulation would contain no more unmasked border triangles\n        with a circle ratio below *min_circle_ratio*, thus improving the\n        mesh quality for subsequent plots or interpolation.\n        '
        ntri = self._triangulation.triangles.shape[0]
        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
        current_mask = self._triangulation.mask
        if current_mask is None:
            current_mask = np.zeros(ntri, dtype=bool)
        valid_neighbors = np.copy(self._triangulation.neighbors)
        renum_neighbors = np.arange(ntri, dtype=np.int32)
        nadd = -1
        while nadd != 0:
            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
            added_mask = wavefront & mask_bad_ratio
            current_mask = added_mask | current_mask
            nadd = np.sum(added_mask)
            valid_neighbors[added_mask, :] = -1
            renum_neighbors[added_mask] = -1
            valid_neighbors = np.where(valid_neighbors == -1, -1, renum_neighbors[valid_neighbors])
        return np.ma.filled(current_mask, True)

    def _get_compressed_triangulation(self):
        if False:
            return 10
        '\n        Compress (if masked) the encapsulated triangulation.\n\n        Returns minimal-length triangles array (*compressed_triangles*) and\n        coordinates arrays (*compressed_x*, *compressed_y*) that can still\n        describe the unmasked triangles of the encapsulated triangulation.\n\n        Returns\n        -------\n        compressed_triangles : array-like\n            the returned compressed triangulation triangles\n        compressed_x : array-like\n            the returned compressed triangulation 1st coordinate\n        compressed_y : array-like\n            the returned compressed triangulation 2nd coordinate\n        tri_renum : int array\n            renumbering table to translate the triangle numbers from the\n            encapsulated triangulation into the new (compressed) renumbering.\n            -1 for masked triangles (deleted from *compressed_triangles*).\n        node_renum : int array\n            renumbering table to translate the point numbers from the\n            encapsulated triangulation into the new (compressed) renumbering.\n            -1 for unused points (i.e. those deleted from *compressed_x* and\n            *compressed_y*).\n\n        '
        tri_mask = self._triangulation.mask
        compressed_triangles = self._triangulation.get_masked_triangles()
        ntri = self._triangulation.triangles.shape[0]
        if tri_mask is not None:
            tri_renum = self._total_to_compress_renum(~tri_mask)
        else:
            tri_renum = np.arange(ntri, dtype=np.int32)
        valid_node = np.bincount(np.ravel(compressed_triangles), minlength=self._triangulation.x.size) != 0
        compressed_x = self._triangulation.x[valid_node]
        compressed_y = self._triangulation.y[valid_node]
        node_renum = self._total_to_compress_renum(valid_node)
        compressed_triangles = node_renum[compressed_triangles]
        return (compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum)

    @staticmethod
    def _total_to_compress_renum(valid):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        valid : 1D bool array\n            Validity mask.\n\n        Returns\n        -------\n        int array\n            Array so that (`valid_array` being a compressed array\n            based on a `masked_array` with mask ~*valid*):\n\n            - For all i with valid[i] = True:\n              valid_array[renum[i]] = masked_array[i]\n            - For all i with valid[i] = False:\n              renum[i] = -1 (invalid value)\n        '
        renum = np.full(np.size(valid), -1, dtype=np.int32)
        n_valid = np.sum(valid)
        renum[valid] = np.arange(n_valid, dtype=np.int32)
        return renum