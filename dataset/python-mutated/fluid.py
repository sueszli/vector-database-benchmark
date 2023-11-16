import numpy as np
import scipy.sparse as sp
from math import factorial
from itertools import cycle
from functools import reduce
from scipy.sparse.linalg import factorized
from scipy.ndimage import map_coordinates, spline_filter

def difference(derivative, accuracy=1):
    if False:
        return 10
    derivative += 1
    radius = accuracy + derivative // 2 - 1
    points = range(-radius, radius + 1)
    coefficients = np.linalg.inv(np.vander(points))
    return (coefficients[-derivative] * factorial(derivative - 1), points)

def operator(shape, *differences):
    if False:
        i = 10
        return i + 15
    differences = zip(shape, cycle(differences))
    factors = (sp.diags(*diff, shape=(dim,) * 2) for (dim, diff) in differences)
    return reduce(lambda a, f: sp.kronsum(f, a, format='csc'), factors)

class Fluid:

    def __init__(self, shape, *quantities, pressure_order=1, advect_order=3):
        if False:
            return 10
        self.shape = shape
        self.dimensions = len(shape)
        self.quantities = quantities
        for q in quantities:
            setattr(self, q, np.zeros(shape))
        self.indices = np.indices(shape)
        self.velocity = np.zeros((self.dimensions, *shape))
        laplacian = operator(shape, difference(2, pressure_order))
        self.pressure_solver = factorized(laplacian)
        self.advect_order = advect_order

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        advection_map = self.indices - self.velocity

        def advect(field, filter_epsilon=0.1, mode='constant'):
            if False:
                for i in range(10):
                    print('nop')
            filtered = spline_filter(field, order=self.advect_order, mode=mode)
            field = filtered * (1 - filter_epsilon) + field * filter_epsilon
            return map_coordinates(field, advection_map, prefilter=False, order=self.advect_order, mode=mode)
        for d in range(self.dimensions):
            self.velocity[d] = advect(self.velocity[d])
        for q in self.quantities:
            setattr(self, q, advect(getattr(self, q)))
        jacobian_shape = (self.dimensions,) * 2
        partials = tuple((np.gradient(d) for d in self.velocity))
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)
        divergence = jacobian.trace()
        curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()
        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)
        self.velocity -= np.gradient(pressure)
        return (divergence, curl, pressure)

def inflow(fluid, angle=0, padding=25, radius=7, velocity=1.5):
    if False:
        print('Hello World!')
    ' Source defnition '
    center = np.floor_divide(fluid.shape, 2)
    points = np.array([angle])
    points = tuple((np.array((np.cos(p), np.sin(p))) for p in points))
    normals = tuple((-p for p in points))
    r = np.min(center) - padding
    points = tuple((r * p + center for p in points))
    inflow_velocity = np.zeros_like(fluid.velocity)
    inflow_dye = np.zeros(fluid.shape)
    for (p, n) in zip(points, normals):
        mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= radius
        inflow_velocity[:, mask] += n[:, None] * velocity
        inflow_dye[mask] = 1
    return (inflow_velocity, inflow_dye)