import numpy as np
import random

def make_batch_select_masks(batch_size, npolygons_range=(1, 10), nvertices_range=(3, 40), vertex_ndim=2, vertex_dtype=np.float32):
    if False:
        return 10
    polygons = []
    vertices = []
    selected_masks = []
    for _ in range(batch_size):
        nmasks = random.randint(*npolygons_range)
        available_masks = list(range(nmasks))
        selected_masks.append(np.array(random.sample(available_masks, random.randint(1, nmasks)), dtype=np.int32))
        vertex_count = 0
        mask_id = 0
        curr_polygons = np.zeros([nmasks, 3], dtype=np.int32)
        for m in range(nmasks):
            nvertices = random.randint(*nvertices_range)
            curr_polygons[m, :] = (mask_id, vertex_count, vertex_count + nvertices)
            vertex_count = vertex_count + nvertices
            mask_id = mask_id + 1
        polygons.append(curr_polygons)
        if np.issubdtype(vertex_dtype, np.integer):
            vertices.append(np.random.randint(low=np.iinfo(vertex_dtype).min, high=np.iinfo(vertex_dtype).max, size=(vertex_count, vertex_ndim), dtype=vertex_dtype))
        else:
            vertices.append(np.array(np.random.rand(vertex_count, vertex_ndim), dtype=vertex_dtype))
    return (polygons, vertices, selected_masks)