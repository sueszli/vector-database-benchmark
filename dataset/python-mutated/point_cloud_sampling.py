"""
Point cloud sampling example codes. This code supports
- Voxel point sampling
- Farthest point sampling
- Poisson disk sampling

"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from collections import defaultdict
do_plot = True

def voxel_point_sampling(original_points: npt.NDArray, voxel_size: float):
    if False:
        while True:
            i = 10
    "\n    Voxel Point Sampling function.\n    This function sample N-dimensional points with voxel grid.\n    Points in a same voxel grid will be merged by mean operation for sampling.\n\n    Parameters\n    ----------\n    original_points :  (M, N) N-dimensional points for sampling.\n                        The number of points is M.\n    voxel_size : voxel grid size\n\n    Returns\n    -------\n    sampled points (M', N)\n    "
    voxel_dict = defaultdict(list)
    for i in range(original_points.shape[0]):
        xyz = original_points[i, :]
        xyz_index = tuple(xyz // voxel_size)
        voxel_dict[xyz_index].append(xyz)
    points = np.vstack([np.mean(v, axis=0) for v in voxel_dict.values()])
    return points

def farthest_point_sampling(orig_points: npt.NDArray, n_points: int, seed: int):
    if False:
        return 10
    '\n    Farthest point sampling function\n    This function sample N-dimensional points with the farthest point policy.\n\n    Parameters\n    ----------\n    orig_points :  (M, N) N-dimensional points for sampling.\n                    The number of points is M.\n    n_points : number of points for sampling\n    seed : random seed number\n\n    Returns\n    -------\n    sampled points (n_points, N)\n\n    '
    rng = np.random.default_rng(seed)
    n_orig_points = orig_points.shape[0]
    first_point_id = rng.choice(range(n_orig_points))
    min_distances = np.ones(n_orig_points) * float('inf')
    selected_ids = [first_point_id]
    while len(selected_ids) < n_points:
        base_point = orig_points[selected_ids[-1], :]
        distances = np.linalg.norm(orig_points[np.newaxis, :] - base_point, axis=2).flatten()
        min_distances = np.minimum(min_distances, distances)
        distances_rank = np.argsort(-min_distances)
        for i in distances_rank:
            if i not in selected_ids:
                selected_ids.append(i)
                break
    return orig_points[selected_ids, :]

def poisson_disk_sampling(orig_points: npt.NDArray, n_points: int, min_distance: float, seed: int, MAX_ITER=1000):
    if False:
        while True:
            i = 10
    '\n    Poisson disk sampling function\n    This function sample N-dimensional points randomly until the number of\n    points keeping minimum distance between selected points.\n\n    Parameters\n    ----------\n    orig_points :  (M, N) N-dimensional points for sampling.\n                    The number of points is M.\n    n_points : number of points for sampling\n    min_distance : minimum distance between selected points.\n    seed : random seed number\n    MAX_ITER : Maximum number of iteration. Default is 1000.\n\n    Returns\n    -------\n    sampled points (n_points or less, N)\n    '
    rng = np.random.default_rng(seed)
    selected_id = rng.choice(range(orig_points.shape[0]))
    selected_ids = [selected_id]
    loop = 0
    while len(selected_ids) < n_points and loop <= MAX_ITER:
        selected_id = rng.choice(range(orig_points.shape[0]))
        base_point = orig_points[selected_id, :]
        distances = np.linalg.norm(orig_points[np.newaxis, selected_ids] - base_point, axis=2).flatten()
        if min(distances) >= min_distance:
            selected_ids.append(selected_id)
        loop += 1
    if len(selected_ids) != n_points:
        print('Could not find the specified number of points...')
    return orig_points[selected_ids, :]

def plot_sampled_points(original_points, sampled_points, method_name):
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], marker='.', label='Original points')
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], marker='o', label='Filtered points')
    plt.legend()
    plt.title(method_name)
    plt.axis('equal')

def main():
    if False:
        i = 10
        return i + 15
    n_points = 1000
    seed = 1234
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 10.0, n_points)
    y = rng.normal(0.0, 1.0, n_points)
    z = rng.normal(0.0, 10.0, n_points)
    original_points = np.vstack((x, y, z)).T
    print(f'original_points.shape={original_points.shape!r}')
    print('Voxel point sampling')
    voxel_size = 20.0
    voxel_sampling_points = voxel_point_sampling(original_points, voxel_size)
    print(f'voxel_sampling_points.shape={voxel_sampling_points.shape!r}')
    print('Farthest point sampling')
    n_points = 20
    farthest_sampling_points = farthest_point_sampling(original_points, n_points, seed)
    print(f'farthest_sampling_points.shape={farthest_sampling_points.shape!r}')
    print('Poisson disk sampling')
    n_points = 20
    min_distance = 10.0
    poisson_disk_points = poisson_disk_sampling(original_points, n_points, min_distance, seed)
    print(f'poisson_disk_points.shape={poisson_disk_points.shape!r}')
    if do_plot:
        plot_sampled_points(original_points, voxel_sampling_points, 'Voxel point sampling')
        plot_sampled_points(original_points, farthest_sampling_points, 'Farthest point sampling')
        plot_sampled_points(original_points, poisson_disk_points, 'poisson disk sampling')
        plt.show()
if __name__ == '__main__':
    main()