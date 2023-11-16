import copy
import logging
import math
from collections import defaultdict
from itertools import combinations
from typing import Optional, Tuple, List, Set, Dict, Iterable, Any
import numpy as np
import scipy.spatial as spatial
from opensfm import bow, context, feature_loader, vlad, geo, geometry
from opensfm.dataset_base import DataSetBase
logger: logging.Logger = logging.getLogger(__name__)

def has_gps_info(exif: Dict[str, Any]) -> bool:
    if False:
        i = 10
        return i + 15
    return exif and 'gps' in exif and ('latitude' in exif['gps']) and ('longitude' in exif['gps'])

def sorted_pair(im1: str, im2: str) -> Tuple[str, str]:
    if False:
        return 10
    if im1 < im2:
        return (im1, im2)
    else:
        return (im2, im1)

def get_gps_point(exif: Dict[str, Any], reference: geo.TopocentricConverter) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    'Return GPS-based representative point. Direction is returned as Z oriented (vertical assumption)'
    gps = exif['gps']
    altitude = 0
    direction = np.array([0, 0, 1])
    return (reference.to_topocentric(gps['latitude'], gps['longitude'], altitude), direction)
DEFAULT_Z = 1.0
MAXIMUM_Z = 8000
SAMPLE_Z = 100

def sign(x: float) -> float:
    if False:
        i = 10
        return i + 15
    "Return a float's sign."
    return 1.0 if x > 0.0 else -1.0

def get_gps_opk_point(exif: Dict[str, Any], reference: geo.TopocentricConverter) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    'Return GPS-based representative point.'
    opk = exif['opk']
    (omega, phi, kappa) = (math.radians(opk['omega']), math.radians(opk['phi']), math.radians(opk['kappa']))
    R_camera = geometry.rotation_from_opk(omega, phi, kappa)
    z_axis = R_camera[2]
    origin = get_gps_point(exif, reference)
    return (origin[0], z_axis / (sign(z_axis[2]) * z_axis[2]) * DEFAULT_Z)

def find_best_altitude(origin: Dict[str, np.ndarray], directions: Dict[str, np.ndarray]) -> float:
    if False:
        i = 10
        return i + 15
    "Find the altitude that minimize X/Y bounding box. Domain is [0, MAXIMUM_Z].\n    'origin' contains per-image positions in worl coordinates\n    'directions' viewing directions expected to be homogenized with z=1.\n\n    We sample it every SAMPLE_Z and regress a second polynomial from which we take its extrema.\n    "
    directions_base = np.array([p for p in directions.values()])
    origin_base = np.array([p for p in origin.values()])
    (samples_x, samples_y) = ([], [])
    for current_z in range(1, MAXIMUM_Z, SAMPLE_Z):
        scaled = origin_base + directions_base / DEFAULT_Z * current_z
        current_size = (np.max(scaled[:, 0]) - np.min(scaled[:, 0])) ** 2 + (np.max(scaled[:, 1]) - np.min(scaled[:, 1])) ** 2
        samples_x.append(current_z)
        samples_y.append(current_size)
    coeffs = np.polyfit(samples_x, samples_y, 2)
    extrema = -coeffs[1] / (2 * coeffs[0])
    if extrema < 0:
        logger.info(f'Altitude is negative ({extrema}) : viewing directions are probably divergent. Using default altitude of {DEFAULT_Z}')
        extrema = DEFAULT_Z
    return extrema

def get_representative_points(images: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter) -> Dict[str, np.ndarray]:
    if False:
        print('Hello World!')
    'Return a topocentric point for each image, that is suited to run distance-based pair selection.'
    origin = {}
    directions = {}
    map_method = {(True, False, False): get_gps_point, (True, True, False): get_gps_opk_point}
    had_orientation = False
    for image in images:
        exif = exifs[image]
        has_gps = 'gps' in exif and 'latitude' in exif['gps'] and ('longitude' in exif['gps'])
        if not has_gps:
            continue
        has_opk = 'opk' in exif
        has_ypr = 'ypr' in exif
        had_orientation |= has_opk or has_ypr
        method_id = (has_gps, has_opk, has_ypr)
        if method_id not in map_method:
            raise RuntimeError(f'GPS / OPK / YPR {(has_gps, has_opk, has_ypr)} tag combination unsupported')
        (origin[image], directions[image]) = map_method[method_id](exif, reference)
    if had_orientation:
        altitude = find_best_altitude(origin, directions)
        logger.info(f'Altitude for orientation based matching {altitude}')
        directions_scaled = {k: v / DEFAULT_Z * altitude for (k, v) in directions.items()}
        points = {k: origin[k] + directions_scaled[k] for k in images}
    else:
        points = origin
    return points

def match_candidates_by_distance(images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, max_neighbors: int, max_distance: float) -> Set[Tuple[str, str]]:
    if False:
        return 10
    'Find candidate matching pairs by GPS distance.\n\n    The GPS altitude is ignored because we want images of the same position\n    at different altitudes to be matched together.  Otherwise, for drone\n    datasets, flights at different altitudes do not get matched.\n    '
    if len(images_cand) == 0:
        return set()
    if max_neighbors <= 0 and max_distance <= 0:
        return set()
    max_neighbors = max_neighbors or 99999999
    max_distance = max_distance or 99999999.0
    k = min(len(images_cand), max_neighbors)
    representative_points = get_representative_points(images_cand + images_ref, exifs, reference)
    difference = abs(len(representative_points) - len(set(images_cand + images_ref)))
    if difference > 0:
        logger.warning(f"Couldn't fetch {difference} images. Returning NO pairs.")
        return set()
    points = np.zeros((len(representative_points), 3))
    for (i, point_id) in enumerate(images_cand):
        points[i] = representative_points[point_id]
    tree = spatial.cKDTree(points)
    pairs = set()
    for image_ref in images_ref:
        nn = k + 1 if image_ref in images_cand else k
        point = representative_points[image_ref]
        (distances, neighbors) = tree.query(point, k=nn, distance_upper_bound=max_distance)
        if type(neighbors) == int:
            neighbors = [neighbors]
        for j in neighbors:
            if j >= len(images_cand):
                continue
            image_cand = images_cand[j]
            if image_cand != image_ref:
                pairs.add(sorted_pair(image_ref, image_cand))
    return pairs

def norm_2d(vec: np.ndarray) -> float:
    if False:
        return 10
    'Return the 2D norm of a vector.'
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

def match_candidates_by_graph(images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, rounds: int) -> Set[Tuple[str, str]]:
    if False:
        return 10
    'Find by triangulating the GPS points on X/Y axises'
    if len(images_cand) < 4 or rounds < 1:
        return set()
    images_cand_set = set(images_cand)
    images_ref_set = set(images_ref)
    images = list(images_cand_set | images_ref_set)
    representative_points = get_representative_points(images, exifs, reference)
    points = np.zeros((len(images), 2))
    for (i, point) in enumerate(representative_points.values()):
        points[i] = point[0:2]

    def produce_edges(triangles):
        if False:
            print('Hello World!')
        for triangle in triangles:
            for (vertex1, vertex2) in combinations(triangle, 2):
                (image1, image2) = (images[vertex1], images[vertex2])
                if image1 == image2:
                    continue
                pair_way1 = image1 in images_cand_set and image2 in images_ref_set
                pair_way2 = image2 in images_cand_set and image1 in images_ref_set
                if pair_way1 or pair_way2:
                    yield (sorted_pair(image1, image2), (vertex1, vertex2))
    pairs = set()
    edge_distances = []
    try:
        triangles = spatial.Delaunay(points).simplices
    except spatial.QhullError:
        triangles = spatial.Delaunay(points, qhull_options='Qbb Qc Qz Q12 QbB').simplices
    for ((image1, image2), (vertex1, vertex2)) in produce_edges(triangles):
        pairs.add((image1, image2))
        edge_distances.append(norm_2d(points[vertex1] - points[vertex2]))
    scale = np.median(edge_distances)
    for _ in range(rounds):
        points_current = copy.copy(points) + np.random.rand(*points.shape) * scale
        triangles = spatial.Delaunay(points_current).simplices
        for ((image1, image2), _) in produce_edges(triangles):
            pairs.add((image1, image2))
    return pairs

def match_candidates_with_bow(data: DataSetBase, images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, max_neighbors: int, max_gps_distance: float, max_gps_neighbors: int, enforce_other_cameras: bool) -> Dict[Tuple[str, str], float]:
    if False:
        print('Hello World!')
    'Find candidate matching pairs using BoW-based distance.\n\n    If max_gps_distance > 0, then we use first restrain a set of\n    candidates using max_gps_neighbors neighbors selected using\n    GPS distance.\n\n    If enforce_other_cameras is True, we keep max_neighbors images\n    with same cameras AND max_neighbors images from any other different\n    camera.\n    '
    if max_neighbors <= 0:
        return {}
    results = compute_bow_affinity(data, images_ref, images_cand, exifs, reference, max_gps_distance, max_gps_neighbors)
    return construct_pairs(results, max_neighbors, exifs, enforce_other_cameras)

def compute_bow_affinity(data: DataSetBase, images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, max_gps_distance: float, max_gps_neighbors: int) -> List[Tuple[str, List[float], List[str]]]:
    if False:
        while True:
            i = 10
    'Compute affinity scores between references and candidates\n    images using BoW-based distance.\n    '
    (preempted_candidates, need_load) = preempt_candidates(images_ref, images_cand, exifs, reference, max_gps_neighbors, max_gps_distance)
    logger.info('Computing %d BoW histograms' % len(need_load))
    histograms = load_histograms(data, need_load)
    (args, processes, batch_size) = create_parallel_matching_args(data, preempted_candidates, histograms)
    logger.info('Computing BoW candidates with %d processes' % processes)
    return context.parallel_map(match_bow_unwrap_args, args, processes, batch_size)

def match_candidates_with_vlad(data: DataSetBase, images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, max_neighbors: int, max_gps_distance: float, max_gps_neighbors: int, enforce_other_cameras: bool, histograms: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    if False:
        print('Hello World!')
    'Find candidate matching pairs using VLAD-based distance.\n     If max_gps_distance > 0, then we use first restrain a set of\n    candidates using max_gps_neighbors neighbors selected using\n    GPS distance.\n\n    If enforce_other_cameras is True, we keep max_neighbors images\n    with same cameras AND max_neighbors images from any other different\n    camera.\n\n    If enforce_other_cameras is False, we keep max_neighbors images\n    from all cameras.\n\n    Pre-computed VLAD histograms can be passed as a dictionary.\n    Missing histograms will be computed and added to it.\n    '
    if max_neighbors <= 0:
        return {}
    results = compute_vlad_affinity(data, images_ref, images_cand, exifs, reference, max_gps_distance, max_gps_neighbors, histograms)
    return construct_pairs(results, max_neighbors, exifs, enforce_other_cameras)

def compute_vlad_affinity(data: DataSetBase, images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, max_gps_distance: float, max_gps_neighbors: int, histograms: Dict[str, np.ndarray]) -> List[Tuple[str, List[float], List[str]]]:
    if False:
        return 10
    'Compute affinity scores between references and candidates\n    images using VLAD-based distance.\n    '
    (preempted_candidates, need_load) = preempt_candidates(images_ref, images_cand, exifs, reference, max_gps_neighbors, max_gps_distance)
    if len(preempted_candidates) == 0:
        logger.warning(f"Couldn't preempt any candidate with GPS, using ALL {len(images_cand)} as candidates")
        preempted_candidates = {image: images_cand for image in images_ref}
        need_load = set(images_ref + images_cand)
    need_load = {im for im in need_load if im not in histograms}
    logger.info('Computing %d VLAD histograms' % len(need_load))
    histograms.update(vlad_histograms(need_load, data))
    (args, processes, batch_size) = create_parallel_matching_args(data, preempted_candidates, histograms)
    logger.info('Computing VLAD candidates with %d processes' % processes)
    return context.parallel_map(match_vlad_unwrap_args, args, processes, batch_size)

def preempt_candidates(images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], reference: geo.TopocentricConverter, max_gps_neighbors: int, max_gps_distance: float) -> Tuple[Dict[str, list], Set[str]]:
    if False:
        while True:
            i = 10
    'Preempt candidates using GPS to reduce set of images\n    from which to load data to save RAM.\n    '
    preempted_cand = {im: images_cand for im in images_ref}
    if max_gps_distance > 0 or max_gps_neighbors > 0:
        gps_pairs = match_candidates_by_distance(images_ref, images_cand, exifs, reference, max_gps_neighbors, max_gps_distance)
        preempted_cand = defaultdict(list)
        for p in gps_pairs:
            if p[0] in images_ref:
                preempted_cand[p[0]].append(p[1])
            if p[1] in images_ref:
                preempted_cand[p[1]].append(p[0])
    need_load = set(preempted_cand.keys())
    for (k, v) in preempted_cand.items():
        need_load.update(v)
        need_load.add(k)
    return (preempted_cand, need_load)

def construct_pairs(results: List[Tuple[str, List[float], List[str]]], max_neighbors: int, exifs: Dict[str, Any], enforce_other_cameras: bool) -> Dict[Tuple[str, str], float]:
    if False:
        while True:
            i = 10
    'Construct final sets of pairs to match'
    pairs = {}
    for (im, distances, other) in results:
        order = np.argsort(distances)
        if enforce_other_cameras:
            pairs.update(pairs_from_neighbors(im, exifs, distances, order, other, max_neighbors))
        else:
            for i in order[:max_neighbors]:
                pairs[sorted_pair(im, other[i])] = distances[i]
    return pairs

def create_parallel_matching_args(data: DataSetBase, preempted_cand: Dict[str, list], histograms: Dict[str, np.ndarray]) -> Tuple[List[Tuple[str, list, Dict[str, np.ndarray]]], int, int]:
    if False:
        print('Hello World!')
    'Create arguments to matching function'
    args = [(im, cands, histograms) for (im, cands) in preempted_cand.items()]
    per_process = 512
    processes = context.processes_that_fit_in_memory(data.config['processes'], per_process)
    batch_size = max(1, len(args) // (2 * processes))
    return (args, processes, batch_size)

def match_bow_unwrap_args(args: Tuple[str, Iterable[str], Dict[str, np.ndarray]]) -> Tuple[str, List[float], List[str]]:
    if False:
        i = 10
        return i + 15
    'Wrapper for parallel processing of BoW'
    (image, other_images, histograms) = args
    return bow_distances(image, other_images, histograms)

def match_vlad_unwrap_args(args: Tuple[str, Iterable[str], Dict[str, np.ndarray]]) -> Tuple[str, List[float], List[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Wrapper for parallel processing of VLAD'
    (image, other_images, histograms) = args
    return vlad.vlad_distances(image, other_images, histograms)

def match_candidates_by_time(images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], max_neighbors: int) -> Set[Tuple[str, str]]:
    if False:
        print('Hello World!')
    'Find candidate matching pairs by time difference.'
    if max_neighbors <= 0:
        return set()
    k = min(len(images_cand), max_neighbors)
    times = np.zeros((len(images_cand), 1))
    for (i, image) in enumerate(images_cand):
        times[i] = exifs[image]['capture_time']
    tree = spatial.cKDTree(times)
    pairs = set()
    for image_ref in images_ref:
        nn = k + 1 if image_ref in images_cand else k
        time = exifs[image_ref]['capture_time']
        (distances, neighbors) = tree.query([time], k=nn)
        if type(neighbors) == int:
            neighbors = [neighbors]
        for j in neighbors:
            if j >= len(images_cand):
                continue
            image_cand = images_cand[j]
            if image_ref != image_cand:
                pairs.add(sorted_pair(image_ref, image_cand))
    return pairs

def match_candidates_by_order(images_ref: List[str], images_cand: List[str], max_neighbors: int) -> Set[Tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    'Find candidate matching pairs by sequence order.'
    if max_neighbors <= 0:
        return set()
    n = (max_neighbors + 1) // 2
    pairs = set()
    for (i, image_ref) in enumerate(images_ref):
        a = max(0, i - n)
        b = min(len(images_cand), i + n)
        for j in range(a, b):
            image_cand = images_cand[j]
            if image_ref != image_cand:
                pairs.add(sorted_pair(image_ref, image_cand))
    return pairs

def match_candidates_from_metadata(images_ref: List[str], images_cand: List[str], exifs: Dict[str, Any], data: DataSetBase, config_override: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'Compute candidate matching pairs between between images_ref and images_cand\n\n    Returns a list of pairs (im1, im2) such that (im1 in images_ref) is true.\n    Returned pairs are unique given that (i, j) == (j, i).\n    '
    overriden_config = data.config.copy()
    overriden_config.update(config_override)
    max_distance = overriden_config['matching_gps_distance']
    gps_neighbors = overriden_config['matching_gps_neighbors']
    graph_rounds = overriden_config['matching_graph_rounds']
    time_neighbors = overriden_config['matching_time_neighbors']
    order_neighbors = overriden_config['matching_order_neighbors']
    bow_neighbors = overriden_config['matching_bow_neighbors']
    bow_gps_distance = overriden_config['matching_bow_gps_distance']
    bow_gps_neighbors = overriden_config['matching_bow_gps_neighbors']
    bow_other_cameras = overriden_config['matching_bow_other_cameras']
    vlad_neighbors = overriden_config['matching_vlad_neighbors']
    vlad_gps_distance = overriden_config['matching_vlad_gps_distance']
    vlad_gps_neighbors = overriden_config['matching_vlad_gps_neighbors']
    vlad_other_cameras = overriden_config['matching_vlad_other_cameras']
    data.init_reference()
    reference = data.load_reference()
    if not all(map(has_gps_info, exifs.values())):
        if gps_neighbors != 0:
            logger.warn('Not all images have GPS info. Disabling matching_gps_neighbors.')
        gps_neighbors = 0
        max_distance = 0
        graph_rounds = 0
    images_ref.sort()
    if max_distance == gps_neighbors == time_neighbors == order_neighbors == bow_neighbors == vlad_neighbors == graph_rounds == 0:
        d = set()
        t = set()
        g = set()
        o = set()
        b = set()
        v = set()
        pairs = {sorted_pair(i, j) for i in images_ref for j in images_cand if i != j}
    else:
        d = match_candidates_by_distance(images_ref, images_cand, exifs, reference, gps_neighbors, max_distance)
        g = match_candidates_by_graph(images_ref, images_cand, exifs, reference, graph_rounds)
        t = match_candidates_by_time(images_ref, images_cand, exifs, time_neighbors)
        o = match_candidates_by_order(images_ref, images_cand, order_neighbors)
        b = match_candidates_with_bow(data, images_ref, images_cand, exifs, reference, bow_neighbors, bow_gps_distance, bow_gps_neighbors, bow_other_cameras)
        v = match_candidates_with_vlad(data, images_ref, images_cand, exifs, reference, vlad_neighbors, vlad_gps_distance, vlad_gps_neighbors, vlad_other_cameras, {})
        pairs = d | g | t | o | set(b) | set(v)
    pairs = ordered_pairs(pairs, images_ref)
    report = {'num_pairs_distance': len(d), 'num_pairs_graph': len(g), 'num_pairs_time': len(t), 'num_pairs_order': len(o), 'num_pairs_bow': len(b), 'num_pairs_vlad': len(v)}
    return (pairs, report)

def bow_distances(image: str, other_images: Iterable[str], histograms: Dict[str, np.ndarray]) -> Tuple[str, List[float], List[str]]:
    if False:
        i = 10
        return i + 15
    'Compute BoW-based distance (L1 on histogram of words)\n    between an image and other images.\n    '
    if image not in histograms:
        return (image, [], [])
    distances = []
    other = []
    h = histograms[image]
    for im2 in other_images:
        if im2 != image and im2 in histograms:
            h2 = histograms[im2]
            distances.append(np.fabs(h - h2).sum())
            other.append(im2)
    return (image, distances, other)

def load_histograms(data: DataSetBase, images: Iterable[str]) -> Dict[str, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Load BoW histograms of given images'
    min_num_feature = 8
    histograms = {}
    bows = bow.load_bows(data.config)
    for im in images:
        filtered_words = feature_loader.instance.load_words(data, im, masked=True)
        if filtered_words is None:
            logger.error('No words in image {}'.format(im))
            continue
        if len(filtered_words) <= min_num_feature:
            logger.warning('Too few filtered features in image {}: {}'.format(im, len(filtered_words)))
            continue
        histograms[im] = bows.histogram(filtered_words[:, 0])
    return histograms

def vlad_histogram_unwrap_args(args: Tuple[DataSetBase, str]) -> Optional[Tuple[str, np.ndarray]]:
    if False:
        print('Hello World!')
    'Helper function for multithreaded VLAD computation.\n\n    Returns the image and its descriptor.\n    '
    (data, image) = args
    vlad_descriptor = vlad.instance.vlad_histogram(data, image)
    if vlad_descriptor is not None:
        return (image, vlad_descriptor)
    else:
        logger.warning(f"Couldn't compute VLAD descriptor for image {image}")
        return None

def vlad_histograms(images: Iterable[str], data: DataSetBase) -> Dict[str, np.ndarray]:
    if False:
        print('Hello World!')
    'Construct VLAD histograms from the image features.\n\n    Returns a dictionary of VLAD vectors for the images.\n    '
    batch_size = 4
    vlads = context.parallel_map(vlad_histogram_unwrap_args, [(data, image) for image in images], data.config['processes'], batch_size)
    return {v[0]: v[1] for v in vlads if v}

def pairs_from_neighbors(image: str, exifs: Dict[str, Any], distances: List[float], order: List[int], other: List[str], max_neighbors: int) -> Dict[Tuple[str, str], float]:
    if False:
        i = 10
        return i + 15
    'Construct matching pairs given closest ordered neighbors.\n\n    Pairs will of form (image, im2), im2 being the closest max_neighbors\n    given by (order, other) having the same cameras OR the closest max_neighbors\n    having from any other camera.\n    '
    (same_camera, other_cameras) = ([], [])
    for i in order:
        im2 = other[i]
        d = distances[i]
        if exifs[im2]['camera'] == exifs[image]['camera']:
            if len(same_camera) < max_neighbors:
                same_camera.append((im2, d))
        elif len(other_cameras) < max_neighbors:
            other_cameras.append((im2, d))
        if len(same_camera) + len(other_cameras) >= 2 * max_neighbors:
            break
    pairs = {}
    for (im2, d) in same_camera + other_cameras:
        pairs[tuple(sorted((image, im2)))] = d
    return pairs

def ordered_pairs(pairs: Set[Tuple[str, str]], images_ref: List[str]) -> List[Tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    'Image pairs that need matching skipping duplicates.\n\n    Returns a list of pairs (im1, im2) such that (im1 in images_ref) is true.\n    '
    per_image = defaultdict(list)
    for (im1, im2) in pairs:
        per_image[im1].append(im2)
        per_image[im2].append(im1)
    ordered = set()
    remaining = set(images_ref)
    if len(remaining) > 0:
        next_image = remaining.pop()
        while next_image:
            im1 = next_image
            next_image = None
            for im2 in per_image[im1]:
                if (im2, im1) not in ordered:
                    ordered.add((im1, im2))
                    if not next_image and im2 in remaining:
                        next_image = im2
                        remaining.remove(im2)
            if not next_image and remaining:
                next_image = remaining.pop()
    return list(ordered)