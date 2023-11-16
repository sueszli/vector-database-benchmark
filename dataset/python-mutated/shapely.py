from typing import List
from shapely.geometry import CAP_STYLE, JOIN_STYLE, MultiPolygon, Polygon, box

def get_shapely_box(x: int, y: int, width: int, height: int) -> Polygon:
    if False:
        while True:
            i = 10
    '\n    Accepts coco style bbox coords and converts it to shapely box object\n    '
    minx = x
    miny = y
    maxx = x + width
    maxy = y + height
    shapely_box = box(minx, miny, maxx, maxy)
    return shapely_box

def get_shapely_multipolygon(coco_segmentation: List[List]) -> MultiPolygon:
    if False:
        while True:
            i = 10
    '\n    Accepts coco style polygon coords and converts it to shapely multipolygon object\n    '
    polygon_list = []
    for coco_polygon in coco_segmentation:
        point_list = list(zip(coco_polygon[0::2], coco_polygon[1::2]))
        shapely_polygon = Polygon(point_list)
        polygon_list.append(shapely_polygon)
    shapely_multipolygon = MultiPolygon(polygon_list)
    return shapely_multipolygon

def get_bbox_from_shapely(shapely_object):
    if False:
        for i in range(10):
            print('nop')
    '\n    Accepts shapely box/poly object and returns its bounding box in coco and voc formats\n    '
    (minx, miny, maxx, maxy) = shapely_object.bounds
    width = maxx - minx
    height = maxy - miny
    coco_bbox = [minx, miny, width, height]
    voc_bbox = [minx, miny, maxx, maxy]
    return (coco_bbox, voc_bbox)

class ShapelyAnnotation:
    """
    Creates ShapelyAnnotation (as shapely MultiPolygon).
    Can convert this instance annotation to various formats.
    """

    @classmethod
    def from_coco_segmentation(cls, segmentation, slice_bbox=None):
        if False:
            return 10
        '\n        Init ShapelyAnnotation from coco segmentation.\n\n        segmentation : List[List]\n            [[1, 1, 325, 125, 250, 200, 5, 200]]\n        slice_bbox (List[int]): [xmin, ymin, width, height]\n            Should have the same format as the output of the get_bbox_from_shapely function.\n            Is used to calculate sliced coco coordinates.\n        '
        shapely_multipolygon = get_shapely_multipolygon(segmentation)
        return cls(multipolygon=shapely_multipolygon, slice_bbox=slice_bbox)

    @classmethod
    def from_coco_bbox(cls, bbox: List[int], slice_bbox: List[int]=None):
        if False:
            return 10
        '\n        Init ShapelyAnnotation from coco bbox.\n\n        bbox (List[int]): [xmin, ymin, width, height]\n        slice_bbox (List[int]): [x_min, y_min, x_max, y_max] Is used\n            to calculate sliced coco coordinates.\n        '
        shapely_polygon = get_shapely_box(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        shapely_multipolygon = MultiPolygon([shapely_polygon])
        return cls(multipolygon=shapely_multipolygon, slice_bbox=slice_bbox)

    def __init__(self, multipolygon: MultiPolygon, slice_bbox=None):
        if False:
            while True:
                i = 10
        self.multipolygon = multipolygon
        self.slice_bbox = slice_bbox

    @property
    def multipolygon(self):
        if False:
            return 10
        return self.__multipolygon

    @property
    def area(self):
        if False:
            print('Hello World!')
        return int(self.__area)

    @multipolygon.setter
    def multipolygon(self, multipolygon: MultiPolygon):
        if False:
            print('Hello World!')
        self.__multipolygon = multipolygon
        area = 0
        for shapely_polygon in multipolygon.geoms:
            area += shapely_polygon.area
        self.__area = area

    def to_list(self):
        if False:
            return 10
        '\n        [\n            [(x1, y1), (x2, y2), (x3, y3), ...],\n            [(x1, y1), (x2, y2), (x3, y3), ...],\n            ...\n        ]\n        '
        list_of_list_of_points: List = []
        for shapely_polygon in self.multipolygon.geoms:
            if shapely_polygon.area != 0:
                x_coords = shapely_polygon.exterior.coords.xy[0]
                y_coords = shapely_polygon.exterior.coords.xy[1]
                if self.slice_bbox:
                    minx = self.slice_bbox[0]
                    miny = self.slice_bbox[1]
                    x_coords = [x_coord - minx for x_coord in x_coords]
                    y_coords = [y_coord - miny for y_coord in y_coords]
                list_of_points = list(zip(x_coords, y_coords))
            else:
                list_of_points = []
            list_of_list_of_points.append(list_of_points)
        return list_of_list_of_points

    def to_coco_segmentation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        [\n            [x1, y1, x2, y2, x3, y3, ...],\n            [x1, y1, x2, y2, x3, y3, ...],\n            ...\n        ]\n        '
        coco_segmentation: List = []
        for shapely_polygon in self.multipolygon.geoms:
            if shapely_polygon.area != 0:
                x_coords = shapely_polygon.exterior.coords.xy[0]
                y_coords = shapely_polygon.exterior.coords.xy[1]
                if self.slice_bbox:
                    minx = self.slice_bbox[0]
                    miny = self.slice_bbox[1]
                    x_coords = [x_coord - minx for x_coord in x_coords]
                    y_coords = [y_coord - miny for y_coord in y_coords]
                coco_polygon = [None] * len(x_coords) * 2
                coco_polygon[0::2] = [int(coord) for coord in x_coords]
                coco_polygon[1::2] = [int(coord) for coord in y_coords]
            else:
                coco_polygon = []
            if coco_polygon[:2] == coco_polygon[-2:]:
                del coco_polygon[-2:]
            coco_polygon = [point for point in coco_polygon] if coco_polygon else coco_polygon
            coco_segmentation.append(coco_polygon)
        return coco_segmentation

    def to_opencv_contours(self):
        if False:
            while True:
                i = 10
        '\n        [\n            [[[1, 1]], [[325, 125]], [[250, 200]], [[5, 200]]],\n            [[[1, 1]], [[325, 125]], [[250, 200]], [[5, 200]]]\n        ]\n        '
        opencv_contours: List = []
        for shapely_polygon in self.multipolygon.geoms:
            if shapely_polygon.area != 0:
                x_coords = shapely_polygon.exterior.coords.xy[0]
                y_coords = shapely_polygon.exterior.coords.xy[1]
                if self.slice_bbox:
                    minx = self.slice_bbox[0]
                    miny = self.slice_bbox[1]
                    x_coords = [x_coord - minx for x_coord in x_coords]
                    y_coords = [y_coord - miny for y_coord in y_coords]
                opencv_contour = [[[int(x_coords[ind]), int(y_coords[ind])]] for ind in range(len(x_coords))]
            else:
                opencv_contour: List = []
            opencv_contours.append(opencv_contour)
        return opencv_contours

    def to_xywh(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        [xmin, ymin, width, height]\n        '
        if self.multipolygon.area != 0:
            (coco_bbox, _) = get_bbox_from_shapely(self.multipolygon)
            if self.slice_bbox:
                minx = self.slice_bbox[0]
                miny = self.slice_bbox[1]
                coco_bbox[0] = coco_bbox[0] - minx
                coco_bbox[1] = coco_bbox[1] - miny
        else:
            coco_bbox: List = []
        return coco_bbox

    def to_coco_bbox(self):
        if False:
            while True:
                i = 10
        '\n        [xmin, ymin, width, height]\n        '
        return self.to_xywh()

    def to_xyxy(self):
        if False:
            i = 10
            return i + 15
        '\n        [xmin, ymin, xmax, ymax]\n        '
        if self.multipolygon.area != 0:
            (_, voc_bbox) = get_bbox_from_shapely(self.multipolygon)
            if self.slice_bbox:
                minx = self.slice_bbox[0]
                miny = self.slice_bbox[1]
                voc_bbox[0] = voc_bbox[0] - minx
                voc_bbox[2] = voc_bbox[2] - minx
                voc_bbox[1] = voc_bbox[1] - miny
                voc_bbox[3] = voc_bbox[3] - miny
        else:
            voc_bbox = []
        return voc_bbox

    def to_voc_bbox(self):
        if False:
            print('Hello World!')
        '\n        [xmin, ymin, xmax, ymax]\n        '
        return self.to_xyxy()

    def get_convex_hull_shapely_annotation(self):
        if False:
            while True:
                i = 10
        shapely_multipolygon = MultiPolygon([self.multipolygon.convex_hull])
        shapely_annotation = ShapelyAnnotation(shapely_multipolygon)
        return shapely_annotation

    def get_simplified_shapely_annotation(self, tolerance=1):
        if False:
            while True:
                i = 10
        shapely_multipolygon = MultiPolygon([self.multipolygon.simplify(tolerance)])
        shapely_annotation = ShapelyAnnotation(shapely_multipolygon)
        return shapely_annotation

    def get_buffered_shapely_annotation(self, distance=3, resolution=16, quadsegs=None, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round, mitre_limit=5.0, single_sided=False):
        if False:
            while True:
                i = 10
        '\n        Approximates the present polygon to have a valid polygon shape.\n        For more, check: https://shapely.readthedocs.io/en/stable/manual.html#object.buffer\n        '
        buffered_polygon = self.multipolygon.buffer(distance=distance, resolution=resolution, quadsegs=quadsegs, cap_style=cap_style, join_style=join_style, mitre_limit=mitre_limit, single_sided=single_sided)
        shapely_annotation = ShapelyAnnotation(MultiPolygon([buffered_polygon]))
        return shapely_annotation

    def get_intersection(self, polygon: Polygon):
        if False:
            print('Hello World!')
        '\n        Accepts shapely polygon object and returns the intersection in ShapelyAnnotation format\n        '
        intersection = self.multipolygon.intersection(polygon)
        if len(polygon.exterior.xy[0]) == 5 and polygon.exterior.xy[0][0] == polygon.exterior.xy[0][1] and (polygon.exterior.xy[0][2] == polygon.exterior.xy[0][3]):
            (coco_bbox, voc_bbox) = get_bbox_from_shapely(polygon)
            slice_bbox = coco_bbox
        else:
            slice_bbox = None
        if intersection.geom_type == 'Polygon':
            intersection_multipolygon = MultiPolygon([intersection])
        elif intersection.geom_type == 'MultiPolygon':
            intersection_multipolygon = intersection
        else:
            intersection_multipolygon = MultiPolygon([])
        intersection_shapely_annotation = ShapelyAnnotation(intersection_multipolygon, slice_bbox)
        return intersection_shapely_annotation