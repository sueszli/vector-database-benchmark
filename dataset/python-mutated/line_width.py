import open3d as o3d
import random
NUM_LINES = 10

def random_point():
    if False:
        print('Hello World!')
    return [5 * random.random(), 5 * random.random(), 5 * random.random()]

def main():
    if False:
        i = 10
        return i + 15
    pts = [random_point() for _ in range(0, 2 * NUM_LINES)]
    line_indices = [[2 * i, 2 * i + 1] for i in range(0, NUM_LINES)]
    colors = [[0.0, 0.0, 0.0] for _ in range(0, NUM_LINES)]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(pts)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    lines.colors = o3d.utility.Vector3dVector(colors)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'unlitLine'
    mat.line_width = 10
    o3d.visualization.draw({'name': 'lines', 'geometry': lines, 'material': mat})
if __name__ == '__main__':
    main()