import open3d as o3d
import numpy as np

def save_point_cloud(xyz, filename='pc.ply', color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(xyz) if color is None else color)
    o3d.io.write_point_cloud(filename, pcd)
    print(f'pc save to {filename}')

def save_box_corners(boxes_corners_points, filename='box.ply', color=(1.,0.,0.)):
    box_lines = np.array([[2,3],[0,3],[4,5],[4,7],[5,6],[6,7],[0,4],[1,5],[2,6],[3,7], [0,1], [1,2]]) # [0,1] front down edge # [1,2] # right down edges
    points = boxes_corners_points.reshape(-1, 3)

    lines = []
    for i, b in enumerate(boxes_corners_points):
        lines.append( box_lines + i * 8 )
    lines = np.concatenate(lines)

    color = np.array([color for j in range(len(lines))])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(filename, lineset)

    print(f'lineset save to {filename}')
    return lineset
