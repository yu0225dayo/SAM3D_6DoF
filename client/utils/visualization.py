"""
pose 推定結果の可視化ユーティリティ
"""

import numpy as np
import cv2 as _cv2


def project_pointcloud_on_image(
    bgr: np.ndarray,
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics,
    point_color=(0, 255, 0),
    bbox_color=(0, 255, 255),
    point_size: int = 2,
    points_unit: str = "mm",
) -> np.ndarray:
    """
    3D点群とbboxをRGB画像に投影して重ね描きする

    Args:
        bgr:         (H, W, 3) カメラ画像 (BGR)
        points_3d:   (N, 3) 物体座標系の点群
        R:           (3, 3) 回転行列 (物体→カメラ座標系)
        t:           (3,)   平行移動 [m]
        intrinsics:  CameraIntrinsics
        point_color: 点群の描画色 (BGR)
        bbox_color:  バウンディングボックスの描画色 (BGR)
        point_size:  点の半径 [px]
        points_unit: 点群の単位 ("mm" or "m"). SAM-3D生成メッシュは "mm"

    Returns:
        (H, W, 3) 投影結果画像
    """
    result = bgr.copy()
    h, w = bgr.shape[:2]

    pts_m = points_3d / 1000.0 if points_unit == "mm" else points_3d
    pts_cam = (R @ pts_m.T).T + t  # (N, 3) [m]

    valid = pts_cam[:, 2] > 0.01
    pts_valid = pts_cam[valid]
    us = (intrinsics.fx * pts_valid[:, 0] / pts_valid[:, 2] + intrinsics.cx).astype(np.int32)
    vs = (intrinsics.fy * pts_valid[:, 1] / pts_valid[:, 2] + intrinsics.cy).astype(np.int32)

    in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    for u, v in zip(us[in_bounds], vs[in_bounds]):
        _cv2.circle(result, (int(u), int(v)), point_size, point_color, -1)

    mins = pts_m.min(axis=0)
    maxs = pts_m.max(axis=0)
    corners = np.array([
        [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]],
    ])
    corners_cam = (R @ corners.T).T + t
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        c1, c2 = corners_cam[i], corners_cam[j]
        if c1[2] > 0.01 and c2[2] > 0.01:
            u1 = int(intrinsics.fx * c1[0] / c1[2] + intrinsics.cx)
            v1 = int(intrinsics.fy * c1[1] / c1[2] + intrinsics.cy)
            u2 = int(intrinsics.fx * c2[0] / c2[2] + intrinsics.cx)
            v2 = int(intrinsics.fy * c2[1] / c2[2] + intrinsics.cy)
            if (0 <= u1 < w and 0 <= v1 < h) or (0 <= u2 < w and 0 <= v2 < h):
                _cv2.line(result, (u1, v1), (u2, v2), bbox_color, 2, _cv2.LINE_AA)

    return result


def render_mesh_on_image(
    bgr: np.ndarray,
    mesh_path: str,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics,
    mesh_unit: str = "mm",
    alpha: float = 0.6,
    mesh_color: tuple = (0.2, 0.8, 0.2),
) -> np.ndarray:
    """
    Open3D でメッシュ面上に点をサンプリングして画像に投影する

    Args:
        bgr:        (H, W, 3) カメラ画像 (BGR)
        mesh_path:  三角メッシュPLYファイルパス
        R:          (3, 3) 回転行列 (物体→カメラ座標系)
        t:          (3,)   平行移動 [m]
        intrinsics: CameraIntrinsics
        mesh_unit:  メッシュの単位 ("mm" or "m"). SAM-3D生成メッシュは "mm"
        alpha:      オーバーレイの不透明度
        mesh_color: メッシュの色 (R, G, B) 各0.0〜1.0

    Returns:
        (H, W, 3) レンダリング結果画像 (BGR)
    """
    pts = None
    try:
        import open3d as o3d
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh_o3d.triangles) > 0:
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=10000)
            pts = np.asarray(pcd.points).astype(np.float32)
        else:
            pts = np.asarray(mesh_o3d.vertices).astype(np.float32)
    except Exception as e:
        print(f"[render_mesh] open3d でのメッシュ読み込み失敗 ({e}). フォールバック")
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(mesh_path)
            pts = np.asarray(pcd.points).astype(np.float32)
        except Exception:
            from utils.pointcloud_utils import load_pointcloud_ply
            pts = load_pointcloud_ply(mesh_path)

    point_color_bgr = (
        int(mesh_color[2] * 255),
        int(mesh_color[1] * 255),
        int(mesh_color[0] * 255),
    )
    return project_pointcloud_on_image(
        bgr, pts, R, t, intrinsics,
        point_color=point_color_bgr,
        bbox_color=(0, 220, 220),
        point_size=1,
        points_unit=mesh_unit,
    )
