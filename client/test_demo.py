"""
テストデータを使った 6DoF pose 推定パイプラインのデモ

RealSense なしでフォルダから RGB + 深度を読み込み、
SAM-3D でメッシュ生成 → 重力方向から高さ推定 → SAM-6D で pose 推定 → 可視化する。

フォルダ構成:
    demo_data/demo1/
        rgb.png
        depth.png
        cam.json   ({"cam_K": [fx,0,cx,0,fy,cy,0,0,1], "depth_scale": 0.001, "gravity": [gx,gy,gz]})

使用方法:
    python test_demo.py demo_data/demo1
    python test_demo.py demo_data/demo1 --gravity 0 -1 0
    python test_demo.py demo_data/demo1 --click-x 320 --click-y 240 --no-show
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import cv2
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intrinsics(data_dir: str, img_w: int, img_h: int,
                    depth_scale_ref: list, gravity_ref: list):
    """
    cam.json から intrinsics を読み込む。
    gravity_ref[0] に cam.json の gravity ベクトルを格納する（存在する場合）。
    """
    from utils.coord_transform import CameraIntrinsics

    cam_json = os.path.join(data_dir, "cam.json")
    if os.path.exists(cam_json):
        with open(cam_json, "r") as f:
            cam = json.load(f)
        K = cam["cam_K"]
        if len(K) != 9:
            raise ValueError(f"cam_K の形式が不正: {K}")
        fx, cx = K[0], K[2]
        fy, cy = K[4], K[5]
        if "depth_scale" in cam:
            depth_scale_ref[0] = cam["depth_scale"]
        if "gravity" in cam:
            gravity_ref[0] = cam["gravity"]
            print(f"[intrinsics] cam.json から gravity 読み込み: {cam['gravity']}")
        print(f"[intrinsics] fx={fx} fy={fy} cx={cx} cy={cy}")
    else:
        fx, fy = 591.0, 590.0
        cx, cy = img_w / 2, img_h / 2
        print("[intrinsics] cam.json なし → デフォルト値を使用")

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=img_w, height=img_h)


def load_depth(depth_path: str, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"深度画像が読み込めません: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale
    print(f"[depth] shape={depth_raw.shape}  range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


# ===========================================================================
# IMU・高さ計算ユーティリティ
# ===========================================================================

def get_gravity_imu(n_samples: int = 30) -> np.ndarray:
    """RealSense 加速度センサーから重力方向の単位ベクトルを取得する。"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError(
            "pyrealsense2 がインストールされていません。\n"
            "  pip install pyrealsense2\n"
            "または --gravity で重力ベクトルを手動指定してください。"
        )

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

    print(f"[IMU] RealSense IMU を起動して重力ベクトルを取得中 ({n_samples} サンプル)...")
    profile = pipeline.start(cfg)
    samples = []
    collected = 0
    try:
        while collected < n_samples:
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if not accel_frame:
                continue
            motion = accel_frame.as_motion_frame().get_motion_data()
            samples.append([motion.x, motion.y, motion.z])
            collected += 1
    finally:
        pipeline.stop()

    g = np.mean(samples, axis=0)
    g = g / np.linalg.norm(g)
    print(f"[IMU] 重力ベクトル g = [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]")
    return g.astype(np.float64)


def get_points_3d_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
) -> np.ndarray:
    """マスク領域内の深度画像を 3D 点群に変換する。"""
    mask_bool = mask.astype(bool)
    vs, us = np.where(mask_bool)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d = depth_m[vs, us].astype(np.float64)
    valid = (d > min_depth) & (d < max_depth)
    vs, us, d = vs[valid], us[valid], d[valid]

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    X = (us - cx) * d / fx
    Y = (vs - cy) * d / fy
    Z = d
    return np.stack([X, Y, Z], axis=1)


def calc_height_from_points(points_3d: np.ndarray, gravity_vec: np.ndarray) -> float:
    """3D 点群を重力方向に射影して物体の高さを計算する。"""
    projections = points_3d @ gravity_vec  # (N,)
    return float(projections.max() - projections.min())


def estimate_height_from_depth_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    gravity_vec: np.ndarray,
):
    """深度画像 + マスク + 重力ベクトル → 物体高さ [m] + 点群"""
    pts = get_points_3d_from_mask(depth_m, mask, fx, fy, cx, cy)
    if len(pts) < 10:
        print(f"[高さ推定] 有効な深度点が不足 ({len(pts)} 点)。高さ推定をスキップ。")
        return 0.0, pts
    h = calc_height_from_points(pts, gravity_vec)
    print(f"[高さ推定] 点数={len(pts)}  高さ={h:.4f} m ({h*100:.1f} cm)")
    return h, pts


def draw_height_pcd(
    rgb_bgr: np.ndarray,
    pts: np.ndarray,
    gravity_vec: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    height_m: float,
) -> np.ndarray:
    """高さで色付けした点群を RGB 画像に描画する（最高点=赤, 最低点=青）。"""
    if len(pts) == 0:
        return rgb_bgr.copy()

    img = rgb_bgr.copy()
    H, W = img.shape[:2]

    proj = pts @ gravity_vec
    Z = pts[:, 2]
    valid = Z > 0
    u = np.where(valid, (pts[:, 0] * fx / Z + cx).astype(np.int32), -1)
    v = np.where(valid, (pts[:, 1] * fy / Z + cy).astype(np.int32), -1)

    for target_idx, color in [(np.argmax(proj), (0, 0, 255)),
                               (np.argmin(proj), (255, 0, 0))]:
        if not valid[target_idx]:
            continue
        ui, vi = int(u[target_idx]), int(v[target_idx])
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        cv2.drawMarker(img, (ui, vi), color,
                       markerType=cv2.MARKER_STAR,
                       markerSize=12, thickness=1, line_type=cv2.LINE_AA)

    cv2.putText(img, f"Height: {height_m*100:.1f} cm",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "HIGH", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "LOW",  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img


# ===========================================================================
# メイン
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="テストデータで 6DoF pose 推定をデモ")
    parser.add_argument("--data-dir", default="demo_data/demo1", dest="data_dir",
                        help="データフォルダ (rgb.png / depth.png / cam.json を含む)")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--mesh-out",    default="meshes/test_object.ply")
    parser.add_argument("--click-x",     type=int, default=-1)
    parser.add_argument("--click-y",     type=int, default=-1)
    parser.add_argument("--no-show",     action="store_true",
                        help="cv2.imshow を使わない (ヘッドレス環境用)")
    parser.add_argument("--gravity", type=float, nargs=3, default=None,
                        metavar=("GX", "GY", "GZ"),
                        help="重力方向ベクトル手動指定。cam.json の gravity より優先。")
    parser.add_argument("--imu-samples", type=int, default=30,
                        help="IMU サンプル数 (デフォルト: 30)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"エラー: フォルダが見つかりません: {args.data_dir}")
        sys.exit(1)

    rgb_path   = os.path.join(args.data_dir, "rgb.png")
    depth_path = os.path.join(args.data_dir, "depth.png")
    for p in (rgb_path, depth_path):
        if not os.path.exists(p):
            print(f"エラー: {p} が見つかりません")
            sys.exit(1)

    config    = load_config(args.config)
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    from pipeline.sam6d_detector import SAM6DClient
    from utils.visualization import project_pointcloud_on_image, draw_pose_axes
    from utils.pointcloud_utils import load_pointcloud_ply

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {rgb_path}")
    print(f"[RGB] {rgb_path}  shape={rgb.shape}")

    h, w = rgb.shape[:2]
    depth_scale_ref = [0.001]
    gravity_ref     = [None]
    intrinsics = load_intrinsics(args.data_dir, w, h, depth_scale_ref, gravity_ref)

    depth = load_depth(depth_path, depth_scale_ref[0])
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    # 重力ベクトル決定 (優先順位: --gravity > cam.json > IMU)
    if args.gravity is not None:
        gravity_vec = np.array(args.gravity, dtype=np.float64)
        gravity_vec /= np.linalg.norm(gravity_vec)
        print(f"[高さ] 重力ベクトル (手動指定): {gravity_vec}")
    elif gravity_ref[0] is not None:
        gravity_vec = np.array(gravity_ref[0], dtype=np.float64)
        gravity_vec /= np.linalg.norm(gravity_vec)
        print(f"[高さ] 重力ベクトル (cam.json): {gravity_vec}")
    else:
        print("[高さ] 重力ベクトル未指定。IMU から自動取得を試みます...")
        try:
            gravity_vec = get_gravity_imu(n_samples=args.imu_samples)
        except Exception as e:
            print(f"[高さ] IMU 取得失敗: {e}\n      高さ推定をスキップします。")
            gravity_vec = None

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path   = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")
    os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)

    # Step 1: mesh 生成
    print("\n[Step 1] SAM-3D でメッシュ生成中...")
    if args.click_x >= 0 and args.click_y >= 0:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            click_x=args.click_x, click_y=args.click_y,
            mesh_method=mesh_method,
        )
        click_x, click_y = args.click_x, args.click_y
    else:
        print("[物体選択] ウィンドウで物体をクリックしてください...")
        _, click_x, click_y, masks, scores = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=mesh_method)
    print(f"[Step 1完了] mesh: {mesh_path}")

    out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    # Step 2: マスク + 深度 → 高さ推定
    object_size_mm = 0.0
    if gravity_vec is not None and masks:
        best_idx  = int(np.argmax(scores)) if scores else 0
        best_mask = masks[best_idx]
        print(f"\n[Step 2] マスクから高さ推定中 (mask_idx={best_idx}, score={scores[best_idx] if scores else '?':.3f})...")

        if best_mask.shape[:2] != depth.shape[:2]:
            best_mask = cv2.resize(best_mask, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        mask_area  = int(np.count_nonzero(best_mask))
        img_area   = best_mask.shape[0] * best_mask.shape[1]
        mask_ratio = mask_area / img_area * 100
        print(f"[Step 2] マスク面積: {mask_area}px / {img_area}px ({mask_ratio:.1f}%)")
        if mask_area > img_area * 0.02:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded_mask  = cv2.erode(best_mask, erode_kernel, iterations=1)
            print("[Step 2] → erosion 適用 (7px)")
        else:
            eroded_mask = best_mask
            print("[Step 2] → 小さいため erosion スキップ")

        height_m, pts_3d = estimate_height_from_depth_mask(
            depth, eroded_mask,
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            gravity_vec,
        )

        if height_m > 0.005:
            object_size_mm = height_m * 1000.0
            print(f"[Step 2完了] 推定高さ: {height_m*100:.1f} cm = {object_size_mm:.0f} mm")

            mask_vis     = cv2.applyColorMap(best_mask, cv2.COLORMAP_JET)
            mask_overlay = cv2.addWeighted(rgb, 0.6, mask_vis, 0.4, 0)
            cv2.imwrite(os.path.join(out_dir, "sam_mask.png"), mask_overlay)

            height_vis = draw_height_pcd(
                rgb, pts_3d, gravity_vec,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                height_m,
            )
            cv2.imwrite(os.path.join(out_dir, "height_estimation.png"), height_vis)
            print(f"[Step 2] 高さ推定画像保存: {out_dir}/height_estimation.png")
        else:
            print("[Step 2] 高さ推定失敗。object_size_mm=0 (サーバ深度推定に委譲)。")
    else:
        if gravity_vec is None:
            print("[Step 2] 重力ベクトルなし。高さ推定スキップ。")
        else:
            print("[Step 2] SAM マスクが空。高さ推定スキップ。")

    # Step 3: pose 推定
    client._object_size_mm = object_size_mm
    print(f"\n[Step 3] SAM-6D で 6DoF pose 推定中 (object_size_mm={object_size_mm:.0f})...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics, click_x=click_x, click_y=click_y)
    print(f"[Step 3完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    if img_pose is not None:
        cv2.imwrite(os.path.join(out_dir, "server_pointcloud.png"), img_pose)
        print(f"[保存] {out_dir}/server_pointcloud.png")
        if not args.no_show:
            cv2.imshow("server: pointcloud", img_pose)
    if img_mesh is not None:
        cv2.imwrite(os.path.join(out_dir, "server_mesh.png"), img_mesh)
        print(f"[保存] {out_dir}/server_mesh.png")
        if not args.no_show:
            cv2.imshow("server: mesh", img_mesh)

    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)
    if object_size_mm > 0:
        mesh_pts = mesh_pts * (object_size_mm / 200.0)
        print(f"[可視化] メッシュスケール補正: 200mm → {object_size_mm:.0f}mm (factor={object_size_mm/200.0:.3f})")

    # bbox + 点群投影
    vis_pts = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite(os.path.join(out_dir, "pose_check_pts.png"), vis_pts)
    print(f"[保存] {out_dir}/pose_check_pts.png")

    # bbox + axis (点群なし — rgb に bbox と座標軸のみ投影)
    axis_len_m = float(np.max(mesh_pts) - np.min(mesh_pts)) / 1000.0 * 0.3
    vis_bbox = project_pointcloud_on_image(
        rgb, mesh_pts, R, t, intrinsics, points_unit="mm", draw_points=False)
    vis_bbox_axis = draw_pose_axes(vis_bbox, R, t, intrinsics, axis_len_m=axis_len_m)
    cv2.imwrite(os.path.join(out_dir, "pose_check_bbox_axis.png"), vis_bbox_axis)
    print(f"[保存] {out_dir}/pose_check_bbox_axis.png")

    if not args.no_show:
        win = "pose check: bbox + axis"
        cv2.imshow(win, vis_bbox_axis)
        while True:
            if cv2.waitKey(100) != -1:
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    if object_size_mm > 0:
        print(f"  推定物体高さ: {object_size_mm/10:.1f} cm  ({object_size_mm:.0f} mm)")
    else:
        print("  推定物体高さ: 自動推定 (高さ入力なし)")
    print(f"  pose t: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print("=" * 50)


if __name__ == "__main__":
    main()
