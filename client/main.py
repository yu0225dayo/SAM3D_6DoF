"""
RealSense RGBD + IMU でリアルタイム 6DoF pose 推定

パイプライン:
    1. IMU から重力ベクトル取得
    2. RealSense ライブプレビュー → クリックで物体指定 → Enter で撮影
    3. tmp_input/ に rgb.png / depth.png / cam.json (gravity 込み) を保存
    4. test_demo.py と同一フロー:
       Step 1: SAM-3D でメッシュ生成
       Step 2: マスク + 深度 + 重力 → 高さ推定
       Step 3: SAM-6D で 6DoF pose 推定 → 可視化・保存

使用方法:
    python main.py
    python main.py --mesh-out meshes/cup.ply
    python main.py --gravity 0 -1 0   # IMU の代わりに手動指定
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

TMP_INPUT = "tmp_input"


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# IMU・高さ計算ユーティリティ (test_demo.py と共通)
# ===========================================================================

def get_gravity_imu(n_samples: int = 30) -> np.ndarray:
    """RealSense 加速度センサーから重力方向の単位ベクトルを取得する。"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError("pyrealsense2 がインストールされていません。")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

    print(f"[IMU] 重力ベクトル取得中 ({n_samples} サンプル)...")
    pipeline.start(cfg)
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


def get_points_3d_from_mask(depth_m, mask, fx, fy, cx, cy,
                             min_depth=0.05, max_depth=5.0):
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
    return np.stack([X, Y, d], axis=1)


def calc_height_from_points(points_3d, gravity_vec):
    projections = points_3d @ gravity_vec
    return float(projections.max() - projections.min())


def estimate_height_from_depth_mask(depth_m, mask, fx, fy, cx, cy, gravity_vec):
    pts = get_points_3d_from_mask(depth_m, mask, fx, fy, cx, cy)
    if len(pts) < 10:
        print(f"[高さ推定] 有効な深度点が不足 ({len(pts)} 点)。スキップ。")
        return 0.0, pts
    h = calc_height_from_points(pts, gravity_vec)
    print(f"[高さ推定] 点数={len(pts)}  高さ={h:.4f} m ({h*100:.1f} cm)")
    return h, pts


def draw_height_pcd(rgb_bgr, pts, gravity_vec, fx, fy, cx, cy, height_m):
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
        if 0 <= ui < W and 0 <= vi < H:
            cv2.drawMarker(img, (ui, vi), color, markerType=cv2.MARKER_STAR,
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
    parser = argparse.ArgumentParser(description="RealSense リアルタイム 6DoF pose 推定")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--mesh-out",    default="meshes/object.ply")
    parser.add_argument("--gravity", type=float, nargs=3, default=None,
                        metavar=("GX", "GY", "GZ"),
                        help="重力ベクトル手動指定 (省略時は IMU から自動取得)")
    parser.add_argument("--imu-samples", type=int, default=30)
    parser.add_argument("--no-show",     action="store_true")
    args = parser.parse_args()

    config    = load_config(args.config)
    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from utils.coord_transform import CameraIntrinsics
    from utils.pointcloud_utils import load_pointcloud_ply
    from utils.visualization import project_pointcloud_on_image, draw_pose_axes

    # ------------------------------------------------------------------
    # Step 0a: IMU から重力ベクトル取得
    # ------------------------------------------------------------------
    if args.gravity is not None:
        gravity_vec = np.array(args.gravity, dtype=np.float64)
        gravity_vec /= np.linalg.norm(gravity_vec)
        print(f"[重力] 手動指定: {gravity_vec}")
    else:
        try:
            gravity_vec = get_gravity_imu(n_samples=args.imu_samples)
        except Exception as e:
            print(f"[重力] IMU 取得失敗: {e}\n      高さ推定をスキップします。")
            gravity_vec = None

    # ------------------------------------------------------------------
    # Step 0b: RealSense ライブプレビュー → クリック → Enter で撮影
    # ------------------------------------------------------------------
    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    clicked = {"cx": -1, "cy": -1}
    WIN = "RealSense Preview  |  Click object → Enter: shoot / ESC: quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["cx"] = x
            clicked["cy"] = y
            print(f"[Camera] クリック座標: ({x}, {y})")

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    print("[Camera] プレビュー表示中 ... 物体をクリック → Enter で撮影、ESC で終了")
    rgb = depth = None
    while True:
        rgb, depth, _ = camera.capture()
        preview = rgb.copy()
        if clicked["cx"] < 0:
            cv2.putText(preview, "Click object  |  ESC: quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.circle(preview, (clicked["cx"], clicked["cy"]), 8, (0, 0, 255), -1)
            cv2.putText(preview, "Enter: shoot & confirm  |  ESC: quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(WIN, preview)
        key = cv2.waitKey(1)
        if key == 27:                           # ESC
            cv2.destroyAllWindows()
            camera.stop()
            sys.exit(0)
        elif key == 13 and clicked["cx"] >= 0:  # Enter → このフレームで確定
            print("[Camera] 撮影しました")
            break

    cv2.destroyAllWindows()
    camera.stop()

    click_x, click_y = clicked["cx"], clicked["cy"]

    # ------------------------------------------------------------------
    # Step 0c: tmp_input/ に保存
    # ------------------------------------------------------------------
    os.makedirs(TMP_INPUT, exist_ok=True)
    cv2.imwrite(os.path.join(TMP_INPUT, "rgb.png"), rgb)
    depth_raw = (depth / camera.depth_scale).astype(np.uint16)
    cv2.imwrite(os.path.join(TMP_INPUT, "depth.png"), depth_raw)
    cam_json = {
        "cam_K": [intrinsics.fx, 0.0, intrinsics.cx,
                  0.0, intrinsics.fy, intrinsics.cy,
                  0.0, 0.0, 1.0],
        "depth_scale": float(camera.depth_scale),
    }
    if gravity_vec is not None:
        cam_json["gravity"] = gravity_vec.tolist()
    with open(os.path.join(TMP_INPUT, "cam.json"), "w") as f:
        json.dump(cam_json, f, indent=2)
    print(f"[保存] {TMP_INPUT}/rgb.png / depth.png / cam.json")

    # ------------------------------------------------------------------
    # 以下 test_demo.py と同一フロー
    # ------------------------------------------------------------------

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
    _, masks, scores = client.save_reference_mesh(
        rgb, mesh_path,
        click_x=click_x, click_y=click_y,
        mesh_method=mesh_method,
    )
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
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, height_m,
            )
            cv2.imwrite(os.path.join(out_dir, "height_estimation.png"), height_vis)
            print(f"[Step 2] 高さ推定画像保存: {out_dir}/height_estimation.png")
        else:
            print("[Step 2] 高さ推定失敗。object_size_mm=0 (サーバ深度推定に委譲)。")
    else:
        print("[Step 2] " + ("重力ベクトルなし。" if gravity_vec is None else "SAM マスクが空。") + "高さ推定スキップ。")

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

    vis_pts = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite(os.path.join(out_dir, "pose_check_pts.png"), vis_pts)
    print(f"[保存] {out_dir}/pose_check_pts.png")

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
