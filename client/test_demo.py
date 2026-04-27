"""
テストデータを使った 6DoF pose 推定パイプラインのデモ

RealSense なしでフォルダから RGB + 深度を読み込み、
SAM-3D でメッシュ生成 → SAM-6D で pose 推定 → 可視化する。

フォルダ構成:
    demo_data/demo1/
        rgb.png
        depth.png
        cam.json   ({"cam_K": [fx,0,cx,0,fy,cy,0,0,1], "depth_scale": 0.001})

使用方法:
    python test_demo.py demo_data/demo1
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


def load_intrinsics(data_dir: str, img_w: int, img_h: int, depth_scale_ref: list):
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


def main():
    parser = argparse.ArgumentParser(description="テストデータで 6DoF pose 推定をデモ")
    parser.add_argument("data_dir", help="データフォルダ (rgb.png / depth.png / cam.json を含む)")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--mesh-out",    default="meshes/test_object.ply")
    parser.add_argument("--click-x",     type=int, default=-1)
    parser.add_argument("--click-y",     type=int, default=-1)
    parser.add_argument("--no-show",     action="store_true",
                        help="cv2.imshow を使わない (ヘッドレス環境用)")
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
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {rgb_path}")
    print(f"[RGB] {rgb_path}  shape={rgb.shape}")

    h, w = rgb.shape[:2]
    depth_scale_ref = [0.001]
    intrinsics = load_intrinsics(args.data_dir, w, h, depth_scale_ref)

    depth = load_depth(depth_path, depth_scale_ref[0])
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

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
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=args.click_x, click_y=args.click_y,
                                   mesh_method=mesh_method)
        click_x, click_y = args.click_x, args.click_y
    else:
        print("[物体選択] ウィンドウで物体をクリックしてください...")
        _, click_x, click_y, _, _ = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=mesh_method)
    print(f"[Step 1完了] mesh: {mesh_path}")

    # Step 2: pose 推定
    print("\n[Step 2] SAM-6D で 6DoF pose 推定中...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics, click_x=click_x, click_y=click_y)
    print(f"[Step 2完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

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
    vis_pts = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite(os.path.join(out_dir, "pose_check_pts.png"), vis_pts)
    print(f"[保存] {out_dir}/pose_check_pts.png")

    vis_mesh = render_mesh_on_image(rgb, mesh_path, R, t, intrinsics, mesh_unit="mm")
    cv2.imwrite(os.path.join(out_dir, "pose_check_mesh.png"), vis_mesh)
    print(f"[保存] {out_dir}/pose_check_mesh.png")

    if not args.no_show:
        cv2.imshow("pose check: pts", vis_pts)
        cv2.imshow("pose check: mesh", vis_mesh)
        print("何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
