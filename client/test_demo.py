"""
テストデータを使った 6DoF pose 推定パイプラインのデモ

RealSense なしでファイルから RGB + 深度を読み込み、
SAM-3D でメッシュ生成 → SAM-6D で pose 推定 → 可視化する。

使用方法:
    python test_demo.py \
        --rgb demo_data/demo1/rgb.png \
        --depth demo_data/demo1/depth.png \
        --cam-json demo_data/demo1/cam.json
"""

import argparse
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


def load_intrinsics(args, img_w: int, img_h: int):
    import json as _json
    from utils.coord_transform import CameraIntrinsics

    if args.cam_json:
        with open(args.cam_json, "r") as f:
            cam = _json.load(f)
        K = cam["cam_K"]
        if len(K) == 9:
            fx, cx = K[0], K[2]
            fy, cy = K[4], K[5]
        else:
            raise ValueError(f"cam_K の形式が不正: {K}")
        if "depth_scale" in cam and args.depth_scale == 0.001:
            args.depth_scale = cam["depth_scale"]
        print(f"[intrinsics] fx={fx} fy={fy} cx={cx} cy={cy}")
    else:
        fx = args.fx
        fy = args.fy
        cx = args.cx if args.cx > 0 else img_w / 2
        cy = args.cy if args.cy > 0 else img_h / 2

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=img_w, height=img_h)


def load_depth(depth_path: str, depth_scale: float = 1.0) -> np.ndarray:
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"深度画像が読み込めません: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale
    print(f"[depth] shape={depth_raw.shape}  range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


def main():
    parser = argparse.ArgumentParser(description="テストデータで 6DoF pose 推定をデモ")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--no-show", action="store_true",
                        help="cv2.imshow を使わない (ヘッドレス環境用)")

    # 入力データ
    parser.add_argument("--rgb",         required=True, help="RGB 画像パス")
    parser.add_argument("--depth",       required=True, help="深度画像パス")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="深度スケール係数 (0.001: mm→m)")

    # カメラパラメータ
    parser.add_argument("--cam-json", default=None,
                        help="camera JSON ({cam_K:[fx,0,cx,0,fy,cy,0,0,1], depth_scale:1.0})")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=-1)
    parser.add_argument("--cy", type=float, default=-1)

    # mesh
    parser.add_argument("--mesh-out",     default="meshes/test_object.ply",
                        help="生成 mesh の保存先 .ply")

    # オプション
    parser.add_argument("--click-x",     type=int,   default=-1)
    parser.add_argument("--click-y",     type=int,   default=-1)
    parser.add_argument("--object-size", type=float, default=0.0,
                        help="物体の高さ [cm] (0=自動推定)")

    args = parser.parse_args()
    config = load_config(args.config)

    from pipeline.sam6d_detector import SAM6DClient
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[RGB] {args.rgb}  shape={rgb.shape}")

    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path      = args.mesh_out
    mesh_method    = sam_cfg.get("mesh_method", "bpa")
    object_size_mm = args.object_size * 10.0 if args.object_size > 0 else 0.0

    # Step 1: mesh 生成
    print("\n[Step 1] SAM-3D でメッシュ生成中...")
    os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)
    if args.click_x >= 0 and args.click_y >= 0:
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=args.click_x, click_y=args.click_y,
                                   mesh_method=mesh_method,
                                   object_size_mm=object_size_mm)
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
