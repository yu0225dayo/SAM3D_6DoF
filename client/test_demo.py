"""
テストデータを使った 6DoF pose 推定パイプラインのデモ

RealSense なしでファイルから RGB + 深度を読み込み、
サーバの SAM-3D + SAM-6D で pose 推定を動作確認する。

使用方法:
    # Step1: reference mesh 生成
    python test_demo.py --mode offline-mesh --rgb test_data/rgb.png

    # Step2: pose 推定 (mesh 指定済みの場合)
    python test_demo.py --mode online \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --mesh meshes/test_object.ply \
        --server-mesh-path "..." --template-dir "..."

    # Step1+2 まとめて実行
    python test_demo.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png
"""

import argparse
import os
import sys
import yaml
import numpy as np
import cv2

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
        print(f"[intrinsics] cam_json から読み込み: fx={fx} fy={fy} cx={cx} cy={cy}")
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
    print(f"[depth] {depth_path}  shape={depth_raw.shape}  "
          f"range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


def run_offline_mesh(args, config):
    """RGB ファイル → サーバ → reference mesh 保存"""
    from pipeline.sam6d_detector import SAM6DClient

    sam_cfg = config["sam3d"]
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
    )

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[offline-mesh] RGB: {args.rgb}  shape={rgb.shape}")

    mesh_path   = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")
    object_size_mm = args.object_size * 10.0 if args.object_size > 0 else 0.0

    if args.click_x >= 0 and args.click_y >= 0:
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=args.click_x, click_y=args.click_y,
                                   mesh_method=mesh_method,
                                   object_size_mm=object_size_mm)
    elif args.interactive:
        client.save_reference_mesh_interactive(rgb, mesh_path, mesh_method=mesh_method)
    else:
        client.save_reference_mesh(rgb, mesh_path, mesh_method=mesh_method,
                                   object_size_mm=object_size_mm)

    print(f"\n[完了] mesh: {mesh_path}")
    print(f"  サーバ mesh: {client._server_mesh_path}")
    print(f"  テンプレート: {client._template_dir}")
    print(f"\n次のコマンド:")
    print(f"  python test_demo.py --mode online \\")
    print(f"    --rgb {args.rgb} --depth <depth_path> \\")
    print(f"    --mesh {mesh_path} \\")
    print(f"    --server-mesh-path \"{client._server_mesh_path}\" \\")
    print(f"    --template-dir \"{client._template_dir}\"")


def run_online(args, config):
    """RGB + 深度ファイル → SAM-6D pose 推定 → 可視化"""
    from pipeline.sam6d_detector import SAM6DClient
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")

    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    client.load_reference_mesh(
        args.mesh,
        server_mesh_path=args.server_mesh_path or "",
        template_dir=args.template_dir or "",
    )

    print("\n[Step 1] SAM-6D で 6DoF pose 推定中...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics,
        click_x=args.click_x, click_y=args.click_y,
    )
    print(f"  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    os.makedirs("output/test", exist_ok=True)

    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
        print("[保存] output/test/server_pointcloud.png")
        if not args.no_show:
            cv2.imshow("Pose: pointcloud", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)
        print("[保存] output/test/server_mesh.png")
        if not args.no_show:
            cv2.imshow("Pose: mesh", img_mesh)

    # ローカル可視化: 点群投影
    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)
    vis_pts = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite("output/test/pose_check_pts.png", vis_pts)
    print("[保存] output/test/pose_check_pts.png")

    # ローカル可視化: メッシュレンダリング
    vis_mesh = render_mesh_on_image(rgb, args.mesh, R, t, intrinsics, mesh_unit="mm")
    cv2.imwrite("output/test/pose_check_mesh.png", vis_mesh)
    print("[保存] output/test/pose_check_mesh.png")

    if not args.no_show:
        cv2.imshow("Pose: pts projection", vis_pts)
        cv2.imshow("Pose: mesh render", vis_mesh)
        print("何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_full(args, config):
    """RGB + 深度ファイル → メッシュ生成 → pose 推定 → 可視化"""
    from pipeline.sam6d_detector import SAM6DClient
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[full] RGB: {args.rgb}  shape={rgb.shape}")

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path      = args.mesh_out
    mesh_method    = sam_cfg.get("mesh_method", "bpa")
    click_x, click_y = args.click_x, args.click_y
    object_size_mm = args.object_size * 10.0 if args.object_size > 0 else 0.0

    print("\n[Step 1] SAM-3D でメッシュ生成中...")
    if click_x >= 0 and click_y >= 0:
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=click_x, click_y=click_y,
                                   mesh_method=mesh_method,
                                   object_size_mm=object_size_mm)
    elif args.interactive:
        _, click_x, click_y, _, _ = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=mesh_method)
    else:
        client.save_reference_mesh(rgb, mesh_path, mesh_method=mesh_method,
                                   object_size_mm=object_size_mm)
    print(f"[Step 1完了] mesh: {mesh_path}")

    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    print("\n[Step 2] SAM-6D で 6DoF pose 推定中...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics, click_x=click_x, click_y=click_y)
    print(f"[Step 2完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    os.makedirs("output/test", exist_ok=True)

    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
        print("[保存] output/test/server_pointcloud.png")
        if not args.no_show:
            cv2.imshow("Pose: pointcloud", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)
        print("[保存] output/test/server_mesh.png")
        if not args.no_show:
            cv2.imshow("Pose: mesh", img_mesh)

    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)
    vis_pts = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite("output/test/pose_check_pts.png", vis_pts)
    vis_mesh = render_mesh_on_image(rgb, mesh_path, R, t, intrinsics, mesh_unit="mm")
    cv2.imwrite("output/test/pose_check_mesh.png", vis_mesh)
    print("[保存] output/test/pose_check_pts.png")
    print("[保存] output/test/pose_check_mesh.png")

    if not args.no_show:
        cv2.imshow("Pose: pts projection", vis_pts)
        cv2.imshow("Pose: mesh render", vis_mesh)
        print("何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="テストデータで 6DoF pose 推定をデモ")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--mode",    choices=["offline-mesh", "online", "full"], default="full")
    parser.add_argument("--no-show", action="store_true",
                        help="cv2.imshow を使わない (ヘッドレス環境用)")

    # 入力データ
    parser.add_argument("--rgb",         required=True, help="RGB 画像パス")
    parser.add_argument("--depth",       default=None,  help="深度画像パス (online/full)")
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
    parser.add_argument("--mesh",             default=None,
                        help="[online] ローカル mesh (.ply)")
    parser.add_argument("--mesh-out",         default="meshes/test_object.ply",
                        help="[offline-mesh/full] 保存先 .ply")
    parser.add_argument("--server-mesh-path", default=None)
    parser.add_argument("--template-dir",     default=None)

    # オプション
    parser.add_argument("--click-x",     type=int,   default=-1)
    parser.add_argument("--click-y",     type=int,   default=-1)
    parser.add_argument("--interactive", action="store_true", default=True)
    parser.add_argument("--object-size", type=float, default=0.0,
                        help="物体の高さ [cm] (0=自動推定)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "offline-mesh":
        run_offline_mesh(args, config)
    elif args.mode == "full":
        if not args.depth:
            print("エラー: --depth を指定してください。")
            sys.exit(1)
        run_full(args, config)
    else:  # online
        if not args.depth:
            print("エラー: --depth を指定してください。")
            sys.exit(1)
        if not args.mesh:
            print("エラー: --mesh を指定してください。")
            sys.exit(1)
        run_online(args, config)


if __name__ == "__main__":
    main()
