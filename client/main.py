"""
6DoF pose 推定メインエントリーポイント

パイプライン:
    [オフライン] 物体ごとに1回:
        RealSense RGB → サーバ (SAM-3D) → reference mesh (.ply) 保存
        サーバが SAM-6D でテンプレートをレンダリングし保存

    [オンライン] キー操作:
        RealSense RGBD → サーバ (SAM-6D) → 6DoF pose → 画像に投影・保存

使用方法:
    # Step1: reference mesh を生成 (物体ごとに1回)
    python main.py --mode offline-mesh --mesh-out meshes/cup.ply

    # Step2: pose 推定 (mesh 指定済みの場合)
    python main.py --mode online --mesh meshes/cup.ply \
        --server-mesh-path "..." --template-dir "..."

    # Step1+2 まとめて実行 (フルパイプライン)
    python main.py --mode full --mesh-out meshes/cup.ply
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_offline_mesh(config: dict, args):
    """RealSense RGB → サーバ (SAM-3D) → mesh 保存"""
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient

    cam_cfg = config["camera"]
    sam_cfg = config["sam3d"]

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
    )

    print("\n[操作方法]")
    print("  [c] この画像で reference mesh を生成")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break
            elif key == ord("c"):
                mesh_path = args.mesh_out
                print(f"\n[mesh生成] → {mesh_path}")
                if sam_cfg.get("interactive", True):
                    client.save_reference_mesh_interactive(rgb, mesh_path)
                else:
                    client.save_reference_mesh(rgb, mesh_path)
                print(f"[完了] {mesh_path}")
                print(f"  サーバ mesh: {client._server_mesh_path}")
                print(f"  テンプレート: {client._template_dir}")
                break
    finally:
        camera.stop()


def run_online(config: dict, args):
    """毎フレーム: RealSense RGBD → SAM-6D pose → 可視化・保存"""
    import cv2 as _cv2
    from datetime import datetime
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from utils.coord_transform import CameraIntrinsics
    from utils.pointcloud_utils import load_pointcloud_ply
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image

    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    client.load_reference_mesh(
        args.mesh,
        server_mesh_path=args.server_mesh_path or "",
        template_dir=args.template_dir or "",
    )

    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    os.makedirs("output", exist_ok=True)
    cam_json = {
        "fx": float(camera.fx), "fy": float(camera.fy),
        "cx": float(camera.cx), "cy": float(camera.cy),
        "width": cam_cfg["width"], "height": cam_cfg["height"],
    }
    with open("camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    print("\n[操作方法]")
    print("  [p] pose 推定 → 画像保存")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break

            elif key == ord("p"):
                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)

                print("\n[pose推定] SAM-6D で 6DoF pose 推定中...")
                R, t, img_pose, img_mesh = client.estimate_pose(
                    rgb, depth, intrinsics,
                    click_x=args.click_x, click_y=args.click_y,
                )
                print(f"  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
                print(f"  R=\n{R}")

                if img_mesh is not None:
                    path = os.path.join(out_dir, "pose_mesh.png")
                    _cv2.imwrite(path, img_mesh)
                    _cv2.imshow("pose: mesh", img_mesh)
                    _cv2.waitKey(1)
                if img_pose is not None:
                    path = os.path.join(out_dir, "pose_pointcloud.png")
                    _cv2.imwrite(path, img_pose)
                    _cv2.imshow("pose: pointcloud", img_pose)
                    _cv2.waitKey(1)

                vis_pts = project_pointcloud_on_image(
                    rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
                _cv2.imwrite(os.path.join(out_dir, "pose_check_pts.png"), vis_pts)
                _cv2.imshow("pose check: pts", vis_pts)
                _cv2.waitKey(1)

                vis_mesh = render_mesh_on_image(
                    rgb, args.mesh, R, t, intrinsics, mesh_unit="mm")
                _cv2.imwrite(os.path.join(out_dir, "pose_check_mesh.png"), vis_mesh)
                _cv2.imshow("pose check: mesh", vis_mesh)
                _cv2.waitKey(1)

                print(f"[完了] 保存先: {out_dir}")

    except KeyboardInterrupt:
        print("\n[main] 中断されました。")
    finally:
        camera.stop()


def run_full(config: dict, args):
    """
    フルパイプライン: カメラ起動 → 物体選択 → mesh 生成 → pose 推定 → 可視化

    操作:
        [c] 物体をクリック選択 → mesh 生成
        [p] pose 推定 → 画像保存
        [q] 終了
    """
    import cv2 as _cv2
    from datetime import datetime
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from utils.coord_transform import CameraIntrinsics
    from utils.pointcloud_utils import load_pointcloud_ply
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image

    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    mesh_path   = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    os.makedirs("output", exist_ok=True)
    cam_json = {
        "fx": float(camera.fx), "fy": float(camera.fy),
        "cx": float(camera.cx), "cy": float(camera.cy),
        "width": cam_cfg["width"], "height": cam_cfg["height"],
    }
    with open("camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    # 状態変数
    mesh_pts   = None
    click_x = click_y = -1
    rgb_frozen = depth_frozen = None

    print("\n[操作方法]")
    print("  [c] 物体をクリック選択 → mesh 生成")
    print("  [p] pose 推定 → 画像保存 ([c] 後に有効)")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break

            elif key == ord("c"):
                print("\n[mesh生成] 物体をクリックして選択してください...")
                rgb_frozen   = rgb.copy()
                depth_frozen = depth.copy()
                os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)
                _, click_x, click_y, sam_masks, sam_scores = \
                    client.save_reference_mesh_interactive(
                        rgb_frozen, mesh_path, mesh_method=mesh_method)
                mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)
                print(f"[mesh生成完了] {mesh_path}")

                # SAM マスク比較画像を保存
                if sam_masks:
                    out_dir = os.path.join(
                        "output", datetime.now().strftime("%Y%m%d_%H%M%S") + "_mesh")
                    os.makedirs(out_dir, exist_ok=True)
                    panels = []
                    for i, (mask, score) in enumerate(zip(sam_masks, sam_scores)):
                        panel = rgb_frozen.copy()
                        colored = np.zeros_like(panel)
                        colored[mask > 127] = (0, 255, 0)
                        panel = _cv2.addWeighted(panel, 0.7, colored, 0.3, 0)
                        ys, xs = np.where(mask > 127)
                        if len(xs) > 0:
                            _cv2.rectangle(panel,
                                           (int(xs.min()), int(ys.min())),
                                           (int(xs.max()), int(ys.max())),
                                           (0, 255, 255), 2)
                        label = f"mask{i} score={score:.3f}"
                        if i == int(np.argmax(sam_scores)):
                            label += " [BEST]"
                        _cv2.putText(panel, label, (10, 30),
                                     _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        panels.append(panel)
                    compare = np.hstack(panels)
                    _cv2.imwrite(os.path.join(out_dir, "masks_compare.png"), compare)
                    _cv2.imshow("SAM masks", compare)
                    _cv2.waitKey(1)
                    print(f"[マスク] 保存: {out_dir}/masks_compare.png")

                print("[次のステップ] [p] を押して pose 推定してください。")

            elif key == ord("p"):
                if mesh_pts is None or rgb_frozen is None:
                    print("[警告] 先に [c] で物体を選択してください。")
                    continue

                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)

                print("\n[pose推定] SAM-6D で 6DoF pose 推定中...")
                try:
                    R, t, img_pose, img_mesh = client.estimate_pose(
                        rgb_frozen, depth_frozen, intrinsics,
                        click_x=click_x, click_y=click_y,
                    )
                except Exception as e:
                    print(f"[エラー] pose 推定失敗: {e}")
                    continue

                print(f"  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
                print(f"  R=\n{R}")

                if img_mesh is not None:
                    _cv2.imwrite(os.path.join(out_dir, "pose_mesh.png"), img_mesh)
                    _cv2.imshow("pose: mesh", img_mesh)
                    _cv2.waitKey(1)
                if img_pose is not None:
                    _cv2.imwrite(os.path.join(out_dir, "pose_pointcloud.png"), img_pose)
                    _cv2.imshow("pose: pointcloud", img_pose)
                    _cv2.waitKey(1)

                vis_pts = project_pointcloud_on_image(
                    rgb_frozen, mesh_pts, R, t, intrinsics, points_unit="mm")
                _cv2.imwrite(os.path.join(out_dir, "pose_check_pts.png"), vis_pts)
                _cv2.imshow("pose check: pts", vis_pts)
                _cv2.waitKey(1)

                vis_mesh = render_mesh_on_image(
                    rgb_frozen, mesh_path, R, t, intrinsics, mesh_unit="mm")
                _cv2.imwrite(os.path.join(out_dir, "pose_check_mesh.png"), vis_mesh)
                _cv2.imshow("pose check: mesh", vis_mesh)
                _cv2.waitKey(1)

                print(f"[完了] 保存先: {out_dir}")

    except KeyboardInterrupt:
        print("\n[main] 中断されました。")
    finally:
        camera.stop()


def main():
    parser = argparse.ArgumentParser(
        description="RealSense + SAM-3D + SAM-6D 6DoF pose 推定",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # mesh 生成のみ
  python main.py --mode offline-mesh --mesh-out meshes/cup.ply

  # pose 推定 (mesh 指定済み)
  python main.py --mode online --mesh meshes/cup.ply \\
      --server-mesh-path "..." --template-dir "..."

  # フルパイプライン (mesh 生成 + pose 推定)
  python main.py --mode full --mesh-out meshes/cup.ply
        """
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["full", "online", "offline-mesh"], default="full",
        help="full: 物体選択から一括 / online: mesh 指定済みで pose のみ / offline-mesh: mesh 生成のみ",
    )
    parser.add_argument("--mesh",             default=None,
                        help="[online] ローカルの reference mesh (.ply)")
    parser.add_argument("--mesh-out",         default="meshes/object.ply",
                        help="[full/offline-mesh] mesh 保存先 .ply")
    parser.add_argument("--server-mesh-path", default=None,
                        help="[online] サーバ側の mesh パス")
    parser.add_argument("--template-dir",     default=None,
                        help="[online] サーバ側テンプレートディレクトリ")
    parser.add_argument("--click-x", type=int, default=-1)
    parser.add_argument("--click-y", type=int, default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    print("=" * 50)
    print("  RealSense + SAM-3D + SAM-6D")
    print(f"  モード: {args.mode}")
    print("=" * 50)

    if args.mode == "full":
        run_full(config, args)
    elif args.mode == "offline-mesh":
        run_offline_mesh(config, args)
    else:  # online
        if args.mesh is None:
            print("エラー: --mesh で reference mesh (.ply) を指定してください。")
            sys.exit(1)
        if not os.path.exists(args.mesh):
            print(f"エラー: mesh が見つかりません: {args.mesh}")
            sys.exit(1)
        run_online(config, args)


if __name__ == "__main__":
    main()
