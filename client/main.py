"""
6DoF pose 推定メインエントリーポイント

パイプライン:
    RealSense 起動 → 物体クリック選択 → mesh 生成 → pose 推定 → 結果表示

使用方法:
    python main.py
    python main.py --mesh-out meshes/cup.ply
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


def main():
    parser = argparse.ArgumentParser(description="RealSense + SAM-3D + SAM-6D 6DoF pose 推定")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--mesh-out", default="meshes/object.ply",
                        help="mesh 保存先 .ply")
    parser.add_argument("--click-x",  type=int, default=-1)
    parser.add_argument("--click-y",  type=int, default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)

    config    = load_config(args.config)
    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from utils.coord_transform import CameraIntrinsics
    from utils.pointcloud_utils import load_pointcloud_ply
    from utils.visualization import project_pointcloud_on_image, render_mesh_on_image

    # カメラ起動・1フレーム取得
    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    os.makedirs("output", exist_ok=True)
    with open("camera.json", "w") as f:
        json.dump({
            "fx": float(camera.fx), "fy": float(camera.fy),
            "cx": float(camera.cx), "cy": float(camera.cy),
            "width": cam_cfg["width"], "height": cam_cfg["height"],
        }, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    # ライブプレビュー: Enter 押下時に1枚撮影
    print("[Camera] プレビュー表示中... 物体をクリックして座標を指定、Enter で撮影・確定、ESC で終了")
    click_x, click_y = args.click_x, args.click_y
    clicked = {"cx": -1, "cy": -1}

    WIN = "RealSense Preview (click object / Enter: shoot / ESC: quit)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["cx"] = x
            clicked["cy"] = y
            print(f"[Camera] クリック座標: ({x}, {y})")

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

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

        if key == 27:   # ESC
            cv2.destroyAllWindows()
            camera.stop()
            print("終了します。")
            sys.exit(0)
        elif key == 13 and clicked["cx"] >= 0:  # Enter: このフレームで確定
            print("[Camera] 撮影しました")
            break

    cv2.destroyAllWindows()
    camera.stop()

    click_x = clicked["cx"]
    click_y = clicked["cy"]

    out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    # RGBD + カメラパラメータを保存
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb)
    depth_mm = (depth / camera.depth_scale).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_mm)
    cam_json_path = os.path.join(out_dir, "cam.json")
    with open(cam_json_path, "w") as f:
        json.dump({
            "cam_K": [intrinsics.fx, 0.0, intrinsics.cx,
                      0.0, intrinsics.fy, intrinsics.cy,
                      0.0, 0.0, 1.0],
            "depth_scale": float(camera.depth_scale),
        }, f, indent=2)
    print(f"[保存] {out_dir}/rgb.png / depth.png / cam.json")

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    # Step 1: 物体選択 → mesh 生成
    mesh_path = args.mesh_out
    os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)

    print("\n[Step 1] SAM-3D でメッシュ生成中...")
    _, sam_masks, sam_scores = client.save_reference_mesh(
        rgb, mesh_path,
        click_x=click_x, click_y=click_y,
        mesh_method=sam_cfg.get("mesh_method", "bpa"),
    )
    print(f"[Step 1完了] mesh: {mesh_path}")

    # SAM マスク確認画像を保存
    if sam_masks:
        best_idx = int(np.argmax(sam_scores))
        panels = []
        for i, (mask, score) in enumerate(zip(sam_masks, sam_scores)):
            panel = rgb.copy()
            colored = np.zeros_like(panel)
            colored[mask > 127] = (0, 255, 0)
            panel = cv2.addWeighted(panel, 0.7, colored, 0.3, 0)
            ys, xs = np.where(mask > 127)
            if len(xs) > 0:
                cv2.rectangle(panel,
                               (int(xs.min()), int(ys.min())),
                               (int(xs.max()), int(ys.max())),
                               (0, 255, 255), 2)
            label = f"mask{i} score={score:.3f}" + (" [BEST]" if i == best_idx else "")
            cv2.putText(panel, label, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(panel)
        compare = np.hstack(panels)
        cv2.imwrite(os.path.join(out_dir, "masks_compare.png"), compare)
        cv2.imshow("SAM masks", compare)
        cv2.waitKey(1)

    # Step 2: pose 推定
    print("\n[Step 2] SAM-6D で 6DoF pose 推定中...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics, click_x=click_x, click_y=click_y)
    print(f"[Step 2完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    if img_pose is not None:
        cv2.imwrite(os.path.join(out_dir, "pose_pointcloud.png"), img_pose)
        cv2.imshow("server: pointcloud", img_pose)
        print(f"[保存] {out_dir}/pose_pointcloud.png")
    if img_mesh is not None:
        cv2.imwrite(os.path.join(out_dir, "pose_mesh.png"), img_mesh)
        cv2.imshow("server: mesh", img_mesh)
        print(f"[保存] {out_dir}/pose_mesh.png")

    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)
    vis_pts = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite(os.path.join(out_dir, "pose_check_pts.png"), vis_pts)
    cv2.imshow("pose check: pts", vis_pts)
    print(f"[保存] {out_dir}/pose_check_pts.png")

    vis_mesh = render_mesh_on_image(rgb, mesh_path, R, t, intrinsics, mesh_unit="mm")
    cv2.imwrite(os.path.join(out_dir, "pose_check_mesh.png"), vis_mesh)
    cv2.imshow("pose check: mesh", vis_mesh)
    print(f"[保存] {out_dir}/pose_check_mesh.png")

    print("何かキーを押すと終了...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
