# Client

ローカル PC 側のコードです。  
画像データをサーバへ送信し、6DoF 姿勢推定結果を受け取ります。

## 実行環境

- Windows / Mac / Linux
- Python 3.9+
- Realsense R455 (IMUつき推奨)

## セットアップ

### conda を使う場合（推奨）

```bash
conda env create -f environment_client.yml
conda activate client_sam3d6dof
```

### pip を使う場合

```bash
pip install -r requirements_client.txt
```

## 設定

`config.yaml` のサーバ URL をサーバの IP アドレスに合わせて変更してください。

```yaml
sam3d:
  server_url: "http://<サーバIP>:8080"
  mesh_method: "knn"   # メッシュ生成方式: bpa / poisson / knn

robot:
  mode: "mock"   # mock (デバッグ) / tcp / ros / serial
```

## 実行方法

### テスト実行（静止画ファイル）

```bash
conda activate client_sam3d6dof
cd client

# データフォルダを指定して実行 (ウィンドウで物体をクリック選択)
python test_demo.py --data-dir demo_data/demo1

# --data-dir 省略時は demo_data/demo1 がデフォルト
python test_demo.py

# クリック座標を手動指定してヘッドレス実行
python test_demo.py --data-dir demo_data/demo1 --click-x 320 --click-y 240 --no-show

# 重力ベクトルを手動指定 (cam.json に gravity がない場合)
python test_demo.py --data-dir demo_data/demo1 --gravity 0 -1 0
```

![テストデモ](test_demo.png)

### 本番実行（RealSense カメラ）

```bash
conda activate client_sam3d6dof
cd client
python main.py
```

![本番実行](main.png)

## データフォルダ構成

```
demo_data/demo1/
├─ rgb.png        カラー画像
├─ depth.png      深度画像 (uint16, mm 単位)
└─ cam.json       カメラパラメータ
```

`cam.json` の例:
```json
{
    "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "depth_scale": 0.001,
    "gravity": [0.0, -0.999, 0.04]
}
```

> `gravity` フィールドは省略可。省略した場合は RealSense IMU から自動取得、または `--gravity` で手動指定。

## 出力ファイル

実行後、`output/<timestamp>/` に保存されます。

| ファイル | 内容 |
|---|---|
| `pose_check_bbox_axis.png` | bbox + 座標軸の投影結果 |
| `pose_check_pts.png` | 点群 + bbox の投影結果 |
| `server_pointcloud.png` | サーバ側点群可視化 |
| `server_mesh.png` | サーバ側メッシュ可視化 |
| `sam_mask.png` | SAM マスクオーバーレイ |
| `height_estimation.png` | 高さ推定点群可視化 |

## オプション一覧 (test_demo.py)

| オプション | 説明 |
|---|---|
| `--data-dir <フォルダ>` | データフォルダ (rgb.png / depth.png / cam.json を含む) (デフォルト: `demo_data/demo1`) |
| `--click-x / --click-y` | 物体指定クリック座標 (省略するとウィンドウでクリック選択) |
| `--gravity GX GY GZ` | 重力方向ベクトル手動指定 (cam.json の gravity より優先) |
| `--no-show` | `cv2.imshow` を使わない (SSH / ヘッドレス環境用) |
| `--mesh-out <パス>` | メッシュ保存先 PLY パス (デフォルト: `meshes/test_object.ply`) |
| `--imu-samples <N>` | IMU から重力取得するサンプル数 (デフォルト: 30) |
| `--config <パス>` | 設定ファイルパス (デフォルト: `config.yaml`) |