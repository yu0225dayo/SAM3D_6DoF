# 使い方

RealSense カメラを使った 6DoF pose 推定スクリプト。

## 前提条件

- Intel RealSense D400系カメラが接続済み
- 計算機サーバ (`server/server.py`) が起動済み
- `config.yaml` の `server_url` が正しく設定済み

---

## main.py（カメラ入力）

```bash
python main.py
python main.py --mesh-out meshes/cup.ply
```

### 流れ

1. RealSense から1フレーム取得
2. ウィンドウが開くので、物体をクリックして選択
3. SAM-3D でメッシュ生成（数十秒）
4. SAM-6D で 6DoF pose 推定
5. 結果ウィンドウが開き、キーを押すと終了

### 出力

起動時刻で `output/<YYYYMMDD_HHMMSS>/` が作成される。

| ファイル | 内容 |
|---------|------|
| `masks_compare.png`     | SAM が検出したマスク候補 |
| `server_pointcloud.png` | サーバが生成した点群投影画像 |
| `server_mesh.png`       | サーバが生成したメッシュ投影画像 |
| `pose_check_pts.png`    | ローカルで点群を投影した確認画像 |
| `pose_check_mesh.png`   | ローカルでメッシュを投影した確認画像 |

起動時に `camera.json`（カメラ内部パラメータ）がカレントディレクトリに保存される。

### 引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--config`   | `config.yaml`      | 設定ファイルパス |
| `--mesh-out` | `meshes/object.ply` | mesh 保存先 |
| `--click-x`  | `-1` | 物体クリック座標 X（指定時はウィンドウ選択をスキップ） |
| `--click-y`  | `-1` | 物体クリック座標 Y |

---

## test_demo.py（テストデータ入力）

カメラなしでファイルから RGB・深度を読み込んで動作確認する。

```bash
python test_demo.py \
    --rgb demo_data/demo1/rgb.png \
    --depth demo_data/demo1/depth.png \
    --cam-json demo_data/demo1/cam.json
```

### 引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--config`      | `config.yaml` | 設定ファイルパス |
| `--rgb`         | （必須）       | RGB 画像パス |
| `--depth`       | （必須）       | 深度画像パス |
| `--depth-scale` | `0.001`        | 深度スケール係数（mm→m） |
| `--cam-json`    | —              | カメラパラメータ JSON |
| `--mesh-out`    | `meshes/test_object.ply` | mesh 保存先 |
| `--click-x/y`   | `-1`           | 物体座標（指定時はウィンドウ選択をスキップ） |
| `--no-show`     | `False`        | ヘッドレス環境用（imshow を無効化） |

`--cam-json` の形式:
```json
{"cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1], "depth_scale": 0.001}
```

---

## config.yaml 主要設定

```yaml
sam3d:
  server_url: "http://10.40.1.126:8080"
  mesh_method: "knn"   # bpa / poisson / knn
  timeout: 300.0

sam6d:
  timeout: 30.0

camera:
  width: 640
  height: 480
  fps: 30
```
