# SAM3D_6DoF

RGB-D カメラで物体を撮影し、GPU サーバ上で 3D 再構築と 6DoF 姿勢推定を行うパイプラインです。  
テンプレート CAD モデル不要・クリックひとつで物体の 6DoF 姿勢 (R, t) を取得できます。　　
**SAM3Dの3DモデリングとSAM6Dの位置姿勢推定をつなげるパイプライン構築**。 


![demo](gif/2026_326.gif)

---

## 概要

```
クライアント (ローカル PC / Windows)
  ↓ RGB + Depth 送信 (HTTP :8080)
サーバ (GPU 計算機 Linux)
  ├─ server.py        ← SAM2 マスク + SAM-3D メッシュ生成
  └─ SAM-6D Docker    ← テンプレートレンダリング + 6DoF 姿勢推定
  ↓ R, t 返却
クライアント
  └─ 把持姿勢生成 (Shape2Gesture) → ロボット送信
```

| コンポーネント | 役割 |
|---|---|
| [SAM2](https://github.com/facebookresearch/segment-anything-2) | クリック点から物体マスクを生成 |
| [SAM-3D](https://github.com/Pointcept/SAM-3D) | マスク内の RGB から 3D Gaussian Splat → PLY メッシュを生成 |
| [SAM-6D](https://github.com/JiehongLin/SAM-6D) | テンプレートマッチングで 6DoF 姿勢 (R, t) を推定 |

---

## ディレクトリ構成

```
SAM3D_6DoF/
├── sam2_checkpoints/
│   └── sam2.1_hiera_large.pt              # ★ SAM2 Large 重み
├── server/                                # GPU 計算機側
│   ├── server.py                          # メインサーバ (FastAPI, port 8080)
│   ├── sam6d_service.py                   # SAM-6D マイクロサービス (Docker, port 8081)
│   ├── sam6d_wrapper.py                   # SAM-6D Python ラッパー
│   ├── docker-compose.yml                 # Docker 設定
│   ├── Dockerfile.sam6d                   # SAM-6D Docker イメージ定義
│   ├── sam-3d-objects/                    # SAM-3D サブモジュール
│   │   └── checkpoints/hf/
│   │       └── pipeline.yaml (+ 重み)     # ★ SAM-3D 重み
│   └── SAM-6D/SAM-6D/
│       ├── Instance_Segmentation_Model/
│       │   └── sam_vit_h_4b8939.pth       # ★ SAM ViT-H 重み (ISM)
│       └── Pose_Estimation_Model/
│           └── checkpoints/
│               └── sam-6d-pem-base.pth    # ★ SAM-6D PEM 重み
│   └── tmp/                               # 実行時の中間ファイル (自動生成)
│       ├── rgb.png / depth.png            #   受信した入力画像
│       ├── camera_custom.json             #   受信したカメラパラメータ
│       └── server_reconstructions/
│           ├── mask_sam2.png              #   SAM2 マスク
│           ├── object_seed42.ply          #   SAM-3D 生成点群
│           ├── object_seed42_mesh.ply     #   生成メッシュ
│           ├── object_seed42_mesh_scaled.ply  # スケール補正済みメッシュ
│           └── object_seed42_mesh_templates/
│               ├── templates/             #   SAM-6D レンダリングテンプレート
│               └── sam6d_results/         #   SAM-6D 推定結果 JSON
├── client/                                # ローカル PC 側
│   ├── test_demo.py                       # テスト用エントリポイント (静止画ファイル)
│   ├── main.py                            # メインエントリポイント (RealSense カメラ)
│   ├── config.yaml                        # 設定ファイル (サーバ URL, カメラパラメータ等)
│   ├── pipeline/
│   │   ├── sam6d_detector.py              # SAM-6D クライアント (HTTP)
│   │   └── sam3d_segmentation.py          # SAM-3D クライアント (HTTP)
│   └── utils/
│       ├── coord_transform.py             # 座標変換ユーティリティ
│       └── visualization.py              # 姿勢・点群の可視化
├── gif/
│   └── 2026_326.gif                       # デモ動画
└── howto.txt                              # 起動手順メモ
```

---

## セットアップ

### 前提条件

| 環境 | 要件 |
|---|---|
| サーバ | Linux, NVIDIA GPU (VRAM 16 GB 以上推奨), Docker, CUDA |
| クライアント | Windows / Linux, Python 3.9+, (RealSense SDK) |

### サーバ側セットアップ

```bash
# 1. リポジトリを展開 (サブモジュールごと)
git clone <このリポジトリ>
cd SAM3D_6DoF
git submodule update --init --recursive

# 2. SAM-6D Docker イメージをビルド (初回のみ)
cd server
docker compose build sam6d
```

### モデル重みのダウンロード

**重みファイルはリポジトリに含まれていません。** 以下の手順で取得してください。

#### 1. SAM2 Large (server.py が使用)

```bash
mkdir -p sam2_checkpoints
wget -P sam2_checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

配置先: `sam2_checkpoints/sam2.1_hiera_large.pt` (リポジトリルート直下)

#### 2. SAM ViT-H (SAM-6D ISM が使用)

```bash
wget -P server/SAM-6D/SAM-6D/Instance_Segmentation_Model/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

配置先: `server/SAM-6D/SAM-6D/Instance_Segmentation_Model/sam_vit_h_4b8939.pth`

#### 3. SAM-6D PEM checkpoint (SAM-6D 姿勢推定モデル)

```bash
mkdir -p server/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints
wget -P server/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints/ \
  https://huggingface.co/JiehongLin/SAM-6D/resolve/main/SAM-6D/Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth
```

配置先: `server/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth`

#### 4. SAM-3D checkpoints (sam-3d-objects)

HuggingFace の [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) からダウンロードします。  
> **注意**: アクセス申請 (氏名・生年月日・所属) が必要です。承認後にダウンロード可能になります。

配置先: `server/sam-3d-objects/checkpoints/hf/pipeline.yaml` (+ モデル重み)

#### 重みファイルの配置まとめ

```
SAM3D_6DoF/                                 ← このリポジトリのルート
├── sam2_checkpoints/
│   └── sam2.1_hiera_large.pt               ← SAM2 Large
└── server/
    ├── sam-3d-objects/
    │   └── checkpoints/hf/
    │       └── pipeline.yaml (+ 重み)      ← SAM-3D
    └── SAM-6D/
        └── SAM-6D/
            ├── Instance_Segmentation_Model/
            │   └── sam_vit_h_4b8939.pth    ← SAM ViT-H (ISM)
            └── Pose_Estimation_Model/
                └── checkpoints/
                    └── sam-6d-pem-base.pth ← SAM-6D PEM
```

### クライアント側セットアップ

```bash
cd client
pip install -r requirements.txt
```

---

## 使い方

詳細はそれぞれの README を参照してください。

- サーバ側: [server/README.md](server/README.md)
- クライアント側: [client/README.md](client/README.md)

---

## 既知の制限

- **スケール不一致**: SAM-3D は単眼 RGB から再構築するためスケールが不定。サーバ側でメッシュの最長辺を 200 mm に正規化して補正しているが、実物と完全には一致しない。
- **モデルロード時間**: サーバ起動後、SAM-6D Docker のモデルロードに 1〜2 分かかる。
