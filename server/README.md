# Server

計算機（GPU 環境）側のコードです。  
ローカル PC から HTTP 経由で画像データを受け取り、GPU 処理を行って結果を返します。

## 実行環境

- Linux + NVIDIA GPU (VRAM 16 GB 以上推奨)
- Python 3.11, CUDA 12.1
- Docker

## セットアップ

### 1. Python 環境

```bash
# conda を使う場合
conda env create -f environment_sam3d.yml
conda activate sam3d

# pip を使う場合
pip install -r requirements_sam3d.txt
pip install -e sam-3d-objects   # sam-3d-objects 本体
```

### 2. モデル重みのダウンロード

#### SAM2 Large

```bash
mkdir -p ../sam2_checkpoints
wget -P ../sam2_checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

#### SAM ViT-H (SAM-6D ISM が使用)

```bash
wget -P SAM-6D/SAM-6D/Instance_Segmentation_Model/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### SAM-6D PEM

```bash
mkdir -p SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints
wget -P SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints/ \
  https://huggingface.co/JiehongLin/SAM-6D/resolve/main/SAM-6D/Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth
```

#### SAM-3D checkpoints

HuggingFace の [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) からダウンロード（アクセス申請が必要）。

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
cd sam-3d-objects
hf download --repo-type model \
    --local-dir checkpoints/hf-download \
    --max-workers 1 \
    facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
```

### 3. Docker イメージのビルド（初回のみ）

```bash
docker compose build sam6d
```

## 起動方法

### STEP 1: SAM-6D Docker を起動

```bash
cd server
docker compose up -d sam6d

# 起動確認 (モデルロードに 1〜2 分かかる)
docker logs -f sam6d_service
curl http://localhost:8081/health
# → {"status":"ok","models_loaded":true} が返れば OK
```

### STEP 2: server.py を起動

```bash
conda activate sam3d
cd server
python server.py

# バックグラウンドで実行する場合
nohup python server.py > server.log 2>&1 &
```

## API エンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/health` | GET | サーバ状態確認 |
| `/reconstruct_mesh` | POST | RGB → SAM2 マスク → SAM-3D メッシュ生成 + SAM-6D テンプレートレンダリング |
| `/pose_estimate` | POST | RGB + depth → SAM2 マスク → SAM-6D 6DoF 姿勢推定 |
