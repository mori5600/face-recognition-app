# Face Recognition App

Windows ローカル専用の顔認証アプリです。依存管理は `uv`、顔検出と特徴抽出は `OpenCV DNN (YuNet + SFace)`、写真突破を下げる生体確認は `MediaPipe Face Landmarker` を前提にしています。

## セットアップ

```powershell
uv sync
uv run face-recognition-download-models
uv run face-recognition-app doctor
uv run face-recognition-app
```

## 用意したもの

- `uv` ベースの依存管理
- `OpenCV FaceDetectorYN / FaceRecognizerSF` のロード確認
- `MediaPipe Face Landmarker` を使った challenge-response 生体確認
- 公式 ONNX モデルのダウンロード導線
- カメラ 1 フレーム取得確認コマンド
- SQLite 永続化
- 名前入力による顔登録
- 既登録人物との顔照合
- 登録・照合の直前に 2 ステップの生体確認
- 顔選択方式の切り替え
- 照合方式と閾値の切り替え
- 登録済み人物の選択と削除
- 設計書に寄せた `domain / gateways / infra / app / strategy / ui` の最小構成

## 使い方

```powershell
uv run face-recognition-app doctor
uv run face-recognition-app download-models
uv run face-recognition-app camera-check
uv run face-recognition-app
```

## 確認

確認時は必ず次のスクリプトを実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

このスクリプトは次を順番に実行します。

- `uv run pyright`
- `uv run ty check .`
- `uv run ruff check .`
- `uv run pytest`
- `uv run python -m compileall app main.py`

## 現時点の制約

- 既定値は `SingleFaceOnlySelector` なので、複数人が映る場合は `最大顔優先` か `中央顔優先` に切り替えてください。
- `最大顔優先` と `中央顔優先` も選択できます。
- `人物単位最近傍` は、同一人物の複数 encoding の平均距離で評価する実装です。
- `FaceEncoding` は設計書に合わせて 128 次元固定です。
- カメラフレームごとに顔検出と特徴抽出を行うため、PC 性能によってはプレビューが重くなります。
- 生体確認は RGB カメラ前提の簡易 PAD です。写真には強くなりますが、厳密な本人確認を保証するものではありません。
