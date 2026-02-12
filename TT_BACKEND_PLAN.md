# SGLang Tenstorrent Backend Implementation Plan

## Overview

SGLangにTenstorrent (TT) ハードウェアサポートを追加するプロジェクト。
T3K環境（Wormhole 8枚）での動作を目標とする。

## 実装済み (commit: aef21ed16)

### 追加ファイル

1. **`python/sglang/srt/hardware_backend/tt/__init__.py`**
   - TT backend の公開API

2. **`python/sglang/srt/hardware_backend/tt/utils.py`**
   - `is_tt()` - デバイス検出
   - `init_tt_backend()` - ttnn初期化、メッシュデバイス管理
   - `set_default_server_args()` - TT用デフォルト設定
   - `get_mesh_device()`, `close_tt_backend()`, `tt_synchronize()`

3. **`python/sglang/srt/hardware_backend/tt/memory_pool_tt.py`**
   - `TTMHATokenToKVPool` - MHA用KVキャッシュプール
   - `TTMLATokenToKVPool` - MLA用KVキャッシュプール

4. **`python/sglang/srt/hardware_backend/tt/tt_model_runner.py`**
   - `TTModelRunner` - TT用モデル実行クラス
   - prefill/decode forward pass
   - トレース機能（将来のTT trace mode用）

5. **`python/sglang/srt/model_loader/tt_loader.py`**
   - `TTModelLoader` - TT最適化モデルローダー
   - tt-metal-modelsからの最適化モデル読み込み
   - フォールバック: 標準モデル実装を使用

### 変更ファイル

1. **`python/sglang/srt/utils/common.py`**
   - `is_tt()` 追加
   - `get_tt_memory_capacity()` 追加
   - `get_device()` に "tt" 追加
   - `get_device_name()` に TT対応追加

2. **`python/sglang/srt/server_args.py`**
   - `--device tt` オプション追加
   - `_handle_tt_backends()` メソッド追加
   - デフォルト設定: torch_native attention, pytorch sampling, CUDA graphs無効

3. **`python/sglang/srt/model_executor/model_runner.py`**
   - TT用分散バックエンド (gloo) 設定
   - `init_tt_backend()` 呼び出し

4. **`python/sglang/srt/model_loader/loader.py`**
   - TTModelLoader への振り分けロジック追加

5. **`python/sglang/srt/configs/device_config.py`**
   - "tt" デバイスタイプ追加

## 使用方法

```bash
python -m sglang.launch_server --model-path <model> --device tt
```

## 検証状況

### 環境
- T3K (Wormhole x 8)
- ttnn 0.65.1 (PyPI)
- KMD version: 2.4.1

### 確認済み
- [x] `ttnn.get_device_ids()` - 8台のデバイス検出成功
- [x] トポロジー検出成功 (ローカル: 0-3, リモート: 4-7)

### 未解決の問題
- [ ] `ttnn.open_device()` / `ttnn.open_mesh_device()` でsegfault
  - 発生箇所: `MetalContext::initialize_firmware`
  - ファームウェア19.1.0でも発生
  - ETH FW: 7.2.0

## 次のステップ

1. **デバイスオープンの問題解決**
   - ttnnバージョンとファームウェア/KMDの互換性確認
   - T3Kメッシュ構成での正しいデバイスオープン方法確認
   - tt-metalリポジトリのサンプルコード参照

2. **基本動作確認**
   - 単純なテンソル操作の動作確認
   - メモリ情報の取得

3. **SGLang統合テスト**
   - `is_tt()` の動作確認
   - サーバー起動テスト（`--device tt`）

4. **モデル実行**
   - 小さいモデル (e.g., Llama-2-7B) でのテスト
   - tt-metal-modelsとの統合

## 参考情報

### ttnn API (T3K向け)
```python
import ttnn

# デバイス検出
device_ids = ttnn.get_device_ids()  # [0, 1, 2, 3, 4, 5, 6, 7]

# メッシュデバイスとして開く (T3K = 2x4)
mesh_shape = ttnn.MeshShape(2, 4)
mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

# 単一デバイス
device = ttnn.open_device(device_id=0)
```

### バージョン互換性
- ttnn 0.65.1: firmware max 19.1.0
- 現在のシステム: firmware 19.1.0, ETH FW 7.2.0, KMD 2.4.1

### 関連リソース
- tt-metal: https://github.com/tenstorrent/tt-metal
- tt-metal-models: https://github.com/tenstorrent/tt-metal-models
