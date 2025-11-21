# Tomato 数据集训练指南 - Point Transformer V3

## ✅ 已完成的配置

我已经为你的 Tomato 数据集配置好了 Point Transformer V3 训练环境：

1. ✅ 创建了自定义数据集类 `TomatoDataset` (`pointcept/datasets/tomato.py`)
2. ✅ 创建了训练配置文件 (`configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py`)
3. ✅ 创建了训练脚本 (`scripts/train_tomato.sh`)
4. ✅ 测试通过 - 数据加载正常（77个训练样本，16个验证样本）

## 🚀 开始训练

**重要**: 必须先激活 conda 环境：
```bash
conda activate pointcept
```

### 方法1：使用训练脚本（推荐）

```bash
cd /home/sheepyd/article_reproduction/Pointcept
bash scripts/train_tomato.sh
```

脚本会自动激活 pointcept 环境并设置 PYTHONPATH。

### 方法2：直接使用Python命令

**必须先激活环境和设置PYTHONPATH**:

```bash
conda activate pointcept
cd /home/sheepyd/article_reproduction/Pointcept
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 单GPU训练
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
    --config-file configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/ptv3_tomato

# 多GPU训练（如果有多个GPU）
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py \
    --config-file configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py \
    --num-gpus 2 \
    --options save_path=exp/tomato/ptv3_tomato batch_size=8
```

## 📊 数据集统计

- **训练样本**: 77个点云文件
- **验证样本**: 16个点云文件  
- **测试样本**: 7个点云文件
- **语义类别**: 3类 (0: background, 1: stem, 2: tomato)
- **任务类型**: 实例分割 (Instance Segmentation)

## ⚙️ 重要参数配置

### 当前配置 (`configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py`)

- **batch_size**: 4（可根据GPU显存调整）
- **num_worker**: 4（数据加载线程）
- **epoch**: 800（训练轮数）
- **optimizer**: AdamW (lr=0.006)
- **grid_size**: 0.02（体素化网格大小）

### 如何修改配置

#### 1. 修改类别名称

如果你的类别名称不是 background/stem/tomato，编辑配置文件：

```python
class_names = [
    "你的类别0",  # class 0
    "你的类别1",  # class 1  
    "你的类别2",  # class 2
]
```

#### 2. GPU显存不足

如果遇到 CUDA out of memory 错误，修改：

```python
batch_size = 2  # 从4减到2
```

或在命令行指定：

```bash
python tools/train.py \
    --config-file configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py \
    --num-gpus 1 \
    --options batch_size=2
```

#### 3. 调整训练轮数

```python
epoch = 400  # 减少训练轮数
```

## 📁 训练输出

训练结果保存在 `exp/tomato/ptv3_tomato/`:

```
exp/tomato/ptv3_tomato/
├── model/
│   ├── model_best.pth      # 最佳模型
│   └── model_last.pth      # 最新模型
├── log.txt                 # 训练日志
└── config.py               # 使用的配置
```

## 📝 监控训练过程

### 实时查看日志

```bash
tail -f exp/tomato/ptv3_tomato/log.txt
```

### 日志包含的信息

- 每个epoch的训练损失
- 验证集评估指标（mIoU, Precision, Recall等）
- 学习率变化
- 训练时间统计

## 🔄 恢复训练

如果训练中断，可以从checkpoint恢复：

```bash
python tools/train.py \
    --config-file configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/ptv3_tomato resume=True
```

## 🧪 测试模型

训练完成后测试模型：

```bash
python tools/test.py \
    --config-file configs/tomato/insseg-pointgroup-v1m2-0-ptv3-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/ptv3_tomato weight=exp/tomato/ptv3_tomato/model/model_best.pth
```

## 🎯 使用预训练模型（可选）

如果想使用在ScanNet上预训练的模型进行微调，下载预训练权重并在配置文件中添加：

```python
hooks = [
    dict(type="CheckpointLoader", 
         keywords="module.", 
         replacement="module.",
         load_path="path/to/pretrained_model.pth"),
    # ... 其他hooks
]
```

## ❓ 常见问题

### 1. CUDA out of memory
**解决方案**: 减小 `batch_size` 到 2 或 1

### 2. 数据加载很慢
**解决方案**: 增加 `num_worker`，但不要超过CPU核心数

### 3. 想加快训练速度
**解决方案**: 
- 使用多GPU: `--num-gpus 2`
- 启用混合精度训练（已默认开启 `enable_amp=True`）
- 增加 `batch_size`（如果显存允许）

### 4. 想修改网格采样大小
在配置中修改 `GridSample` 的 `grid_size`:
```python
dict(type="GridSample", grid_size=0.02)  # 减小会保留更多点
```

## 📚 更多信息

- 配置详解: `configs/tomato/README.md`
- 数据集类: `pointcept/datasets/tomato.py`
- 训练脚本: `scripts/train_tomato.sh`

## 🎉 下一步

1. 运行 `bash scripts/train_tomato.sh` 开始训练
2. 使用 `tail -f exp/tomato/ptv3_tomato/log.txt` 监控训练
3. 训练完成后使用 `tools/test.py` 评估模型
4. 根据结果调整超参数进行优化

祝训练顺利！🍅
