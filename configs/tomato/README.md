# Tomato Dataset Semantic Segmentation

## 数据集信息

- 数据路径: `datasets/tomato_20480/`
- 语义类别数: 3 (soil, stem, tomato)
- 训练样本: 根据train文件夹中的数据
- 验证样本: 根据val文件夹中的数据
- 测试样本: 根据test文件夹中的数据

## 可用配置文件

| 配置文件 | 骨干网络 | 显存占用 | 推荐场景 |
|---------|---------|---------|---------|
| `semseg-pt-v3m1-0-tomato.py` | Point Transformer v3 | 高 | 高性能GPU |
| `semseg-spunet-v1m1-0-tomato.py` | SpUNet | 低 | 显存有限 |

## 快速开始

### 1. 激活conda环境

```bash
conda activate pointcept
```

### 2. 开始训练

**使用 SpUNet（推荐，显存占用低）:**
```bash
# 单GPU训练
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
    --config-file configs/tomato/semseg-spunet-v1m1-0-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/spunet_tomato

# 多GPU训练（例如使用2个GPU）
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py \
    --config-file configs/tomato/semseg-spunet-v1m1-0-tomato.py \
    --num-gpus 2 \
    --options save_path=exp/tomato/spunet_tomato
```

**使用 Point Transformer v3（需要更多显存）:**
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
    --config-file configs/tomato/semseg-pt-v3m1-0-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/ptv3_tomato
```

### 3. 恢复训练（如果中断了）

```bash
python tools/train.py \
    --config-file configs/tomato/semseg-spunet-v1m1-0-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/spunet_tomato resume=True
```

### 4. 测试模型

```bash
python tools/test.py \
    --config-file configs/tomato/semseg-spunet-v1m1-0-tomato.py \
    --num-gpus 1 \
    --options save_path=exp/tomato/spunet_tomato weight=exp/tomato/spunet_tomato/model/model_best.pth
```

## 配置说明

### 修改类别名称

如果你的3个类别不是 soil, stem, tomato，请在配置文件中修改：

```python
class_names = [
    "your_class_0",  # class 0
    "your_class_1",  # class 1  
    "your_class_2",  # class 2
]
```

### 调整训练参数

在配置文件中可以修改：

- `batch_size`: 批次大小（SpUNet推荐8-16，PTv3推荐2-4）
- `num_worker`: 数据加载线程数
- `epoch`: 训练轮数（默认800）
- `optimizer` 和 `scheduler`: 优化器和学习率调度器参数

### GPU显存不足怎么办？

1. **使用 SpUNet**: 比 PTv3 显存占用低很多
2. 减小 `batch_size`（例如改为2或4）
3. 增大 `grid_size`（例如从0.02改为0.04，会降低精度）
4. 使用混合精度训练 `enable_amp=True`（默认已启用）

## 输出文件

训练过程会在 `exp/tomato/` 目录下生成：

- `model/`: 保存的模型权重
  - `model_best.pth`: 最佳模型
  - `model_last.pth`: 最后一次的模型
- `log.txt`: 训练日志
- `config.py`: 使用的配置文件副本

## 监控训练

可以使用 `tail` 命令实时查看训练日志：

```bash
tail -f exp/tomato/ptv3_tomato/log.txt
```

## 常见问题

1. **CUDA out of memory**: 减小batch_size
2. **数据加载慢**: 增加num_worker（但不要超过CPU核心数）
3. **想使用预训练模型**: 在配置文件的hooks中添加预训练权重路径

## 性能提示

- 首次运行会编译CUDA算子，可能需要几分钟
- 建议先用小数据集测试配置是否正确
- 可以添加 `enable_amp=True` 启用混合精度训练以节省显存
