# MambaPulse

基于 Vision Mamba 的无人机航拍小目标检测框架。模型采用纯 Mamba 主干与 DETR 风格的查询解码器，在 VisDrone2019-DET 数据集上进行小目标检测。



## 方法概览

MambaPulse 由编码器和解码器两部分组成：

**编码器（`encoder.py`）**
- **ViM 主干**：基于 Vision Mamba 的 backbone，从 24 层中提取 4 个尺度的特征。
- **HiRes-Mamba 分支（`vim/hires_mamba.py`）**：高分辨率纯 Mamba 分支，对原图做 4×4 patch embedding 后经双向 Mamba 处理，提供 stride=4 的高分辨率特征，专门服务于小目标。
- **P2 融合路径**：用 PixelShuffle + 双向 Mamba 做上采样，再用纯 Mamba 模块（`MambaFusion`）融合 HiRes 分支与主干特征，全程不依赖卷积特征提取。
- **BiFPN neck**：对 5 个尺度（P2 到 P6）做加权双向特征融合。

**解码器（`model.py`）**
- DETR 风格查询解码器，使用 group queries（默认 6 组、每组 400 个查询）。
- **MQSI**：双向 Mamba 实现的查询自交互模块（替代自注意力）。
- **MQI**：双向 CrossMamba 实现的查询-特征交叉交互模块（替代交叉注意力），对每个尺度做 2D 对齐后融合。
- 每层输出分类与边界框预测，训练时使用辅助损失（aux loss）。

训练使用 Hungarian 匹配 + Focal Loss + L1/GIoU 边界框损失，支持类别平衡的 alpha 加权。

## 目录结构

```
MambaPulse/
├── encoder.py                      # 编码器：ViM 主干 + HiRes-Mamba + P-1 融合 + BiFPN
├── model.py                        # 完整模型：编码器 + 解码器
├── train.py                        # 训练脚本（VisDrone 数据集、mAP 评估）
├── mamba_block.py                  # Block / CrossBlock 包装器
├── cross_mamba.py                  # CrossMamba mixer
├── selective_scan_interface_ca.py  # CrossMamba 的 CUDA 选择性扫描接口
├── hires_mamba.py                  # HiRes-Mamba 分支
└── README.md
```

## 环境配置

`mamba-ssm` 与 `causal-conv1d` 需要编译 CUDA 算子，对 CUDA / PyTorch 版本较为敏感，版本不匹配会导致编译或运行失败。`selective_scan_interface_ca.py` 针对 `causal-conv1d 1.1.0` 的接口做了适配，使用其他版本时若报参数数量不符，需检查该文件中的 `causal_conv1d_fwd` / `causal_conv1d_bwd` 调用。

- Python: `3.10`
- CUDA: `11.8`
- PyTorch: `2.1.1`

主要依赖：

```bash
pip install torch torchvision          # 与你的 CUDA 版本匹配
pip install causal-conv1d==1.1.0       # 版本需与 selective_scan_interface_ca.py 适配
pip install mamba-ssm
pip install timm einops
pip install albumentations opencv-python scipy numpy
```

此外还需要 `selective_scan_cuda` 与 `causal_conv1d_cuda`（随 `mamba-ssm` / `causal-conv1d` 编译安装）。

## 数据集

使用 VisDrone2019-DET，按如下结构组织：

```
<data_root>/
├── VisDrone2019-DET-train/
│   ├── images/         
│   └── annotations/    
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

标注沿用 VisDrone 官方格式（每行 `x,y,w,h,score,category,...`），脚本会自动忽略 `category` 为 0（ignored regions）和 11（others）的标注，其余映射为 10 个检测类别。

## 训练

```bash
python train.py \
    --data_root /path/to/visdrone \
    --batch_size 6 --grad_accum_steps 3 \
    --epochs 300 --lr 1e-4 \
    --num_queries 400 --num_groups 6 \
    --amp
```

常用参数：

- `--num_queries`：每组查询数，必须是完全平方数（默认 400 = 20²）。
- `--num_groups`：group queries 组数（默认 6）。
- `--img_size`：输入分辨率（默认 640）。
- `--backbone_lr_mult` / `--hires_lr_mult`：主干与 HiRes 分支相对于其余模块的学习率倍数。
- `--class_balanced`：启用基于类别频率的 Focal Loss alpha 加权。
- `--amp`：混合精度训练。


## 致谢

这项工作以 [Vim](https://github.com/hustvl/Vim)（Zhu等人）和[Mamba](https://github.com/state-spaces/mamba)（Gu 和 Dao）为基础。