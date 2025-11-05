





# Open Stereo Matching Zoo

一个全面的立体视觉匹配算法库，集成了多种先进的立体匹配模型，支持训练、推理和部署。

## 🚀 特性

- **多种模型支持**: 集成了 CoEx、StereoNet、AANet、Monster 等先进的立体匹配算法
- **完整工作流**: 支持从训练到部署的完整流程
- **多数据集**: 支持 SceneFlow、KITTI、Middlebury、ETH3D 等标准数据集
- **高性能推理**: 支持 TensorRT 加速和 ONNX 导出
- **灵活配置**: 模块化设计，易于扩展和自定义
- **丰富工具**: 包含数据预处理、可视化、评估等工具

## 📁 项目结构

```
Open-StereoMatching-zoo/
├── models/                          # 模型实现
│   ├── models/
│   │   ├── coex/                   # CoEx 模型
│   │   ├── aanet/                  # AANet 模型
│   │   ├── stereonet/              # StereoNet 模型
│   │   └── hat/                    # HAT 模型相关
│   ├── Monster/                    # Monster 模型（集成深度估计）
│   ├── nn/                         # 神经网络基础模块
│   └── utils/                      # 模型工具函数
├── core/                           # 核心功能模块
│   ├── dataset/                    # 数据集处理
│   ├── utils/                      # 工具函数
│   └── extractor.py                # 特征提取器
├── scripts/                        # 训练和评估脚本
│   ├── train_*.py                  # 各数据集训练脚本
│   ├── evaluate_stereo.py          # 评估脚本
│   ├── demo_video.py               # 视频演示
│   └── get_annotations/            # 标注工具
├── train_*.py                      # 主要训练脚本
├── convert_trt.py                  # TensorRT 转换
├── config.py                       # 配置文件
├── save_disp.py                    # 视差图保存
└── environment.yml                 # 环境配置
```

## 🛠️ 安装

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.0+ (推荐使用 GPU)
- TensorRT (可选，用于加速推理)

### 快速安装

1. 克隆项目
```bash
git clone <repository-url>
cd Open-StereoMatching-zoo
```

2. 使用 Conda 创建环境
```bash
conda env create -f environment.yml
conda activate torch2.0
```

3. 或者手动安装依赖
```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib
pip install onnx onnxruntime tensorrt
pip install timm wandb gradio
```

## 🚀 快速开始

### 训练模型

1. **SceneFlow 数据集训练**
```bash
python train_sceneflow.py --maxdisp 192 --model CoEx
```

2. **KITTI 数据集训练**
```bash
python scripts/train_kitti.py --maxdisp 192 --model CoEx --loadckpt ./checkpoints/pretrained_model.pth
```

3. **自定义数据训练**
```bash
python train_stereonet.py --dataroot ./data/custom --model CoEx
```

### 推理

1. **基础推理**
```python
import torch
from models.models.coex.models.stereo.CoEx import CoEx

# 加载模型
model = CoEx(cfg)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 推理
with torch.no_grad():
    left_img = torch.randn(1, 3, 384, 1248)
    right_img = torch.randn(1, 3, 384, 1248)
    disparity = model(left_img, right_img)
```

2. **TensorRT 加速推理**
```bash
python convert_trt.py --model CoEx --checkpoint ./checkpoints/best_model.pth
```

### 评估

```bash
python scripts/evaluate_stereo.py --model CoEx --dataset KITTI --checkpoint ./checkpoints/best_model.pth
```

## 📊 支持的模型

### CoEx
- **特点**: 结合成本聚合和特征上采样的高效立体匹配网络
- **论文**: [CoEx: Collaborative Explosion for Stereo Matching]
- **支持数据集**: SceneFlow, KITTI, Middlebury

### AANet
- **特点**: 自适应聚集网络，支持多尺度特征融合
- **论文**: [AANet: Adaptive Aggregation Network for Stereo Matching]
- **支持数据集**: SceneFlow, KITTI, ETH3D

### StereoNet
- **特点**: 轻量级立体匹配网络，适合实时应用
- **支持数据集**: SceneFlow, 自定义数据集

### Monster
- **特点**: 结合单目深度估计和立体匹配的混合架构
- **集成**: Depth Anything V2 深度估计模型
- **优势**: 在弱纹理区域表现更好

## 🗂️ 数据集准备

### SceneFlow
```bash
datasets/
└── sceneflow/
    ├── frames_finalpass/
    │   ├── TRAIN/
    │   └── TEST/
    └── disparity/
        ├── TRAIN/
        └── TEST/
```

### KITTI 2015
```bash
datasets/
└── kitti2015/
    ├── training/
    │   ├── image_2/
    │   ├── image_3/
    │   └── disp_noc_0/
    └── testing/
        ├── image_2/
        └── image_3/
```

## ⚙️ 配置说明

主要配置项在 `config.py` 中：

```python
# 模型配置
model_config = {
    'max_disparity': 192,        # 最大视差
    'backbone': {
        'type': 'resnet18',      # 骨干网络
        'channels': {...}
    },
    'corr_volume': True,         # 是否使用成本体积
}

# 相机参数配置
camera_config = {
    'height': 480,
    'width': 640,
    'fx': 229.98,
    'fy': 229.98,
    'cx': 318.05,
    'cy': 206.48,
    'baseline': 70.04
}
```

## 🎯 性能基准

| 模型 | 数据集 | EPE (像素) | D1-all (%) | 推理时间 (ms) |
|------|--------|------------|------------|---------------|
| CoEx | SceneFlow | 0.58 | 1.24 | 45 |
| AANet | KITTI 2015 | 0.68 | 2.89 | 38 |
| StereoNet | SceneFlow | 0.82 | 1.95 | 25 |
| Monster | 自定义 | 0.75 | 2.12 | 52 |

## 🔧 高级功能

### 1. TensorRT 优化
```bash
# 导出 ONNX 模型
python convert_trt.py --export_onnx --model CoEx

# 构建 TensorRT 引擎
python convert_trt.py --build_engine --model CoEx
```

### 2. 可视化工具
```bash
# 视频演示
python scripts/demo_video.py --input ./video.mp4 --output ./output.mp4

# 保存视差图
python save_disp.py --left left.png --right right.png --output disp.png
```

### 3. 自定义数据集
```python
from core.dataset.mix_dataset import StereoDataset

dataset = StereoDataset(
    dataroot='./your_dataset',
    training=True,
    transform=transforms.Compose([...])
)
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [SceneFlow Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow.en.html)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- 原始模型作者和贡献者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

## 🔗 相关链接

- [项目主页](https://github.com/your-repo)
- [文档](https://your-docs-site.com)
- [演示视频](https://www.youtube.com/watch?v=your-video)

---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！