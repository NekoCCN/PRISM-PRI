# PRISM 系统：风力发电机叶片智能缺陷检测平台

**版本**: 1.0.0
**日期**: 2025年10月15日

---

## 目录

1.  [项目概述](#1-项目概述)
2.  [技术架构](#2-技术架构)
3.  [环境安装](#3-环境安装)
4.  [数据集准备](#4-数据集准备)
5.  [使用流程](#5-使用流程)
    - [步骤一：训练模型](#步骤一训练模型)
    - [步骤二：评估模型性能](#步骤二评估模型性能)
    - [步骤三：部署为API服务](#步骤三部署为api服务)
    - [步骤四：调用API进行预测](#步骤四调用api进行预测)
6.  [项目文件说明](#6-项目文件说明)

---

## 1. 项目概述

PRISM (Progressive Recognition and Intelligent Semantic Mapping) 是一个创新的多模态智能检测平台，专为风力发电机叶片的缺陷识别与分析而设计。本项目基于您提供的技术说明文档，实现了PRISM系统的核心功能，包括模型训练、性能评估和API服务部署。

## 2. 技术架构

-   **前端特征提取**: 集成 `OverLoCK` 模型 作为强大的特征提取骨干。
-   **语义特征增强**: 集成 `DINOv2` 模型 以提供全局语义特征。
-   **高效检测头**: 采用 `YOLOv10` 的检测头，在融合后的特征图上进行精确、高效的目标检测。
-   **智能分析后端**: 预留 `VLM (视觉语言模型)` 接口，用于对检测结果进行深度语义分析和报告生成。

## 3. 环境安装

首先，创建一个新的Python虚拟环境，然后按照以下步骤安装所有依赖。

### 3.1 安装核心依赖

将项目中的 `requirements.txt` 文件准备好，然后运行：

```bash
pip install -r requirements.txt
```

### 3.2 安装特殊依赖

部分依赖需要特殊方式安装以保证性能。

**1. `natten` (Neighborhood Attention Transformer)**

这是 `OverLoCK` 的关键依赖。请根据您的CUDA和PyTorch版本，从[官方发布页面](https://shi-labs.com/natten/wheels/)选择对应的wheel文件进行安装。例如：

```bash
# 适用于 PyTorch 2.3.x 和 CUDA 12.1 的示例
pip install natten==0.17.1+torch230cu121 -f [https://shi-labs.com/natten/wheels/](https://shi-labs.com/natten/wheels/)
```

**2. `depthwise_conv2d_implicit_gemm` (高效大核卷积)**

为了最大化 `OverLoCK` 的运行效率，强烈建议安装此自定义算子。

```bash
# 1. 克隆 SLaK 仓库
git clone [https://github.com/VITA-Group/SLaK.git](https://github.com/VITA-Group/SLaK.git)

# 2. 进入算子目录并编译安装
cd SLaK/slak_cuda
python setup.py install

# 3. 返回您的PRISM项目目录
cd ../..
```

## 4. 数据集准备

本项目使用标准的YOLO格式数据集。请按以下结构组织您的数据，并在项目根目录创建 `data.yaml` 文件。

**目录结构:**
```
/path/to/your_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

**`data.yaml` 文件示例 (请放在PRISM项目根目录):**
```yaml
# data.yaml
# 请务必使用绝对路径或相对于脚本运行位置的正确相对路径
train: /path/to/your_dataset/images/train
val: /path/to/your_dataset/images/val

# 类别数量
nc: 3

# 类别名称 (顺序必须与标签文件中的类别ID对应)
names: ['crack', 'peel', 'wear']
```

## 5. 使用流程

### 步骤一：训练模型

使用 `train_prism_v2.py` 脚本对PRISM系统的融合模块和检测头进行微调。脚本将自动加载您的 `data.yaml` 配置。

**命令:**
```bash
python train_prism_v2.py
```

训练过程中，模型权重会定期保存为 `.pth` 文件（例如 `prism_finetuned_epoch_10.pth`）。

### 步骤二：评估模型性能

训练完成后，使用 `evaluate_prism.py` 脚本在验证集上评估模型的性能（mAP等指标）。

**修改脚本:**
打开 `evaluate_prism.py` 文件，修改 `weights_file` 变量为您想要评估的权重文件路径。

```python
# evaluate_prism.py
...
if __name__ == '__main__':
    # --- 如何使用 ---
    # 1. 指定训练好的权重文件路径
    weights_file = 'prism_finetuned_epoch_50.pth' # <-- 修改这里
    ...
```

**命令:**
```bash
python evaluate_prism.py
```
评估脚本会输出 mAP@0.5, mAP@0.5:0.95 等标准指标。

### 步骤三：部署为API服务

使用 `deploy_server.py` 脚本将训练好的模型部署为一个HTTP服务器。

**修改脚本:**
打开 `deploy_server.py` 文件，修改 `WEIGHTS_PATH` 变量为您最终选定的模型权重。

```python
# deploy_server.py
...
# --- 2. 全局变量和配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_PATH = 'prism_finetuned_epoch_50.pth' # <-- 修改这里
...
```

**命令:**
```bash
python deploy_server.py
```
服务启动后，会监听本地的 `8000` 端口。

### 步骤四：调用API进行预测

服务器运行后，您可以通过多种方式调用API。

**方式一：使用自动生成的API文档 (推荐)**

1.  打开浏览器，访问 `http://127.0.0.1:8000/docs`。
2.  展开 `/predict/` 接口，点击 "Try it out"。
3.  点击 "Choose File"，上传一张您要检测的叶片图片。
4.  点击 "Execute"。服务器将返回JSON格式的详细分析报告。

**方式二：使用 `curl` 命令**

在终端中执行以下命令，将 `path/to/your/blade_image.jpg` 替换为您的图片路径。

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/blade_image.jpg;type=image/jpeg'
```

## 6. 项目文件说明

-   `README.md`: 本文档。
-   `requirements.txt`: Python依赖列表。
-   `overlock_local.py`: `OverLoCK` 模型的本地化实现，作为项目的核心组件。
-   `train_prism_v2.py`: 负责模型训练的脚本，集成了 `ultralytics` 的官方损失函数。
-   `evaluate_prism.py`: 负责模型性能评估的脚本，可计算mAP等指标。
-   `deploy_server.py`: 基于FastAPI的部署脚本，将模型封装为HTTP服务。