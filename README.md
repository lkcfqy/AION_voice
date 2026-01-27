# AION Voice

AION Voice 是一个基于 **液体状态机 (Liquid State Machine, LSM)** 和 **超维计算 (Hyperdimensional Computing, HDC)** 的认知音频智能体项目。它旨在模拟类似生物的听觉感知、认知处理和语音生成过程。

## 核心特性

*   **液体状态机 (LSM)**: 使用脉冲神经网络 (SNN) 处理音频流，提取时空特征。
*   **全局工作空间 (GWT)**: 基于 HDC 的全局工作空间理论实现，用于信息交换和广播。
*   **现代 Hopfield 网络 (MHN)**: 用于情景记忆存储和检索。
*   **社会驱动 (Social Drive)**: 基于自由能原理 (FEP) 和社交需求驱动智能体行为。
*   **Visdom 仪表盘**: 实时可视化大脑内部活动（脉冲、概念相似度、情感状态）。

## 安装指南

建议使用 Conda 创建虚拟环境：

```bash
conda create -n aion_voice python=3.9
conda activate aion_voice
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行指南

### 1. 启动可视化服务器 (可选但推荐)

为了查看实时仪表盘，请在单独的终端中启动 Visdom 服务器：

```bash
python -m visdom.server
```

### 2. 运行交互循环

这是智能体的主程序，它会监听麦克风输入并尝试进行交互：

```bash
python scripts/interaction_loop.py
```

### 3. 训练生成模型

如果需要重新训练 LSM 的读出层权重：

```bash
python scripts/train_generative.py --data <path_to_wav_dataset>
```

## 目录结构

*   `src/`: 核心源代码 (LSM, GWT, HDC, Drive 等)。
*   `scripts/`: 运行脚本 (交互循环, 训练脚本)。
*   `datasets/`: 存放数据集。
