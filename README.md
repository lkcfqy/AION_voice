# AION Voice: 认知音频智能体 (GPU 架构版)

AION (Artificial Intelligence on Neuromorphic-principles) 是一个受生物大脑启发的认知智能体。
它不使用传统的深度学习 (Transformer/LLM)，而是结合了 **液体状态机 (Liquid State Machine, SNN)** 和 **超维计算 (HDC)**，在 GPU 上模拟大规模神经动力学。

---

## 🚀 核心架构升级 (Phase 2)

本项目已完成从 CPU (Nengo) 到 **GPU (PyTorch Native)** 的底层重构。

### 1. 20k 神经元大脑 (LSM)
*   **旧版**: 2,000 神经元 (CPU), 响应慢，容易过载。
*   **新版**: **20,000 神经元 (GPU)**, 并行加速。能够捕捉极细微的语音时空特征。
*   **特性**: 混沌边缘 (Edge of Chaos) 初始化，支持复杂的非线性映射。

### 2. 在线睡眠与做梦 (Online Sleep)
*   **机制**: 当系统检测到环境长时间安静（>30秒）时，会自动进入 **梦境模式 (Dreaming State)**。
*   **功能**: 在梦中，海马体 (MHN) 会随机回放之前的记忆片段，激活大脑皮层 (LSM) 进行无监督的强化学习（巩固记忆）。

### 3. 注意力全域工作空间 (Attention GWT)
*   **机制**: 引入了类似 Transformer 的 Key-Query-Value 注意力机制。
*   **功能**: 智能体不再是被动接收所有信息，而是根据当前的内部驱动力 (Social Drive) **主动聚焦**于听觉或记忆信号。

### 4. 谐振器网络 (Resonator Networks)
*   **机制**: 实现了 HDC 的迭代解码算法。
*   **功能**: 能够从叠加的概念向量中精准解析出复合结构（例如区分 "Cat bites Dog" 和 "Dog bites Cat"）。

---

## 🛠️ 安装指南

**前置要求**:
*   Windows / Linux
*   NVIDIA 显卡 (支持 CUDA 11.8+)
*   Miniconda 或 Anaconda

### 1. 创建环境

```bash
conda create -n aion_voice python=3.10
conda activate aion_voice
```

### 2. 安装依赖 (关键步骤)

为了确保 GPU 加速生效，请严格按照以下顺序安装：

```bash
# 1. 优先安装 CUDA 版 PyTorch (修正版)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 安装其余音频与可视化依赖
pip install -r requirements.txt
```

---

## ▶️ 运行指南

### 1. 启动可视化仪表盘 (可选)

AION 拥有一个实时的大脑内部监控系统。
在独立的终端运行：

```bash
python -m visdom.server
```
*访问地址: http://localhost:8097*

### 2. 训练生成模型 (初次运行必须)

让大脑适应你的麦克风环境和语音特征。这会训练 LSM 的读出层权重。

```bash
# 训练 500 个样本 (速度非常快, GPU 加速)
python scripts/train_generative.py --data datasets/LJSpeech-1.1 --limit 500
```
*成功标志: 终端显示 `✅ 权重已保存至...` 且 `Max Weight` 正常 (约 0.01~0.1)。*

### 3. 启动交互 (唤醒 AION)

这是主程序。它会监听麦克风，并尝试与你互动。

```bash
python scripts/interaction_loop.py
```

**如何体验：**
*   **说话**: 对着麦克风说 "Hello" 或读一段文字，观察终端的音量反馈。
*   **观察做梦**: **保持房间完全安静 30 秒**。你会看到终端显示 `💤 Dreaming...`，并听到它在“自言自语”（这是它在整理记忆）。

---

## 📂 目录清理说明

目前项目结构已完成净化，**没有冗余文件**。
所有 `src/*.py` 和 `scripts/*.py` 均为适配 GPU 的最新版本。无需删除任何代码文件。

*   `src/lsm.py`: PyTorch GPU 实现
*   `src/gwt.py`: Attention GWT 实现
*   `src/hrr.py`: Resonator 实现

---

## ⚠️ 常见问题

**Q: 报错 `No module named 'visdom'` 或类似库缺失?**
A: 这是因为之前强制重装 PyTorch 时依赖链断了。请运行 `pip install visdom requests Pillow` 修复。

**Q: 启动时报错 `Matmul size mismatch`?**
A: 请确保您使用的是最新的 `interaction_loop.py`，我们已修复了张量扁平化 (Flatten) 的问题。

**Q: 必须有显卡吗?**
A: 是的。虽然代码能在 CPU 上跑，但 20k 神经元的规模在 CPU 上会非常卡顿（<1Hz），无法实时交互。
