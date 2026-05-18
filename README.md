# AION Voice

面向语音交互的脑启发智能体实验原型。它把实时麦克风输入转换为 Mel 频谱，经 Liquid State Machine、超维编码、注意力式 Global Workspace 和 Modern Hopfield Network 处理，再尝试生成语音响应。

## 当前状态

仓库已经包含实时交互循环、生成式读出层训练脚本和核心模块实现。当前代码可以作为语音版 AION 的研究骨架使用，但默认仓库没有附带训练数据集，也没有提交 `lsm_readout_weights.pt` 读出层权重；未训练时主循环会提示模型无法正常说话。

系统支持在用户沉默一段时间后进入 sleep/dream 风格的记忆整理流程。`datasets/**/*.wav` 下如有音频文件，主循环会尝试预加载为语音记忆。

## 主要模块

- `scripts/interaction_loop.py`：实时麦克风输入、记忆检索、响应生成主循环。
- `scripts/train_generative.py`：训练 LSM 读出层并保存 `lsm_readout_weights.pt`。
- `src/lsm.py`：Liquid State Machine。
- `src/hdc.py`：语音超维编码。
- `src/gwt.py`：注意力式 Global Workspace。
- `src/mhn.py`：Modern Hopfield Network 记忆。
- `src/codec.py`：音频特征与重建相关工具。
- `src/config.py`：采样率、Mel 维度、设备和权重路径配置。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如 `torch` 安装因为 CUDA 源配置失败，建议先按本机 CUDA/CPU 环境单独安装 PyTorch，再安装其余依赖。

训练生成式读出层：

```bash
python scripts/train_generative.py --data datasets
```

启动实时交互：

```bash
python scripts/interaction_loop.py
```

## 运行提示

- 需要可用的麦克风和扬声器。
- `sounddevice` 依赖系统音频后端；macOS 或 Linux 上可能需要额外安装 PortAudio。
- CUDA 可显著提升体验；CPU 模式会自动降低 LSM 神经元规模。
- Visdom 只作为可选可视化，不是主流程必需项。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
