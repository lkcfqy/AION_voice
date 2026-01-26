# AION Voice - 语音交互认知智能体

本项目是一个基于 **液体状态机 (LSM)** 和 **超维计算 (HDC)** 的语音交互智能体框架。它能够感知音频输入，通过联想记忆进行思考，并利用运动皮层驱动参数化语音合成器进行回复。

## 核心架构

- **感知层 (Cochlea & LSM)**: 将原始音频频谱转换为神经脉冲序列，并进一步投影为 HDC 向量。
- **认知层 (GWT & HDC)**: 使用全局工作空间 (Global Workspace) 管理意图，利用现代 Hopfield 网络 (MHN) 存储情节记忆。
- **驱动层 (Social Drive)**: 基于社交渴望驱动智能动发起或响应对话，而非传统的电池/饥饿感逻辑。
- **执行层 (Motor Cortex & WORLD)**: 将大脑意图转换为 WORLD 声码器的发音参数。

## 快速开始

### 1. 启动 Visdom 仪表盘
```bash
python -m visdom.server
```

### 2. 运行交互主循环
```bash
python scripts/interaction_loop.py
```

### 3. 正式运动校准
如果需要重新校准发音动作：
```bash
python scripts/motor_babbling.py
```


