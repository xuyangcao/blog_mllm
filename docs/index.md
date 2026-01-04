# 前言

这不是一本"教程型"的书，而是一本**面向工程实践与研究思考的技术笔记集**。

## 本书目标

很多技术文章要么只讲公式不谈实现，要么只贴代码不解释原理。这本书试图架起两者之间的桥梁：

1. **公式拆解**：不只列出公式，而是逐层拆解，解释每个符号的含义和设计意图
2. **类比辅助**：用熟悉的概念（如监督学习）类比陌生的领域（如强化学习），降低理解门槛
3. **工程视角**：解释"为什么这样设计"——比如为什么要做归一化、为什么要 clip、如何防止梯度爆炸
4. **代码对照**：提供伪代码或核心实现，帮助将数学公式映射到实际代码

## 适合谁读

如果你正在做以下方向的研究或工程工作：

- 🔥 多模态大模型（VLM、MLLM）
- 🧠 大语言模型训练与优化
- ⚡ 分布式训练与推理加速
- 🎯 强化学习对齐（RLHF、GRPO、PPO）

希望这本书能帮你少走弯路，把原理搞透。

---

## 📚 内容目录

```{toctree}
:maxdepth: 3
:caption: 第一部分：基础理论与背景

intro/llm_overview
intro/dl_basics
intro/multimodal_basics
```

```{toctree}
:maxdepth: 3
:caption: 第二部分：核心模型架构

architecture/transformer
architecture/transformer/position_encoding
architecture/llm_design
architecture/mllm_arch
```

```{toctree}
:maxdepth: 3
:caption: 第三部分：训练方法与优化

training/data_preparation
training/distributed
training/rl_alignment
training/grpo
training/inference
```

```{toctree}
:maxdepth: 3
:caption: 第四部分：多模态应用与实战

applications/image_text
applications/video_audio
applications/digital_human
applications/engineering
applications/trends
```

```{toctree}
:maxdepth: 3
:caption: 附录

appendix/python_pytorch
appendix/math_basics
appendix/tools
appendix/references
```

---

## 进度概览

| 部分 | 章节 | 状态 |
|------|------|------|
| 基础理论 | 大模型概述 / 深度学习基础 / 多模态基础 | 📝 0/3 |
| 核心架构 | Transformer / 大模型设计 / 多模态架构 | 📝 0/3 |
| 训练优化 | 数据准备 / 分布式训练 / RL对齐 / GRPO / 推理部署 | ✅ 1/5 (GRPO 已完成) |
| 应用实战 | 图文 / 视频语音 / 数字人 / 工程实战 / 前沿趋势 | 📝 0/5 |
| 附录 | Python/PyTorch / 数学基础 / 工具 / 参考资料 | 📝 0/4 |

**总进度：** 1 / 20 已完成
