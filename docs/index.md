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

## 阅读建议

每篇文章通常按以下结构组织：

1. **核心思想**：一句话概括技术本质
2. **公式回顾**：列出关键公式
3. **深入理解**：逐项拆解公式，解释设计动机
4. **伪代码/实现**：对照代码理解实现细节
5. **参考资料**：论文、文档、工具链接

---

## 📚 内容目录

```{toctree}
:maxdepth: 2
:caption: 分布式训练

dist/mixed_precision
dist/gradient_accumulation
dist/pipeline_parallel
dist/data_parallel
dist/model_parallel
dist/parallel_training_optimization
```

```{toctree}
:maxdepth: 2
:caption: Transformer

transformer/attention
transformer/kv_cache
transformer/rope
transformer/position_encoding
transformer/self_attention
transformer/multi_head_attention
```

```{toctree}
:maxdepth: 2
:caption: 强化学习

RL/ppo
RL/grpo
```

```{toctree}
:maxdepth: 2
:caption: 模型解析

model/qwen-vl
model/intervl
model/deepseek-r1
```

---

### 进度概览

| 分类 | 状态 |
|------|------|
| 分布式训练 | 📝 0/6 |
| Transformer | 📝 0/6 |
| 强化学习 | ✅ 1/2 (GRPO 已完成) |
| 模型解析 | 📝 0/3 |

**总进度：1 / 17 已完成**

