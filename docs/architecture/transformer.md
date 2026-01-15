# Transformer 架构详解

Transformer 是当前 LLM/MLLM 的核心骨干架构。它的关键特点是：用**注意力（Attention）**替代循环结构，使得序列建模更易并行、可扩展，并且在大规模训练中表现出稳定的 scaling 特性。

本章目标：

- 给出 Transformer 的**最小可用数学定义**（你能看懂论文与实现）
- 解释关键模块背后的**设计动机与工程权衡**
- 为后续“KV Cache / RoPE / 分布式训练与推理优化”建立共同语言

---

## Transformer 的最小结构图

以 Decoder-only（GPT 系）为例，每层（block）可以抽象为：

\[
h \leftarrow h + \mathrm{Attn}(\mathrm{LN}(h))
\]
\[
h \leftarrow h + \mathrm{MLP}(\mathrm{LN}(h))
\]

其中：

- \(h \in \mathbb{R}^{T \times d}\) 是长度为 \(T\) 的 token 表示（hidden states）
- LN 是 LayerNorm（通常是 Pre-LN 结构）
- Attn 是（带 causal mask 的）自注意力
- MLP 是前馈网络（FFN/MLP）

你会发现：**残差连接 + 归一化 + 两个子模块（Attention/MLP）**构成了几乎所有现代 LLM 的“层级骨架”。

---

## 为什么 Attention 有效：一句话直觉

Attention 可以看成“内容相关的动态加权聚合”：每个 token 根据自身 query 与其他 token key 的匹配程度，对 value 做加权求和，从而把“与当前生成最相关的信息”搬运到当前位置。

---

## 工程视角：Transformer 的关键瓶颈

- **训练**：显存（激活 + 优化器状态）、通信（并行策略）、数值稳定（混合精度）
- **推理**：prefill 的 \(O(T^2)\) attention、decode 阶段的 KV cache 带宽、batching/并发与延迟

这也是为什么后面章节会重点讲：KV Cache、位置编码（RoPE/ALiBi）、并行训练与推理加速。

---

## 📖 详细章节

```{toctree}
:maxdepth: 2

transformer/attention
transformer/self_attention
transformer/multi_head_attention
transformer/ffn
transformer/layernorm_residual
transformer/position_encoding
transformer/rope_alibi
transformer/kv_cache
```
