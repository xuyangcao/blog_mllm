# RoPE 与 ALiBi（位置编码的两条主线）

位置编码的作用是：让模型区分“同一组 token 的不同排列”，并表达相对顺序信息。对于 Transformer 来说，位置相关性最终会体现在 attention logits 上。

本节聚焦两类在 LLM 中非常常见的方法：

- **RoPE（Rotary Position Embedding）**：通过旋转把位置信息注入到 \(Q,K\) 中
- **ALiBi（Attention with Linear Biases）**：直接给 attention logits 加一个随距离变化的 bias

> 绝对位置编码的基础与示意图见 `position_encoding.md`。

---

## RoPE：把位置变成“旋转”

RoPE 的核心做法是：对每个位置 \(t\)，用一个与 \(t\) 相关的旋转矩阵 \(R_t\) 作用在 \(Q,K\) 上：

\[
Q_t' = R_t Q_t,\quad K_t' = R_t K_t
\]

然后 attention logits 使用旋转后的 \(Q',K'\)：

\[
Q_t' (K_j')^\top
\]

直觉上：旋转把绝对位置编码成“相位”，从而让 \(t-j\) 的相对位置信息自然地出现在内积中。

### 工程收益

- 相对位置建模能力强
- 在长上下文（以及外推）上表现通常更好
- 实现上只作用于 \(Q,K\)，与注意力算子融合友好

---

## ALiBi：在 logits 上加距离偏置

ALiBi 的思想非常直接：对每个注意力分数加上与相对距离相关的线性 bias：

\[
\mathrm{score}_{t,j} = \frac{Q_t K_j^\top}{\sqrt{d_k}} + b(t-j)
\]

其中 \(b(\Delta)\) 通常是负斜率乘以距离，使得模型天然偏好近邻 token（但仍允许远距离关注）。

### 工程收益

- 极其简单：不改 \(Q,K\) 的表示，只改 logits
- 长上下文外推往往更稳（取决于训练与实现细节）

---

## RoPE vs ALiBi：怎么选？

从工程角度给一个经验对比：

- **想要通用 LLM 配方、社区成熟实现、与主流 checkpoint 兼容**：RoPE 更常见
- **想要更简单的实现、并在某些长上下文外推场景更稳**：ALiBi 很有吸引力

最终选择通常受以下因素影响：

- 你使用的基座模型/生态是否已经固定（Qwen/LLaMA 等多数是 RoPE）
- 推理框架对 RoPE 融合与长上下文支持是否成熟
- 目标场景是否强依赖超长上下文外推


