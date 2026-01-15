# 多头注意力（Multi-Head Attention）

单头注意力用一组 \(Q,K,V\) 做一次检索与聚合。多头注意力的动机是：**让模型在不同子空间并行地学习不同的对齐关系**（例如语法依赖、实体指代、局部模式等）。

标准形式：

\[
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W_O
\]

其中每个 head：

\[
\mathrm{head}_h=\mathrm{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})
\]

通常取 \(d_k=d_v=d/H\)，使得拼接后维度回到 \(d\)。

## 为什么多头有效：一个工程直觉

如果只有一头，注意力权重矩阵（\(T\times T\)）只有一张“关系图”。多头相当于有 \(H\) 张关系图并行学习，然后再线性组合回主干表示。这让模型既能捕获：

- **局部模式**（例如邻近 token）
- **长程依赖**（例如跨句依赖）
- **不同类型的相关性**（语法/语义/格式/引用等）

## 实现要点：QKV 打包与 reshape

工程实现常见做法：

- 一次线性层得到 \(QKV \in \mathbb{R}^{T \times 3d}\)
- reshape 为 \((T, H, d/H)\)，并转置到适合 GEMM 的布局

这样做的理由是：减少 kernel launch、便于算子融合、提升吞吐。

## 注意：多头不是“越多越好”

在固定 \(d\) 下增大头数会让每头维度 \(d/H\) 变小，可能造成表达瓶颈；另一方面头数也会影响 KV cache 带宽与推理性能。实践里常见的选择由模型规模与硬件决定（例如 7B/13B/70B 的头数通常不同）。

## Grouped-Query Attention（GQA）与 Multi-Query Attention（MQA）

这是推理优化中的高频改动，核心是：**减少 K/V 的头数**以降低 KV cache 的显存与带宽开销。

- **MHA**：每个 head 都有独立的 \(K,V\)
- **GQA**：多个 query head 共享一组 \(K,V\)（按 group 共享）
- **MQA**：所有 query head 共享同一组 \(K,V\)

代价是表达能力可能下降，但对长上下文推理的性价比很高（尤其在吞吐受 KV 带宽限制时）。


