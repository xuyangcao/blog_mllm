# Attention 机制

Attention 的标准（scaled dot-product）形式可以写成：

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
\]

其中：

- \(Q \in \mathbb{R}^{T_q \times d_k}\)：queries
- \(K \in \mathbb{R}^{T_k \times d_k}\)：keys
- \(V \in \mathbb{R}^{T_k \times d_v}\)：values
- \(M\)：mask（例如 causal mask / padding mask），把不允许关注的位置加上 \(-\infty\)
- \(\sqrt{d_k}\)：缩放项，防止点积随维度增大导致 softmax 饱和

## 直觉：在做什么？

对每个 query（例如“当前 token”），我们计算它与所有 key 的相似度，得到权重，再对 value 做加权求和。你可以把它看成：

- **检索**：用 query 在 key 空间里找相关内容
- **聚合**：把相关 value 搬运到当前位置

## 为什么要除以 \(\sqrt{d_k}\)？

如果 \(Q\) 和 \(K\) 的每个维度近似零均值、方差为 1，则点积 \(q \cdot k\) 的方差随 \(d_k\) 增大而增大，softmax 会更容易进入“极端尖峰”区间，导致梯度不稳定。缩放后可以让 logits 的尺度更可控。

## Mask：Transformer “自回归”的关键

在语言模型的自回归生成里，token \(t\) 不能看见未来 token，因此使用 causal mask：

- 若 \(j > t\)，则 \(M_{t,j}=-\infty\)
- 否则 \(M_{t,j}=0\)

这样 softmax 后未来位置权重为 0。

## 复杂度：为什么推理会卡？

对于长度 \(T\) 的序列（\(T_q=T_k=T\)），注意力矩阵 \(QK^\top\) 是 \(T \times T\)，因此：

- **时间复杂度**：\(O(T^2 d)\)
- **显存/缓存压力**：注意力权重与中间张量对长上下文非常昂贵

这推动了后续的优化方向：FlashAttention、长上下文稀疏注意力、以及推理阶段的 KV Cache（把 \(K,V\) 缓存起来把 decode 复杂度从 \(O(T^2)\) 变成每步 \(O(T)\)）。

## 工程要点（你写代码时会踩的坑）

- **数值稳定**：softmax 前通常会减去 max（log-sum-exp trick）
- **mask 的 dtype/广播**：注意 -inf 的 dtype 与 fp16/bf16 的兼容性（实践中常用一个很小的负数近似）
- **padding mask**：batch 内不同长度需要屏蔽 padding，否则会污染聚合


