# 自注意力（Self-Attention）

自注意力是 Attention 的一个特例：\(Q,K,V\) 都来自同一个序列的表示 \(X \in \mathbb{R}^{T \times d}\)。

典型实现是三组线性投影：

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

其中 \(W_Q,W_K \in \mathbb{R}^{d \times d_k}\)，\(W_V \in \mathbb{R}^{d \times d_v}\)。

然后：

\[
\mathrm{SA}(X)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
\]

## Self-Attention 与“信息混合”

你可以把 Self-Attention 看成一种“全局可学习的混合算子”：每个位置都能从任意位置拉取信息。与之对比：

- 卷积：局部感受野（需要堆叠多层扩大范围）
- RNN：顺序传递（难并行，长依赖更难）

## Causal Self-Attention（Decoder-only 的核心）

对 GPT 类模型，必须使用 causal mask，保证生成第 \(t\) 个 token 时只使用 \(x_{<t}\)：

\[
p(x_{1:T})=\prod_{t=1}^T p(x_t \mid x_{<t})
\]

因此 Self-Attention 的可见性约束不是“可选项”，而是概率建模假设的一部分。

## 训练 vs 推理：为何 KV Cache 必不可少

### 训练（teacher forcing）

训练时我们一次性输入整个序列，计算所有位置的 logits，并行高效。但注意力仍然是 \(O(T^2)\)。

### 推理（自回归解码）

推理时每生成一个新 token，如果每次都重新计算所有 \(K,V\)，成本会随步数累积变得不可接受。KV Cache 的做法是：

- 历史 token 的 \(K,V\) 只算一次并缓存
- 每一步只计算新 token 的 \(Q,K,V\)，并与缓存的 \(K,V\) 做 attention

这样把 decode 的重复计算压到最低（详见 `kv_cache.md`）。


