# KV Cache 原理

KV Cache 是自回归解码（decode）阶段最重要的加速手段之一：**缓存历史 token 的 Key/Value，避免每一步重复计算与重复读取**。

## 为什么会慢：从“每步重算”说起

在第 \(t\) 步解码时，模型输入长度是 \(t\)。如果你每一步都对整个长度重新计算 attention，那么总成本大致是：

\[
\sum_{t=1}^{T} O(t^2) = O(T^3)
\]

这在长输出时不可接受（实际实现会有细节差异，但“重复计算随步数累积爆炸”的趋势是一样的）。

## KV Cache 的做法

对每一层注意力，把历史 token 的 \(K,V\) 缓存起来：

- prefill（处理 prompt）阶段：一次性算出所有位置的 \(K,V\) 并缓存
- decode 阶段：每来一个新 token，只算这个新 token 的 \(K_t,V_t\)，并 append 到缓存

在第 \(t\) 步：

- \(Q_t\)：只对当前 token 计算
- \(K_{1:t},V_{1:t}\)：直接从 cache 读取

于是每步 attention 成本从“重算全部”变成：

\[
O(t) \quad (\text{与序列长度线性相关})
\]

总成本约为：

\[
\sum_{t=1}^T O(t) = O(T^2)
\]

这就是 KV Cache 在推理中必不可少的原因。

## KV Cache 的代价：显存与带宽

KV cache 的显存大致与以下因素成正比：

- batch size
- 序列长度（prompt + 已生成）
- 层数 \(L\)
- 头数与每头维度（以及是否使用 GQA/MQA）
- dtype（fp16/bf16/int8 等）

很多推理系统的瓶颈不是算力而是 **KV cache 带宽**（读写 cache 的开销）。

## 常见工程优化

- **GQA/MQA**：减少 K/V 头数，显著降低 cache 体积与带宽压力
- **Paged KV Cache**：把 cache 做成分页/块状管理，减少碎片并支持高并发变长请求（vLLM 等常用）
- **量化 KV**：把 KV cache 低比特存储以换显存（需要权衡精度与吞吐）
- **FlashAttention**：在训练/prefill 阶段减少中间张量与提升带宽利用率

## 与“并发/吞吐/延迟”的关系

在服务端推理中，KV cache 直接决定：

- 能同时服务多少请求（并发）
- 每个 token 的生成速度（吞吐）
- 是否能进行高效 batching（动态 batch / continuous batching）

因此你会看到：推理系统的很多工程设计（batching、调度、memory manager）都围绕 KV cache 展开。


