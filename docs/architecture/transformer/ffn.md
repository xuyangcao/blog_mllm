# 前馈网络（FFN / MLP）

Transformer 中的 MLP（也常叫 FFN）负责“逐 token 的非线性变换”，与 Attention 的“跨 token 信息混合”互补。

经典的两层 FFN：

\[
\mathrm{FFN}(x)=W_2 \, \sigma(W_1 x + b_1) + b_2
\]

对序列 \(X \in \mathbb{R}^{T\times d}\) 是逐行独立地应用。

## 现代 LLM 常用的变体：SwiGLU / GeGLU

很多 LLM 使用门控 MLP（GLU family）提升效果与稳定性，例如 SwiGLU：

\[
\mathrm{SwiGLU}(x) = (W_a x)\odot \mathrm{SiLU}(W_b x)
\]
\[
\mathrm{MLP}(x)=W_o \,\mathrm{SwiGLU}(x)
\]

其中 \(\odot\) 是逐元素乘。

工程直觉：

- 门控结构给了模型一个“可学习的通道选择”，表达更灵活
- 在相近参数量下常带来更好的效果

## 宽度（hidden size）为什么通常比 d 大？

FFN 的中间维度常取 \(d_\text{ff}\approx 4d\)（不同模型略有差异）。原因是：

- Attention 更像“路由/搬运”，FFN 更像“计算/变换”
- 扩大 FFN 宽度提升逐 token 的非线性容量

## 工程优化点

- **算子融合**：FFN 主要是 GEMM + activation + GEMM，适合 fuse
- **混合精度**：GEMM 用 BF16/FP16，累加用 FP32（依实现而定）
- **张量并行**：FFN 的两次线性层非常适合做 TP（按列/按行切分）


