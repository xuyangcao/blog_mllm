# LayerNorm 与残差连接

如果只记住一句话：**残差连接保证可训练性，归一化保证数值稳定与梯度尺度可控**。

## 残差连接（Residual Connection）

Transformer block 中的残差结构：

\[
h \leftarrow h + f(h)
\]

它让网络更容易优化（梯度可以沿着恒等路径传播），并显著缓解深层网络的退化问题。

工程上残差的直接收益：

- 深层模型更稳定
- 训练更不容易“突然崩掉”
- 对学习率/初始化更鲁棒

## LayerNorm（LN）

LayerNorm 对每个 token 的特征维做归一化（而不是对 batch 维）：

\[
\mathrm{LN}(x)=\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

其中 \(\mu,\sigma^2\) 在特征维上计算。

### Pre-LN vs Post-LN

两种常见结构：

- **Post-LN（原始 Transformer）**：\(h \leftarrow \mathrm{LN}(h + f(h))\)
- **Pre-LN（现代 LLM 常用）**：\(h \leftarrow h + f(\mathrm{LN}(h))\)

实践经验：Pre-LN 通常更容易训练非常深的网络（梯度更稳定），因此在 LLM 中很常见。

## RMSNorm

一些模型使用 RMSNorm 代替 LayerNorm（省掉均值项，计算更省）：

\[
\mathrm{RMSNorm}(x)=\gamma \odot \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}
\]

## 工程细节：数值与性能

- LN/RMSNorm 在推理中是高频算子，很多推理框架会做 fuse
- 混合精度下 \(\epsilon\) 的选择会影响稳定性（尤其是很深的网络）
- 残差路径的 dtype（fp16/bf16/fp32）与累加策略也会影响稳定性与精度


