# 大模型架构设计

本章讨论“LLM 作为一个工程系统”的架构设计：为什么主流是 Decoder-only？参数量与训练/推理成本如何估算？稀疏化与量化各自解决什么问题？tokenization 为什么会影响成本与上限？

## Decoder-only、Encoder-only、Encoder-Decoder 架构

把 Transformer 用在 NLP 上，大体有三类范式：

### Encoder-only（BERT 系）：理解为主

训练目标常见是 MLM（masked language modeling）。特点：

- 强理解、强表征
- 生成能力弱（不天然支持自回归生成）
- 适合分类/检索/表示学习

### Encoder-Decoder（T5 系）：条件生成

Encoder 编码输入，Decoder 生成输出：

- 适合“输入到输出”的条件生成（翻译、摘要、问答）
- 结构更通用，但推理时通常更重（两套堆叠）

### Decoder-only（GPT 系）：统一的生成接口

以自回归建模：

\[
p(x_{1:T})=\prod_{t=1}^{T} p(x_t \mid x_{<t})
\]

它的工程优势是：

- **统一接口**：几乎所有任务都能写成“条件生成”
- **训练与推理路径一致**（训练 teacher forcing，推理自回归）
- **系统更简单**：一套堆叠、易扩展、生态成熟

这也是为什么大部分通用 LLM（以及很多 MLLM 的语言主干）采用 Decoder-only。
## 模型规模与参数量对比

你在做工程决策时，至少要能粗估三件事：**参数量、训练 FLOPs、推理 KV cache 成本**。

### 参数量的粗估（以 Decoder-only Transformer 为例）

不同实现细节会影响常数项，但量级上：

- Attention 的投影（QKV + Out）参数规模约为 \(O(d^2)\)
- FFN（尤其是 4d 宽度/门控 MLP）参数规模也约为 \(O(d^2)\)，且通常占比更大
- 总参数量约为 \(O(L d^2)\)

工程直觉：**在现代 LLM 里，MLP 参数与算力占比通常不低于 Attention**。

### 训练计算量（FLOPs）

训练总 FLOPs 近似与（参数量 × token 数）同阶。更具体的估算会考虑：

- attention 的 \(T^2\) 项（长序列更昂贵）
- MLP 的 GEMM
- 反向传播是前向的 ~2-3 倍

你最终会关心：在目标上下文长度与 batch 配置下，训练吞吐是多少、能否稳定扩展到多机多卡。
## 激活/参数稀疏化与高效推理

稀疏化的目标通常是：**在不显著损伤质量的前提下，降低推理计算与带宽成本**。

### 参数稀疏化：MoE（Mixture of Experts）

MoE 的基本形式是：把 FFN 替换为多个 expert，每个 token 只路由到少量 expert 计算：

- 优点：在相同训练 token 下，参数量可显著变大，同时每 token 计算量可控
- 缺点：系统复杂（路由、负载均衡、通信），推理的 latency 也更难做稳定

### 激活稀疏化与结构化稀疏

包括：

- 剪枝（结构化剪枝更利于硬件加速）
- 低秩分解
- 激活稀疏（依赖特定算子与硬件支持）

工程上最常见、最稳定的“高效推理”路线反而是：

- 量化（int8/int4）
- KV cache 优化（GQA/MQA、paged cache）
- 算子融合（FlashAttention、fused MLP）
## Tokenization 与词表设计

tokenization 影响三个关键指标：

1. **序列长度**：同一句话被切成多少 token，直接影响 attention 成本与 KV cache 体积
2. **表示效率**：子词粒度影响模型对罕见词、数字、代码等的建模
3. **多语言与领域覆盖**：词表与训练语料分布不匹配会造成额外 token 膨胀与质量下降

### BPE / SentencePiece 的核心直觉

- 从字符出发逐步合并高频片段
- 高频片段变成单 token，降低序列长度

### 工程建议

- 若目标场景包含大量代码/公式/特殊符号，tokenization 的设计会显著影响成本与效果
- 多模态模型里常会引入“特殊 token”（例如 `<image>`）或视觉 token，这要求词表与系统协议一致

---

## 本章小结

- Decoder-only 之所以成为主流，是因为“统一条件生成接口 + 工程系统简化 + 生态成熟”。
- 规模估算的最低要求：能大致判断参数量、训练成本、推理 KV cache 成本。
- 高效推理的主线通常是：量化 + KV cache 优化 + 算子融合；稀疏化（MoE）更像“高投入高回报”的系统工程。
