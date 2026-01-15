# vLLM：原理与使用

vLLM 是一个高性能 LLM 推理框架，目标是把“模型能跑”升级为“**能稳定高并发、成本可控地在线服务**”。它在工程上最核心的贡献通常来自：

- **Paged KV Cache（PagedAttention）**：解决 KV cache 的碎片与复用问题
- **Continuous Batching**：让 GPU 在请求持续到达时保持高利用率

---

## 1. 为什么 vLLM 能跑得快：问题从 KV Cache 说起

LLM 自回归解码需要 KV cache（缓存历史 token 的 Key/Value）。在在线服务中，KV cache 往往是瓶颈：

- 请求长度不一（变长 prompt / 变长输出）→ 显存分配与回收频繁
- 并发高 → cache 读写带宽吃紧
- 如果按“每个请求一段连续显存”管理，很容易产生**碎片**，导致“显存还够但分配失败/吞吐下降”

---

## 2. 核心原理一：Paged KV Cache（PagedAttention）

### 2.1 核心思想

把 KV cache 从“每个请求一块连续的大数组”改成“**按固定大小 block（页）分配**”，类似操作系统的分页内存：

- 每个请求的 KV cache 由多个 block 组成
- block 从一个全局的 block pool 里申请/归还
- 请求结束即可归还 block，供其他请求复用

### 2.2 带来的直接收益

- **显存碎片显著降低**：不用为变长序列反复申请不同大小的连续块
- **更好的并发与吞吐**：更容易把显存利用率打满
- **更稳的服务行为**：减少 OOM/分配失败导致的抖动

> 工程直觉：在线推理的很多“不稳定”，根源不是模型，而是变长请求下的 KV cache 内存管理。

---

## 3. 核心原理二：Continuous Batching（持续动态拼 batch）

传统 batch 的问题是：要等一批请求齐了才能一起跑，或者为了低延迟只能小 batch，导致 GPU 吃不满。

vLLM 的 continuous batching 让系统可以：

- 请求持续进入
- 调度器在每个 decoding step 动态决定“当前这一步拼哪些请求一起算”
- 让 GPU 更接近“持续满载”

结果通常是：

- 吞吐上升（tokens/s）
- 延迟更可控（尤其是尾延迟 P95/P99）

---

## 4. 使用方式（最常见三种）

> 下面命令仅作为“你在自己环境里怎么用”的示例，本书工程里不强依赖安装 vLLM。

### 4.1 安装

建议先参考 vLLM 官方文档的安装说明（不同 CUDA/平台组合差异很大）：

- `https://docs.vllm.ai/`

### 4.2 Python 离线推理（本地批量生成）

典型流程是：加载模型 → 传入 prompts → 得到生成结果。你会主要关注：

- batch size 与吞吐
- max tokens、stop tokens、temperature/top-p 等采样参数

### 4.3 启动 OpenAI 兼容服务（在线推理）

vLLM 常用形态是启动一个 OpenAI-compatible 的 HTTP 服务，然后你的业务侧按 OpenAI API 方式调用（便于替换与灰度）。

上线时你需要额外补齐：

- 鉴权、限流、配额
- 审计日志
- 监控（tokens/s、queue time、KV cache 使用率、OOM/重试）

---

## 5. 关键配置与调参（工程角度）

### 5.1 上下文长度与 KV cache

上下文越长，KV cache 体积越大，往往首先卡在显存与带宽。常见策略：

- 控制 max context / max new tokens
- 用 GQA/MQA 的模型（K/V 头数更少）
- 必要时考虑 KV cache 量化（取决于框架支持与精度容忍度）

### 5.2 吞吐 vs 延迟

这是永恒权衡：

- 更激进的 batching 与更高吞吐，可能增加排队时间
- 更低延迟策略可能降低 GPU 利用率

建议上线前明确 SLA，并分桶看 P50/P95/P99。

---

## 6. 常见坑位（你很可能会遇到）

- **显存看似够但仍 OOM**：通常是 KV cache + 并发 + 长上下文组合导致
- **吞吐不如预期**：多半是 batch 没拼起来、或被 I/O/CPU 前处理限制
- **长输出拖慢整个服务**：需要限额（max tokens）与调度策略
- **多模态模型更慢**：视觉 encoder 与跨注意力成本高，必须做缓存与 token 控制

---

## 本章小结

- vLLM 的工程核心是：用 Paged KV Cache + Continuous Batching 提升在线推理的吞吐与稳定性。
- 上线时要把“可观测 + 可控失败 + 权限边界”作为第一等公民，而不是只看离线生成效果。


