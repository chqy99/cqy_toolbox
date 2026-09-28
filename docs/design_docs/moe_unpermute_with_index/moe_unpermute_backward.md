# moe_unpermute_backward 实现设计方案

* #### 文档基本信息
| 算子名称 | moe_unpermute |
| ------ | ------------ |
| 编制人/日期 | 陈其阳/2025-2-7 |

* #### 修改记录
| 版本号 | 修订人  | 修订日期   | 修订描述 |
| ----- | ------ | -------   | ------- |
| V 1.0 | 陈其阳  | 2025-2-7 | 首次提交 |
| V 2.0 | 陈其阳  | 2025-5-21 | v2 接口，性能优化 |

* #### 内容描述

本文档为`moe_unpermute_backward`算子 pad 分支实现的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。

## 1 需求分析

### 1.1 算子需求分析
| 算子功能简介| moe 模块的 token_unpermutation 步骤中的 local unpermute 功能模块的融合反向算子 |
|-------------|--------------------------------------------------------------|
| 需求来源    | PJ_LAB |
| 应用网络    | 浦语3 |
| 输入数据类型 | indices: int64; others: half, float, bfloat16 |
| 输入shape | 见[算子shape展开说明](#111-算子shape展开说明) |
| 输入Layout | ARRAY |
| 输出数据类型 | half, float, bfloat16 |
| 输出Shape   | 见[算子shape展开说明](#111-算子shape展开说明) |
| 输出Layout  | ARRAY |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否 |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否|
| 是否需要支持原位        | 否   |
| 是否需要支持stride机制  | 否   |
| 是否需要支持广播        | 否   |
| 0元素检查是否直接返回    | 否   |

#### 1.1.1 算子shape展开说明

上层传参说明：
- total_seq = mbs * seqlen * topk * t, mbs 是 micro_batch_size；t = [0.9, 1) 大概这个范围
- cap = ceil((seqlen * topk/expert_num) * cap_factor)

| 模式 | permuted_tokens | indices | probs | indices 是否唯一 | unpermuted_tokens |
|-----|-----------------|---------|-------|-----------------|------------------|
| Allgather(不感知padded_mode) | [total_seq, H] | [total_seq] | - | 唯一 | [total_seq, H] |
| AlltoAll(默认,unpad) | [seqlen * topk, H] | [seqlen * topk] | [seqlen, topk] | 唯一 | [seqlen, H] |
| AlltoAll方法(cap_factor=1.0,unpad) | [total_seq, H] | [total_seq] | [seqlen, topk] | 唯一 | [seqlen, H] |
| AlltoAll方法(cap_factor=1.0,pad) | [expert_num * cap, H] | [expert_num, cap] | [expert_num, cap] | 不唯一 | [seqlen, H] |

permuted_tokens_grad 与 permuted_tokens 的 shape 完全相同<br>
probs_grad 与 probs 的 shape 完全相同。

注：
- 算子层面不感知 mbs, t, cap_factor
- 第三个模式, total_seq <= seqlen * topk，如果取=号，第三个模式就是第二个模式。

记 pad 模式为 padded_mode。<br>
观察这四种模式的shape特点：
- permuted_tokens 高维度数量与 indices 的总数量一定相等, 可以用一个参数 permuted_total_tokens 表示。
- permuted_tokens 低纬度数量与 unpermuted_tokens 的低维度数量一定相等，可以用一个参数 hidden 表示。
- unpad 模式下，topk 必须指定，需要用一个参数 topk 表示。<br>
  topk 在 unpad 模式等于 probs 的低维度数量，其余情况下不使能（默认为1）。
- unpermuted_tokens 高维度数量乘以 topk 与 permuted_tokens 高维度数量不一定相等，<br>
  unpermuted_tokens 高维度数量需要用一个参数 unpermuted_total_tokens 表示。

知道 permuted_total_tokens, hidden, unpermuted_total_tokens 数值之后，<br>
可以直接表示 permuted_tokens，unpermuted_tokens 的 shape，和 indices 的总数量。<br>
在 pad 模式下，permuted_total_tokens 也可直接表示 probs 的总数量。<br>
在 unpad 模式下，知道 topk 数值之后，unpermuted_total_tokens, topk 可以表示 probs 的 shape。<br>

因此，padded_mode, permuted_total_tokens, hidden, unpermuted_total_tokens, topk 五个参数可完成对 moe_unpermute 算子的必要描述。


### 1.2 算子功能和应用场景描述

MOE 模块主要分为四个步骤：
- router：路由器对每个token计算专家分配分数，并通过 Softmax 选出 top‑k 个专家及其概率
- token_permutation：根据路由分配，对token特征进行重新排列，将分配给同一专家的token聚集到一起，通常需要计算掩码并进行前缀和操作以生成排列索引
- experts computation：对分组后的tokens在各自专家上并行执行前馈网络或其他算子，常见做法是使用 batched GEMM 或 block‑sparse 矩阵乘法
- token_unpermutation：将专家输出逆向重排回原始token顺序，并按路由概率对多个专家输出加权求和，得到最终的 MoE 层输出

token_unpermutation 中的主要步骤：
- shape 变换
- local unpermute, 概率缩放（moe_unpermute 算子功能）
- EP，TP 的规约

本算子为 moe_unpermute 的反向实现。token_unpermutation 有多种模式，多种模式下的 moe_unpermute 的实现细节有所区别，<br>
可分为：allgather 分支，all2all(unpad) 分支，all2all(pad)分支。<br>

#### 1.2.1 allgather 分支
前向公式
$$
刷0: \forall {i,j},\quad \text{y}_{i,j} = 0 \\[1.5ex]
令 k = {\text{index}_i} \\[1.5ex]
\forall {i,j},\quad \text{y}_{k,j} \mathrel{+}= \text{x}_{i,j}
$$

反向公式推导
$$
令 k = {\text{index}_i} \\[1.5ex]
前向公式求偏导得:\frac{\partial y_{k,j}}{\partial x_{i,j}} = 1 \\[1.5ex]
变换过程:\frac{\partial L}{\partial x_{i,j}} = \frac{\partial L}{\partial y_{k,j}} * \frac{\partial y_{k,j}}{\partial x_{i,j}} = \frac{\partial L}{\partial y_{k,j}} * 1\\[1.5ex]
x的梯度:\nabla{x_{i,j}} = \nabla{y_{k,j}}
$$

#### 1.2.2 all2all(unpad) 分支
前向公式
$$
\forall {i,j},\quad \text{t}_{i,j} = 0 \\[1.5ex]
令 k = {\text{index}_i} \\[1.5ex]
\forall {i,j},\quad \text{t}_{k,j} \mathrel{+}= \text{x}_{i,j} \\[1.5ex]
\forall {i,j},\quad \text{y}_{i/topk,j} \mathrel{+}= \text{t}_{i,j} * p_i
$$
简化得
$$
\forall {i,j},\quad \text{y}_{i,j} = 0 \\[1.5ex]
令 k = {\text{index}_i} \\[1.5ex]
\forall {i,j},\quad \text{y}_{k/topk,j} \mathrel{+}= \text{x}_{i,j} * p_k
$$
反向公式推导
$$
令 k = {\text{index}_i} \\[1.5ex]
前向公式求偏导得:\frac{\partial y_{k/topk,j}}{\partial x_{i,j}} = p_k \quad
\frac{\partial y_{k/topk,j}}{\partial p_k} = \text{x}_{i,j} \\[1.5ex]
变换过程:\frac{\partial L}{\partial x_{i,j}} = \frac{\partial L}{\partial y_{k/topk,j}} * \frac{\partial y_{k/topk,j}}{\partial x_{i,j}}= \frac{\partial L}{\partial y_{k/topk,j}} * p_k \\[1.5ex]
\frac{\partial L}{\partial p_k} = \sum_j\frac{\partial L}{\partial y_{k/topk,j}} * \frac{\partial y_{k/topk,j}}{\partial p_k} = \sum_j\frac{\partial L}{\partial y_{k/topk,j}} * \text{x}_{i,j} \\[1.5ex]
x的梯度:\nabla{x_{i,j}} = \nabla{y_{k/topk,j}} * p_k \\[1.5ex]
由于 k 可能重复, probs的梯度: \\[1.5ex]
\forall i,\quad \nabla{p_i} = 0 \\[1.5ex]
\nabla{p_k} \mathrel{+}= \sum_j\nabla{y_{k/topk,j}} * \text{x}_{i,j} \\[1.5ex]
$$

#### 1.2.3 all2all(pad) 分支
前向公式
$$
刷0: \forall {i,j},\quad \text{y}_{i,j} = 0 \\[1.5ex]
令 k = {\text{index}_i} \\[1.5ex]
\forall {i,j},\quad \text{y}_{k,j} \mathrel{+}= \text{x}_{i,j} * p_i
$$
反向公式推导
$$
令 k = {\text{index}_i} \\[1.5ex]
前向公式求偏导得:\frac{\partial y_{k,j}}{\partial x_{i,j}} = p_i \quad
\frac{\partial y_{k,j}}{\partial p_i} = \text{x}_{i,j} \\[1.5ex]
变换过程:\frac{\partial L}{\partial x_{i,j}} = \frac{\partial L}{\partial y_{k,j}} * \frac{\partial y_{k,j}}{\partial x_{i,j}}= \frac{\partial L}{\partial y_{k,j}} * p_i \\[1.5ex]
\frac{\partial L}{\partial p_i} = \sum_j\frac{\partial L}{\partial y_{k,j}} * \frac{\partial y_{k,j}}{\partial p_i} = \sum_j\frac{\partial L}{\partial y_{k,j}} * \text{x}_{i,j} \\[1.5ex]
x的梯度:\nabla{x_{i,j}} = \nabla{y_{k,j}} * p_i \\[1.5ex]
probs的梯度:\nabla{p_i} = \sum_j\nabla{y_{k,j}} * \text{x}_{i,j}
$$

### 1.3 算子输入输出参数要求
| 参数          | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| ------------- | ---- | ----------------- | ----------- | -------- | -------- |
| queue        | CNRT 队列，保存运行的上下文信息     | 输入              | /          | /        | 无       |
| unpermuted_tokens_grad | 输入数据，指向输入词向量的mlu首地址 | 输入 | half, float, bfloat16 | ARRAY | dims=2 |
| indices | 输入数据，指向索引数据的mlu首地址 | 输入 | int64 | ARRAY |   |
| permuted_tokens | 输入数据，指向permuted_tokens数据的mlu首地址 | 输入 | half, float, bfloat16 | ARRAY | dims=2 |
| probs | 输入数据，指向probs数据的mlu首地址 | 输入 | half, float, bfloat16 | ARRAY |  |
| permuted_tokens_grad | 输出数据，指向输出词向量的mlu首地址 | 输出 | half, float, bfloat16 | ARRAY | 同 permuted_tokens |
| probs_grad | 输出数据，指向输出probs_grad的首地址 | 输出 | half, float, bfloat16 | ARRAY | 同 probs |
| padded_mode | 参数，pad 的模式 | 输入 | bool | / | 无 |
| permuted_total_tokens | 参数，permuted_tokens_grad 第一维数量 | 输入 | size_t | / | 无 |
| hidden | 参数，permuted_tokens_grad 第二维数量 | 输入 | size_t | / | 无 |
| topk | 参数，padded_mode 为 false 且 probs 不为空时，topk为 probs 第二维数量，否则为 1 | 输入 | int | / | 无 |
| unpermuted_total_tokens | 参数，unpermuted_tokens_grad 第一维数量 | 输出 | size_t | / | 无 |

### 1.4 算子限制
- 本身有 nan/inf 的得对齐，如果计算溢出造成的 nan/inf 这种可以不对齐。
- sorted_indices 中元素值范围为 [0, topk*unpermuted_total_tokens - 1]。
- all2all(unpad) probs_grad 会有累加计算，累加次数过多时，精度可能无法对齐。<br>
  注：累加次数过多时，静态阈值可能通不过，尤其是bfloat16类型。

### 1.5 验收标准

#### 1.5.1 精度验收标准

- 采用动态阈值：
  diff=[diff1, diff2, diff4], threshold_rate=[10, 10, 1]。
- largeTensor, 或含有 nan/inf 时，使用静态阈值：<br>
  diff=[diff1, diff2], threshold=[3e-3, 3e-3]。

## 2 算子接口设计

### 2.1 参考接口
permute和unpermute的算子融合：https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/permute.cu <br>
前向算子功能参考：http://gitlab.software.cambricon.com/neuware/oss/nvidia/Megatron-LM/-/blob/core_r0.9.0_mlu/megatron/core/transformer/moe/moe_utils.py <br>

没有拼接算子源码实现，推导算法示意：
```python
def unpermuted_grad_pad(
    unpermuted_tokens_grad: torch.Tensor,
    indices: torch.Tensor,
    permuted_tokens: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    unpermuted_shape = unpermuted_tokens_grad.shape
    indices = indices.view(-1, 1).expand(-1, unpermuted_shape[1])
    temp = torch.gather(unpermuted_tokens_grad, 0, indices)
    probs_grad = None

    if probs is not None:
        probs_shape = probs.shape
        probs = probs.view(-1).unsqueeze(-1)
        permuted_tokens_grad = temp * probs
        probs_grad = temp * permuted_tokens
        probs_grad = probs_grad.sum(dim=1).reshape(probs_shape)
    else:
        permuted_tokens_grad = temp

    return permuted_tokens_grad, probs_grad
```
```python
def unpermuted_grad_unpad(
    unpermuted_tokens_grad: torch.Tensor,
    sorted_indices: torch.Tensor,
    permuted_tokens: torch.Tensor,
    probs: torch.Tensor = None,
):
    if probs is not None:
        # permuted_tokens_grad
        probs_shape = probs.shape
        topk = probs.size(1)
        probs = probs.view(-1)
        sorted_indices = sorted_indices.view(-1)
        probs_gather = torch.gather(probs, 0, sorted_indices)

        sorted_indices = sorted_indices.unsqueeze(1).expand(
            -1, unpermuted_tokens_grad.size(-1)
        )
        unpermuted_tokens_grad = unpermuted_tokens_grad.repeat_interleave(topk, dim=0)
        output_size = unpermuted_tokens_grad.shape
        permuted_tokens_grad = torch.gather(unpermuted_tokens_grad, 0, sorted_indices)
        permuted_tokens_grad = permuted_tokens_grad * probs_gather.unsqueeze(-1)
        # probs_grad
        output = torch.zeros(
            output_size, dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )
        unpermuted_tokens = output.scatter_add_(0, sorted_indices, permuted_tokens)
        probs_grad = unpermuted_tokens * unpermuted_tokens_grad
        probs_grad = probs_grad.sum(dim=1).reshape(probs_shape)
    else:
        sorted_indices = sorted_indices.unsqueeze(1).expand(
            -1, unpermuted_tokens_grad.size(-1)
        )
        permuted_tokens_grad = torch.gather(unpermuted_tokens_grad, 0, sorted_indices)
        probs_grad = None

    return permuted_tokens_grad, probs_grad
```


### 2.2 接口设计
```c++
bangcKernelsStatus_t BANGC_KERNELS_WIN_API
mluGetMoeUnpermuteBackwardWorkspaceSize(const cnrtQueue_t queue,
                                        const bool has_probs,
                                        const bool padded_mode,
                                        const uint32_t data_size,
                                        const size_t permuted_total_tokens,
                                        size_t *workspace_size);

template <typename T>
bangcKernelsStatus_t BANGC_KERNELS_WIN_API mluMoeUnpermuteBackward_v2(
    const cnrtQueue_t queue, const T *unpermuted_tokens_grad,
    const int64_t *sorted_indices, const T *permuted_tokens, const T *probs,
    T *permuted_tokens_grad, T *probs_grad, void *workspace,
    const bool padded_mode, const size_t permuted_total_tokens,
    const size_t hidden, const int topk, const size_t unpermuted_total_tokens);
```

# 实现方案设计

### 3.1 实现方案

记单次处理的数据段为（[seg_tokens, seg_hidden]）
(1) no probs 分支<br>
该分支只需要求 permuted_tokens_grad, 实现简单
- memcpy1d load indices
- indices 转换成 indices_offset
- 根据 indices_offset, gather load unpermuted_tokens_grad
- memcpy2d store unpermuted_tokens_grad 到 permuted_tokens_grad

(2) pad 分支<br>
该分支使用流水实现。
- 单独起kernel，将 indices 计算成 indices_offset 并输出到 workspace 上。
- LOAD 逻辑：
  - memcpy1d load indices_offset
  - memcpy1d load probs
  - memcpy2d load permuted_tokens
  - 根据 indices_offset， gather load unpermuted_tokens_grad
- COMPUTE 逻辑：
  - probs_grad 计算：hidden_offset 为 0 时，probs_grad 刷 0。
  - probs_grad 计算：unpermuted_tokens_grad * permuted_tokens --> permuted_tokens
  - probs_grad 计算：permuted_tokens[seg_tokens, seg_hidden] 低维度求和并加到 probs_grad
  - permuted_tokens_grad 计算：for 循环处理 unpermuted_tokens_grad[seg_tokens, seg_hidden] * probs[seg_tokens]
- STORE 逻辑：
  - 最后一段 hidden 处理时，atomic_add probs_grad
  - memcpy2d store permuted_tokens_grad

(3) unpad 分支<br>
该分支使用流水实现。
- 单独起kernel，将 indices 计算成 indices_offset 并输出到 workspace 上。<br>
  注：需要使用 int64 型的 bang_div，依赖 toolkit 4.2
- 单独起kernel， 按 indices gather probs 成 probs_g, 并输出到 workspace 上。
- LOAD 逻辑：
  - memcpy1d load indices_offset
  - memcpy1d load probs_g
  - memcpy2d load permuted_tokens
  - 根据 indices_offset， gather load unpermuted_tokens_grad
- COMPUTE 逻辑：
  - probs_grad 计算：hidden_offset 为 0 时，probs_grad 刷 0。
  - probs_grad 计算：unpermuted_tokens_grad * permuted_tokens --> permuted_tokens
  - probs_grad 计算：permuted_tokens[seg_tokens, seg_hidden] 低维度求和并加到 probs_grad
  - permuted_tokens_grad 计算：for 循环处理 unpermuted_tokens_grad[seg_tokens, seg_hidden] * probs[seg_tokens]
- STORE 逻辑：
  - 最后一段 hidden 处理时，for 循环单个处理 atomic_add probs_grad
  - memcpy2d store permuted_tokens_grad

### 3.2 伪代码实现

### 3.3 拆分

#### 3.3.1 核间拆分
二维拆分，大的长方形被拆分成 core_num 个小长方形。<br>
沿高维度（tokens 维度）方向拆分的小长方形数量记为 tokens_part，<br>
沿低维度（hidden 维度）方向拆分的小长方形数量记为 hidden_part。<br>

每个IPU处理的数据块对应一个小长方形，要使得负载尽可能均衡，需要满足 tokens_part * hidden_part = core_num。<br>
也就是说 tokens_part 与 hidden_part 都为 core_num 的因数。<br>

core_num 的因数集合为 Q, 例如： core_num = 32 时， Q = {1,2,4,8,16,32}<br>

默认 tokens_part = core_num, hidden_part = 1。（即默认只做 tokens 维度的拆分）<br>
当 core_num > permuted_total_tokens （tokens 维度本身数量太少）时，也去拆分 hidden 维度。<br>
从 Q 中选取小于 permuted_total_tokens 的最大因数， 这个因数作为 tokens_part。<br>
确定 tokens_part 后，hidden_part = core_num / tokens_part。

#### 3.3.2 核内拆分
tokens 维度平均拆分，hidden 维度按512B对齐拆分。

### 3.4 算子防呆说明
- 指针限制:
  - probs、probs_grad 可为空，其余输入和输出指针均不能为空。
- 数据类型限制:
  - permuted_tokens、probs（不为空时）、unpermuted_tokens 的 dtype 相同，可为 half/float/bfloat16。bfloat16 在mtp_592以上板卡支持。
  - sorted_indices 的 dtype 为 int64。
- shape限制：见[算子shape展开说明](#111-算子shape展开说明)
- 真值限制：
  - sorted_indices 中元素值范围为 [0, topk*unpermuted_total_tokens-1]。

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明
全规模优化，permuted_tokens_grad 和 probs_grad 合为一个kernel，省去一次 unpermuted_tokens_grad 的 IO。
采用流水实现，掩盖计算时间。