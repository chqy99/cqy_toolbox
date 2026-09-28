# chenqiyang (chqy99) BangC 代码风格画像

## 基调：思想为主，不是风格教条

**这篇画像记的是编程思想与结构取向，不是格式教条。** 风格细节（缩进、对齐、命名大小写）不是重点，重点是 device 侧怎么把一段逻辑**结构化**地落下来。更要紧的是：**编程思想是为了服务编程逻辑，不是反过来限制逻辑本身。** 思想是 servant 不是 master——当某条思想在某场景下会捆住逻辑、让代码变差时，偏离它是对的，只要写注释说明为什么偏离。所以下面所有"理念/元原则"都是**默认追求**，不是不可破的铁律；不能陷入教条主义。

另外，我以前的写法或多或少可能有部分与下面描述不符，但整体编程思想是一脉的——**以本画像当前描述的成熟思想为准绳**，不逐行考古。下面的思想是我当前成熟形态的总结，不是对旧代码的验收报告。

## 编程思想：结构式编程为底色

这一节是画像的根，比任何格式细节都重要。我写 BangC device 代码的底层思想是**结构式编程（structured programming）**，具体落在四条:

1. **结构先于细节**：先把骨架段落摆正——分层、阶段、循环边界——再填每段内容。骨架错了，细节再巧也是歪的。
2. **单向自上而下推进**：控制流是一条自上而下的单向链，读我的 `coreProcess` 像读一篇分段论述，每段吃完上段结论往下走，不回跳、不交织。
3. **分层分工，各层只做一件事**：入口只分割 task，core 只编排流水，叶子只做逐段计算。层次之间靠参数衔接，不越权。
4. **段落式凝聚**：每一步是一段凝聚的代码块，按序排开不交叉（详见下文元原则）。

逻辑简单的算子我不做过多区分；但**逻辑复杂的算子**，device 侧永远按下面这条固定链路写，顺序几乎不变——这是结构式编程在 device 侧的具体化身:

```
1. 校验 → 不满足立即返回（早退）
2. 算 tile 切片尺寸（seg_*）
3. 由 tile 尺寸推导空间划分的具体大小（chunk_size / pingpong_size / 各 buffer 偏移）
4. 内存空间确定后，统一把 for 循环之前所需的所有控制流变量 + 重复性计算一次性算完
5. 进入 pipeline，每条流封装成函数/宏，循环里直接调用
```

> 早退机制是存在的，只是**分布在合适的层、在计算开始之前**，不重复。复杂算子的 `coreProcess` 通常已被入口/上游 guard 过，core 内不重复早退是分工，不是缺校验——其他函数（entry、叶子）都挂了早退。hadamard 的早退就在 entry（`_union1.mlu:499` `if (cluster_batch_num <= 0) return;`）和叶子（`:338`/`:407`），coreProcess 直接进 tile。

### 元原则：段落式凝聚，不串流程

上面五步在我代码里是**段落式**的——每一步是一段凝聚的代码块，按序排开，**不交叉、不回插**。绝不会出现"写一点控制→插一段分配→又回头写控制"那种想到哪写到哪的乱串：控制逻辑归控制、空间划分归划分、循环预计算归预计算，各自成段，段与段之间靠前一段的结论衔接。读我的 `coreProcess` 是自上而下单向推进的，每段读完就知道下一段要吃什么。这比五步本身更是我的底色——**结构先于细节**，先把骨架段落摆正，再填每段内容。

### 元原则：严格区分"纯计算"与"因空间不足而 for 循环切片"

我把 device 侧的循环分得很清楚，两类绝不混为一谈：

- **纯计算循环**：数据天然就在片上/可一次处理，for 只是算法本身的迭代（如蝶形 `handled_dim *= 2`、逐 block 做 H_512）。这类循环与片上空间上限无关，是算法结构。
- **空间不足切片循环**：因为 NRAM/SRAM/WRAM 放不下整块数据，才被迫按 `seg_*` 切片、套 ping-pong 流水。这类循环的存在理由是空间，tile 尺寸由空间反推（见理念二）。

两者在我代码里是分开的 for、分开的函数，宁可多一层调用也不揉在一起。pipeline 那层大循环永远属于"空间不足切片"；纯计算的迭代收敛在叶子流函数内部（hadamard 的 `butterfly` `_union1.mlu:95-106`、`matmul` `:81-84` 是纯计算叶子，`coreProcess` `:259` 的 `repeat+4` 循环是空间切片）。

**例外与纪律**：当针对特定场景做优化、确认某路径不再需要考虑空间上限时（比如数据量已知很小、或某分支已保证一次放得下），我会直接写成不切片的纯计算形态，**但此时一定会写注释说明为什么可以放弃空间上限**——不写注释的"无 for 切片"会被我视为隐患。hadamard entry 的小维快路径就挂了 `// 5级流水时，片上能一次性处理完`（`_union1.mlu:503-504`）说明为何不切片。

### 元原则：与编程思想冲突时允许，但必注释说明

上面这套思想是我**默认追求**的形态，不是不可破的铁律。现实里为了性能、为了特定场景，确实会有偏离——**偏离本身允许，但一定会写注释说明为什么偏离**。注释就是任何偏离的合法性凭证，不写注释的偏离会被我视为隐患。这条原则统摄了画像里所有"例外"：

- hadamard 的 `kernelHadamardTransform` 入口**以 task 级别分配为主**，但类似"多次起 kernel"那种**大编排**也允许放在 entry 里（大维两阶段的 block 级循环 `:542`/`:569` 就属此类），不强行拆出 core 函数。判断标准是编排的粒度：task 分配 + 跨 stage 大编排归 entry，逐段计算归 core/叶子。**内存超大的情况会在 entry 里做注释说明**（大维两阶段的设计理由"stage1 出口截回 T 是精度损失元凶"、缓冲数实测择优就写在入口诊断注释里）。
- hadamard mpu 侧 IO 流是**内联**的（`__memcpy_io1_async` 直调，`:262-286`），没封装成流函数——只有 ipu 侧封装了（`loadMove`/`postTranspose`/`matmul`/`butterfly`）。判断标准很简单：**本身就一句话的流，不需要再封装**；封装是为了藏复杂度，没复杂度可藏就别加一层。

（也有没注释的小毛刺，见下文"真实小毛刺"——那是未收敛的瑕疵，不是合法例外。）

### 理念一：先校验、早退，不拖泥带水

"早退"是个广义习惯，不止入口能不能干的 guard，还包括循环里**没活干就跳过**的各种短路：余数段为 0 时跳过尾段处理、mpu/ipu 角色不匹配时 `__is_mpu()` 直接 return、`batch == 0` / `cluster_batch_num <= 0` 这类空输入立即返回。校验逻辑和计算逻辑绝不交织，能早退就早退，不让下游去处理"本不该到这"的情况。host 侧同理——早返回 guard 优先。

### 理念二：tile 尺寸先行，空间划分由 tile 推导出来

不是先拍脑袋定 buffer 大小再迁就 tile，而是**先算 tile 能切多大**，再用 tile 尺寸反推每个 buffer 占多少、ping/pong 怎么分。空间划分是 tile 的结论，不是前提。hadamard 里这段推导甚至带长注释记录"为什么 chunk_num=3"和实测性能（300→133us）：

```cpp
// hadamard_transform_union1.mlu:206-216
constexpr uint32_t chunk_num = 3;
uint32_t nram_chunk_size = FLOOR_ALIGN(MAX_NRAM_SIZE / (chunk_num + 1), 64);
uint32_t wram_chunk_size = FLOOR_ALIGN(MAX_WRAM_SIZE / 16 / chunk_num, 256);
uint32_t sram_output_offset = MAX_SRAM_SIZE / 2;
uint32_t sram_chunk_size    = FLOOR_ALIGN(MAX_SRAM_SIZE / 2 / chunk_num, 64);

uint32_t nram_seg_batch = nram_chunk_size / align_dim / sizeof(float);
uint32_t sram_seg_batch = sram_chunk_size / taskDimX / align_dim / sizeof(T);
uint32_t seg_batch = std::min(nram_seg_batch, sram_seg_batch);   // tile 由空间反推
```

> 早期 ftrl/lamb 还没把 tile 和空间拆成独立两步——span 直接从 `MAX_NRAM_SIZE` 反推（`span_size = FLOOR_ALIGN(pingpong_size/8, ALIGNSIZE)`），tile 与空间耦合在一起。这是"最早阶段"的简化形态（见"成熟度弧线"），不矛盾，是演化前身。

### 理念三：内存空间一确定，循环前的控制流变量与重复计算一次性算完

这是我最强的个人印记。tile 和空间一旦定，**所有** for 循环要用的 repeat、尾段处理、offset、stride、elem 折算……全部在进循环前集中算掉，循环体里只做"调流函数 + sync"，绝不把推导塞进循环里反复算：

```cpp
// hadamard_transform_union1.mlu:229-247
uint32_t cluster_seg_batch     = taskDimX * seg_batch;
uint32_t rem_cluster_seg_batch = batch % cluster_seg_batch;
int64_t repeat = batch / cluster_seg_batch + ((rem_cluster_seg_batch > 0) ? 1 : 0);

// 最后一段拆分方法
uint32_t last_cluster_seg_batch = taskDimX * seg_batch;
uint32_t last_seg_batch         = seg_batch;
uint32_t last_batch_offset      = seg_batch * taskIdX;
if (rem_cluster_seg_batch > 0) {
  last_cluster_seg_batch = rem_cluster_seg_batch;
  uint32_t last_repeat   = rem_cluster_seg_batch / taskDimX;
  uint32_t last_rem      = rem_cluster_seg_batch % taskDimX;
  last_seg_batch         = last_repeat + (taskIdX < last_rem ? 1 : 0);
  last_batch_offset      = last_repeat * taskIdX + (taskIdX < last_rem ? taskIdX : last_rem);
}

uint32_t in_elem  = in_stride / sizeof(T);
uint32_t out_elem = out_stride / sizeof(T);

for (int64_t i = 0; i < repeat + 4; ++i) {   // ← 循环体里只剩调流 + sync
```

moe 里同理，`repeat_hidden`/`repeat_tokens`/`final_hidden`/`final_tokens` 全在循环前算好：

```cpp
// moe_unpermute_unpad_element_wise.mlu:139-153
size_t repeat_hidden = CEIL_DIV(hidden_num, max_seg_hidden);
size_t repeat_tokens = CEIL_DIV(tokens_num, max_seg_tokens);
size_t repeat = repeat_hidden * repeat_tokens;

uint32_t final_hidden = hidden_num % max_seg_hidden;
if (final_hidden == 0) { final_hidden = max_seg_hidden; }
uint32_t final_tokens = tokens_num % max_seg_tokens;
if (final_tokens == 0) { final_tokens = max_seg_tokens; }
size_t high_idx = 0, low_idx = 0;
size_t tokens_offset = 0, hidden_offset = 0;
uint32_t seg_tokens = 0, seg_hidden = 0;
// 紧接着就是 pipeline3
```

### 理念四：每条流封装成函数，pipeline 里直接调用

复杂算子里，LOAD/COMPUTE/STORE 各自是一条独立的"流"，我会把每条流封装成一个 `__mlu_func__` 或宏，pipeline 循环体只负责按序调用它们 + sync——流的内部实现和流水编排彻底解耦。这是"能抽象就抽象"在控制流层面的体现：

```cpp
// moe_unpermute_unpad_element_wise.mlu:44-83  每条流一个宏
#define _LOAD(idx)   getLoopInfo(...); __memcpy_async(...); __gather_async(...);
#define _COMPUTE(idx) getLoopInfo(...); computeSeg(...); __bang_sumpool(...);
#define _STORE(idx)  getLoopInfo(...); st1_store(...);

// 循环体只剩调度（:155-184）
if (repeat > 0) { _LOAD(0) PINGPONGSWAP __sync(); }
if (repeat > 1) { _LOAD(1) _COMPUTE(0) PINGPONGSWAP __sync(); }
for (size_t i = 2; i < repeat; ++i) {
  _STORE(i - 2) _LOAD(i) _COMPUTE(i - 1) PINGPONGSWAP __sync();
}
```

hadamard 里 ipu 侧 MOVE/COMPUTE 流分别封装成 `loadMove`/`postTranspose`/`matmul`/`butterfly`，循环体里只是按角色调用（mpu 侧 IO 流本身一句话，内联直调，见上文"冲突时允许"）：

```cpp
// hadamard_transform_union1.mlu:287-323 (ipu 侧)
// ipu MOVE1
postTranspose((T*)(sram_buffer + ...) + move1_batch_offset * align_dim,
              (float*)(nram_buffer + chunk_id * nram_chunk_size), ...);
// ipu MOVE0
loadMove((float*)(wram_buffer + chunk_id * wram_chunk_size),
         (T*)(sram_buffer + ...) + move0_batch_offset * align_dim, ...);
// ipu COMPUTE
matmul(...); butterfly(...);
```

**四条理念合起来**就是我的 device 侧写作法：校验早退 → tile 先行 → 空间由 tile 推 → 循环前一次性预计算 → 流函数化 + pipeline 只调度。逻辑简单的算子可能省略分层，但复杂算子这五步一定在。

---

## 结构骨架（三层分解）

所有 op 共享同一套函数分解，是最稳定的习惯：

```
__mlu_global__ kernelXxx      → task 分割 + tail clamp + 调 core
__mlu_func__  coreProcess     → NRAM 划分 + tiling + 流水循环
__mlu_func__  computeSeg/     → 叶子：逐段 load/compute/store
              storeSeg/loadData/storeData/ftrlCompute
```

`coreProcess`/`coreHandle` 永远是流水循环的拥有者，叶子函数被它调用。入口里**以 task 级别分配为主，默认不放逐段计算逻辑**——但类似"多次起 kernel"那种大编排允许放在 entry（如 hadamard 大维两阶段），内存超大的情况做注释说明（见"冲突时允许"）。

### 1. `__mlu_global__` 入口：只做 cluster/core 级分割

- **moe**：`taskId` 二维切成 tokens × hidden，尾段 clamp，再 `coreProcess<...>(...)`。MPU guard 用 `__is_mpu()`（旧文件）或 `coreId == 0x80`（新文件），两种都用过——甚至同一 forward op 内并存（element_wise 用 `coreId == 0x80` `:192`，pad 用 `__is_mpu()` `:140`）。
- **hadamard**：mpu/ipu 角色分流——`if (__is_mpu()) { mpu IO } else { ipu MOVE+COMPUTE }`，每轮 `__sync_cluster()`：

```cpp
// hadamard_transform_union1.mlu:259-328
for (int64_t i = 0; i < repeat + 4; ++i) {
  if (__is_mpu()) {
    // mpu IO1 ... mpu IO0 (move)
  } else {
    // ipu MOVE1 ... ipu MOVE0 ... ipu COMPUTE
  }
  __sync_cluster();
}
```

- **ftrl/lamb 原版**：对齐切分后用 `if (weights == nullptr)` 运行时分发 `coreHandle<..., false/true>`。ftrl_group_lasso 还多了第二分发轴 `weight_dim >= 128` → `is_enough_weight_dim`，同一套"运行时分支驱动模板裁剪"手法的扩展。

### 2. NRAM 划分仪式

- **file-scope `__nram__ int8_t nram_buffer[MAX_NRAM_SIZE];`**，永远紧跟 include 之后。hadamard 还额外声明 `__wram__`/`__mlu_shared__` 三平面：

```cpp
// hadamard_transform_union1.mlu:23-25
__nram__ int8_t nram_buffer[MAX_NRAM_SIZE];
__wram__ int8_t wram_buffer[MAX_WRAM_SIZE];
__mlu_shared__ int8_t sram_buffer[MAX_SRAM_SIZE];
```

- **从 int8_t 基址指针算术切出命名子区**：`(float*)(nram_buffer + offset)`，而非类型化 buffer。
- **ASCII 内存布局表**置于每个 `coreProcess`/`coreHandle` 顶部，是招牌：

```cpp
// moe_unpermute_unpad_element_wise.mlu:89-91
// | permuted_tokens | probs  | indices_s  | output                 |
// | T               | T      | int64_t    | T                      |
// | tokens * hidden | tokens | tokens     | tokens / topk * hidden |
```

```cpp
// hadamard_transform_union1.mlu:185-186
// | name          | dtype | dim0       | dim1      |
// | nram_chunk(i) | float | align_dim  | seg_batch |
```

- **NRAM sizing 固定舞步**（moe）：先 `max_seg_hidden` FLOOR_ALIGN 到 `ALIGNSIZE/sizeof(T)` 并 clamp，再算 `max_seg_tokens`，最后一段溢出回退——"对齐后溢出就减 1"，配中文 why 注释：

```cpp
// moe_unpermute_pad.mlu:71-83
// 如果对齐后内存使用量溢出，此时 nram_max_tokens 没有对齐
// 低于 ALIGNSIZE / sizeof(T) 时，隐含了 hidden_num 极大
// nram_max_tokens 减 1, 则不会溢出
```

- **hidden 维走 `512/sizeof(T)` 对齐**（cache-line），其它地方 `ALIGNSIZE=128`——刻意的差异。

### 3. 三级 ping-pong 流水（最强 fingerprint）

两种等价写法，跨 8+ 文件完全同形：

**早期/通用 if-band 形式**（ftrl/lamb 原版）：

```cpp
for (size_t i = 0; i < repeat + 2; ++i) {
  uint32_t pingpong_gap = (i % 2) * pingpong_size;
  // COMPUTE  if (i >= 1 && i < repeat + 1)
  // STORE    if (i >= 2)
  // LOAD     if (i < repeat)
  __sync();
}
```

**成熟期 macro 形式**（moe element_wise/unique/backward v2）：

```cpp
// moe_unpermute_unpad_element_wise.mlu:155-184
if (repeat > 0) { _LOAD(0) PINGPONGSWAP __sync(); }
if (repeat > 1) { _LOAD(1) _COMPUTE(0) PINGPONGSWAP __sync(); }
for (size_t i = 2; i < repeat; ++i) {
  _STORE(i - 2)
  _LOAD(i)
  // _COMPUTE 中有大量标量运算，放在IO后面有较好的性能
  _COMPUTE(i - 1)
  PINGPONGSWAP
  __sync();
}
if (repeat > 1) { _STORE(repeat - 2) }
if (repeat > 0) { _COMPUTE(repeat - 1) }
PINGPONGSWAP
__sync();
if (repeat > 0) { _STORE(repeat - 1) }
```

buffer 永远 `ping_*`/`pong_*`。hadamard 升级到 `chunk_num=3` 三缓冲 + 4 轮 prologue，并注释"5 缓冲也测过，3 胜出"。hadamard 走的是 if-band 形式（按 `chunk_id` 分支），没用 `_LOAD` 宏——两种形式都视作合法。

### 4. `if constexpr` 类型分发 + 内联 asm

- `if constexpr (std::is_same_v<T, float>)` 梯子是**通用 dtype 分支手段**（绕开 BangC device 侧不能用函数指针的限制）：asm 后缀、convert、大维 float-vs-low-prec 策略全走它：

```cpp
// hadamard_transform_union1.mlu:35-43
if constexpr (std::is_same_v<float, T>) {
  __asm__ volatile(HADAMARD_MOVE_TPL("b32") ";\n\t" ::__VA_ARGS__);
} else if constexpr (std::is_same_v<half, T>) {
  __asm__ volatile(HADAMARD_MOVE_TPL("b16") ", .cvt.rn.f32.f16();\n\t" ::__VA_ARGS__);
} else {
  __asm__ volatile(HADAMARD_MOVE_TPL("b16") ", .cvt.rn.f32.bf16();\n\t" ::__VA_ARGS__);
}
```

- builtin 不够时直接 `__asm__ volatile`：`st.stride.gdram.nram.async.io1`、`sync.pmv.cio`、`sync.psimd.cio`、`fuse.nram.u32`，配中文 why 注释。ftrl/lamb 原版的 `fuse.nram.u32` 就挂了"用 u32, 不会出现 float 的 NaN,Inf 乘法错误"（`ftrl_union1.mlu` ftrlZToW、`lamb_union.mlu` lambFunctor 里都有）。hadamard 的 asm 多用 shape 标注（`// raw shape: [seg_batch, align_dim]`）+ 对齐说明，中文 why 偏少——同一原则的两种密度。
- 形状标注注释先于每条 DMA：`// raw shape: [seg_batch, align_dim]`、`// wram shape: [co / 64, 16, 4, ci]`。

### 5. Host launcher：扁平、状态返回、最少校验

- 早返回 guard 优先 → 设备查询 → 计算 → launch，`return BANGC_KERNELS_STATUS_*;`，无 try/catch/logging/fallback：

```cpp
// hadamard_transform.mlu:83-115
if (dim > 32768) {
  return BANGC_KERNELS_STATUS_NOT_SUPPORTED;
}
int ordinal = -1, cluster_num, core_dim;
CNRT_CHECK(cnrtGetDevice(&ordinal));
CNRT_CHECK(cnrtDeviceGetAttribute(&core_dim, cnrtAttrMcorePerCluster, ordinal));
cluster_num               = getClusterDim();
cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;
cnrtDim3_t k_dim{.x = (uint32_t)core_dim, .y = (uint32_t)cluster_num, .z = 1};
using TDevice = bang_unwrap_data_t<std::decay_t<decltype(*x)>>;
...
return BANGC_KERNELS_STATUS_SUCCESS;
```

- **designated initializer**：`cnrtDim3_t k_dim{.x=..., .y=..., .z=1}` 是固定习惯。
- 设备类型推导 `bang_unwrap_data_t<std::decay_t<decltype(*ptr)>>`。
- 模板实例化用宏驱动，文件底部每类型一行：

```cpp
// hadamard_transform.mlu:118-125
template FUNCTION_HADAMARD_TRANSFORM(float);
template FUNCTION_HADAMARD_TRANSFORM(bang_half_t);
template FUNCTION_HADAMARD_TRANSFORM(bang_bfloat16_t);
```

- **moe launcher 里手搓 core-split 除数循环**：forward 已把它抽进共享 `baseMoeUnpermute`（`moe_unpermute.mlu:60-69`），两个 forward launcher 共用一份；backward 两个 launcher 仍各写一份（`moe_unpermute_backward.mlu:58-67`/`:117-127`）。共 3 处，forward 比"复制"更抽象。

### 6. 注释风格

- **英文 Doxygen**（`@param`/`@par Data Type`/`@par Scale Limitation`/`@par Reference` 全齐）+ **中文实现 why**。
- `@par Reference`：moe 挂 Megatron-LM 内网链接；hadamard/ftrl/lamb 原版写 `- None.`——**不是每篇都挂 Megatron**，挂不挂取决于该算子是否有对应参考实现。
- 把**实测性能数字写进注释当论据**："排流水前 300us，排流水后 133 us"（`hadamard_transform_union1.mlu:205`）。
- block comment 像"事后诊断书"而非流水账——大维两阶段的设计注释直接写"stage1 出口截回 T 是元凶"。

### 7. 命名约定（按类别分大小写）

命名是我刻意维护的语义层，按"是什么/做什么"分类别用大小写：

| 类别 | 风格 | 例子 |
|---|---|---|
| kernel | PascalCase | `kernelHadamardTransform`、`kernelUnpermutePad`、`kernelFtrlUnion1` |
| `__mlu_func__` 流函数 | camelCase + 动词 | `loadMove`、`postTranspose`、`computeSeg`、`storeSeg`、`ftrlZToW`、`lambFunctor` |
| `__mlu_func__` 核心 | `coreProcess`/`coreHandle`/`clusterProcess` | 固定名，流水循环拥有者；按编排粒度选名 |
| host helper | snake_case | `next_power_of_two`、`baseMoeUnpermute`（host helper 也用 camelCase，见小毛刺） |
| 局部变量 | snake_case | `seg_batch`、`align_dim`、`cluster_seg_batch`、`pingpong_gap` |
| constexpr 常量 | 短小写 | `ci`、`p1_block`、`chunk_num` |
| 流水阶段宏 | `_` 前缀大写 | `_LOAD`/`_COMPUTE`/`_STORE` |
| 其它宏 | UPPER_SNAKE + 前缀 | `HADAMARD_MOVE_TPL`、`IMPL_PAD`、`ENTRY_UNIQUE` |
| 类型 | 单字母 | `T`、`TDevice`、`GradT`/`WeightT`/`LargerT` |

**命名上的个人印记——"流"用动词命名：** 复杂算子里每条流封装成函数，名字直接是动作（`loadMove`、`postTranspose`、`computeSeg`、`storeSeg`、`st1_store`），pipeline 里 `_LOAD`/`_COMPUTE`/`_STORE` 宏是对这些流函数的调度入口。函数名读起来就是"做什么"，而不是"是什么"。流水拥有者固定叫 `coreProcess`/`coreHandle`（`core` 指 core 级，因为 `core` 一般就指 ipu）；当编排粒度抬到 cluster 级、拥有者负责的是 cluster 维度的流水时，叫 `clusterProcess`——名字跟随实际编排粒度，跨 op 复用同一名强化"这是流水核心"的语义。

**真实小毛刺**（tic，非系统化，未收敛的瑕疵而非合法例外）：
- `UnPad`/`Unpad` 大小写不一致（`kernelMoeUnpermuteUnPadElementWise` vs `kernelUnpermuteUnpad`）。
- `IMPL_`/`ENTRY_` 宏前缀混用。
- dtype 实例化顺序不一致（`bfloat16,half,float` vs `float,half,bfloat16`）。
- `st1_store` 用 lower_snake 而非 camelCase+动词（流函数里唯一的破例）。
- `matmul`/`butterfly` 是名词命名而非动词（hadamard 纯计算叶子，偏离"流用动词"）。
- `coreProcessElementWise` 带后缀，不是裸 `coreProcess`（moe element_wise 唯一带后缀的 core）。
- `@par note` 小写 vs `@par Note`（lamb.h vs ftrl.h）。
- `baseMoeUnpermute` host helper 用 camelCase 而非 snake_case。

### 8. 格式

- 2 空格缩进、同行左括号、单语句 `if` 不加括号。
- 赋值与尾注释**手工列对齐**：

```cpp
// hadamard_transform_union1.mlu:142-144
uint32_t w1                = seg_batch / 16 * 16;
uint32_t w2                = seg_batch % 16;
constexpr uint32_t t_align = sizeof(float) / sizeof(T);
```

- 行宽不卡（asm 串超 120 也放着）。
- `// -------- NAME --------` 横幅分阶段（`_union1.mlu:27` `// -------- MOVE --------`、`:74` `// -------- COMPUTE --------`）。
- 模板参数列表逐行换行、`const`/`&` 对齐。

### 9. 重用哲学：能抽象就抽象，提炼出公共件

核心取向是**优先抽象、提炼公共件**，而非复制粘贴。早期 op 里出现的重复，是当时框架/算子差异的约束产物，不是刻意选择；到了 hadamard（成熟期）这套取向已经完全显形。

**成熟期（hadamard）—— 抽象提炼的实际形态：**

- `coreProcess` 一份实现被小维路径 **和** 大维两阶段的 stage1 同时复用。低精度（half/bf16）路径不复制 kernel 体，只强制 `coreProcess<float>` 实例化——函数体原样不动，靠模板实例化解耦精度与算法：

```cpp
// hadamard_transform_union1.mlu:574  (大维低精度 stage1 复用同一 coreProcess)
coreProcess<float>(ws_f + t * p1_block, ws_f + t * p1_block, 1.0f, cluster_batch_num,
                   p1_block, read_width, p1_block, ws_row_bytes, ws_row_bytes);
```

- `CvtToFloat`/`CvtFromFloat` 集中所有 dtype 转换为 `if constexpr` 梯子（`hadamard_transform_util.h:66-82`），`preCvtToFloatWorkspace` 与 `postProcessLargeDim` 都调它，转换逻辑唯一源。
- `hadamard16` 矩阵在 `_util.h` 声明一次，`coreProcess` matmul 与 host 共享引用，不重复定义。
- 框架公共件（`CEIL_ALIGN`/`FLOOR_ALIGN`/`MAX_NRAM_SIZE`/`CNRT_CHECK`/`bang_unwrap_data_t`）一律复用，不自己造同义宏。
- 宏卫生：`HADAMARD_MOVE_ASM`/`HADAMARD_MOVE_TPL` 用完即 `#undef`（`loadMove` 底部 `_union1.mlu:70-71`、`postTranspose` 底部 `:168-169`），保持文件级干净。

**早期（moe/ftrl）的重复 ≠ 偏好，是约束：**

- `getLoopInfo`/`swapPtr` 在 moe 两个 op 的 `_kernel.h` 各定义一遍——当时没有跨 op 公共头可放，而非不想抽。
- `computeSeg`/`storeSeg` 在 pad/unpad/no_probs 间近似复制+微调——各 variant 的尾段处理确有差异，当时尚未找到统一参数化的切口。
- `v1_kernel/` 与 `*_kernel_union1/` 并存——**仅见于 moe_unpermute_backward**（forward 已无 v1），保旧版以便回退，不是拒绝收敛。

**演化结论：** 从 moe 的"复制改"到 hadamard 的"coreProcess 一份多场景复用 + 转换集中 + 宏用完即 undef"，方向一致——**只要能抽象就抽象，提炼成单一公共件；不能抽象时才退而复制，且留下回头收敛的余地（v1/v2 并存）**。forward moe 把 core-split 抽进 `baseMoeUnpermute` 共享，正是这条方向的中间证据。

### 10. 成熟度弧线

- **ftrl/lamb（最早）**：标量参数穿到底、float-only、手写设备查询（`cnrtGetDevice`/`cnrtDeviceGetAttribute`）、无 `static_assert`、无 `CNRT_CHECK`；tile 与空间尚未拆成独立两步（span 直接由 `MAX_NRAM_SIZE` 反推）。`bang_unwrap_data_t` 已在 host 用上。lamb 的 device 代码至今完整存活在当前树；ftrl/ftrl_group_lasso 的 device 代码被他人重写（`opt_functor`/`policyFuncU1`/bf16 支持等），本人原版只存历史 commit + `.h` 头。
- **moe**：流水宏化、if-band 稳定成形、core-split 抽共享 helper。
- **hadamard（最新）**：`#undef` 后宏卫生、`CNRT_CHECK`、surgical 精度修复（大维两阶段 + `preCvtToFloatWorkspace`/`postProcessLargeDim` 把"stage1 出口截回 T 的精度损失"手术式切掉）、缓冲数实测择优、tile 与空间拆成独立两步、零死代码零 TODO——"交的是第三四遍"。

---

## 证据来源（附录）

本画像基于以下算子中**真实归属本人**的代码提炼（git blame + 源码精读），作为画像落地的参照，不作为"是否满足"的验收。git 仓库为嵌套的 `cambricon_proj/gitlab/mlu-ops/`（顶层 `.git` 为空，历史只在嵌套仓内）。

| 算子 | 阶段 | 当前树存活情况 | 证据强度 |
|---|---|---|---|
| `ftrl` / `ftrl_group_lasso` | 最早（2025-08~09，commit `45e165db`/`b26c762a`） | **device 代码已被他人重写**（`MLUUnionKernelFtrlVec`/`opt_functor`/`policyFuncU1`），本人原版仅存历史 commit；当前树仅保留 `.h` Doxygen 头 + 版权头 | 原版实现；靠 `git show <commit>:kernels/<op>/...mlu` 取证 |
| `lamb` | 最早（2025-10，commit `22b0550f`） | **device 代码至今完整存活**于 `kernels_light/lamb/lamb.mlu`（`l2norm`/`lambFunctor`/`coreHandle`/`kernelLambUnion1` 逐字保留，仅 blame 挂在 luoran 的文件搬移 commit `22b0550f` 之下） | 原版实现且仍在树上，证据强 |
| `moe_unpermute` / `moe_unpermute_backward` | 中期（2025-01 + 2026 改进） | 本人占各文件 75–95%，他人仅加版权头/assert/弃用桩 | 中期主力，证据强 |
| `hadamard_transform` | 最新最成熟（2026，commit `b6d83b50`/`3830864a`/`fcdb09c2`） | 984/988 行本人 | 画像主要锚点，证据最强 |

源码路径：`cambricon_proj/gitlab/mlu-ops/kernels_light/<op>/`（lamb）/`.../kernels_light/moe_unpermute/`（moe）/`.../kernels_light/hadamard_transform/`（hadamard）。ftrl/ftrl_group_lasso 原版取自 `git show 45e165db:kernels/ftrl/ftrl_union1.mlu`、`git show b26c762a:kernels/ftrl_group_lasso/ftrl_group_lasso_union1.mlu`。本文行号锚点均指向真实文件。
