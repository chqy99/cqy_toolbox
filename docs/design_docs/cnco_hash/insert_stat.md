# insert_and_evict 实现设计方案--基于统计信息设计

* #### 文档基本信息
| 算子名称 | insert_and_evict |
| ------ | ------ |
| 编制人/日期 | 陈其阳/2026-9-18 |

* #### 修改记录
| 版本号 | 修订人 | 修订日期 | 修订描述 |
| ------ | ------ | ------ | ------ |
| V 1.0 | 陈其阳 | 2026-9-18 | 首次提交 |

* #### 内容描述
本文档为 hkv 库中 insert_and_evict 行为移植到寒武纪 MLU 的基于统计信息设计的实现方案。

* #### 先验知识
 - 同批次插入的 key 不会重复。
 - 同批次插入的 score 为 timestamp，值相同；下一次批次的 score 大于上一批次。
 - 不会插入无效的 key。

## 1 需求分析

### 1.1 算子需求分析

| 算子功能简介 | 向 bucket 化哈希表批量 insert |
| ------ | ------ |
| 需求来源 |  大别山，参考仓库：merlin `upsert_and_evict` |
| 应用网络 | 推荐系统 embedding 两级缓存（device 热表 + host 大表差分同步）；LLM 推理 KV cache |
| 输入数据类型 | key: uint64_t;score: uint64_t;value: 定长 <= 4096B embedding;delta_diff/delta_diff_ad: float;delta_version: uint64_t;delta_on_device/find_status:bool |
| 输入shape | keys `[num]`、scores `[num]`、values `[num, V]`、delta_diff `[num]`、delta_diff_ad `[num]`、delta_version `[num]`、delta_on_device `[num]`、find_status `[num]` |
| 输入Layout  | ARRAY（各段独立连续） |
| 输出数据类型 | evicted_keys: uint64_t;evicted_scores:uint64_t;evicted_values:同value;evicted_delta_diff:float;evicted_delta_version:uint64_t;evicted_counter:uint32_t;key_exist:u8|
| 输出shape | evicted_keys `[num]`、evicted_scores `[num]`、evicted_values `[num, V]`、evicted_delta_diff `[num]`、evicted_delta_version `[num]`、key_exist `[num]` |
| 输出Layout | ARRAY |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | 否 |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | 否 |
| 是否需要支持原位        | 是。表存储（key 槽 / value 块 / digest / delta_diff / 计数）为持久状态，insert 直接原位改写 |
| 是否需要支持stride机制  | 否 |
| 是否需要支持广播        | 否 |
| 0元素检查是否直接返回    | 是。len = 0 直接返回 SUCCESS（不清零 counter、不分配 patch_id） |

### 1.2 算子功能和应用场景描述

**单条插入状态机**（merlin OccupyResult → V1 RetType）——本算子的语义核心：

| RetType (u8) | 触发 | 数据动作 |
|---|---|---|
| INVALID(0) | reserved key | 跳过 |
| EXIST(1) | key 已在桶内（dup） | 原位更新 score/value；delta 累加；**不驱逐** |
| NOT_EXIST(2) | 桶未满，写入空槽 | 新条目写入，桶计数 +1 |
| EVICT(3) | 桶满，insert score 严格大于桶内最小 score | 驱逐桶内最小 score 槽；旧条目 key/score/value/delta **从表内拷出**进 evicted 输出 |
| EVICT_DROP(4) | 桶满，无驱逐资格 | 新 key 不进表，其 key/score/value **从 batch 输入直接拷**进 evicted 输出 |

### 1.3 算子输入输出参数要求

**表构造参数（Map 构造函数）**：

| 参数 | 语义 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制 | 持久化 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| initial_size | 初始容量（槽数），向上取 2 的幂 | 输入 | uint32_t | / | >= bucket_size | 是（构造期） |
| load_factor | 负载因子 | 输入 | float | / | (0, 1] | 是 |
| default_key | 空槽 key 标记（sentinel） | 输入 | uint64_t | / | 1 | 是 |
| value_type_size | value 定长 V | 输入 | uint32_t | / | <= 4096B（内部对齐 2 的幂） | 是 |
| default_value（device 侧名 default_embedding） | 空 value 模板（device 指针） | 输入 | 同 value | / | 1 | 是（caller 持有） |
| allocator | 设备内存分配器 | 输入 | / | / | / | / |
| value_blocks_size | value 块大小 | 输入 | uint64_t | / | 默认 512MB | 是 |
| shuffle_dim | value 写入洗牌维度 | 输入 | uint32_t | / | 默认 1024 | 是 |
| max_capacity | 容量上限 | 输入 | uint32_t | / | 2 的幂，默认 2^31 | 是 |
| bucket_size | 单桶槽位数 | 输入 | uint32_t | / | 2 的幂，[8, 128]，默认 128 | 是 |

**表持久存储（map 生命周期常驻，insert 原位改写）**：

| 参数 | 语义 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制 | 持久化 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| key_slot_map | key_slot 相关槽 | 输入+输出 | 自定义struct | 32B连续 | - | 是 |
| value_map（device 侧名 d_values_table） | 独立 value 存储（键值分离）| 输入+输出 | 定长V | - | 是 |
| counter | 表占用计数 | 输出 | uint32_t | - | 1 | 是 |
| current_patch_id | 当前批次号 | 输入+输出 | uint32_t | - | 单调递增 | 是 |

**单次 insert_with_evict 调用的批量输入输出**：

| 参数 | 语义 | 类型（输入/输出） | 支持类型 | 物理布局 | 规模限制 | 持久化 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| d_counter | 表占用总槽数 | 输入+输出 | uint32_t | device 标量 | 1 | 是 |
| d_key_slots | key_slot 槽数组基址 | 输入+输出 | 自定义 struct | ARRAY，[capacity]，slot_size 连续 | <= capacity | 是 |
| d_values | value 存储基址（与 key_slot 槽位一一对应） | 输入+输出 | 定长 V | ARRAY，[capacity]，槽位 i ↔ 第 i 条 | V <= 4096B | 是 |
| d_valid_num | 每桶有效槽数（免扫描桶计数） | 输入+输出 | uint8_t | ARRAY，[num_buckets] | <= bucket_size | 是 |
| slot_size | 单槽字节数 | 输入 | uint32_t | / | 32 | 是 |
| bucket_size | 单桶槽位数 | 输入 | uint32_t | / | 2 的幂，[8, 128]，默认 128 | 是 |
| d_keys | 待插入 key 批 | 输入 | uint64_t | ARRAY | [num]，批内不重复 | 否 |
| d_embeddings（device 侧名 d_values） | 待插入 value 批 | 输入 | 定长 V | ARRAY | [num, V] | 否 |
| d_scores | 驱逐优先级 score（timestamp） | 输入 | uint64_t | ARRAY | [num]，批内同值、跨批递增 | 否 |
| default_embedding | 空 value 模板 | 输入 | 同 value | device 指针 | 1 | 是（caller 持有） |
| key_exist | 每 key 插入结果码 RetType | 输出 | uint8_t | ARRAY | [num] | 否 |
| empty_key | 空槽 key 标记（sentinel） | 输入 | uint64_t | / | 1 | 是 |
| d_evicted_keys | 被驱逐 / 被丢弃 key | 输出 | uint64_t | ARRAY | [num]，可 nullptr | 否 |
| d_evicted_values | 被驱逐 / 被丢弃 value | 输出 | 定长 V | ARRAY | [num, V]，可 nullptr | 否 |
| d_evicted_scores | 被驱逐 / 被丢弃 score | 输出 | uint64_t | ARRAY | [num]，可 nullptr | 否 |
| d_evicted_counter | 本批 evicted 条数 | 输出 | uint32_t | device 标量 | 1，<= num，可 nullptr | 否 |
| num | 本批 key 数 | 输入 | uint32_t | / | 0 直接返回 SUCCESS | 否 |
| capacity | 表总槽数 | 输入 | uint32_t | / | 2 的幂，= num_buckets * bucket_size | 是 |
| embedding_type_size | value 单条字节数 V | 输入 | uint32_t | / | <= 4096B | 是 |
| unique_key | 批内 key 唯一性提示 | 输入 | bool | / | 1，reserved | 否 |
| ignore_evict_strategy | 跳过驱逐资格比较 | 输入 | bool | / | 1，reserved（驱逐恒走 min-score） | 否 |
| delta_diff | 差分累积值 | 输入 | float | ARRAY | [num]，reserved | 否 |
| delta_diff_ad | 差分累积值（ad 分支） | 输入 | float | ARRAY | [num]，reserved | 否 |
| delta_version | 差分版本号 | 输入 | uint64_t | ARRAY | [num]，reserved | 否 |
| evicted_delta_diff | 被驱逐 delta_diff | 输出 | float | ARRAY | [num]，reserved，可 nullptr | 否 |
| evicted_delta_version | 被驱逐 delta_version | 输出 | uint64_t | ARRAY | [num]，reserved，可 nullptr | 否 |
| base_delta_version | 版本号基准 | 输入 | uint64_t | / | 1，reserved | 否 |
| delta_on_device | value 是否驻留 device | 输入 | bool | ARRAY | [num]，reserved | 否 |
| find_status | 查找命中状态 | 输入 | bool | ARRAY | [num]，reserved | 否 |
| priority_push_times | 优先级推挤次数 | 输入 | int | / | >= 0，reserved | 否 |
| current_patch_id | 当前批次号 | 输入 | uint32_t | / | 单调递增 | 是（表侧状态，见上表） |
| queue | CNRT 执行队列 | 输入 | cnrtQueue_t | / | 非空 | / |

### 1.4 算子限制

- value 定长 <= 4096B（内部对齐到 2 的幂）。
- bucket_size 为 2 的幂 ∈ [8, 128]（默认 128）；capacity 为 2 的幂。
- **单写者假设**：同一时刻仅一个 insert 在飞（host mutex 串行化）。
- priority_push_times 当前必须 = 0；
- reserved key（sentinel key）跳过，不占任何状态与计数。
- evicted_* 输出缓冲由调用方分配，容量须 >= num（最坏全批驱逐/拒插）。

## 2 算子接口设计

### 2.1 参考实现

device 侧接口
```c++
__mlu_global__ void kernelInsertScoreIsTimestamp(
  // pending
  uint64_t *d_keys,
  uint64_t *d_scores, // 特化分支不处理
  int8_t *d_values,
  uint32_t *sorted_raw_idx,
  uint32_t *hist_bucket_no,
  uint32_t *hist_bucket_begin,
  uint32_t *hist_bucket_cnt,
  uint32_t *core_hist_num,
  // delta
  float *delta_diff,
  float *delta_diff_ad,
  uint64_t *delta_version,
  uint64_t base_delta_version,
  int8_t *delta_on_device,
  // map
  uint64_t *d_key_slots,
  int8_t *d_values_table,
  uint8_t *d_valid_num,
  uint32_t *d_counter,
  // evict
  void *d_bucket_evict_key_ws,
  void *d_bucket_evict_score_ws,
  void *d_bucket_evict_value_ws,
  void *d_bucket_evict_dd_ws,
  void *d_bucket_evict_dv_ws,
  uint32_t *core_bucket_evict_cnt,
  // param
  uint32_t key_num,
  size_t value_size,
  uint32_t bucket_size,
  size_t capacity,
  uint32_t current_patch_id
);
```

key_slot 结构体定义（`d_key_slots` 的单个元素，数组 [capacity] × slot_size 32B）：

```cpp
struct key_slot {
  uint64_t key;            // 8B
  uint64_t score;          // 8B  驱逐优先级（timestamp）
  uint64_t delta_version;  // 8B  增量版本号；0 = 已被 host 消费并 reset
  float    delta_diff;     // 4B  差分累积值
  uint32_t patch_id;       // 4B  插入批次号
};
```

## 3 实现方案设计

### 3.1 实现方案

#### 3.1.1 预处理

| 阶段 | kernel | 输入 → 输出 | 职责 |
|---|---|---|---|
| K1 | hash | d_keys → d_key_bucket_no | key → bucket_no（hash & (num_buckets-1)） |
| K2 | CNNL topK 稳定排序 | bucket_no → d_sorted_bno, sorted_raw_idx | 按 bucket_no 稳定排序（保批内原序） |
| K3 | RLE 直方图 | d_sorted_bno → hist_bucket_no, hist_bucket_begin, hist_bucket_cnt, core_hist_num | 按 key 均拆，各核 RLE 压缩出桶段 |
| K4 | 去重 + 前缀和 | hist（in-place，hist_bucket_no 只读） | 跨核同桶合并（删前留后）；处理后不同核不处理同一 bucket |
| K5 | 截断 + filter + evict_drop 收集 | hist（in-place 截断 hist_bucket_cnt）、sorted_raw_idx（原位改写为 valid_raw_idx）→ evict_drop_raw_idx, core_evict_drop_cnt | hist_bucket_cnt 截断至 bucket_size，valid_raw_idx 原位 filter，主流程无需再考虑 evict_drop |

数值示例（4 核、16 个 key、bucket_size = 4）：排序后 bucket_no 序列（d_sorted_bno）= 1,1,1,1,1,2,2,3,3,3,4,4,4,4,5,6，按 key 均拆每核 4 个。hist 数组为 1D 摆放，按 key_num 均拆直接切片，下表按核分行展示（行宽 = 均拆 key 数，~ 表示无效槽位、不关心其内容）；切片内容由 K3 一次写入，K4/K5 只原地改写数值，不搬移条目，活条目恒为段前 core_hist_num 个。切片长度即各核段内 key 数上界，hist 永不溢出切片，后续合并（K4）与 filter（K5）所需内存空间更少；各核的处理范围按自身 hist_bucket_begin 起点、hist_bucket_cnt 偏移确定，写入落在自身区间内，不会影响下一个核 hist_bucket_begin 位置的数据，不存在写冲突。

K3 各核 RLE 直方图：

| | core0 | core1 | core2 | core3 |
|---|---|---|---|---|
| core_hist_num | 1 | 3 | 2 | 3 |
| hist_bucket_no | 1,~,~,~ | 1,2,3,~ | 3,4,~,~ | 4,5,6,~ |
| hist_bucket_begin | 0,~,~,~ | 4,5,7,~ | 8,10,~,~ | 12,14,15,~ |
| hist_bucket_cnt | 4,~,~,~ | 1,2,1,~ | 2,2,~,~ | 2,1,1,~ |

K4 删前留后合并跨核同桶：bucket 1 删 core0 末尾 entry、core1 保留（begin 4→0）；bucket 3 删 core1 末尾 entry、core2 保留（begin 8→7）；bucket 4 删 core2 末尾 entry、core3 保留（begin 12→10）。删除只递减 core_hist_num（死条目以 ~ 占位），hist_bucket_begin / hist_bucket_cnt 原地刷新，各核 hist 数组访问起点不变：

| | core0 | core1 | core2 | core3 |
|---|---|---|---|---|
| core_hist_num | 0 | 2 | 1 | 3 |
| hist_bucket_no | ~,~,~,~ | 1,2,~,~ | 3,~,~,~ | 4,5,6,~ |
| hist_bucket_begin | ~,~,~,~ | 0,5,~,~ | 7,~,~,~ | 10,14,15,~ |
| hist_bucket_cnt | ~,~,~,~ | 5,2,~,~ | 3,~,~,~ | 4,1,1,~ |

K5 截断 + filter（bucket_size = 4）：bucket 1 的 cnt 由 5 截为 4，core1 区间内排序后位置 4 判 evict_drop；valid_raw_idx 为 sorted_raw_idx 的原位改写（无独立缓冲），各核 key 区间内剔除 evict_drop 后前移 filter，非首个 hist_bucket_begin 按前置 drop 数前移刷新（core1 的 bucket 2 begin 5→4），各核首个 hist_bucket_begin 不动。

| | core0 | core1 | core2 | core3 |
|---|---|---|---|---|
| core_hist_num | 0 | 2 | 1 | 3 |
| hist_bucket_no | ~,~,~,~ | 1,2,~,~ | 3,~,~,~ | 4,5,6,~ |
| hist_bucket_begin | ~,~,~,~ | 0,4,~,~ | 7,~,~,~ | 10,14,15,~ |
| hist_bucket_cnt | ~,~,~,~ | 4,2,~,~ | 3,~,~,~ | 4,1,1,~ |

拆分结果（valid_raw_idx 按 K4 后各核 key 区间分段：core1 区间 [0,7) 7 槽 filter 为 6、尾槽 pad ~；core2 区间 [7,10)；core3 区间 [10,16)；下标为排序后序列内位置）：

```text
valid_raw_idx       = [0,1,2,3,5,6,~, 7,8,9, 10,11,12,13,14,15] （原位修改 sorted_raw_idx）
core_evict_drop_cnt = [0, 1, 0, 0]  // 每核 evict_drop 总数，供 K7 evict 后处理
```

evict_drop_raw_idx 不占独立缓冲，复用 hist_bucket_no 切片：前 core_hist_num 个仍为桶编号，其后紧跟 core_evict_drop_cnt 个 drop 原始下标，两者联合定位（段首 offset + core_hist_num，段长 core_evict_drop_cnt）。K5 后 hist_bucket_no 各核切片最终内容：

| evict_drop_raw_idx（复用 hist_bucket_no） | core0 | core1 | core2 | core3 |
|---|---|---|---|---|
| （桶编号 + drop 下标） | ~,~,~,~ | 1,2,4,~ | 3,~,~,~ | 4,5,6,~ |

evict_drop_raw_idx 落点为本核 hist_bucket_no 切片内活条目之后（offset + core_hist_num 起）：活条目 cnt ≥ 1，故 core_hist_num ≤ Σ hist_bucket_cnt = valid；又 valid + drop = split_num（本核 key 数即切片长），两者合并得 core_hist_num + drop ≤ split_num，最坏恰好写满本核切片、不越入下一核。K7 由 offset + core_hist_num 定位每核 drop 段首，core_evict_drop_cnt 定位段长，无需额外数组。

K5 只做核内 filter，不跨核 filter：drop 腾出的槽位不向后核借数据填补，各核首个 hist_bucket_begin 与段边界保持不动，省去跨核搬移操作；主流程只读各核段内前 Σ hist_bucket_cnt 个槽位（core1 读 [0,6)），段尾剩余槽位内容不关心（~）。

对应 device 侧接口，预处理阶段实际上获取了：
- hist_bucket_no 直方图统计的桶编号
- hist_bucket_begin，直方图统计的桶偏移
- hist_bucket_cnt，直方图统计的桶数量
- core_hist_num，每个核需要处理直方图的数量
- valid_raw_idx，K5 拆分与 filter 产物，排序结果的有效原始坐标
- evict_drop_raw_idx，evict_drop 的原始坐标
- core_evict_drop_cnt，每核 evict_drop 的总数量（K7 evict 后处理用）
- 隐含语义，key_num 均拆得到的 offset 是 hist_bucket_no, hist_bucket_begin, hist_bucket_cnt 的起点

##### 3.1.1.1 核间拆分

hist_bucket_no / hist_bucket_begin / hist_bucket_cnt 为 1D 摆放，按 key_num 均拆直接切片（见数值示例），各核只访问自己那一段切片，处理数量由本核 core_hist_num 给出。

valid_raw_idx 是排序序列上的一维数组，处理范围由本行 hist_bucket_begin 直接定位：首个 hist_bucket_begin 即本核 key 区间起点，各桶段按 begin + cnt 偏移展开，不越出自身区间。

#### 3.1.2 主流程

主流程按数据流分为 5 个阶段：LOAD(HIST,PENDING,MAP) -> FIND_DUP_IDX -> GET_EVICT_IDX -> UPDATE -> STORE。先口语化介绍每个阶段做什么：

- **LOAD(HIST,PENDING,MAP)**：把三类数据搬上片。HIST 告诉本核要处理哪些桶、每桶落了多少待插入 key（hist_bucket_no / hist_bucket_begin / hist_bucket_cnt）；PENDING 是待插入批中落入这些桶的 key / value（valid_raw_idx 定位）；MAP 是表内这些桶的现有槽位数据（d_key_slots 对应段 + d_valid_num）。
- **FIND_DUP_IDX**：pending key 与桶内已有 key 逐对比对，找出重复项——这些 key 是原位更新，不是新插入。
- **GET_EVICT_IDX**：桶满且还有新 key 待进时，按驱逐策略选出被腾出的槽位索引。进不了桶的 evict_drop 已在预处理 K5 拆出，主流程不再处理。
- **UPDATE**：按三类位置改写槽位——重复项原位更新 score / value / delta，驱逐项换主写入新 key，空槽直接写入。
- **STORE**：改写后的 key_slot / value 拷回显存，d_valid_num 同步更新。

下面按两种处理粒度展开细节：单桶方案（3.1.2.1）、多桶方案（3.1.2.2），delta 逻辑单列（3.1.2.3）。

切片级 LOAD 公共流程：
1. 加载标量 core_hist_num。
2. 连续加载向量 hist_bucket_no, hist_bucket_begin, hist_bucket_cnt，向量切片长度定义为 once_load_hist_num（默认为 128）。
3. 加载向量d_valid_num。最后一个 bucket_no - 第一个 bucket_no = once_load_hist_num - 1 时，说明是连续的 bucket_no。 
  - 连续 bucket_no 时，连续加载向量 d_valid_num。
  - 非连续 bucket_no 时，gather 加载向量 d_valid_num。

##### 3.1.2.1 单次循环处理单个 bucket 方案

1. pending 数据加载，hist_bucket_begin 和 hist_bucket_cnt 是对应的偏移和数量。预处理 K5 已把 hist_bucket_cnt 截断至 bucket_size、valid_raw_idx 原位 filter，主流程无需再处理 evict_drop。
  - 连续加载向量 valid_raw_idx，长度即 hist_bucket_cnt[i]，上限为 bucket_size。
  - valid_pending_num = hist_bucket_cnt[i]。
  - 根据 valid_raw_idx gather 得到 pending 序列。
  - 思考：单桶 gather 时 transfer_num 可能很小（小于 96），向量化性能不足。
2. map 数据加载，按 d_valid_num 有效长度直接连续加载。
3. pending 数据与 map 数据查重，获取重复的位置索引。
  - 构造 inc_idx(0,1,2...,bucket_size*bucket_size - 1)，这个只需要构造一次。
  - 将 d_key_slots transpose 取出 d_keys，memcpy 复制 valid_pending_num 次，然后 transpose，得到 d_keys_2d。再使用 bang_cycle_eq 对比 d_keys_2d 与 pending_keys 得到 mask_2d，mask_2d 先转为与 inc_idx 一致的 u32，再 bang_filter 筛选得到 dup_num 和 dup_idx_2d。再使用 bang_div 和 bang_rem 得到 dup_map_idx 和 dup_pending_idx。
  - 数值示例（valid_num = 5，valid_pending_num = 3，桶内 map keys = (1,3,4,6,8)，pending keys = (4,5,8)）。map 复制 valid_pending_num 份得 [3 × 5]，transpose 后 d_keys_2d 形状为 [valid_num × valid_pending_num]，展开下标 = m * valid_pending_num + p（m 为 map 槽序，p 为 pending 序）：

| d_keys_2d | p=0 (4) | p=1 (5) | p=2 (8) |
|---|---|---|---|
| m=0 (1) | 1 | 1 | 1 |
| m=1 (3) | 3 | 3 | 3 |
| m=2 (4) | **4** | 4 | 4 |
| m=3 (6) | 6 | 6 | 6 |
| m=4 (8) | 8 | 8 | **8** |

  cycle_eq 以 valid_pending_num 为周期对齐 pending_keys，即 mask_2d[i] = (d_keys_2d[i] == pending_keys[i mod valid_pending_num])：

```text
i          :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
d_keys_2d  :  1  1  1  3  3  3  4  4  4  6  6  6  8  8  8
pending    :  4  5  8  4  5  8  4  5  8  4  5  8  4  5  8
mask_2d    :  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1
```

  filter 以 mask_2d 筛 inc_idx（inc_idx 预构造 0..bucket_size*bucket_size-1，本桶取前 valid_num * valid_pending_num = 15 个），得 dup_num = 2，dup_idx = (6, 14)。div/rem 的除数与模数均为 valid_pending_num：

| dup_idx | / 3 → dup_map_idx | % 3 → dup_pending_idx | 命中（map 值, pending 值） |
|---|---|---|---|
| 6 | 2（map 槽 2） | 0（pending 序 0） | 4 = 4 |
| 14 | 4（map 槽 4） | 2（pending 序 2） | 8 = 8 |

  矩阵视角交叉验证：命中格 (m=2, p=0) 展开为 2*3+0 = 6，(m=4, p=2) 展开为 4*3+2 = 14，与 dup_idx 一致。

思考：除法和取模都很慢，可以除法后，再用乘加指令算模数。

4. 空槽的位置索引。当 valid_num < bucket_size 时，对应 [valid_num, bucket_size) 都是空槽。
5. evict 数据处理，获取待驱逐的位置索引。
  - need_evict_num = valid_num + valid_pending_num - dup_num - bucket_size。need_evict_num 最低为 0。
  - 如果 need_evict_num > 0，需要先更新重复数据的 patch_id：按 dup_map_idx 做偏移使用 scatter 指令赋值 cur_patch_id。
  - 如何获取 patch_id 最小的 need_evict_num 个数据，方案对比：
  （1）排序方法。多轮基数排序，可以用 cur_patch_id 来减少轮数。
  （2）二分法，如果选用此方法的话，可以维护 d_last_evict_max_patch_id（每桶一个，类似 d_valid_num，[num_buckets] 数组）， 这样二分时候可以缩小下界。这个也是逐桶处理。
6. map 数据状态更新与拷回。
  - 思考：查重和驱逐位置是离散的，空槽是连续的（连续段可能长度不够）。
  - key_slot 的处理，片上使用离散指令处理，拷出时使用 memcpy 指令。
    - key 和 patch_id 更新，先 gather 再 scatter（dup 索引），scatter（evict 索引） or memcpy（空槽索引）。
    - delta 相关更新，先 gather 对应数据（槽上 old_dv/old_dd、输入 insert_* 系列），按 [章节 3.1.2.3 delta 逻辑处理](#3123-delta-逻辑处理) 的乘加形态公式计算 new_dv/new_dd，再 scatter 写回。
  - value_table 的处理，片上和拷出都使用离散指令处理，减少IO总数据量。
    - gather value 到 nram 上，然后 scatter 回 gdram。
7. d_valid_num 更新。<br>
 new_valid_num = old_valid_num + valid_pending_num - dup_num - need_evict_num。 old_valid_num = bucket_size 时，跳过更新。写回指令与读入指令对应。

思考：为了解决单桶难以打满 IO 带宽的问题，提出多桶方案。

##### 3.1.2.2 单次循环处理多个 bucket 方案
核心思想：多个 bucket 一起 IO，但计算可能是逐个处理。

1. 按组级别来加载 pending 数据和 map 数据，组长定义为 group_num(默认为8)。
2. pending 数据加载，hist_bucket_begin 和 hist_bucket_cnt 是对应的偏移和数量。预处理 K5 已把 hist_bucket_cnt 截断至 bucket_size、valid_raw_idx 原位filter，主流程无需再处理 evict_drop。
  - 组内 valid_raw_idx 连续有效，直接 memcpy 连续拷贝，长度 = Σ hist_bucket_cnt[i]。
  - 使用 gather 指令加载向量 pending_keys，其余 IO 延迟加载。
3. map 数据加载，bang_argmax 可以获取 group_max_valid_num（调研显式引用 bang_device_function_extra.h 可以支持 u8 类型），最后一个 bucket_no - 第一个 bucket_no = group_num - 1 时，说明是连续的 bucket_no。
  - 连续 bucket_no 时，使用 memcpy2d，单次片长 group_max_valid_num * slot_size，src_stride 与 dst_stride 为 bucket_size * slot_size。
  - 非连续 bucket_no 时， gather指令加载 d_key_slots，transfer_size = group_max_valid_num * slot_size，transfer_num = group_num。
  - 剪枝情况： cur_patch_id == 1 时可跳过。
4. 逐桶计算，pending 数据与 map 数据查重，获取重复的位置索引。
5. 空槽的位置索引。当 valid_num < bucket_size 时，对应 [valid_num, bucket_size) 都是空槽。
6. evict 数据处理，获取待驱逐的位置索引。
  - 多组同时排序方法。空槽刷值成 cur_patch_id，逐桶偏移 i * cur_patch_id，group_num 个桶一起排序。
  - 思考：表负载程度低时（d_counter/capacity来评估），此时可能有些桶无需驱逐，只需要 valid_num 个数据排序，逐桶排序可能更优。表负载程度较高时，此时大概率驱逐，也需要全排序，但是需要多考虑空槽刷值成 cur_patch_id，需要更大的buffer空间。大概率只考虑逐桶排序。
7. map 数据状态更新与拷回
  - 思考：多桶的话，需要算全局偏移，更新时候可以用 mask 的乘加指令，应该可以提升向量化程度。
8. d_valid_num 更新。

思考：多桶的方案难以排流水。一是，内存大小限制；二是，哪怕内存足够，看上去也难以分类成 3 个阶段（甚至N个阶段），不太可能能排流水掩盖不同流的时间。并且有较多的标量逻辑，排流水的效果可能有限。

##### 3.1.2.3 delta 逻辑处理

```c++
// ---------- 输入(随本条 key 一起进来, 开头一次性载入) ----------
uint64_t insert_dv    = delta_version[key_idx];    // host 侧带来的版本号(时间戳), 缺省 0
float    insert_dd    = delta_diff[key_idx];       // host 侧"总和型"增量, 缺省 0.0
float    insert_dd_ad = delta_diff_ad[key_idx];    // device 侧"加法型"增量, 缺省 0.0
bool     dod          = delta_on_device[key_idx];  // 本条 delta 是否来自 device
uint64_t base         = base_delta_version;        // 当前全局基准版本(时间戳)

// ---------- 槽上现存值(load 读出) ----------
// dup 时 = 同 key 的历史累积(可能非 0: 尚未被 host 消费);
// 空槽/墓碑时 = 0(初始化/删除时已清零);
// old_dv == 0 <=> 该槽 delta 已被 host 消费并 reset
uint64_t old_dv = bucket.delta_version[pos];
float    old_dd = bucket.delta_diff[pos];

// ---------- 输出(决策出的新值, 随后 store 落盘) ----------
uint64_t new_dv;
float    new_dd;

if (!evicted) {  // OCCUPIED_EMPTY / OCCUPIED_RECLAIMED / DUPLICATE
  if (old_dv == 0) {  // 槽上 delta 已 reset
    new_dv = dod ? base
                 : (insert_dv != 0 ? insert_dv : base);
    new_dd = dod ? insert_dd_ad    // device: 换用加法型增量
                 : insert_dd;      // host:   用输入总和
    bucket.delta_version[pos] = new_dv;  // 此分支 dv 必写回
  } else {  // 槽上还有未消费的累积
    new_dv = old_dv;                      // 版本不动, 不写回
    new_dd = dod ? insert_dd_ad + old_dd  // device: 在旧值上累加
                 : insert_dd;             // host:   总和覆盖
  }
  bucket.delta_diff[pos] = new_dd;  // dd 两种情况都写回
} else {  // EVICT: 新 key 落在被驱逐者的槽
  // 槽上旧 delta 已随被驱逐 key 写入 evicted_delta_*, 与新 key 无关,
  // 故基于输入自身决策, 无旧值可参考
  new_dv = (insert_dv != 0) ? insert_dv : base;
  new_dd = insert_dd;
  bucket.delta_version[pos] = new_dv;  // 此路径 dv/dd 都写回
  bucket.delta_diff[pos]    = new_dd;
}
```

思考：

- 公式变形1
```c++
uint64_t tmp_dv = (insert_dv != 0 ? insert_dv : base);
if (!evicted) {
  uint64_t *mask1 = (old_dv == 0); // 这个不确定是否只有空槽才有，如果是还能简化
  uint64_t *mask2 = dod;
  // new_dv 化简得到
  new_dv = mask1 * (dod * (base - tmp_dv) + tmp_dv - old_dv) + old_dv;
  // new_dd = mask1 * (dod * insert_dd_ad * (1 - dod) * insert_dd) + (1-mask1) * (dod * (insert_dd_ad + old_dd) + (1-dod) * insert_dd);
  new_dd = - mask1 * dod * old_dd + dod * (insert_dd_ad + old_dd - insert_dd) + insert_dd;
} else {
  new_dv = tmp_dv;
  new_dd = insert_dd;
}
```

- 公式变形2（通道归一化：dod 的分支判据提前到载入阶段消化，主循环公式收敛为一条乘加 + 一条 select）

dod 在 delta 决策中有两个作用：A 选输入字段（device 取 `ad`，host 取 `dd`）；B 选合并方式（device 累加到 `old_dd`，host 总和覆盖）。作用 A 与版本兜底只依赖 per-key 输入，在 gather pending 载入阶段即可完成，主循环不再出现 dod 分支。

```c++
// ---------- 载入阶段(gather pending 时, dod 已知), per-key ----------
dd_in  = dod * (insert_dd_ad - insert_dd) + insert_dd; // 作用A: 选字段, 3 条指令
tmp_dv = (insert_dv != 0) * (insert_dv - base) + base;
dv_in  = dod * (base - tmp_dv) + tmp_dv;              // dod 时基准版本权威, 3 条指令

// ---------- 主循环(通道已归一, 只剩槽上状态) ----------
mask_reset = (old_dv == 0);         // 槽上 delta 已被 host 消费并 reset
mask_acc   = (!mask_reset) & dod;   // dod 唯一残留: 未消费 dup 才累加 old_dd
new_dd = dd_in + mask_acc * old_dd;             // 一条乘加
new_dv = mask_reset * (dv_in - old_dv) + old_dv; // 乘加形态(与 new_dd 同构)
```

- 注意事项：dd 是 float 类型，需要考虑 naninf 的问题。

流程约定：
1. 聚焦核心逻辑，片上内存空间预算未严格核算：once_load_hist_num（默认 128）、group_num（默认 8）的取值有一定依据，不会大幅超出片上限制，但后续流程细节仍需推定，不确定完全满足。
2. 未考虑极端规模情况。
3. 未详细描述余数段（末次切片、末组不满）的处理。
4. 部分步骤的描述存在轻度省略。上述简化不影响核心逻辑。
5. 循环骨架：核 → 切片（once_load_hist_num）→ 组（group_num）→ 组内视数据量再切片。切片边界确定、不会重叠；组级与逐桶两种粒度混杂（加载按组、查重与驱逐逐桶），粒度切换的边界需要事先约定。
6. gather / scatter 类指令的 offset 均为字节偏移（元素下标需乘以元素字节宽度），不再逐一标注。

#### 3.1.3 后处理 evict 数据

K7 后处理 filter evict 数据。核心逻辑：
1. filter 数据。
2. 根据 evict_drop_raw_idx 把 pending 序列数据 gather 拷贝到 evict 序列。

core_bucket_evict_cnt, core_evict_drop_cnt 记录了每个核 evict 和 evict_drop 的数量，容易做 filter 处理。

### 3.2 拆分
#### 3.2.1 核内内存划分

考虑纯串行时的内存划分：
