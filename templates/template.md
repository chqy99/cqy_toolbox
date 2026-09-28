# Placeholder{op} 实现设计方案

* #### 文档基本信息
| 算子名称 | Placeholder{op} |
| ------ | ------ |
| 编制人/日期 | Placeholder{author/datatime} |

* #### 修改记录
| 版本号 | 修订人 | 修订日期 | 修订描述 |
| ------ | ------ | ------ | ------ |
| V 1.0 | Placeholder{author} | Placeholder{datatime} | Placeholder{fix descripe} |

* #### 内容描述

本文档为`Placeholder{op}`算子实现的设计文档，包括需求分析、接口设计、方案设计、性能优化记录。
Placeholder{op introduction}

## 1 需求分析

### 1.1 算子需求分析
| 算子功能简介 | Placeholder{op brief introduction} |
|-------------|--------------------------------------------------------------|
| 需求来源    | Placeholder{.} |
| 应用网络    | Placeholder{.} |
| 输入数据类型 | Placeholders{...} |
| 输入Shape  |  Placeholders{...} |
| 输入Layout  | Placeholders{...}(default: ARRAY) |
| 输出数据类型 | Placeholders{...} |
| 输出Shape  | Placeholders{...} |
| 输入Layout  | Placeholders{...}(default: ARRAY) |
| 是否含有dim/axis等类似语义的参数且该参数支持负数/其他特殊处理 | Placeholder{.} |
| 是否含有labels/index等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | Placeholder{.}|
| 是否需要支持原位        | Placeholder{.}   |
| 是否需要支持stride机制  | Placeholder{.}       |
| 是否需要支持广播  | Placeholder{.}                       |
| 0元素检查是否直接返回  | Placeholder{.}                   |


### 1.2 算子功能和应用场景描述

### 1.3 算子输入输出参数要求
| 参数          | 语义 | 类型（输入/输出） | 支持类型    | 物理布局 | 规模限制 |
| ------------- | ---- | ----------------- | ----------- | -------- | -------- |
| ...... |

### 1.4 算子限制

### 1.5 验收标准

#### 1.5.1 精度验收标准

## 2 算子接口设计

### 2.1 参考接口
```python
```
or
```cuda
```

### 2.2 接口设计
```c++
```
（接口与之前的参数要一一对应）

## 3 实现方案设计

### 3.1 实现方案

### 3.2 伪代码实现
```c++
```

### 3.3 拆分

#### 3.3.1 核间拆分

#### 3.3.2 核内拆分

### 3.4 算子防呆说明

## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

### 4.2 已经过优化的规模说明
