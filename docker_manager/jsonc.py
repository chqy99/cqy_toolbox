#!/usr/bin/env python3
"""jsonc.py — JSONC 解析模块

支持 // 行注释和尾逗号的 JSON 解析，零外部依赖。
"""

import json
import re


def strip_comments(text: str) -> str:
    """剥离 JSONC 中的 // 行注释，保留字符串内的 //。

    逐字符扫描，遇到 " 进入字符串模式，字符串内不剥离；
    字符串外遇到 // 则删除到行尾。
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '"':
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            result.append(text[i:j])
            i = j
        elif text[i:i + 2] == '//':
            end = text.find('\n', i)
            if end == -1:
                break
            i = end
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def strip_trailing_commas(text: str) -> str:
    """剥离 JSON 中的尾逗号：, ] 或 , } → ] 或 }"""
    return re.sub(r',\s*([}\]])', r'\1', text)


def load(path: str) -> dict:
    """加载 JSONC 配置文件（支持 // 注释和尾逗号）"""
    with open(path, "r") as f:
        raw = f.read()
    cleaned = strip_trailing_commas(strip_comments(raw))
    return json.loads(cleaned)


def save(path: str, data: dict) -> None:
    """保存数据到 JSON 文件（保持可读格式，带尾部换行）"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
