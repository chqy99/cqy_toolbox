from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. REFINED PLACEHOLDER VARIABLE
# ---------------------------------------------------------------------------
class PlaceholderVar():
    def __init__(self, name: str, sampler: Callable[[], Any]):
        self.name: str = name
        self._sampler: Callable[[], Any] = sampler

    def sample(self) -> Any:
        """采样变量值"""
        return self._sampler()

# ---------------------------------------------------------------------------
# 2. REFINED EVALUATION ENVIRONMENT
# ---------------------------------------------------------------------------
class PlaceholderEnv:
    """Manages variable sampling, caching, and expression evaluation."""

    def __init__(self, var_specs: List[PlaceholderVar]):
        self.var_specs: Dict[str, PlaceholderVar] = {v.name: v for v in var_specs}
        self._cache: Dict[str, Any] = {}

    def _draw(self, name: str) -> Any:
        if name not in self.var_specs:
            raise NameError(
                f"Variable '{name}' is used in a template but not defined in var_specs."
            )
        if name not in self._cache:
            self._cache[name] = self.var_specs[name].sample()
        return self._cache[name]

    def evaluate(self, expr: Any) -> Any:
        # 辅助函数1：递归过滤列表中的 None 元素
        def filter_list(lst):
            filtered = []
            for item in lst:
                evaluated_item = self.evaluate(item)  # 先递归解析子元素
                if evaluated_item is not None:  # 仅保留非 None 元素
                    filtered.append(evaluated_item)
            return type(lst)(filtered)  # 保持原类型（list/tuple）

        # 辅助函数2：递归过滤字典中值为 None 的键
        def filter_dict(dct):
            filtered = {}
            for k, v in dct.items():
                evaluated_k = self.evaluate(k)  # 解析键（若有P{}）
                evaluated_v = self.evaluate(v)  # 解析值（若有P{}）
                if evaluated_v is not None:  # 仅保留值为非 None 的键值对
                    filtered[evaluated_k] = evaluated_v
            return filtered

        # 1) 基础类型直接返回
        if isinstance(expr, (int, float, bool, type(None))):
            return expr

        # 2) list / tuple / dict 递归处理
        if isinstance(expr, (list, tuple)):
            return filter_list(expr)
        if isinstance(expr, dict):
            return filter_dict(expr)

        # 3) 非字符串直接透传
        if not isinstance(expr, str):
            return expr

        # 4) 字符串处理 -------------------------------------------------
        import re

        # 4-a) 如果没有 P{...} 占位符，直接返回原字符串
        if not re.search(r"P\{[^}]+\}", expr):
            return expr

        # 4-b) 统一采样变量
        var_values = {name: self._draw(name) for name in self.var_specs}

        # 4-c) 用正则把 ${var} 或 ${expr} 替换为对应的 repr 值
        def repl(m: re.Match) -> str:
            code = m.group(1)  # 花括号里的表达式
            try:
                # 在受限环境下计算表达式
                value = eval(code, {"__builtins__": {}}, var_values)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to evaluate expression '{code}' in '{expr}'. "
                    f"Variables: {var_values}. Error: {e}"
                ) from e
            return repr(value) if isinstance(value, str) else str(value)

        # 4-d) 执行替换
        replaced = re.sub(r"P\{([^}]+)\}", repl, expr)

        # 4-e) 如果替换后仍是纯数字 / 布尔 / None，再 eval 一次还原真实类型
        try:
            return eval(replaced, {"__builtins__": {}}, {})
        except Exception:
            # 说明替换后不是合法字面量，直接返回字符串即可
            return replaced

    def get_sampled_values(self) -> Dict[str, Any]:
        """Returns the current cache of sampled values."""
        # Ensure all variables are sampled before returning
        for name in self.var_specs:
            self._draw(name)
        return self._cache.copy()


# ---------------------------------------------------------------------------
# 3. GENERATOR (mostly unchanged, just uses the new PlaceholderEnv)
# ---------------------------------------------------------------------------
class SmartGenerator:
    def __init__(
        self,
        var_specs: List[PlaceholderVar],
        case_check: Optional[Callable[[Dict[str, Any]], bool]] = None,
        max_retry: int = 10,
    ):
        self.env: PlaceholderEnv = PlaceholderEnv(var_specs)
        self.case_check: Callable[[Dict[str, Any]], bool] = case_check or (lambda _: True)
        self.max_retry: int = max_retry  # 最大重新尝试轮数

    def _try_once(self, templates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A single attempt to sample and build a case dict (fully generic, no field knowledge)."""
        self.env._cache.clear()  # Crucial: clear cache for a fresh sample
        try:
            # Recursively evaluate the whole template dict
            result = self.env.evaluate(templates)
            return result
        except Exception:
            return None

    def generate(
        self, templates: Dict[str, Any], num_cases: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate `num_cases` valid case dicts that satisfy the constraints."""
        results: List[Dict[str, Any]] = []
        attempts = 0
        while len(results) < num_cases and attempts < self.max_retry * num_cases:
            item = self._try_once(templates)
            if item and self.case_check(self.env.get_sampled_values()):
                results.append(item)
            attempts += 1
        if len(results) < num_cases:
            raise RuntimeError(
                f"Failed to generate {num_cases} valid cases (only {len(results)} success after {attempts} attempts)."
            )
        return results


def GeneratorMain(
    top_level_dict: Dict[str, Any],
    manual_template: Dict[str, Any],
    num_cases: int,
    var_specs: List["PlaceholderVar"],
    case_check: Optional[Callable[[Dict[str, Any]], bool]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a top-level dict with manual_data sampled from manual_template.
    top_level_dict: 顶层字段（非 manual_data）
    manual_template: 单个 manual_data 的模板 dict
    num_cases: 生成的 case 数量
    var_specs: PlaceholderVar 列表
    case_check: 可选过滤函数
    output_path: 可选，生成的 JSON 文件保存路径
    """
    from _random_generator.generator_case import GenerateStructure, ManualDataItem, json_beautify
    import os

    gen = SmartGenerator(var_specs, case_check=case_check)
    manual_cases = gen.generate(manual_template, num_cases=num_cases)

    # manual_data 检查
    manual_data_items = []
    for item in manual_cases:
        # 只允许 inputs, outputs, extra
        inputs = item.get("inputs", []) if isinstance(item.get("inputs", []), list) else []
        outputs = item.get("outputs", []) if isinstance(item.get("outputs", []), list) else []
        extra = {k: v for k, v in item.items() if k not in ("inputs", "outputs")}
        extra = extra if extra else None
        manual_data_items.append(ManualDataItem(inputs=inputs, outputs=outputs, extra=extra))

    # 顶层结构检查
    result = dict(top_level_dict) if top_level_dict is not None else {}
    result["manual_data"] = [m.to_dict() for m in manual_data_items]
    # 用 GenerateStructure 检查整体结构
    _ = GenerateStructure(manual_data=manual_data_items, extra={k: v for k, v in result.items() if k != "manual_data"})

    # 美化 JSON
    pretty_json = json_beautify(result)

    # 写入文件
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(pretty_json)
        except Exception as e:
            print(f"[GeneratorMain] Failed to write to {output_path}: {e}")

    return result
