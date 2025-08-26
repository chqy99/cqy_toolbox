import os
import shutil
import re
from typing import List, Optional, Set
from pathlib import Path

class TextFileProcessor:
    """文本文件批量处理工具类，支持正则文本替换。

    功能描述:
    该类默认使用 replacetext 规则，支持正则表达式替换文本内容。
    用户必须提供两个正则表达式参数：old_str 和 new_str。
    默认备份原文件，除非指定 --no-backup。

    参数:
    - path: 文件或目录路径
    - args: 必须提供两个参数：old_str 和 new_str
    - nobackup: 是否跳过备份原文件（默认备份）
    - extensions: 要处理的文件扩展名（默认 .h, .hpp, .c, .cpp）
    """

    def __init__(
        self,
        path: str,
        args: List[str],
        nobackup: bool = False,
        extensions: Optional[Set[str]] = None,
    ):
        self.path = Path(path)
        self.args = args
        self.nobackup = nobackup
        self.extensions = extensions or {".h", ".hpp", ".c", ".cpp"}
        self.processed_files: Set[Path] = set()

        if len(self.args) != 2:
            raise ValueError("必须提供两个参数：old_str 和 new_str")

    def _replace_text(self, content: str) -> str:
        """使用正则表达式替换文本"""
        old_str, new_str = self.args
        return re.sub(old_str, new_str, content)

    def _backup_file(self, file_path: Path) -> None:
        """备份原文件（除非 nobackup=True）"""
        if self.nobackup:
            return
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        shutil.copy2(file_path, backup_path)
        print(f"备份: {file_path} -> {backup_path}")

    def _process_file(self, file_path: Path) -> None:
        """处理单个文件"""
        if file_path.suffix not in self.extensions:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"警告: 无法读取文件 {file_path}: {e}")
            return

        new_content = self._replace_text(content)
        if new_content == content:
            print(f"无变化: {file_path}")
            return

        self._backup_file(file_path)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            self.processed_files.add(file_path)
            print(f"处理完成: {file_path}")
        except (PermissionError, OSError) as e:
            print(f"错误: 无法写入文件 {file_path}: {e}")

    def process(self) -> None:
        """处理文件或目录"""
        if not self.path.exists():
            raise FileNotFoundError(f"路径不存在: {self.path}")

        if self.path.is_file():
            self._process_file(self.path)
        elif self.path.is_dir():
            for file_path in self.path.rglob("*"):
                if file_path.is_file():
                    self._process_file(file_path)
        else:
            raise ValueError(f"无效路径: {self.path}")

    def summary(self) -> None:
        """打印处理摘要"""
        print("\n处理摘要:")
        print(f"处理文件数: {len(self.processed_files)}")
        for file_path in self.processed_files:
            print(f"- {file_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="文本文件批量替换工具")
    parser.add_argument("path", help="文件或目录路径")
    parser.add_argument("old_str", help="要替换的正则表达式")
    parser.add_argument("new_str", help="替换后的字符串")
    parser.add_argument("--no-backup", action="store_true", help="跳过备份原文件")
    parser.add_argument("--extensions", nargs="*", default=[".h", ".hpp", ".c", ".cpp"], help="要处理的文件扩展名")

    args = parser.parse_args()

    processor = TextFileProcessor(
        path=args.path,
        args=[args.old_str, args.new_str],
        nobackup=args.no_backup,
        extensions=set(args.extensions),
    )

    try:
        processor.process()
        processor.summary()
    except Exception as e:
        print(f"错误: {e}")
