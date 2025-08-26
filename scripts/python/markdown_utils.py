import os
from typing import Dict, List, Optional, Union
import markdown
from bs4 import BeautifulSoup

class MarkdownAnalyzer:
    """Markdown 文件分析工具类，支持提取标题结构、指定标题内容等操作。

    功能描述:
        该类封装了多个 Markdown 文件分析方法，包括提取标题层级结构、提取指定标题内容等，
        适用于文档自动化处理、内容提取等场景。

    Args:
        directory (str): 要分析的 Markdown 文件所在目录路径

    Raises:
        FileNotFoundError: 如果目录不存在
        PermissionError: 如果没有权限访问目录或文件

    注意事项:
        - 默认递归扫描子目录中的所有 .md 文件
        - 使用 markdown + BeautifulSoup 解析 Markdown，支持标准语法

    示例:
        >>> analyzer = MarkdownAnalyzer("/path/to/docs")
        >>> structure = analyzer.extract_heading_structure()
        >>> print(structure)
        {
            "file1.md": [
                {"level": 1, "text": "第一章", "children": [
                    {"level": 2, "text": "1.1 节", "children": []}
                ]}
            ]
        }
    """

    def __init__(self, directory: str):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")
        self.directory = directory

    def _parse_markdown_file(self, filepath: str) -> BeautifulSoup:
        """解析 Markdown 文件为 BeautifulSoup 对象"""
        with open(filepath, 'r', encoding='utf-8') as f:
            html = markdown.markdown(f.read())
        return BeautifulSoup(html, 'html.parser')

    def extract_heading_structure(self) -> Dict[str, List[Dict]]:
        """提取所有 Markdown 文件的标题层级结构（目录大纲）。

        Returns:
            Dict[str, List[Dict]]: 返回字典，键为文件路径，值为标题结构列表（嵌套字典）
        """
        results = {}
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if not filename.endswith('.md'):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    soup = self._parse_markdown_file(filepath)
                    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    structure = []
                    stack = []

                    for tag in headings:
                        level = int(tag.name[1])
                        text = tag.get_text().strip()
                        node = {"level": level, "text": text, "children": []}

                        # 调整栈，确保当前节点的父节点是栈顶
                        while stack and stack[-1]["level"] >= level:
                            stack.pop()

                        if stack:
                            stack[-1]["children"].append(node)
                        else:
                            structure.append(node)

                        stack.append(node)

                    results[filepath] = structure

                except (IOError, UnicodeDecodeError) as e:
                    print(f"警告: 无法读取文件 {filepath}: {str(e)}")
                    continue

        return results

    def extract_heading_content(self, target_heading: str) -> Dict[str, List[str]]:
        """提取所有 Markdown 文件中指定标题下的内容。

        Args:
            target_heading (str): 要提取的目标标题文本（精确匹配）

        Returns:
            Dict[str, List[str]]: 返回字典，键为文件路径，值为匹配到的内容列表
        """
        results = {}
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if not filename.endswith('.md'):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    soup = self._parse_markdown_file(filepath)
                    sections = []
                    current_heading = None
                    current_content = []

                    for element in soup.children:
                        if element.name and element.name.startswith('h') and element.get_text().strip() == target_heading:
                            if current_heading is not None:
                                sections.append('\n'.join(current_content))
                                current_content = []
                            current_heading = element
                        elif current_heading is not None:
                            current_content.append(str(element))

                    if current_content:
                        sections.append('\n'.join(current_content))

                    results[filepath] = sections

                except (IOError, UnicodeDecodeError) as e:
                    print(f"警告: 无法读取文件 {filepath}: {str(e)}")
                    continue

        return results


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Markdown 文件分析工具，支持提取标题结构和指定标题内容",
        epilog="示例：\n" +
               "  python markdown_analyzer.py /path/to/docs --mode structure\n" +
               "  python markdown_analyzer.py /path/to/docs --mode content --heading '注意事项'",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "directory",
        help="要分析的 Markdown 文件所在目录路径"
    )

    parser.add_argument(
        "--mode",
        choices=["structure", "content"],
        required=True,
        help="选择运行模式：\n" +
             "  structure: 提取所有 Markdown 文件的标题层级结构\n" +
             "  content: 提取指定标题下的内容（需配合 --heading 使用）"
    )

    parser.add_argument(
        "--heading",
        help="在 content 模式下，指定要提取的标题文本（精确匹配）"
    )

    args = parser.parse_args()

    try:
        analyzer = MarkdownAnalyzer(args.directory)

        if args.mode == "structure":
            structure = analyzer.extract_heading_structure()
            print(json.dumps(structure, ensure_ascii=False, indent=2))

        elif args.mode == "content":
            if not args.heading:
                print("错误：content 模式下必须使用 --heading 指定目标标题")
                exit(1)
            content = analyzer.extract_heading_content(args.heading)
            print(json.dumps(content, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)
