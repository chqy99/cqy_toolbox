import os

def replace_pragma_with_include_guard(file_path, trunc_path="kernels/"):
    """将文件中的 #pragma once 替换为标准的 #ifndef/#define/#endif 头文件保护宏。

    功能描述:
        本函数用于自动将 C/C++ 头文件中的 `#pragma once` 指令替换为传统的头文件保护宏，
        宏名基于文件路径生成，使用下划线拼接路径并转为大写。适用于需要兼容不支持 `#pragma once`
        的编译器或跨平台项目的场景。

    Args:
        file_path (str): 待处理的头文件路径，可以是相对路径或绝对路径。
        trunc_path (str, optional): 用于生成宏名的路径前缀，截断该前缀后剩余部分用于构建宏名。
            默认为 "kernels/"，例如路径为 "kernels/foo/bar.h" 时，宏名将为 "FOO_BAR_H"。

    Returns:
        None: 本函数无返回值，直接修改文件内容。

    Raises:
        ValueError: 当 `trunc_path` 不是 `file_path` 的有效前缀时抛出。
        IOError: 当文件无法读取或写入时抛出。
        Exception: 其他文件操作过程中可能出现的异常。

    注意事项:
        - 函数会原地修改文件，建议提前备份重要文件。
        - 如果文件中未找到 `#pragma once`，则跳过处理并输出提示信息。
        - 宏名生成逻辑会移除路径中的 `.` 字符，以避免宏名中包含非法字符。
        - 仅适用于 UTF-8 编码的文本文件。

    示例:
        >>> replace_pragma_with_include_guard("kernels/utils/helper.h")
        文件 'kernels/utils/helper.h' 替换完成！

        >>> replace_pragma_with_include_guard("src/foo.h", trunc_path="src/")
        文件 'src/foo.h' 替换完成！
    """

    # 获取文件的绝对路径
    absolute_path = os.path.abspath(file_path)

    if trunc_path not in file_path:
        raise ValueError(f"{trunc_path} 不是 {file_path} 的有效前缀")

    # 截取路径字符串
    absolute_path = file_path.split(trunc_path, 1)[-1]

    # 替换路径分隔符为下划线
    macro_name = absolute_path.replace(os.sep, '_').replace('.', '_').upper()

    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 检查是否包含 #pragma once
        found_pragma = any(line.strip() == '#pragma once' for line in lines)

        if not found_pragma:
            print(f"文件 '{file_path}' 中没有找到 #pragma once，跳过替换。")
            return

        # 替换 #pragma once
        modified_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == '#pragma once':
                # 构造新的头文件保护
                modified_lines.append(f"#ifndef {macro_name}\n")
                modified_lines.append(f"#define {macro_name}\n\n")
            else:
                modified_lines.append(line)

        # 添加 #endif 和注释
        modified_lines.append(f"\n#endif // {macro_name}\n")

        # 将修改后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)

        print(f"文件 '{file_path}' 替换完成！")

    except Exception as e:
        print(f"处理文件 '{file_path}' 时出错: {e}")

if __name__ == "__main__":
    import argparse
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="替换 #pragma once 为头文件保护宏")
    parser.add_argument("file_path", type=str, help="要处理的文件路径")
    parser.add_argument("--trunc_path", type=str, default="kernels/", help="用于生成宏名的路径前缀，默认为 'kernels/'")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数
    replace_pragma_with_include_guard(args.file_path, args.trunc_path)