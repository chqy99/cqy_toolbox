from pathlib import Path
import datetime
import platform
import argparse
from typing import List, Tuple, Union, Optional


def scan_dir_by_ctime(
    directory: Union[str, Path],
    *,
    include_subdirs: bool = False,
    return_datetime: bool = False
) -> List[Tuple[Path, Union[str, datetime.datetime]]]:
    """
    扫描指定目录下所有文件的创建时间，并按创建时间倒序返回结果。

    功能描述:
        遍历给定目录（或递归子目录）中的所有文件，获取其创建时间，
        然后按创建时间从晚到早排序并返回。

    Args:
        directory (Union[str, Path]): 需要扫描的目录路径。
        include_subdirs (bool, 可选): 是否递归包含子目录中的文件，默认 False。
        return_datetime (bool, 可选): 若 True 返回 datetime 对象；False 返回格式化字符串，默认 False。

    Returns:
        List[Tuple[Path, Union[str, datetime.datetime]]]:
            返回一个列表，每个元素为 (Path 对象, 创建时间)。

    Raises:
        FileNotFoundError: 指定的目录不存在。
        NotADirectoryError: 指定的路径不是目录。

    注意事项:
        - 在 Windows 上，返回的是真正的“创建时间”；
          在 Unix 系统上，优先使用 st_birthtime（若文件系统支持），否则退回到
          st_ctime（状态变更时间）。
        - 遇到同名符号链接不会进入循环。
        - 返回结果仅包含普通文件（is_file() 为 True）。

    示例:
        >>> scan_dir_by_ctime("/tmp", include_subdirs=True, return_datetime=False)
        [(PosixPath('/tmp/b.txt'), '2025-08-28 15:30:00'), ...]
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录: {directory}")

    # 根据平台选择获取创建时间的函数
    system = platform.system()
    if system == "Windows":
        def _get_ctime(p: Path) -> float:
            return p.stat().st_ctime
    else:
        def _get_ctime(p: Path) -> float:
            st = p.stat()
            return getattr(st, "st_birthtime", st.st_ctime)

    # 选择遍历方式
    iterator = directory.rglob("*") if include_subdirs else directory.iterdir()

    records: List[Tuple[Path, Union[str, datetime.datetime]]] = []
    for item in iterator:
        if item.is_file():
            ctime_ts = _get_ctime(item)
            ctime = (
                datetime.datetime.fromtimestamp(ctime_ts)
                if return_datetime
                else datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")
            )
            records.append((item, ctime))

    # 按创建时间倒序
    records.sort(key=lambda t: t[1], reverse=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="扫描目录下所有文件的创建时间，并按倒序输出。"
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="要扫描的目录路径，默认为当前目录"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归包含子目录中的文件"
    )
    parser.add_argument(
        "-dt",
        "--datetime",
        action="store_true",
        help="以原始 datetime 对象形式输出（默认输出格式化字符串）"
    )

    args = parser.parse_args()

    try:
        results = scan_dir_by_ctime(
            directory=args.directory,
            include_subdirs=args.recursive,
            return_datetime=args.datetime
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"错误: {e}")
    else:
        for path, ctime in results:
            print(f"{ctime}\t{path}")
