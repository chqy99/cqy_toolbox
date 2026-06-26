#!/usr/bin/env python3
"""docker_mgr.py — Docker 镜像管理与调度脚本

管理 dockers.jsonc 中的镜像注册表，提供列表、启动、状态、清理、提交等功能。
实际容器启动由 docker_run.sh 执行。

配置文件使用 JSONC 格式 (JSON + // 行注释 + 尾逗号)，零外部依赖。
"""

import argparse
import os
import socket
import sys
from datetime import datetime

import jsonc
import docker_ops

# 脚本所在目录，用于定位 dockers.jsonc 和 docker_run.sh
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "dockers.jsonc")
DOCKER_RUN_SH = os.path.join(SCRIPT_DIR, "docker_run.sh")


# ===============================
# 配置辅助
# ===============================

def _get_host_ip() -> str:
    """获取本机出口 IP 地址（不会真正发包）"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.0.0.1", 1))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostname()


def _get_last_docker(config: dict) -> str:
    """获取当前机器上次使用的 docker 别名"""
    host_id = _get_host_ip()
    mapping = config.get("last_docker", {})
    if isinstance(mapping, str):
        # 兼容旧格式: last_docker 是字符串
        return mapping
    return mapping.get(host_id, "")


def _set_last_docker(config: dict, alias: str) -> None:
    """设置当前机器上次使用的 docker 别名"""
    host_id = _get_host_ip()
    mapping = config.get("last_docker", {})
    if isinstance(mapping, str):
        # 兼容旧格式: 迁移到新格式
        mapping = {}
    mapping[host_id] = alias
    config["last_docker"] = mapping


def resolve_default_mounts(config: dict) -> list[str]:
    """根据 default_mounts 和 default_tool_mounts 配置，自动检测主机上存在的目录/文件，
    返回 --mount 参数列表。

    - default_mounts: 目录存在则挂载为 dir:dir (读写)
    - default_tool_mounts: 文件存在则挂载为 file:file (只读)
    """
    mounts = []
    for d in config.get("default_mounts", []):
        if os.path.isdir(d):
            mounts.append(f"{d}:{d}")
    for f in config.get("default_tool_mounts", []):
        if os.path.isfile(f):
            mounts.append(f"{f}:{f}:ro")
    return mounts


def check_mount_paths(mounts: list[str]) -> bool:
    """检查挂载路径中主机侧路径是否存在，不存在的提示用户确认。"""
    missing = []
    for m in mounts:
        host_path = m.split(":")[0]
        if not os.path.exists(host_path):
            missing.append(host_path)

    if not missing:
        return True

    print("⚠️  以下挂载路径在主机上不存在:")
    for p in missing:
        print(f"   - {p}")
    try:
        answer = input("是否继续? [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except EOFError:
        return False


def resolve_alias(images: dict, alias: str) -> str:
    """解析别名，支持模糊匹配。

    优先级：精确匹配 alias → 模糊匹配 alias → container 名 → desc
    """
    if alias in images:
        return alias

    candidates = []
    for k in images:
        if alias in k:
            candidates.append((k, "alias"))
    for k, info in images.items():
        container = info.get("container", "")
        if alias in container and not any(c[0] == k for c in candidates):
            candidates.append((k, "container"))
    for k, info in images.items():
        desc = info.get("desc", "")
        if alias.lower() in desc.lower() and not any(c[0] == k for c in candidates):
            candidates.append((k, "desc"))

    if not candidates:
        print(f"错误: 未找到匹配 '{alias}' 的 docker", file=sys.stderr)
        print(f"可用别名: {', '.join(images.keys())}", file=sys.stderr)
        sys.exit(1)

    if len(candidates) == 1:
        matched_alias, match_field = candidates[0]
        match_value = {
            "alias": matched_alias,
            "container": images[matched_alias].get("container", ""),
            "desc": images[matched_alias].get("desc", ""),
        }[match_field]
        print(f"🔍 模糊匹配: '{alias}' → {matched_alias} (匹配: {match_field}='{match_value}')")
        return matched_alias

    print(f"⚡ '{alias}' 匹配到多个 docker，请选择:")
    for i, (a, field) in enumerate(candidates, 1):
        info = images[a]
        print(f"  {i}) {a:<18} container={info.get('container', ''):<28} desc={info.get('desc', '')}")

    try:
        choice = input(f"请输入编号 (1-{len(candidates)}) 或回车取消: ").strip()
        if not choice:
            print("已取消", file=sys.stderr)
            sys.exit(0)
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx][0]
        print(f"错误: 无效编号 '{choice}'", file=sys.stderr)
        sys.exit(1)
    except (ValueError, EOFError):
        print("已取消", file=sys.stderr)
        sys.exit(1)


# ===============================
# 子命令实现
# ===============================

def cmd_list(config: dict, args) -> None:
    """列出所有已注册的 docker 镜像"""
    images = config.get("images", {})
    last = _get_last_docker(config)

    print(f"{'*':<2} {'Alias':<18} {'Dev':<5} {'Use':>4} {'Last Use':<20} {'Desc'}")
    print("-" * 100)

    for alias, info in images.items():
        is_last = "👉" if alias == last else "  "
        device = info.get("device", "mlu")
        use_count = info.get("use_count", 0)
        last_use = info.get("last_use", "")
        if not last_use or last_use.startswith("1970"):
            last_use_short = "-"
        else:
            last_use_short = last_use[:10]
        desc = info.get("desc", "")
        print(f"{is_last:<2} {alias:<18} {device:<5} {use_count:>4} {last_use_short:<20} {desc}")

    host_id = _get_host_ip()
    print(f"\nlast_docker: {last} (on {host_id})" if last else f"\n(未设置 last_docker on {host_id})")

    auto_mounts = resolve_default_mounts(config)
    if auto_mounts:
        print(f"\n自动挂载: {', '.join(auto_mounts)}")


def cmd_info(config: dict, args) -> None:
    """查看某个 docker 的详细信息"""
    alias = args.alias
    images = config.get("images", {})
    alias = resolve_alias(images, alias)

    info = images[alias]
    print(f"别名:       {alias}")
    print(f"镜像:       {info.get('image', '')}")
    print(f"容器名:     {info.get('container', '')}")
    print(f"描述:       {info.get('desc', '')}")
    print(f"设备:       {info.get('device', 'mlu')}")
    print(f"SHM Size:   {info.get('shm_size', '64g')}")
    print(f"使用次数:   {info.get('use_count', 0)}")
    last_use = info.get('last_use', '-')
    if last_use and last_use.startswith("1970"):
        last_use = '-'
    print(f"上次使用:   {last_use}")
    mounts = info.get("mounts", [])
    print(f"显式挂载:   {', '.join(mounts) if mounts else '(无)'}")
    print(f"Home 挂载:  {info.get('home_mount', False)}")
    print(f"自动删除:   {info.get('auto_remove', False)}")
    print(f"用户模式:   {info.get('user_mode', 'root')}")
    if info.get("extra_env"):
        env_str = ", ".join(f"{k}={v}" for k, v in info["extra_env"].items())
        print(f"额外环境:   {env_str}")
    if info.get("dockerfile"):
        print(f"Dockerfile: {info['dockerfile']}")

    auto_mounts = resolve_default_mounts(config)
    if auto_mounts:
        print(f"自动挂载:   {', '.join(auto_mounts)}")


def cmd_run(config: dict, args, config_path: str) -> None:
    """启动指定 docker（调用 docker_run.sh）"""
    alias = args.alias
    images = config.get("images", {})
    alias = resolve_alias(images, alias)

    info = images[alias]

    cmd = [DOCKER_RUN_SH]
    cmd.extend(["--image", info["image"]])
    cmd.extend(["--container", info["container"]])
    cmd.extend(["--device", info.get("device", "mlu")])
    cmd.extend(["--shm-size", info.get("shm_size", "64g")])

    # 挂载
    all_mounts = resolve_default_mounts(config) + info.get("mounts", [])
    if not check_mount_paths(all_mounts):
        print("已取消", file=sys.stderr)
        sys.exit(0)
    for mount in all_mounts:
        cmd.extend(["--mount", mount])

    if info.get("home_mount", False):
        cmd.append("--home-mount")
    if info.get("auto_remove", False):
        cmd.append("--auto-remove")
    if args.rm:
        cmd.append("--rm")
    if args.no_pull:
        cmd.append("--no-pull")

    cmd.extend(["--user-mode", info.get("user_mode", "root")])

    for key, val in info.get("extra_env", {}).items():
        cmd.extend(["--env", f"{key}={val}"])

    # 更新统计
    _set_last_docker(config, alias)
    info["last_use"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    info["use_count"] = info.get("use_count", 0) + 1
    jsonc.save(config_path, config)

    print(f"🚀 启动 docker: {alias} ({info['image']}) [第 {info['use_count']} 次使用]")
    os.execvp(cmd[0], cmd)


def cmd_last(config: dict, args, config_path: str) -> None:
    """快速启动上次使用的 docker"""
    last = _get_last_docker(config)
    if not last:
        print("错误: 未设置 last_docker，请先使用 'run <alias>' 启动一个容器", file=sys.stderr)
        sys.exit(1)
    print(f"📌 上次使用的 docker: {last}")
    args.alias = last
    cmd_run(config, args, config_path)


def cmd_status(config: dict, args) -> None:
    """显示所有已注册容器的运行状态"""
    images = config.get("images", {})
    all_containers = docker_ops.docker_ps_all()
    container_status = {c["name"]: c["status"] for c in all_containers}

    print(f"{'Alias':<18} {'Container':<28} {'Status'}")
    print("-" * 80)
    for alias, info in images.items():
        container = info.get("container", "")
        status = container_status.get(container, "not created")
        print(f"{alias:<18} {container:<28} {status}")


def cmd_clean(config: dict, args) -> None:
    """清理用户容器"""
    images = config.get("images", {})
    registered = {info.get("container", "") for info in images.values() if info.get("container")}
    prefix = docker_ops.infer_user_prefix(registered)

    # 确定要清理的容器列表
    if args.alias:
        # 指定别名：只清理对应的容器
        alias = resolve_alias(images, args.alias)
        container = images[alias].get("container", "")
        if not container:
            print(f"错误: 别名 '{alias}' 未配置 container", file=sys.stderr)
            sys.exit(1)
        all_containers = docker_ops.docker_ps_all()
        target = [c for c in all_containers if c["name"] == container]
        if not target:
            print(f"容器 '{container}' 不存在，无需清理")
            return
    else:
        # 清理所有用户容器
        all_containers = docker_ops.docker_ps_all()
        target = [c for c in all_containers if docker_ops.is_user_container(c["name"], registered, prefix)]
        if not target:
            print("没有找到需要清理的容器")
            return

    # 列表显示
    print(f"找到 {len(target)} 个容器:\n")
    print(f"{'Name':<32} {'Status':<24} {'In Config'}")
    print("-" * 80)
    for c in target:
        in_config = "✅" if c["name"] in registered else "❓ (旧名)"
        print(f"{c['name']:<32} {c['status']:<24} {in_config}")

    if args.dry_run:
        print("\n[dry-run] 以上容器将被 stop + rm")
        return

    if not args.yes:
        try:
            answer = input(f"\n是否 stop 并 rm 以上 {len(target)} 个容器? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("已取消")
                return
        except EOFError:
            print("已取消")
            return

    # 执行清理
    for c in target:
        name = c["name"]
        if "Up" in c["status"]:
            print(f"  ⏹  docker stop {name}")
            docker_ops.docker_stop(name)
        print(f"  🗑  docker rm {name}")
        docker_ops.docker_rm(name)

    print(f"\n✅ 已清理 {len(target)} 个容器")


def cmd_commit(config: dict, args) -> None:
    """将容器保存为镜像"""
    images = config.get("images", {})
    all_containers = docker_ops.docker_ps_all()
    container_status = {c["name"]: c["status"] for c in all_containers}

    # 确定要 commit 的目标
    if args.alias:
        alias = resolve_alias(images, args.alias)
    elif args.list:
        # --list 模式：筛选可 commit 的容器供选择
        available = []
        for a, info in images.items():
            container = info.get("container", "")
            if container in container_status:
                available.append((a, info, container_status[container]))

        if not available:
            print("没有已存在的容器可 commit")
            return

        print("可 commit 的容器:\n")
        print(f"{'#':<4} {'Alias':<18} {'Container':<28} {'Status':<20} {'Image'}")
        print("-" * 110)
        for i, (a, info, status) in enumerate(available, 1):
            img = info.get("image", "")
            if len(img) > 40:
                img = img[:18] + "..." + img[-18:]
            print(f"{i:<4} {a:<18} {info.get('container', ''):<28} {status:<20} {img}")

        try:
            choice = input(f"\n请输入编号 (1-{len(available)}) 或回车取消: ").strip()
            if not choice:
                print("已取消")
                return
            idx = int(choice) - 1
            if not (0 <= idx < len(available)):
                print(f"错误: 无效编号 '{choice}'", file=sys.stderr)
                sys.exit(1)
            alias = available[idx][0]
        except (ValueError, EOFError):
            print("已取消")
            return
    else:
        print("错误: 请指定别名或使用 --list 选择", file=sys.stderr)
        sys.exit(1)

    info = images[alias]
    container = info.get("container", "")
    image = info.get("image", "")

    if not container:
        print(f"错误: 别名 '{alias}' 未配置 container", file=sys.stderr)
        sys.exit(1)

    if container not in container_status:
        print(f"错误: 容器 '{container}' 不存在，请先 run", file=sys.stderr)
        sys.exit(1)

    message = args.message or ""

    if args.dry_run:
        print(f"[dry-run] docker commit {container} → {image}" + (f" -m '{message}'" if message else ""))
        if args.push:
            print(f"[dry-run] docker push {image}")
        return

    # 执行 commit
    print(f"📦 docker commit {container} → {image}")
    if not docker_ops.docker_commit(container, image, message):
        sys.exit(1)
    print(f"✅ 已保存为镜像: {image}")

    if args.push:
        print(f"📤 docker push {image}")
        if not docker_ops.docker_push(image):
            sys.exit(1)
        print(f"✅ 已推送到 registry")


# ===============================
# 主入口
# ===============================

def main():
    parser = argparse.ArgumentParser(
        prog="docker_mgr",
        description="Docker 镜像管理与调度工具 — 管理 dockers.jsonc，调用 docker_run.sh 启动容器",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"配置路径 (默认: {DEFAULT_CONFIG})")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有已注册的 docker 镜像 (等同于 list 子命令)")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # list
    p_list = subparsers.add_parser("list", aliases=["ls"], help="列出所有已注册的 docker 镜像")
    p_list.set_defaults(func=cmd_list)

    # run
    p_run = subparsers.add_parser("run", help="启动指定 docker")
    p_run.add_argument("alias", help="docker 别名，支持模糊匹配")
    p_run.add_argument("--rm", action="store_true", help="容器退出时自动删除 (覆盖 auto_remove)")
    p_run.add_argument("--no-pull", action="store_true", help="跳过启动前的 docker pull")
    p_run.set_defaults(func=cmd_run)

    # last
    p_last = subparsers.add_parser("last", help="快速启动上次使用的 docker")
    p_last.add_argument("--rm", action="store_true", help="容器退出时自动删除")
    p_last.add_argument("--no-pull", action="store_true", help="跳过启动前的 docker pull")
    p_last.set_defaults(func=cmd_last)

    # info
    p_info = subparsers.add_parser("info", help="查看某个 docker 的详细信息")
    p_info.add_argument("alias", help="docker 别名，支持模糊匹配")
    p_info.set_defaults(func=cmd_info)

    # status
    p_status = subparsers.add_parser("status", help="显示所有容器运行状态")
    p_status.set_defaults(func=cmd_status)

    # clean
    p_clean = subparsers.add_parser("clean", help="清理用户容器 (stop + rm)")
    p_clean.add_argument("alias", nargs="?", help="只清理指定别名（支持模糊匹配），不写则清理所有用户容器")
    p_clean.add_argument("--dry-run", action="store_true", help="只列出，不执行 stop/rm")
    p_clean.add_argument("-y", "--yes", action="store_true", help="跳过确认，直接清理")
    p_clean.set_defaults(func=cmd_clean)

    # commit
    p_commit = subparsers.add_parser("commit", help="保存容器为镜像")
    p_commit.add_argument("alias", nargs="?", help="docker 别名（支持模糊匹配），不写则需配合 --list")
    p_commit.add_argument("-m", "--message", default="", help="commit message")
    p_commit.add_argument("--push", action="store_true", help="commit 后 push 到 registry")
    p_commit.add_argument("--dry-run", action="store_true", help="只显示将要执行的操作")
    p_commit.add_argument("--list", action="store_true", help="列出可 commit 的容器并选择")
    p_commit.set_defaults(func=cmd_commit)

    args = parser.parse_args()

    if not args.command and not args.list:
        parser.print_help()
        sys.exit(0)

    config = jsonc.load(args.config)

    # --list 顶层快捷方式
    if args.list:
        cmd_list(config, args)
        return

    # 分发子命令
    cmd_map = {
        "list": cmd_list, "ls": cmd_list,
        "run": cmd_run,
        "last": cmd_last,
        "info": cmd_info,
        "status": cmd_status,
        "clean": cmd_clean,
        "commit": cmd_commit,
    }
    handler = cmd_map.get(args.command)
    if handler:
        # run/last 需要 config_path
        if args.command in ("run", "last"):
            handler(config, args, args.config)
        else:
            handler(config, args)


if __name__ == "__main__":
    main()
