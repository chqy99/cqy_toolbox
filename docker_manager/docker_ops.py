#!/usr/bin/env python3
"""docker_ops.py — Docker 操作工具模块

提供容器查询、停止、删除、提交等通用操作，零外部依赖。
"""

import subprocess
import sys


def docker_ps_all() -> list[dict]:
    """获取所有容器信息，返回 [{name, status, image}, ...]"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"],
            capture_output=True, text=True, check=True,
        )
        containers = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                containers.append({
                    "name": parts[0],
                    "status": parts[1],
                    "image": parts[2] if len(parts) >= 3 else "",
                })
        return containers
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 无法执行 docker ps，请确认 docker 已安装且可用", file=sys.stderr)
        sys.exit(1)


def docker_stop(name: str) -> bool:
    """停止容器，返回是否成功"""
    result = subprocess.run(["docker", "stop", name], capture_output=True)
    return result.returncode == 0


def docker_rm(name: str) -> bool:
    """删除容器，返回是否成功"""
    result = subprocess.run(["docker", "rm", name], capture_output=True)
    return result.returncode == 0


def docker_commit(container: str, image: str, message: str = "") -> bool:
    """将容器提交为镜像，返回是否成功"""
    cmd = ["docker", "commit"]
    if message:
        cmd.extend(["-m", message])
    cmd.extend([container, image])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ docker commit 失败: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def docker_push(image: str) -> bool:
    """推送镜像到 registry，返回是否成功"""
    result = subprocess.run(["docker", "push", image], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ docker push 失败: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def infer_user_prefix(registered: set[str]) -> str:
    """从注册容器名推断用户前缀。

    取所有容器名的最长公共前缀到第一个 _ 之后。
    如 ["chqy_gen", "chqy_torch"] → "chqy_"
    """
    if not registered:
        return ""
    names = sorted(registered)
    first, last = names[0], names[-1]
    # 找公共前缀
    prefix = []
    for a, b in zip(first, last):
        if a == b:
            prefix.append(a)
        else:
            break
    common = ''.join(prefix)
    # 截取到最后一个 _ 之后（包含 _）
    idx = common.rfind('_')
    if idx >= 0:
        return common[:idx + 1]
    return common


def is_user_container(name: str, registered: set[str], prefix: str) -> bool:
    """判断容器是否属于当前用户"""
    if name in registered:
        return True
    if prefix and name.startswith(prefix):
        return True
    return False
