#!/bin/bash
# docker_run.sh — 纯执行层，接收参数启动/进入 docker 容器
# 不关心 YAML 配置，由 docker_mgr.py 调用或手动使用
#
# 用户/权限处理吸收自 spica/tools/run_in_docker.sh，解决 docker 内外权限/用户不一致问题：
#   - 组名清洗：LDAP 组名含特殊字符时自动替换
#   - gosu 优先：比 su 更可靠地传递信号和 PTY
#   - HOME=/tmp/$USER：避免 /home 权限问题
#   - PATH 修复：su 切换用户后不丢失 PATH
#   - 主机环境变量透传：容器内保留 PATH、自定义变量等
#   - sudo 安全拦截：容器内 sudo 时打印警告
#   - PS1 标识：prompt 前缀显示 (docker)，一眼区分内外
#   - 容器名+镜像一致性检查：防止误进旧容器
#   - 启动前自动 pull：docker pull 更新镜像，--no-pull 跳过

set -e

PROG_NAME=$(basename "$0")

# ===============================
# 默认值
# ===============================
IMAGE=""
CONTAINER=""
DEVICE="mlu"       # mlu (默认) / gpu / cpu
SHM_SIZE="64g"
MOUNTS=()
HOME_MOUNT=false
AUTO_REMOVE=false
FORCE_RM=false
USER_MODE="root"   # root: 以 root 进入后创建用户切换; direct: 以 -u uid:gid 直接进入
EXTRA_ENV=()
TEMP_FILES=()
DO_PULL=true       # 默认启动前自动 pull

MY_NAME=$(whoami)
MY_UID=$(id -u)
MY_GID=$(id -g)
MY_GROUP=$(id -g -n)
CONTAINER_HOME="/tmp/$MY_NAME"  # 容器内 HOME 目录，避免 /home 权限问题

# ===============================
# 清理临时文件
# ===============================
cleanup() {
    rm -rf "${TEMP_FILES[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# ===============================
# 帮助信息
# ===============================
usage() {
    cat <<EOF
Usage: $PROG_NAME [options]

启动或进入 docker 容器。若容器不存在则创建并进入，若已存在则重启并进入。

设备类型 (--device):
  mlu   --privileged (默认，寒武纪 MLU 设备)
  gpu   --privileged --gpus all --ulimit (NVIDIA GPU)
  cpu   无设备透传

用户模式 (--user-mode):
  root    以 root 进入，创建同名用户后通过 gosu/su 切换 (默认，推荐)
          自动处理：组名清洗、HOME 设置、PATH 修复、sudo 配置、环境变量透传
  direct  以 -u uid:gid 直接进入，跳过所有用户配置

安全检查:
  - 容器已存在时，校验镜像 hash 是否一致，不一致则报错退出
  - 启动前自动 docker pull 更新镜像 (可用 --no-pull 跳过)

Options:
  -i, --image IMAGE        Docker 镜像名 (必填)
  -n, --container NAME     容器名 (必填)
  -d, --device TYPE        设备类型: mlu (默认) / gpu / cpu
  -s, --shm-size SIZE      共享内存大小 (默认: 64g)
  -m, --mount HOST:CONT    挂载目录 (可多次指定)
      --home-mount         挂载 /tmp/\$USER_home 到容器 HOME (/tmp/\$USER)
                           用于持久化 bash 历史、配置等
      --auto-remove        容器退出时自动删除 (--rm)
      --rm                 强制启用 --rm (优先级高于 --auto-remove)
  -u, --user-mode MODE     用户模式: root (默认) 或 direct
  -e, --env KEY=VAL        设置环境变量 (可多次指定)
      --no-pull            跳过启动前的 docker pull
  -h, --help               显示帮助

Examples:
  # MLU 设备 (默认)
  $PROG_NAME -i pytorch_mlu:latest -n my_mlu

  # GPU 设备，带挂载和自动删除
  $PROG_NAME -i chqy_pt2.8:latest -n my_dev \\
      --device gpu --shm-size 128g --rm \\
      --mount /MLU_OPS:/MLU_OPS --home-mount

  # 纯 CPU
  $PROG_NAME -i ubuntu:22.04 -n my_container --device cpu

  # 跳过自动 pull
  $PROG_NAME -i my_image:latest -n my_container --no-pull

  # 直接以当前用户身份进入 (跳过用户创建)
  $PROG_NAME -i ubuntu:22.04 -n my_container -u direct
EOF
    exit "${1:-0}"
}

# ===============================
# 参数解析
# ===============================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--image)       IMAGE="$2"; shift 2 ;;
        -n|--container)   CONTAINER="$2"; shift 2 ;;
        -d|--device)      DEVICE="$2"; shift 2 ;;
        -s|--shm-size)    SHM_SIZE="$2"; shift 2 ;;
        -m|--mount)       MOUNTS+=("$2"); shift 2 ;;
        --home-mount)     HOME_MOUNT=true; shift ;;
        --auto-remove)    AUTO_REMOVE=true; shift ;;
        --rm)             FORCE_RM=true; shift ;;
        -u|--user-mode)   USER_MODE="$2"; shift 2 ;;
        -e|--env)         EXTRA_ENV+=("$2"); shift 2 ;;
        --no-pull)        DO_PULL=false; shift ;;
        -h|--help)        usage 0 ;;
        *)                echo "未知选项: $1"; usage 1 ;;
    esac
done

# 校验必填参数
if [[ -z "$IMAGE" ]]; then
    echo "错误: 必须指定 --image" >&2
    usage 1
fi
if [[ -z "$CONTAINER" ]]; then
    echo "错误: 必须指定 --container" >&2
    usage 1
fi

# 校验 device 参数
case "$DEVICE" in
    mlu|gpu|cpu) ;;
    *) echo "错误: --device 必须为 mlu/gpu/cpu, 当前值: $DEVICE" >&2; usage 1 ;;
esac

# ===============================
# 容器名+镜像一致性检查
# ===============================
if docker ps -a --format '{{.Names}}' | grep -qxF "$CONTAINER"; then
    # 获取已有容器的镜像 hash
    existing_img=$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || true)
    # 获取目标镜像的 hash (如果本地有的话)
    target_img=$(docker inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)

    if [[ -n "$existing_img" && -n "$target_img" && "$existing_img" != "$target_img" ]]; then
        existing_name=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null || true)
        echo "❌ 错误: 容器 '$CONTAINER' 已存在，但镜像不一致!" >&2
        echo "   容器当前镜像: $existing_name ($existing_img)" >&2
        echo "   目标镜像:     $IMAGE ($target_img)" >&2
        echo "   建议: 使用其他容器名，或先删除已有容器 (docker rm $CONTAINER)" >&2
        exit 1
    fi

    echo "容器 '$CONTAINER' 已存在，重启并进入..."
    docker restart "$CONTAINER"
    until [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER") == true ]]; do
        sleep 0.5
    done
    exec docker exec -u "$(id -u):$(id -g)" -it "$CONTAINER" /bin/bash
fi

# ===============================
# 启动前自动 pull
# ===============================
if [[ "$DO_PULL" == true ]]; then
    echo "🔄 尝试更新镜像: $IMAGE"
    docker pull "$IMAGE" 2>/dev/null || echo "⚠️  docker pull 失败，使用本地镜像继续"
else
    echo "⏭️  跳过 docker pull (--no-pull)"
fi

# 确认镜像存在
if ! docker inspect -f '{{.Id}}' "$IMAGE" >/dev/null 2>&1; then
    echo "❌ 错误: 镜像 '$IMAGE' 不存在" >&2
    exit 1
fi

# ===============================
# 生成主机环境变量文件
# ===============================
generate_host_env() {
    local envfile
    envfile=$(mktemp --suffix _docker_env.sh -t tmpXXX)
    TEMP_FILES+=("$envfile")

    # 过滤并导出环境变量（排除不应透传的变量）
    if printenv -0 >/dev/null 2>&1; then
        printenv -0 | while IFS='=' read -r -d '' k v; do
            case "$k" in
                LD_*|LC_*|PATH|HOME|USER|LS_COLORS|HOSTNAME|PWD|PROMPT_COMMAND|PS1|SHLVL|XDG_*|TERM|DBUS_*|DISPLAY|WAYLAND_*|SSH_*|OLDPWD|SHELL|LOGNAME|MAIL|LESSOPEN|LESSCLOSE|_)
                    ;;
                *)
                    printf 'export %s=%q\n' "$k" "$v" >> "$envfile"
                    ;;
            esac
        done
    fi

    # 导出 alias
    alias -p 2>/dev/null >> "$envfile" || true

    echo "$envfile"
}

# ===============================
# 生成入口脚本（解决用户/权限问题）
# ===============================
generate_entrypoint() {
    local entry
    entry=$(mktemp --suffix _docker_entry.sh -t tmpXXX)
    TEMP_FILES+=("$entry")

    # 注意: 此 heredoc 没有 'EOF' 引号，宿主机变量会被展开
    # 容器端变量用 \$ 转义
    cat > "$entry" <<EOF
#!/bin/bash
set -e

echo "|NOTE| 已进入容器环境，提醒：在\"非挂载进来的目录\"里产生的数据在容器销毁时都会被清除"

# ============ PS1 定制：prompt 前缀显示 (docker) ============
cat >> /etc/skel/.bashrc << 'PS1EOF'

_docker_prompt() {
  if [ "\${VIRTUAL_ENV+x}" ]; then
    echo -e "\033[1;36m($(basename "\$VIRTUAL_ENV")) \033[0m"
  else
    echo -e "\033[1;31m(docker)\033[0m"
  fi
}
PS1='\$(_docker_prompt)\${debian_chroot:+(\$debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\\$ '
PS1EOF

# ============ 组名清洗：LDAP 组名可能含特殊字符 ============
sanitize_group_name() {
    local raw="\$1"
    echo "\$raw" | sed -E 's/[^a-zA-Z0-9_-]/_/g'
}
main_group=\$(sanitize_group_name "$MY_GROUP")
groupadd -f -g $MY_GID \$main_group 2>/dev/null || true
if [ "\$main_group" != "$MY_GROUP" ]; then
    main_group=docker
    groupadd -f docker 2>/dev/null || true
fi

# ============ 创建用户：UID/GID 与宿主机一致 ============
groupadd -f sudo-without-passwd 2>/dev/null || true
useradd -s /bin/bash -o -m -d $CONTAINER_HOME -u $MY_UID -g \$main_group -G root,sudo-without-passwd $MY_NAME 2>/dev/null || true

# ============ sudo 配置：使用 sudoers.d，不破坏原有配置 ============
mkdir -p /etc/sudoers.d
cat > /etc/sudoers.d/99_docker_sudoers << 'SUEOF'
%sudo-without-passwd ALL=(ALL:ALL) NOPASSWD: ALL
SUEOF
chmod 0440 /etc/sudoers.d/99_docker_sudoers

# sudo wrapper：拦截 sudo 调用，打印安全警告
mkdir -p /usr/local/sbin
cat > /usr/local/sbin/sudo << 'SUEOF'
#!/bin/bash
echo -e "\033[1;31m======================================================"
echo -e "⚠️  注意：您正在 Docker 容器内执行 sudo 命令"
echo -e "   - 容器内的更改在删除容器后不会保留"
echo -e "   - \033[1;33m请勿修改挂载进来的宿主机目录内容，以免造成数据污染\033[0m"
echo -e "=========================================================\033[0m"
exec /usr/bin/sudo "\$@"
SUEOF
chmod +x /usr/local/sbin/sudo

# ============ PATH 修复：su 切换用户后不丢失 PATH ============
sed -r -i '/^ENV_PATH/d' /etc/login.defs 2>/dev/null || true
sed -r -i '/^ENV_SUPATH/d' /etc/login.defs 2>/dev/null || true
echo "ENV_PATH PATH=\$PATH" >> /etc/login.defs
echo "ENV_SUPATH PATH=\$PATH" >> /etc/login.defs

# ============ 环境变量透传：将主机环境变量持久化到容器内 ============
if [ -f /.host_env ]; then
    cp /.host_env /etc/docker_host_env 2>/dev/null || true
fi

cat >> /etc/bash.bashrc << 'ENVEOF'
source /etc/docker_host_env 2>/dev/null || true
alias sudo='sudo -E'
ENVEOF

# ============ 初始化用户环境 ============
if [ ! -f $CONTAINER_HOME/.bashrc ]; then
    cp /etc/skel/.bashrc $CONTAINER_HOME/.bashrc 2>/dev/null || \
    cp /etc/bash.bashrc $CONTAINER_HOME/.bashrc 2>/dev/null || true
fi
chown -R $MY_UID:$MY_GID $CONTAINER_HOME 2>/dev/null || true

# ============ /torch 加写权限（如存在） ============
if [ -d /torch ]; then
    chmod -R g+w /torch 2>/dev/null &
fi

# ============ 容器名快捷命令 ============
echo '#!/bin/bash' > /bin/myname
echo "echo $CONTAINER" >> /bin/myname
chmod +x /bin/myname

# ============ 生成用户入口脚本 ============
echo '#!/bin/bash' > /tmp/user_entry.sh
echo 'set -e' >> /tmp/user_entry.sh
echo "cd $CONTAINER_HOME" >> /tmp/user_entry.sh
echo 'if [ "$HOME" = "/root" ]; then export HOME='"$CONTAINER_HOME"'; fi' >> /tmp/user_entry.sh
echo 'exec bash' >> /tmp/user_entry.sh
chmod +x /tmp/user_entry.sh

# ============ 切换用户：优先 gosu，回退 su ============
if command -v gosu >/dev/null 2>&1; then
    gosu $MY_NAME /tmp/user_entry.sh
elif [ -x /usr/local/sbin/gosu ]; then
    /usr/local/sbin/gosu $MY_NAME /tmp/user_entry.sh
else
    su -p -l $MY_NAME -c /tmp/user_entry.sh
fi
EOF

    chmod +x "$entry"
    echo "$entry"
}

# ===============================
# 容器不存在 → docker run
# ===============================
echo "容器 '$CONTAINER' 不存在，创建并启动 (image: $IMAGE, device: $DEVICE)..."

# 生成环境变量文件和入口脚本
HOST_ENV_FILE=$(generate_host_env)
ENTRY_FILE=$(generate_entrypoint)

# 组装 docker run 参数
RUN_ARGS=()

# --rm 逻辑
if [[ "$FORCE_RM" == true || "$AUTO_REMOVE" == true ]]; then
    RUN_ARGS+=(--rm)
fi

RUN_ARGS+=(
    -it
    --name "$CONTAINER"
    --hostname "$CONTAINER"
    --network=host
    --ipc=host
    --pid=host
    --shm-size "$SHM_SIZE"
    -e USER="$MY_NAME"
    -e PYTHONIOENCODING=utf-8
)

# 设备透传
case "$DEVICE" in
    mlu)
        RUN_ARGS+=(--privileged)
        ;;
    gpu)
        RUN_ARGS+=(--privileged)
        RUN_ARGS+=(--gpus all)
        RUN_ARGS+=(--ulimit memlock=-1 --ulimit stack=67108864)
        ;;
    cpu)
        # 无设备透传
        ;;
esac

# 挂载
for m in "${MOUNTS[@]}"; do
    RUN_ARGS+=(-v "$m")
done

# Home 挂载：挂载到容器 HOME ($CONTAINER_HOME = /tmp/$USER)
if [[ "$HOME_MOUNT" == true ]]; then
    RUN_ARGS+=(-v "/tmp/${MY_NAME}_home:$CONTAINER_HOME")
fi

# 环境变量
for e in "${EXTRA_ENV[@]}"; do
    RUN_ARGS+=(-e "$e")
done

# 挂载入口脚本和环境变量文件
RUN_ARGS+=(-v "$ENTRY_FILE:/.entry_wrapper.sh:ro")
RUN_ARGS+=(-v "$HOST_ENV_FILE:/.host_env:ro")

# 挂载 gosu（如果宿主机上有）
GOSU_PATH=""
if command -v gosu >/dev/null 2>&1; then
    GOSU_PATH="$(which gosu)"
elif [[ -x /tools/container/gosu-$(uname -m) ]]; then
    GOSU_PATH="/tools/container/gosu-$(uname -m)"
fi
if [[ -n "$GOSU_PATH" ]]; then
    RUN_ARGS+=(-v "$GOSU_PATH:/usr/local/sbin/gosu:ro")
fi

# ===============================
# 用户模式: direct → 以当前用户身份直接进入
# ===============================
if [[ "$USER_MODE" == "direct" ]]; then
    RUN_ARGS+=(-u "$MY_UID:$MY_GID")
    exec docker run "${RUN_ARGS[@]}" "$IMAGE" /bin/bash
fi

# ===============================
# 用户模式: root → 以 root 进入，执行 entrypoint 创建用户后切换
# ===============================
RUN_ARGS+=(-u root)
exec docker run "${RUN_ARGS[@]}" "$IMAGE" bash -c "/.entry_wrapper.sh"
