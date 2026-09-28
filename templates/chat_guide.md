# AI 终端命令执行系统

## 1. 背景
在部分研发环境中（如远端服务器、Docker或虚拟环境），出于安全或环境一致性考虑，AI无法直接通过SSH连接。这需要一种方式让AI能够向已由人工配置好的本地或远端终端发送命令，并可靠地获取执行结果。

## 2. 方案
本方案通过GUI自动化工具 `xdotool` 模拟键盘输入，将AI生成的命令发送到指定的终端窗口执行，并通过文件系统同步执行状态和输出结果。
核心机制与优化包括：
* **固定 Window ID**：通过 `xdotool selectwindow` 获取唯一窗口ID，避免多终端环境下的窗口定位歧义。
* **执行状态同步（DONE 哨兵）**：引入后缀为 `.done` 的哨兵文件，完全避免AI读取未执行完毕输出的 Race Condition 问题。
* **完整命令重定向作用域**：使用命令组 `{ cmd; } &> log` 确保复杂组合命令的输出被完整捕获。
* **减少焦点劫持风险**：执行命令前强制激活目标窗口，确保持续输入过程不受用户鼠标点击或系统弹窗的干扰。

## 3. 可执行代码
保存为 `ai_exec.py`：

```python
import subprocess
import time
import os

# 用户手动获取并替换此处的 Window ID
TERMINAL_WINDOW_ID = "REPLACE_WITH_WINDOW_ID"

def activate_terminal():
    """激活目标终端窗口"""
    subprocess.run(["xdotool", "windowmap", "--sync", TERMINAL_WINDOW_ID])
    subprocess.run(["xdotool", "windowactivate", "--sync", TERMINAL_WINDOW_ID])

def run_command(cmd: str, output_file: str = "/tmp/ai_output.log"):
    """向终端发送命令并重定向输出与完成状态"""
    done_file = output_file + ".done"

    if os.path.exists(done_file):
        os.remove(done_file)

    full_cmd = f"{{ {cmd}; }} &> {output_file}; echo DONE > {done_file}"
    activate_terminal()

    subprocess.run(["xdotool", "type", "--delay", "1", full_cmd])
    subprocess.run(["xdotool", "key", "Return"])

def wait_done(output_file: str = "/tmp/ai_output.log", timeout: int = 300):
    """检测哨兵文件，等待命令执行完成"""
    done_file = output_file + ".done"
    start = time.time()

    while True:
        if os.path.exists(done_file):
            return
        if time.time() - start > timeout:
            raise TimeoutError("Command execution timeout")
        time.sleep(0.2)

def get_output(output_file: str = "/tmp/ai_output.log"):
    """读取命令的执行输出"""
    with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
```

## 4. 使用说明
**步骤 1：安装依赖**
在基于Linux和X11的系统（Ubuntu等）中安装工具：
```bash
sudo apt install xdotool -y
```

**步骤 2：获取终端 Window ID**
在任意终端运行以下命令，然后用鼠标点击目标终端窗口：
```bash
xdotool selectwindow
```
将控制台输出的数字（如 `50331671`）填入上述代码的 `TERMINAL_WINDOW_ID` 变量中。

**步骤 3：准备目标执行环境**
在被刚才选中的目标终端中，人工进行登录和环境初始化（例如 `ssh server`, 挂载 `docker exec -it container bash`, 或 `source venv/bin/activate`）。

**步骤 4：在Python中调用执行**
在AI环境中调用工具链执行自动化命令封装，并等待执行结果：
```python
from ai_exec import run_command, wait_done, get_output

# 发送任务并指定日志文件
run_command("make", "/tmp/build.log")

# 阻塞等待哨兵文件生成
wait_done("/tmp/build.log")

# 执行完成，读取日志
output = get_output("/tmp/build.log")
print(output)
```

**⚠️ 注意事项：**
1. 目标终端不能被关闭，否则 Window ID 失效，需要重新获取。
2. 自动化工具输入命令期间（约一两秒内），不要操作鼠标/键盘以防焦点被抢占。
3. 请确保本方案只在 Linux 环境 + X11 桌面系统中运行（不支持无桌面环境或Wayland）。
