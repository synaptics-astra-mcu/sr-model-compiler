"""Utilities to help the library"""

import subprocess
import platform


def call_shell_cmd(cmd):
    """Run a shell command"""

    cmd_list = cmd.split(" ")
    result_log = ""
    try:
        result = subprocess.run(cmd_list, capture_output=True, check=True)
        result_log += result.stdout.decode("utf-8")
        result_log += "\n"
        result_log += result.stderr.decode("utf-8")

    except subprocess.CalledProcessError as e:
        print(f"ERROR in command {e}")
        result_log += e.stdout.decode("utf-8")
        result_log += "\n"
        result_log += e.stderr.decode("utf-8")
        return False, result_log

    print(f"utils = shell_cmd = {result}")
    return True, result_log


def get_platform_path(unix_path):
    """Gets a UNIX style path and converts to Windows format if needed"""

    if platform.system() == "Windows":
        return unix_path.replace("/", "\\")
    return unix_path
