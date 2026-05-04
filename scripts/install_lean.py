import shutil
import subprocess
import sys


def has_lean():
    return shutil.which("lean") is not None


def run(cmd):
    print("$", cmd)
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")


def main():
    if has_lean():
        print("lean already installed")
        return

    # Install elan + lean toolchain (Linux/Colab).
    run("curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y")

    # Make elan binaries visible in this process.
    run('bash -lc "source $HOME/.profile; source $HOME/.elan/env; elan default stable; lean --version"')

    if not has_lean():
        print("warning: lean not found in PATH for current process. Try restarting runtime or sourcing ~/.elan/env.")
    else:
        print("lean installed successfully")


if __name__ == "__main__":
    main()
