"""
launch_gui.py — Kill previous Streamlit instances and open the GUI.
Compiled to .exe via PyInstaller for one-click launching.
"""

import os
import subprocess
import sys
import time
import webbrowser

PORT = 8501
SCRIPT = "gui.py"


def get_project_dir():
    """Return the directory where this script (or .exe) lives."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def kill_streamlit_on_port(port: int):
    """Find and kill any process listening on the given port."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                try:
                    pid_int = int(pid)
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid_int)],
                        capture_output=True, timeout=10
                    )
                    print(f"  Killed previous process (PID {pid_int}) on port {port}")
                except (ValueError, subprocess.SubprocessError):
                    pass
    except subprocess.SubprocessError:
        pass


def find_python():
    """Return path to the Python interpreter."""
    # Try the same Python that's running this script
    if not getattr(sys, "frozen", False):
        return sys.executable

    # When frozen (.exe), locate Python on PATH
    for cmd in ["python", "python3", "py"]:
        try:
            r = subprocess.run(
                [cmd, "--version"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.SubprocessError):
            continue

    return "python"


def main():
    project_dir = get_project_dir()
    gui_path = os.path.join(project_dir, SCRIPT)

    if not os.path.exists(gui_path):
        print(f"ERROR: {SCRIPT} not found in {project_dir}")
        input("Press Enter to exit...")
        sys.exit(1)

    print("=" * 50)
    print("  MTG Card Price Predictor — GUI Launcher")
    print("=" * 50)

    # Step 1: Kill any previous Streamlit on this port
    print(f"\n[1/3] Stopping previous instances on port {PORT}...")
    kill_streamlit_on_port(PORT)
    time.sleep(1)

    # Step 2: Launch Streamlit
    python = find_python()
    print(f"\n[2/3] Starting Streamlit (port {PORT})...")
    print(f"  Python: {python}")
    print(f"  Script: {gui_path}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [python, "-m", "streamlit", "run", gui_path,
         "--server.port", str(PORT),
         "--server.headless", "true"],
        cwd=project_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Step 3: Wait for Streamlit to be ready, then open browser
    print(f"\n[3/3] Waiting for server to start...")
    url = f"http://localhost:{PORT}"
    started = False

    for _ in range(30):  # up to 30 seconds
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            started = True
            break
        except Exception:
            pass

    if started:
        print(f"\n  Server ready! Opening {url} in your browser...\n")
        webbrowser.open(url)
    else:
        print(f"\n  Server may still be loading — opening {url} anyway...\n")
        webbrowser.open(url)

    print("  The GUI is running. Close this window to stop the server.")
    print("  (Or press Ctrl+C)\n")

    try:
        # Stream Streamlit output so the console stays informative
        for line in proc.stdout:
            print(f"  {line}", end="")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        proc.terminate()


if __name__ == "__main__":
    main()
