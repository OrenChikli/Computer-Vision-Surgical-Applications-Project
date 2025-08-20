"""
Pip-based installation script for the Surgical Instrument Pose Estimation project.
Creates a virtual environment and installs dependencies.
"""
import subprocess
import sys

PYTHON_VERSION = "3.10"
VENV_NAME = ".venv"

def run(cmd):
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    # 1. Check if Python version is available
    try:
        run(["py", f"-{PYTHON_VERSION}", "--version"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"ERROR: Python {PYTHON_VERSION} not found!")
        print(f"Please install Python {PYTHON_VERSION} from https://www.python.org/downloads/")
        sys.exit(1)

    # 2. Create virtual environment
    try:
        run(["py", f"-{PYTHON_VERSION}", "-m", "venv", VENV_NAME])
        print(f"Virtual environment '{VENV_NAME}' created successfully")
    except subprocess.CalledProcessError:
        print(f"Virtual environment '{VENV_NAME}' may already exist. Continuing...")

    # 3. Install dependencies
    python_exe = f"{VENV_NAME}\\Scripts\\python.exe" if sys.platform == "win32" else f"{VENV_NAME}/bin/python"
    print("Installing dependencies (this may take several minutes)...")
    run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"])

    print("\nInstallation complete!")
    if sys.platform == "win32":
        print(f"To activate the environment, run: {VENV_NAME}\\Scripts\\activate")
    else:
        print(f"To activate the environment, run: source {VENV_NAME}/bin/activate")

if __name__ == "__main__":
    main()