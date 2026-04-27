#!/usr/bin/env python3
"""
Wrapper script to build MyST documentation.
This ensures the myst command is available in the correct Python environment.
"""
import subprocess
import sys
import os

def main():
    # Get the directory where Python scripts are installed
    python_scripts_dir = os.path.join(os.path.dirname(sys.executable), 'bin')
    myst_path = os.path.join(python_scripts_dir, 'myst')
    
    if os.path.exists(myst_path):
        # Use the full path to myst
        cmd = [myst_path, 'build']
    else:
        # Fall back to just 'myst'
        cmd = ['myst', 'build']
    
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error running myst build: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("myst command not found. Please ensure mystmd is installed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()