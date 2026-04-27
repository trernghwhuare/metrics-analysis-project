#!/usr/bin/env python3
"""
Wrapper script to build MyST documentation.
This directly calls the mystmd_py.main module to avoid PATH issues.
"""
import sys
import os

def main():
    # Add current directory to Python path to ensure local modules are found
    sys.path.insert(0, os.getcwd())
    
    try:
        # Import and run the MyST CLI main function directly
        from mystmd_py.main import main as myst_main
        # Set argv to mimic 'myst build' command
        sys.argv = ['myst', 'build']
        myst_main()
    except ImportError as e:
        print(f"MyST CLI module not found: {e}", file=sys.stderr)
        print("Please ensure mystmd is installed correctly.", file=sys.stderr)
        sys.exit(1)
    except SystemExit as e:
        # myst_main calls sys.exit(), so we need to handle it properly
        sys.exit(e.code)
    except Exception as e:
        print(f"Error running MyST CLI: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()