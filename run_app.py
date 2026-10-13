"""
Launcher for the FTIR Feature Perceptibility & Discriminatory Power Annotator.
Run:
    python run_app.py [port]
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory so perceptibility_app can be imported either from inside the folder or parent
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from perceptibility_app.server import run_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8055
    run_server(port)
