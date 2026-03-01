#!/usr/bin/env python3
"""Start the dev server on an available port (avoids 'Address already in use')."""
import socket
import subprocess
import sys

def find_free_port(start=8000, end=9000):
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    port = find_free_port()
    if port is None:
        print("ERROR: No free port between 8000 and 9000.", file=sys.stderr)
        sys.exit(1)
    print(f"Starting server on http://127.0.0.1:{port} (port {port} was free)")
    sys.exit(subprocess.call([
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--reload", "--host", "0.0.0.0", "--port", str(port)
    ]))
