#!/usr/bin/env python3
"""Extracto Desktop App - double-click to launch.

Opens a web browser automatically. No terminal needed.
"""

import os
import sys
import time
import webbrowser
import threading

# Ensure we can find our modules when running as a bundled app
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
    sys.path.insert(0, os.path.dirname(sys.executable))

def open_browser(port):
    """Wait for server to start, then open browser."""
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")

def main():
    port = 8080

    # Try ports until we find one available
    import socket
    for p in [8080, 8081, 8082, 8888, 9090]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', p))
            s.close()
            port = p
            break
        except OSError:
            continue

    # Open browser in background
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start the web server
    from extracto.web.app import app
    print(f"Extracto running at http://localhost:{port}")
    print("Close this window to stop.")
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    main()
