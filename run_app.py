"""CropLabel launcher: starts the server and opens the browser."""
import threading
import time
import webbrowser

import uvicorn

PORT = 8323


def _open():
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run("croplabel.server:app", host="127.0.0.1", port=PORT,
                log_level="warning")
