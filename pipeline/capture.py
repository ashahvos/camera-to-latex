"""Talk to the board over USB serial and get grayscale frames.

Protocol (see firmware/capture/capture.ino):
  laptop -> 'p'   board -> "OK 320 240 1\\n"          (ping: width, height, bytes per pixel)
  laptop -> 'c'   board -> b"FRM1" | width u16 | height u16 | bpp u8 | count u32
                           | count pixel bytes | checksum u16
                  all little-endian; checksum = sum of the pixel bytes mod 65536
Anything else the board prints is skipped while looking for the header; a line
starting with "ERR" is reported.

Command line:
  python -m pipeline.capture --list                 serial ports
  python -m pipeline.capture --ping                 is the board answering?
  python -m pipeline.capture --grab frame.png       save one frame
  python -m pipeline.capture --soak 100             reliability test (gate G1: 100/100)
  python -m pipeline.capture --series DIR --names-file names.txt
                                                    photograph a list (sheets, test lines)
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

MAGIC = b"FRM1"
HEADER = struct.Struct("<HHBI")      # width, height, bytes per pixel, pixel count
CHECKSUM = struct.Struct("<H")


class CaptureError(RuntimeError):
    pass


def _read_exact(stream, n: int, deadline: float) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if chunk:
            buf += chunk
        elif time.monotonic() > deadline:
            raise CaptureError(f"timeout: got {len(buf)} of {n} bytes")
    return bytes(buf)


def read_frame(stream, timeout: float | None = None) -> np.ndarray:
    """Read one FRM1 packet from a file-like object with .read(n)."""
    timeout = config.SERIAL_TIMEOUT_S if timeout is None else timeout
    deadline = time.monotonic() + timeout
    window = bytearray()
    line = bytearray()
    while True:
        b = stream.read(1)
        if not b:
            if time.monotonic() > deadline:
                raise CaptureError("timeout waiting for frame header")
            continue
        window += b
        if len(window) > 4:
            del window[0]
        if bytes(window) == MAGIC:
            break
        if b == b"\n":
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("ERR"):
                raise CaptureError(f"board says: {text}")
            line.clear()
        else:
            line += b
    w, h, bpp, count = HEADER.unpack(_read_exact(stream, HEADER.size, deadline))
    if bpp not in (1, 2) or count != w * h * bpp or count > 4_000_000:
        raise CaptureError(f"bad header: {w}x{h}, {bpp} B/px, {count} B")
    data = _read_exact(stream, count, deadline)
    (expected,) = CHECKSUM.unpack(_read_exact(stream, CHECKSUM.size, deadline))
    got = int(np.frombuffer(data, np.uint8).sum(dtype=np.uint64)) & 0xFFFF
    if got != expected:
        raise CaptureError(f"checksum mismatch (got {got}, board sent {expected})")
    img = np.frombuffer(data, np.uint8).reshape(h, w, bpp)
    return np.ascontiguousarray(img[:, :, 0])


def encode_frame(img: np.ndarray) -> bytes:
    """The bytes the firmware sends for a grayscale image (used by tests)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    data = img.tobytes()
    checksum = int(img.sum(dtype=np.uint64)) & 0xFFFF
    return MAGIC + HEADER.pack(w, h, 1, len(data)) + data + CHECKSUM.pack(checksum)


def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError as exc:
        raise CaptureError("pyserial is not installed: pip install pyserial") from exc
    return list(lp.comports())


def find_port() -> str:
    """The board's serial port: config.SERIAL_PORT, else the only Arduino-looking port."""
    if config.SERIAL_PORT:
        return config.SERIAL_PORT
    ports = list_ports()
    arduino = [p for p in ports if (p.vid == 0x2341) or "arduino" in (p.description or "").lower()
               or "nano" in (p.description or "").lower()]
    if len(arduino) == 1:
        return arduino[0].device
    usb = [p for p in ports if any(k in p.device.lower() for k in ("usbmodem", "ttyacm", "com"))]
    if len(arduino) == 0 and len(usb) == 1:
        return usb[0].device
    names = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    raise CaptureError(f"cannot pick the board's port automatically; ports: {names}. "
                       "Set SERIAL_PORT in config.py or pass --port.")


class Camera:
    def __init__(self, port: str | None = None, baud: int | None = None):
        self.port = port
        self.baud = baud or config.SERIAL_BAUD
        self.ser = None
        self.lock = threading.Lock()

    def open(self) -> "Camera":
        try:
            import serial
        except ImportError as exc:
            raise CaptureError("pyserial is not installed: pip install pyserial") from exc
        self.port = self.port or find_port()
        self.ser = serial.Serial(self.port, self.baud, timeout=0.2, write_timeout=2)
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        return self

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _command_line(self, cmd: bytes, timeout: float = 3.0) -> str:
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = self.ser.readline().decode("ascii", errors="replace").strip()
                if line.startswith("OK") or line.startswith("ERR"):
                    return line
        raise CaptureError(f"no answer to {cmd!r} within {timeout} s")

    def ping(self) -> tuple[int, int, int]:
        line = self._command_line(b"p")
        parts = line.split()
        if parts[0] != "OK" or len(parts) < 4:
            raise CaptureError(f"unexpected ping answer: {line}")
        return int(parts[1]), int(parts[2]), int(parts[3])

    def grab(self) -> np.ndarray:
        last = None
        for _ in range(config.SERIAL_RETRIES):
            with self.lock:
                try:
                    self.ser.reset_input_buffer()
                    self.ser.write(b"c")
                    return read_frame(self.ser)
                except CaptureError as exc:
                    last = exc
                    time.sleep(0.2)
        raise CaptureError(f"no good frame after {config.SERIAL_RETRIES} tries: {last}")


class FrameGrabber(threading.Thread):
    """Keeps grabbing frames in the background so the window stays responsive."""

    def __init__(self, camera: Camera):
        super().__init__(daemon=True)
        self.camera = camera
        self.frame = None
        self.error = None
        self.count = 0
        self.paused = False
        self._halt = threading.Event()
        self._lock = threading.Lock()

    def run(self):
        while not self._halt.is_set():
            if self.paused:
                time.sleep(0.05)
                continue
            try:
                f = self.camera.grab()
                with self._lock:
                    self.frame, self.error = f, None
                    self.count += 1
            except Exception as exc:  # keep the app alive; show the error instead
                with self._lock:
                    self.error = str(exc)
                time.sleep(0.5)

    def latest(self):
        with self._lock:
            return self.frame, self.error

    def stop(self):
        self._halt.set()


# ------------------------------------------------------------------ CLI ---
def _save_png(path: Path, img: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise CaptureError(f"could not write {path}")


def soak(cam: Camera, n: int) -> bool:
    ok = bad = 0
    t0 = time.perf_counter()
    for i in range(n):
        try:
            with cam.lock:
                cam.ser.reset_input_buffer()
                cam.ser.write(b"c")
                img = read_frame(cam.ser)
            ok += 1
            if i == 0:
                print(f"first frame {img.shape[1]}x{img.shape[0]}, mean level {img.mean():.0f}")
        except CaptureError as exc:
            bad += 1
            print(f"frame {i + 1}: {exc}")
    dt = (time.perf_counter() - t0) / max(1, n)
    print(f"{ok}/{n} frames OK, {bad} failed, {dt:.2f} s per frame")
    return bad == 0


def series(cam: Camera, out_dir: Path, names: list[str]) -> None:
    """Preview window; SPACE saves the next name, b = back, n = skip, q = quit."""
    import cv2
    from pipeline.preprocess import orient
    grabber = FrameGrabber(cam)
    grabber.start()
    i = 0
    try:
        while i < len(names):
            frame, err = grabber.latest()
            view = np.full((480, 640, 3), 40, np.uint8)
            if frame is not None:
                g = orient(frame)
                view = cv2.cvtColor(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                    cv2.COLOR_GRAY2BGR)
            exists = (out_dir / f"{names[i]}.png").exists()
            msg = f"{i + 1}/{len(names)}  next: {names[i]}{'  (exists)' if exists else ''}"
            cv2.putText(view, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(view, "SPACE save  b back  n skip  q quit", (8, view.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if err:
                cv2.putText(view, err[:70], (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("capture series", view)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" ") and frame is not None:
                _save_png(out_dir / f"{names[i]}.png", frame)
                print(f"saved {names[i]}.png")
                i += 1
            elif k == ord("b"):
                i = max(0, i - 1)
            elif k == ord("n"):
                i += 1
            elif k in (ord("q"), 27):
                break
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: config.SERIAL_PORT or auto)")
    ap.add_argument("--list", action="store_true", help="list serial ports")
    ap.add_argument("--ping", action="store_true", help="check the board answers")
    ap.add_argument("--grab", type=Path, help="save one frame to this PNG")
    ap.add_argument("--soak", type=int, help="grab this many frames and count failures")
    ap.add_argument("--series", type=Path, help="folder to save a named series of frames into")
    ap.add_argument("--names", help="comma-separated names for --series")
    ap.add_argument("--names-file", type=Path, help="file with one name per line for --series")
    args = ap.parse_args(argv)

    if args.list:
        for p in list_ports():
            print(f"{p.device:24s} {p.description}  vid={p.vid} pid={p.pid}")
        return 0
    with Camera(args.port) as cam:
        print(f"port {cam.port}")
        if args.ping:
            print("board answers: %dx%d, %d byte(s) per pixel" % cam.ping())
        if args.grab:
            img = cam.grab()
            _save_png(args.grab, img)
            print(f"saved {args.grab} ({img.shape[1]}x{img.shape[0]}, mean {img.mean():.0f})")
        if args.soak:
            return 0 if soak(cam, args.soak) else 1
        if args.series:
            names = []
            if args.names:
                names = [n.strip() for n in args.names.split(",") if n.strip()]
            if args.names_file:
                names += [n.strip() for n in args.names_file.read_text().splitlines()
                          if n.strip() and not n.startswith("#")]
            if not names:
                ap.error("--series needs --names or --names-file")
            series(cam, args.series, names)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CaptureError as exc:
        print(f"camera problem: {exc}")
        sys.exit(1)
