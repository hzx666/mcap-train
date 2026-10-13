#!/usr/bin/env python3
"""
Long-running MCAP extraction service.

N_WORKERS (default: cpu_count) independent worker threads each loop:
  pop one task from Redis → process → repeat.
Workers are fully independent — a fast worker never waits on a slow one.
When the queue is empty workers sleep for POLL_INTERVAL before retrying.

Path mapping matches run_extract_job.py:
  oss://ss-oss-data-platform/<key> -> /data/ss-oss-data-platform/<key>
Unlike the Job variant, this process never exits on idle — it keeps polling.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HEAD_MP4_SCRIPT = SCRIPT_DIR / "extract_head_train_mp4.py"
ORIN_MP4_SCRIPT = SCRIPT_DIR / "extract_orin_mp4.py"
HDF5_SCRIPT = SCRIPT_DIR / "hdf5_extract.py"

# OSS -> local mount mappings
# oss://ss-oss-data-platform/<key>  ->  /data/ss-oss-data-platform/<key>
OSS_URI_SCHEME = "oss://"
LOCAL_DATA_MOUNT = "/data/"
OSS_HEAD_CAM_PREFIX = "oss://ss-oss-data-platform/original_data/ego_human_data/head_cam/"
LOCAL_HEAD_CAM_MOUNT = "/data/ss-oss-data-platform/original_data/ego_human_data/head_cam/"
OSS_TELEOP_PREFIX = "oss://ss-oss-data-platform/original_data/teleoperation/"
LOCAL_TELEOP_MOUNT = "/data/ss-oss-data-platform/original_data/teleoperation/"
OSS_ASTRIBOT_PREFIX = "oss://ss-oss-data-platform/original_data/teleoperation/Astribot/"
LOCAL_ASTRIBOT_MOUNT = "/data/ss-oss-data-platform/original_data/teleoperation/Astribot/"

# ---- config ----

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
N_WORKERS = int(os.environ.get("N_WORKERS", str(os.cpu_count() or 1)))

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_QUEUE_KEY = os.environ.get("REDIS_QUEUE_KEY", "mcap:queue")
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_TIMEOUT = int(os.environ.get("REDIS_TIMEOUT", "5"))
TZ = timezone(timedelta(hours=8))

# Task logs under /app (WORKDIR). Override with EXTRACT_LOG_DIR.
LOG_DIR = Path(os.environ.get("EXTRACT_LOG_DIR", "/app"))
LOG_PREFIX = "run_extract"

_shutdown_event = threading.Event()
_print_lock = threading.Lock()
_log_lock = threading.Lock()

# ---- helpers ----


def _now() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _log_path() -> Path:
    day = datetime.now(TZ).strftime("%Y%m%d")
    return LOG_DIR / f"{LOG_PREFIX}_{day}.log"


def _append_log(text: str):
    """Append a record to today's log file under LOG_DIR (default ./)."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _log_path()
        with _log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
    except Exception as e:
        _tprint(f"[WARN] failed to write log: {e}")


_SKIP_RE = re.compile(
    r"SKIP\s*\[(?P<code>[A-Z0-9_]+)\]\s*(?P<msg>.*)",
    re.IGNORECASE,
)
_WARN_RE = re.compile(r"^\s*WARNING:\s*(?P<msg>.*)", re.IGNORECASE | re.MULTILINE)


def _extract_script_errors(stdout: str = "", stderr: str = "") -> dict:
    """Parse child-script stdout/stderr for SKIP [CODE] / WARNING lines."""
    text = "\n".join(x for x in (stdout or "", stderr or "") if x)
    skips = []
    for m in _SKIP_RE.finditer(text):
        skips.append({
            "code": m.group("code").upper(),
            "message": m.group("msg").strip(),
        })
    warnings = [m.group("msg").strip() for m in _WARN_RE.finditer(text) if m.group("msg").strip()]
    # Prefer last SKIP as the primary error (final outcome).
    primary = skips[-1] if skips else None
    return {
        "error_code": primary["code"] if primary else "",
        "error_message": primary["message"] if primary else "",
        "skips": skips,
        "warnings": warnings[-5:],  # keep log bounded
    }


def _log_task(worker_id: int, oss_path: str, result: dict, status: str):
    """Write one processed-task block to the daily log."""
    rc = result.get("returncode", "")
    elapsed = result.get("total_elapsed_s", result.get("elapsed_s", ""))
    script = result.get("script", "")
    mcap_path = result.get("mcap_path", "")
    err_info = _extract_script_errors(
        result.get("stdout", ""), result.get("stderr", ""),
    )
    # If child exited 0 but printed SKIP [CODE], treat log status as SKIP.
    if status.lower() == "ok" and err_info["error_code"]:
        status = "skip"

    lines = [
        "-" * 72,
        f"time       : {_now()}",
        f"worker     : {worker_id}",
        f"status     : {status.upper()}",
        f"script     : {script}",
        f"oss_path   : {oss_path}",
        f"mcap_path  : {mcap_path}",
        f"exit       : {rc}",
        f"elapsed_s  : {elapsed}",
    ]
    if err_info["error_code"]:
        lines.append(f"error_code : {err_info['error_code']}")
        lines.append(f"error_msg  : {err_info['error_message']}")
    elif status.lower() not in ("ok",):
        # Failed without SKIP tag — keep a short stderr/stdout hint.
        hint = (result.get("stderr") or result.get("stdout") or "").strip()
        if hint:
            snippet = hint if len(hint) <= 500 else hint[:500] + "...[truncated]"
            lines.append(f"error_msg  : {snippet}")

    if err_info["warnings"]:
        lines.append("warnings   :")
        for w in err_info["warnings"]:
            lines.append(f"  - {w}")

    if len(err_info["skips"]) > 1:
        lines.append("all_skips  :")
        for s in err_info["skips"]:
            lines.append(f"  - [{s['code']}] {s['message']}")

    stderr = (result.get("stderr") or "").strip()
    if stderr and status.lower() == "failed":
        snippet = stderr if len(stderr) <= 2000 else stderr[:2000] + "\n...[truncated]"
        lines.append("stderr     :")
        lines.append(snippet)
    lines.append("")
    _append_log("\n".join(lines))


def _read_stream(stream, lines_list, print_prefix, lock):
    """Read lines from a subprocess stream in a thread-safe manner."""
    try:
        for line in iter(stream.readline, ""):
            lines_list.append(line)
            with lock:
                if print_prefix:
                    print(print_prefix + line, end="")
                else:
                    print(line, end="")
    finally:
        stream.close()


def _run_script(script_path, mcap_path_str, label, mount_base="", oss_source_path=""):
    """Run a sub-script with MCAP_PATH set. Returns result dict."""
    _tprint(f"\n{'='*60}")
    _tprint(f"[{_now()}] Starting: {label}")
    _tprint(f"[{_now()}] MCAP_PATH = {mcap_path_str}")
    if oss_source_path:
        _tprint(f"[{_now()}] OSS_SOURCE_PATH = {oss_source_path}")
    _tprint(f"{'='*60}")

    env = os.environ.copy()
    env["MCAP_PATH"] = mcap_path_str
    if mount_base:
        env["LOCAL_MOUNT_BASE"] = mount_base
    if oss_source_path:
        env["OSS_SOURCE_PATH"] = oss_source_path

    stdout_lines = []
    stderr_lines = []

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        t_stdout = threading.Thread(
            target=_read_stream,
            args=(proc.stdout, stdout_lines, "", _print_lock),
            daemon=True,
        )
        t_stderr = threading.Thread(
            target=_read_stream,
            args=(proc.stderr, stderr_lines, "[STDERR] ", _print_lock),
            daemon=True,
        )
        t_stdout.start()
        t_stderr.start()

        proc.wait()
        t_stdout.join()
        t_stderr.join()

        elapsed = time.time() - t0
        stdout = "".join(stdout_lines).strip()
        stderr = "".join(stderr_lines).strip()

        success = proc.returncode == 0
        msg = (f"[{_now()}] {label}: {'OK' if success else 'FAILED'} "
               f"(exit={proc.returncode}, elapsed={elapsed:.1f}s)")
        if stderr and not success:
            msg += f" — stderr: {stderr}"
        _tprint(msg)

        return {
            "script": label,
            "success": success,
            "returncode": proc.returncode,
            "mcap_path": mcap_path_str,
            "elapsed_s": round(elapsed, 1),
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as e:
        elapsed = time.time() - t0
        _tprint(f"[{_now()}] {label}: ERROR - {e}")
        return {
            "script": label,
            "success": False,
            "mcap_path": mcap_path_str,
            "returncode": -1,
            "elapsed_s": round(elapsed, 1),
            "stdout": "",
            "stderr": str(e),
        }


def _print_summary(summary):
    _tprint(f"\n{'='*60}")
    _tprint(f"SUMMARY  [{summary['timestamp']}]")
    _tprint(f"{'='*60}")
    _tprint(f"  Path:    {summary['mcap_path']}")
    _tprint(f"  Status:  {summary['status'].upper()}")
    if summary.get("error_code"):
        _tprint(f"  Error:   [{summary['error_code']}] {summary.get('error_message', '')}")
    _tprint(f"  Elapsed: {summary['total_elapsed_s']}s")
    _tprint(f"  Scripts:")
    for r in summary["scripts"]:
        icon = "[OK]" if r["success"] else "[FAIL]"
        if summary.get("status") == "skip":
            icon = "[SKIP]"
        _tprint(f"    {icon} {r['script']}  (exit={r['returncode']}, {r['elapsed_s']}s)")
    _tprint(f"{'='*60}")


def _normalize_oss_path(raw) -> str:
    """Clean OSS URI that may have binary junk prefix/suffix.

    Accepts str or bytes. Redis producers sometimes push values with a few
    non-UTF-8 leading bytes (e.g. b'\\x03\\xfd\\x01oss://...'), which breaks
    decode_responses=True — callers should pass raw bytes from Redis.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        marker = b"oss://"
        idx = raw.find(marker)
        if idx >= 0:
            raw = raw[idx:]
        # Decode only the URI portion; drop trailing non-text bytes.
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)

    path = text.strip().strip("\x00").strip('"').strip("'")
    idx = path.find("oss://")
    if idx > 0:
        path = path[idx:]
    elif idx < 0:
        return path.strip()
    end = len(path)
    while end > 0 and ord(path[end - 1]) < 32:
        end -= 1
    return path[:end].rstrip()


def _oss_to_local(oss_path: str) -> str:
    """Map oss://ss-oss-data-platform/<key> -> /data/ss-oss-data-platform/<key>."""
    if oss_path.startswith(OSS_URI_SCHEME):
        return LOCAL_DATA_MOUNT + oss_path[len(OSS_URI_SCHEME):]
    return oss_path


def _resolve_path(oss_path: str):
    """Determine script and local path for an OSS path.

    Returns (script_path, mcap_path_str, mount_base) or (None, None, None).
    """
    oss_path = _normalize_oss_path(oss_path)
    low = oss_path.lower()
    if oss_path.startswith(OSS_HEAD_CAM_PREFIX):
        local_path = _oss_to_local(oss_path)
        return HEAD_MP4_SCRIPT, local_path, LOCAL_HEAD_CAM_MOUNT
    if low.endswith(".hdf5") or low.endswith(".h5"):
        local_path = _oss_to_local(oss_path)
        return HDF5_SCRIPT, local_path, LOCAL_ASTRIBOT_MOUNT
    if oss_path.startswith(OSS_ASTRIBOT_PREFIX) and not low.endswith(".mcap"):
        local_path = _oss_to_local(oss_path)
        return HDF5_SCRIPT, local_path, LOCAL_ASTRIBOT_MOUNT
    if oss_path.startswith(OSS_TELEOP_PREFIX):
        local_path = _oss_to_local(oss_path)
        return ORIN_MP4_SCRIPT, local_path, LOCAL_TELEOP_MOUNT
    return None, None, None


def _handle_signal(signum, frame):
    sig_name = signal.Signals(signum).name
    _tprint(f"\n[{_now()}] Received {sig_name}, shutting down gracefully...")
    _tprint(f"[{_now()}] Waiting for in-flight tasks to finish...")
    _shutdown_event.set()

# ---- worker ----


def _worker_loop(worker_id: int):
    """Independent worker: pop one task, process, repeat forever until shutdown."""
    try:
        import redis
    except ImportError:
        _tprint(f"[ERROR] Worker-{worker_id}: redis package not installed.")
        return

    r = None
    try:
        # decode_responses=False: queue values may contain non-UTF-8 junk bytes
        # before the oss:// URI (e.g. b'\\x03\\xfd\\x01oss://...').
        r = redis.Redis(
            username=REDIS_USERNAME or None,
            password=REDIS_PASSWORD or None,
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            socket_connect_timeout=REDIS_TIMEOUT,
            decode_responses=False,
        )
        r.ping()
    except Exception as e:
        _tprint(f"[ERROR] Worker-{worker_id}: Redis connection failed: {e}")
        return

    try:
        while not _shutdown_event.is_set():
            # -- pop one task --
            try:
                raw_item = r.lpop(REDIS_QUEUE_KEY)
            except Exception as e:
                _tprint(f"[ERROR] Worker-{worker_id}: Redis pop failed: {e}")
                _shutdown_event.wait(POLL_INTERVAL)
                continue

            if raw_item is None:
                _shutdown_event.wait(POLL_INTERVAL)
                continue

            oss_path = _normalize_oss_path(raw_item)
            if isinstance(raw_item, bytes) and not raw_item.startswith(b"oss://"):
                _tprint(f"[{_now()}] Worker-{worker_id} normalized path "
                        f"from bytes {raw_item[:32]!r}... -> {oss_path!r}")
            elif isinstance(raw_item, str) and oss_path != raw_item:
                _tprint(f"[{_now()}] Worker-{worker_id} normalized path "
                        f"(stripped junk prefix/suffix): {oss_path!r}")
            _tprint(f"[{_now()}] Worker-{worker_id} popped: {oss_path}")

            # -- route --
            resolved = _resolve_path(oss_path)
            if resolved[0] is None:
                preview = raw_item[:64] if isinstance(raw_item, (bytes, str)) else raw_item
                _tprint(f"[{_now()}] Worker-{worker_id}: "
                        f"WARNING unrecognized path, skipping: {preview!r}")
                _log_task(worker_id, oss_path, {
                    "script": "",
                    "mcap_path": "",
                    "returncode": "",
                    "elapsed_s": 0,
                    "stderr": f"unrecognized path: {preview!r}",
                }, "skip")
                continue

            script, mcap_path_str, mount_base = resolved

            # -- run --
            t0 = time.time()
            result = _run_script(script, mcap_path_str, script.stem, mount_base, oss_path)
            result["mcap_path"] = mcap_path_str
            result["total_elapsed_s"] = round(time.time() - t0, 1)
            status = "ok" if result["success"] else "failed"
            err_info = _extract_script_errors(
                result.get("stdout", ""), result.get("stderr", ""),
            )
            if result["success"] and err_info["error_code"]:
                status = "skip"

            _print_summary({
                "timestamp": _now(),
                "mcap_path": mcap_path_str,
                "status": status,
                "total_elapsed_s": result["total_elapsed_s"],
                "scripts": [result],
                "error_code": err_info.get("error_code", ""),
                "error_message": err_info.get("error_message", ""),
            })
            _log_task(worker_id, oss_path, result, status)
    finally:
        if r is not None:
            try:
                r.close()
            except Exception:
                pass

# ---- main ----


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[{_now()}] MCAP Extract Service started")
    print(f"  Redis:         {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} queue={REDIS_QUEUE_KEY}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print(f"  Workers:       {N_WORKERS}")
    print(f"  Log dir:       {LOG_DIR.resolve()}  ({LOG_PREFIX}_YYYYMMDD.log)")
    print(f"  Mounts:")
    print(f"    scheme:   {OSS_URI_SCHEME}<bucket>/<key> -> {LOCAL_DATA_MOUNT}<bucket>/<key>")
    print(f"    head_cam: {OSS_HEAD_CAM_PREFIX} -> {LOCAL_HEAD_CAM_MOUNT}")
    print(f"    teleop:   {OSS_TELEOP_PREFIX} -> {LOCAL_TELEOP_MOUNT}")
    print(f"    Astribot: {OSS_ASTRIBOT_PREFIX} -> {LOCAL_ASTRIBOT_MOUNT}")
    _append_log(
        f"{'='*72}\n"
        f"[{_now()}] service started  workers={N_WORKERS} "
        f"redis={REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} queue={REDIS_QUEUE_KEY}\n"
    )

    threads = []
    for i in range(N_WORKERS):
        t = threading.Thread(target=_worker_loop, args=(i,), name=f"worker-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{_now()}] Service stopped.")


if __name__ == "__main__":
    main()
