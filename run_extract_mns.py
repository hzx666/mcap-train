#!/usr/bin/env python3
"""
Long-running MCAP extraction service that consumes Alibaba Cloud MNS.

Same worker model and OSS path mapping as run_extract.py. Redis consumption
in run_extract.py is unchanged. This process only replaces the queue source.

Messages are an event envelope. This service handles event "mcap_extract_train"
and acks manually (DeleteMessage). Other events are not acked; their
visibility is shortened so another consumer can take them.

After a task finishes, one result envelope is sent to MNS_RESULT_QUEUE_NAME
(default: data-platfrom-event-resp). Success is code 200 and msg "". A failed
attempt is retried without writing. Only the last attempt writes code 500,
then acks. Set MNS_WRITE_RESULT=false to skip the result queue. Base64
encoding follows MNS_BASE64 (default on), matching the Java SDK.

Envelope:
  {"event": "mcap_extract_train", "code": 200, "msg": "", "traceId": "", "data": {"ossPath": "oss://..."}}

Environment:
  MNS_ENDPOINT            required, e.g. http://<uid>.mns.cn-shanghai-internal.aliyuncs.com
  MNS_QUEUE_NAME          required consume queue name
  MNS_WRITE_RESULT        write the result queue (default: true)
  MNS_RESULT_QUEUE_NAME   result queue (default: data-platfrom-event-resp)
  MNS_ACCESS_KEY_ID       default: OSS_ACCESS_KEY_ID
  MNS_ACCESS_KEY_SECRET   default: OSS_ACCESS_KEY_SECRET
  MNS_SECURITY_TOKEN      optional STS token
  MNS_EVENT               event this process handles (default: mcap_extract_train)
  MNS_BASE64              decode message body as base64 (default: true, SDK default)
  MNS_WAIT_SECONDS        long-poll seconds, 0-30 (default: min(POLL_INTERVAL, 30))
  MNS_VISIBILITY_TIMEOUT  hide a mcap_extract_train message while processing (default: 3600)
  MNS_RETRY_SECONDS       when not acked, become visible again after this many seconds (default: 30)
  MNS_MAX_ATTEMPTS        receive count that may write one code 500 (default: 3)
  POLL_INTERVAL           reconnect backoff, shared with run_extract.py (default: 10)
  N_WORKERS               worker threads (default: cpu count)
"""

import json
import os
import signal
import sys
import threading
import time
from urllib.parse import unquote

import run_extract as rx

MNS_ENDPOINT = os.environ.get("MNS_ENDPOINT", "").strip()
MNS_QUEUE_NAME = os.environ.get("MNS_QUEUE_NAME", "").strip()
MNS_RESULT_QUEUE_NAME = os.environ.get("MNS_RESULT_QUEUE_NAME", "data-platfrom-event-resp").strip()
MNS_ACCESS_KEY_ID = (
    os.environ.get("MNS_ACCESS_KEY_ID")
    or os.environ.get("OSS_ACCESS_KEY_ID")
    or ""
).strip()
MNS_ACCESS_KEY_SECRET = (
    os.environ.get("MNS_ACCESS_KEY_SECRET")
    or os.environ.get("OSS_ACCESS_KEY_SECRET")
    or ""
).strip()
MNS_SECURITY_TOKEN = os.environ.get("MNS_SECURITY_TOKEN", "").strip()

_PATH_KEYS = ("oss_path", "ossPath", "path", "oss_uri", "ossUri", "mcap_path", "mcapPath")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


MNS_EVENT = os.environ.get("MNS_EVENT", "mcap_extract_train").strip() or "mcap_extract_train"
MNS_WRITE_RESULT = _env_bool("MNS_WRITE_RESULT", True)
MNS_BASE64 = _env_bool("MNS_BASE64", True)
MNS_WAIT_SECONDS = max(0, min(30, _env_int("MNS_WAIT_SECONDS", min(rx.POLL_INTERVAL, 30))))
MNS_VISIBILITY_TIMEOUT = _env_int("MNS_VISIBILITY_TIMEOUT", 3600)
MNS_RETRY_SECONDS = _env_int("MNS_RETRY_SECONDS", 30)
MNS_MAX_ATTEMPTS = _env_int("MNS_MAX_ATTEMPTS", 3)


def _mns_error_type(exc: BaseException) -> str:
    return str(getattr(exc, "type", "") or "")


def _oss_event_path(data: dict) -> str:
    events = data.get("events")
    if not isinstance(events, list):
        return ""
    for event in events:
        if not isinstance(event, dict):
            continue
        oss = event.get("oss")
        if not isinstance(oss, dict):
            continue
        bucket_info = oss.get("bucket")
        object_info = oss.get("object")
        bucket = bucket_info.get("name") if isinstance(bucket_info, dict) else ""
        key = object_info.get("key") if isinstance(object_info, dict) else ""
        if bucket and key:
            return f"oss://{bucket}/{unquote(str(key)).lstrip('/')}"
    return ""


def _find_oss_path(data) -> str:
    if isinstance(data, str):
        text = data.strip()
        if "oss://" in text or text.lower().endswith((".mcap", ".hdf5", ".h5")):
            return text
        return ""
    if isinstance(data, dict):
        for key in _PATH_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        event_path = _oss_event_path(data)
        if event_path:
            return event_path
        for value in data.values():
            found = _find_oss_path(value)
            if found:
                return found
    if isinstance(data, list):
        for item in data:
            found = _find_oss_path(item)
            if found:
                return found
    return ""


def _parse_envelope(body) -> dict | None:
    """Parse {"event", "code", "msg", "traceId", "data"}. Return None when it is not that object."""
    if body is None:
        return None
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="ignore").strip()
    elif isinstance(body, dict):
        data = body
        text = ""
    else:
        text = str(body).strip()
        data = None
    if data is None:
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
    if not isinstance(data, dict) or "event" not in data:
        return None
    return data


def _code_ok(code) -> bool:
    if code is None or code == "":
        return True
    try:
        return int(code) == 200
    except (TypeError, ValueError):
        return False


def _open_queue(queue_name: str):
    try:
        from mns.account import Account
    except ImportError:
        raise RuntimeError("mns package not installed. Run: pip install aliyun-mns-sdk")

    account = Account(
        MNS_ENDPOINT,
        MNS_ACCESS_KEY_ID,
        MNS_ACCESS_KEY_SECRET,
        MNS_SECURITY_TOKEN,
    )
    queue = account.get_queue(queue_name)
    queue.set_encoding(MNS_BASE64)
    return queue


def _failure_msg(result: dict, err_info: dict) -> str:
    if err_info.get("error_code"):
        text = f"[{err_info['error_code']}] {err_info.get('error_message', '')}".strip()
    else:
        text = (result.get("stderr") or result.get("stdout") or "").strip()
    text = text or "extract failed"
    if len(text) > 2000:
        text = text[:2000] + "...[truncated]"
    return text


def _publish_result(result_queue, envelope: dict, code: int, msg: str):
    """Send {event, code, msg, traceId, data} to the result queue. Body is Base64 when MNS_BASE64 is on."""
    from mns.queue import Message

    data = envelope.get("data")
    if data is None:
        data = {}
    trace_id = envelope.get("traceId")
    body = {
        "event": envelope.get("event") or MNS_EVENT,
        "code": int(code),
        "msg": msg or "",
        "traceId": "" if trace_id is None else str(trace_id),
        "data": data,
    }
    result_queue.send_message(Message(json.dumps(body, ensure_ascii=False)))


def _report_failure(worker_id: int, message_id: str, envelope: dict, msg: str,
                    dequeue_count: int, publish, ack, retryable: bool):
    """Write code 500 at most once. Earlier retryable failures are not published."""
    if retryable and dequeue_count < MNS_MAX_ATTEMPTS:
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: failed "
            f"{dequeue_count}/{MNS_MAX_ATTEMPTS}, not writing result  "
            f"message_id={message_id}"
        )
        return
    if dequeue_count > MNS_MAX_ATTEMPTS or (not retryable and dequeue_count > 1):
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: failure already reported, "
            f"ack without writing  message_id={message_id}"
        )
        ack()
        return
    _finish(worker_id, message_id, envelope, 500, msg, publish, ack)


def _finish(worker_id: int, message_id: str, envelope: dict, code: int, msg: str, publish, ack):
    """Ack the source message. Publish the result only when MNS_WRITE_RESULT is on."""
    if MNS_WRITE_RESULT:
        publish(envelope, code, msg)
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: result code={code} sent and acked  "
            f"message_id={message_id}"
        )
    else:
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: acked, result not written  "
            f"code={code} message_id={message_id}"
        )
    ack()


def _delete_message(queue, receipt_handle: str, worker_id: int):
    try:
        queue.delete_message(receipt_handle)
    except Exception as e:
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: failed to delete MNS message: {e}"
        )


def _change_visibility(queue, receipt_handle: str, seconds: int, worker_id: int) -> str:
    if seconds <= 0:
        return receipt_handle
    seconds = max(1, min(43200, int(seconds)))
    try:
        changed = queue.change_message_visibility(receipt_handle, seconds)
        return changed.receipt_handle or receipt_handle
    except Exception as e:
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: "
            f"failed to change visibility to {seconds}s: {e}"
        )
        return receipt_handle


def _handle_mcap_extract_train(worker_id: int, envelope: dict, message_id: str, ack, extend, publish,
                          dequeue_count: int):
    """Extract oss_path from data. Publish the result envelope, then ack."""
    code = envelope.get("code", 200)
    msg = envelope.get("msg") or ""
    if not _code_ok(code):
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: mcap_extract_train code={code} "
            f"msg={msg!r}, ack without extract  message_id={message_id}"
        )
        ack()
        return

    data = envelope.get("data")
    oss_path = rx._normalize_oss_path(_find_oss_path(data))
    rx._tprint(
        f"[{rx._now()}] Worker-{worker_id} mcap_extract_train: {oss_path}  "
        f"message_id={message_id}"
    )
    if not oss_path:
        _report_failure(
            worker_id, message_id, envelope, "data has no oss_path",
            dequeue_count, publish, ack, retryable=False,
        )
        return

    extend()
    resolved = rx._resolve_path(oss_path)
    if resolved[0] is None:
        reason = f"unrecognized path: {oss_path}"
        rx._tprint(f"[{rx._now()}] Worker-{worker_id}: WARNING {reason}")
        rx._log_task(worker_id, oss_path, {
            "script": "",
            "mcap_path": "",
            "returncode": "",
            "elapsed_s": 0,
            "stderr": reason,
        }, "skip")
        _report_failure(
            worker_id, message_id, envelope, reason,
            dequeue_count, publish, ack, retryable=False,
        )
        return

    script, mcap_path_str, mount_base = resolved
    t0 = time.time()
    result = rx._run_script(script, mcap_path_str, script.stem, mount_base, oss_path)
    result["mcap_path"] = mcap_path_str
    result["total_elapsed_s"] = round(time.time() - t0, 1)
    status = "ok" if result["success"] else "failed"
    err_info = rx._extract_script_errors(
        result.get("stdout", ""), result.get("stderr", ""),
    )
    if result["success"] and err_info["error_code"]:
        status = "skip"

    rx._print_summary({
        "timestamp": rx._now(),
        "mcap_path": mcap_path_str,
        "status": status,
        "total_elapsed_s": result["total_elapsed_s"],
        "scripts": [result],
        "error_code": err_info.get("error_code", ""),
        "error_message": err_info.get("error_message", ""),
    })
    rx._log_task(worker_id, oss_path, result, status)
    if status == "ok":
        _finish(worker_id, message_id, envelope, 200, "", publish, ack)
        return
    _report_failure(
        worker_id, message_id, envelope, _failure_msg(result, err_info),
        dequeue_count, publish, ack, retryable=True,
    )


_HANDLERS = {
    "mcap_extract_train": _handle_mcap_extract_train,
}


def _dispatch(worker_id: int, raw_body, message_id: str, ack, extend, publish,
              dequeue_count: int) -> str:
    """Return 'release' for other events, 'handled' when this process owns the event."""
    envelope = _parse_envelope(raw_body)
    if envelope is None:
        preview = raw_body[:200] if isinstance(raw_body, str) else raw_body
        raise ValueError(f"message is not an event envelope: {preview!r}")

    event = str(envelope.get("event") or "").strip()
    rx._tprint(
        f"[{rx._now()}] Worker-{worker_id} event={event} "
        f"code={envelope.get('code')!r} msg={envelope.get('msg')!r} "
        f"traceId={envelope.get('traceId')!r}  message_id={message_id}"
    )
    if event != MNS_EVENT:
        rx._tprint(
            f"[{rx._now()}] Worker-{worker_id}: event {event!r} is not {MNS_EVENT}, "
            f"not acked"
        )
        return "release"

    handler = _HANDLERS.get(event)
    if handler is None:
        raise RuntimeError(f"no handler for event {event}")
    handler(worker_id, envelope, message_id, ack, extend, publish, dequeue_count)
    return "handled"


def _worker_loop(worker_id: int):
    queue = None
    result_queue = None
    while not rx._shutdown_event.is_set():
        if queue is None or (MNS_WRITE_RESULT and result_queue is None):
            try:
                queue = _open_queue(MNS_QUEUE_NAME)
                if MNS_WRITE_RESULT:
                    result_queue = _open_queue(MNS_RESULT_QUEUE_NAME)
            except Exception as e:
                rx._tprint(f"[ERROR] Worker-{worker_id}: MNS connection failed: {e}")
                queue = None
                result_queue = None
                rx._shutdown_event.wait(rx.POLL_INTERVAL)
                continue

        try:
            recv = queue.receive_message_with_str_body(MNS_WAIT_SECONDS)
        except Exception as e:
            if _mns_error_type(e) == "MessageNotExist":
                continue
            rx._tprint(f"[ERROR] Worker-{worker_id}: MNS receive failed: {e}")
            queue = None
            if MNS_WRITE_RESULT:
                result_queue = None
            rx._shutdown_event.wait(rx.POLL_INTERVAL)
            continue

        if rx._shutdown_event.is_set():
            return

        state = {"receipt": recv.receipt_handle, "acked": False}

        def ack():
            state["acked"] = True

        def extend():
            state["receipt"] = _change_visibility(
                queue, state["receipt"], MNS_VISIBILITY_TIMEOUT, worker_id,
            )

        def publish(envelope, code, msg):
            if MNS_WRITE_RESULT:
                _publish_result(result_queue, envelope, code, msg)

        try:
            action = _dispatch(
                worker_id, recv.message_body, recv.message_id, ack, extend, publish,
                int(getattr(recv, "dequeue_count", 1) or 1),
            )
        except Exception as e:
            rx._tprint(
                f"[{rx._now()}] Worker-{worker_id}: process failed, not acked: {e}"
            )
            action = "nack"
            if MNS_WRITE_RESULT:
                result_queue = None

        if state["acked"]:
            _delete_message(queue, state["receipt"], worker_id)
        elif action == "release":
            _change_visibility(queue, state["receipt"], 1, worker_id)
        else:
            _change_visibility(queue, state["receipt"], MNS_RETRY_SECONDS, worker_id)


def _check_config():
    missing = []
    if not MNS_ENDPOINT:
        missing.append("MNS_ENDPOINT")
    if not MNS_QUEUE_NAME:
        missing.append("MNS_QUEUE_NAME")
    if MNS_WRITE_RESULT and not MNS_RESULT_QUEUE_NAME:
        missing.append("MNS_RESULT_QUEUE_NAME")
    if MNS_WRITE_RESULT and MNS_QUEUE_NAME and MNS_QUEUE_NAME == MNS_RESULT_QUEUE_NAME:
        print("[ERROR] MNS_RESULT_QUEUE_NAME must be a different queue from MNS_QUEUE_NAME")
        sys.exit(1)
    if not MNS_ACCESS_KEY_ID:
        missing.append("MNS_ACCESS_KEY_ID or OSS_ACCESS_KEY_ID")
    if not MNS_ACCESS_KEY_SECRET:
        missing.append("MNS_ACCESS_KEY_SECRET or OSS_ACCESS_KEY_SECRET")
    if missing:
        print("[ERROR] missing MNS config: " + ", ".join(missing))
        sys.exit(1)
    if not (1 <= MNS_VISIBILITY_TIMEOUT <= 43200) and MNS_VISIBILITY_TIMEOUT != 0:
        print("[ERROR] MNS_VISIBILITY_TIMEOUT must be 0 or 1-43200")
        sys.exit(1)
    if not 1 <= MNS_RETRY_SECONDS <= 43200:
        print("[ERROR] MNS_RETRY_SECONDS must be 1-43200")
        sys.exit(1)
    if not 1 <= MNS_MAX_ATTEMPTS <= 100:
        print("[ERROR] MNS_MAX_ATTEMPTS must be 1-100")
        sys.exit(1)
    if MNS_EVENT not in _HANDLERS:
        print(f"[ERROR] no handler for MNS_EVENT={MNS_EVENT}")
        sys.exit(1)


def main():
    _check_config()
    rx.LOG_PREFIX = "run_extract_mns"
    signal.signal(signal.SIGINT, rx._handle_signal)
    signal.signal(signal.SIGTERM, rx._handle_signal)

    try:
        queue = _open_queue(MNS_QUEUE_NAME)
        meta = queue.get_attributes()
        if MNS_WRITE_RESULT:
            _open_queue(MNS_RESULT_QUEUE_NAME).get_attributes()
    except Exception as e:
        print(f"[ERROR] MNS connect failed: {e}")
        sys.exit(1)

    print(f"[{rx._now()}] MCAP Extract Service (MNS) started")
    print(f"  MNS:           {MNS_ENDPOINT}  queue={MNS_QUEUE_NAME}")
    if MNS_WRITE_RESULT:
        print(f"  Result queue:  {MNS_RESULT_QUEUE_NAME}")
    else:
        print("  Result queue:  off")
    print(f"  Event:         {MNS_EVENT}  (manual ack)")
    print(f"  Base64 body:   {MNS_BASE64}")
    print(f"  Wait seconds:  {MNS_WAIT_SECONDS}")
    print(f"  Visibility:    {MNS_VISIBILITY_TIMEOUT}s  (queue default {meta.visibility_timeout}s)")
    print(f"  Retry delay:   {MNS_RETRY_SECONDS}s")
    print(f"  Max attempts:  {MNS_MAX_ATTEMPTS}  (failure result written once)")
    print(f"  Workers:       {rx.N_WORKERS}")
    print(f"  Log dir:       {rx.LOG_DIR.resolve()}  ({rx.LOG_PREFIX}_YYYYMMDD.log)")
    print(f"  Mounts:")
    print(f"    scheme:   {rx.OSS_URI_SCHEME}<bucket>/<key> -> {rx.LOCAL_DATA_MOUNT}<bucket>/<key>")
    print(f"    head_cam: {rx.OSS_HEAD_CAM_PREFIX} -> {rx.LOCAL_HEAD_CAM_MOUNT}")
    print(f"    teleop:   {rx.OSS_TELEOP_PREFIX} -> {rx.LOCAL_TELEOP_MOUNT}")
    print(f"    Astribot: {rx.OSS_ASTRIBOT_PREFIX} -> {rx.LOCAL_ASTRIBOT_MOUNT}")
    rx._append_log(
        f"{'='*72}\n"
        f"[{rx._now()}] service started  workers={rx.N_WORKERS} "
        f"mns queue={MNS_QUEUE_NAME} result={'off' if not MNS_WRITE_RESULT else MNS_RESULT_QUEUE_NAME}\n"
    )

    threads = []
    for i in range(rx.N_WORKERS):
        t = threading.Thread(target=_worker_loop, args=(i,), name=f"mns-worker-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[{rx._now()}] Service stopped.")


if __name__ == "__main__":
    main()
