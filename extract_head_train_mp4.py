#!/usr/bin/env python3
"""
MCAP -> full-length MP4 for all four head cameras (FR / FL / RR / RL).

Unlike extract_head_mp4.py this script:
  - converts every frame (not the middle 10s preview clip)
  - writes all four camera topics (H264 or H265, from CompressedVideo.format)
  - muxes the original bitstream into MP4 (no resize, no re-encode)
  - stores outputs under train_data/ instead of preview/
  - does not generate first-frame JPG thumbnails
  - writes {CAM}_timestamp.txt with original message.log_time in integer ns
  - validates one display frame per nonempty payload and final MP4 frame count

All parameters via environment variables:
  MCAP_PATH        MCAP file path or folder containing .mcap files (required)
  OUT_DIR          local output directory (default: ./output)
  FPS              MP4 timestamp fps, 0 = auto from timestamps (default: 0)
  OVERWRITE        rebuild MP4/timestamp pairs (default: 1; 0 skips complete pairs,
                   fails on incomplete pairs)
  N_WORKERS        parallel files to process (default: 2)
  OSS_OUTPUT_PREFIX  when set, upload to OSS (original_data -> this prefix)

When MCAP_PATH is a folder or an oss:// prefix, all .mcap files under it are processed.

Output layout (local or OSS)::

    .../<mcap_stem>/train_data/FR.mp4
    .../<mcap_stem>/train_data/FL.mp4
    .../<mcap_stem>/train_data/RR.mp4
    .../<mcap_stem>/train_data/RL.mp4
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path, PurePosixPath

import av
from mcap.reader import make_reader
import oss2
import tempfile
import shutil
import gc


CAMERA_TOPICS = {
    "FR": "/camera/FR/video",
    "FL": "/camera/FL/video",
    "RR": "/camera/RR/video",
    "RL": "/camera/RL/video",
}
TOPIC_TO_CAM = {topic: cam for cam, topic in CAMERA_TOPICS.items()}
OUTPUT_SUBDIR = "train_data"


# ---- protobuf decode helpers (for FR H265 format) ----

def _read_varint(data: bytes, pos: int):
    v = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return v, pos


def skip_field(data: bytes, pos: int):
    if pos >= len(data):
        return pos
    tag = data[pos]
    wire_type = tag & 0x07
    pos += 1
    if wire_type == 0:
        _, pos = _read_varint(data, pos)
    elif wire_type == 2:
        sz, pos = _read_varint(data, pos)
        pos += sz
    elif wire_type == 5:
        pos += 4
    elif wire_type == 1:
        pos += 8
    else:
        raise ValueError(f"Unknown wire_type {wire_type} at pos {pos - 1}")
    return pos


def extract_field(data: bytes, field_num: int):
    tag_byte = (field_num << 3) | 2
    pos = 0
    while pos < len(data):
        if data[pos] == tag_byte:
            pos += 1
            sz, pos = _read_varint(data, pos)
            return data[pos:pos + sz]
        pos = skip_field(data, pos)
    return None


# ---- format / fps helpers ----

def list_camera_topics(mcap_path: Path) -> dict[str, str]:
    """Return {cam: topic} for cameras present in the MCAP."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        topics = {ch.topic for ch in summary.channels.values()}

    found = {cam: topic for cam, topic in CAMERA_TOPICS.items() if topic in topics}
    if not found:
        print(f"  WARNING: none of the four camera topics found, skipping: {mcap_path}")
        print(f"    expected: {list(CAMERA_TOPICS.values())}")
    return found


def _normalize_codec(format_str: str | None) -> str | None:
    """Map CompressedVideo.format / codec string to a PyAV decoder name."""
    if not format_str:
        return None
    s = format_str.strip().lower()
    if s.startswith("video/"):
        s = s[6:]
    if s in ("h264", "avc", "avc1", "avc3"):
        return "h264"
    if s in ("h265", "hevc", "hvc1", "hev1"):
        return "hevc"
    return None


def _codec_from_annexb(data: bytes) -> str | None:
    """Guess h264 vs hevc from an Annex-B NAL start code."""
    if data.startswith(b"\x00\x00\x00\x01") and len(data) > 4:
        nal = data[4]
    elif data.startswith(b"\x00\x00\x01") and len(data) > 3:
        nal = data[3]
    else:
        return None
    hevc_type = (nal >> 1) & 0x3F
    if hevc_type in (32, 33, 34):  # VPS / SPS / PPS
        return "hevc"
    h264_type = nal & 0x1F
    if h264_type in (1, 5, 6, 7, 8, 9):
        return "h264"
    return None


def inspect_camera_topics(mcap_path: Path, topics: list[str],
                          fps_override: int, sample: int = 30) -> dict[str, tuple[float, str]]:
    """One pass: fps + bitstream codec per topic. Codec from format field, else Annex-B."""
    time_needed = sample if fps_override <= 0 else 1
    times: dict[str, list[int]] = {t: [] for t in topics}
    codecs: dict[str, str | None] = {t: None for t in topics}
    pending = set(topics)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _sch, ch, msg in reader.iter_messages(topics=topics):
            topic = ch.topic
            if topic not in pending:
                continue
            data = extract_field(msg.data, 3)
            if not data:
                continue
            if codecs[topic] is None:
                fmt = extract_field(msg.data, 4)
                fmt_str = fmt.decode("utf-8", errors="replace") if fmt else None
                codecs[topic] = _normalize_codec(fmt_str)
                if codecs[topic] is None:
                    codecs[topic] = _codec_from_annexb(data)
            if len(times[topic]) < time_needed:
                times[topic].append(msg.log_time)
            if len(times[topic]) >= time_needed and codecs[topic] is not None:
                pending.discard(topic)
            if not pending:
                break

    specs: dict[str, tuple[float, str]] = {}
    for topic in topics:
        fps = float(fps_override) if fps_override > 0 else 30.0
        ts = times[topic]
        if fps_override <= 0 and len(ts) >= 2:
            dur_s = (ts[-1] - ts[0]) / 1e9
            if dur_s > 0:
                fps = (len(ts) - 1) / dur_s
        # Old FR recordings were HEVC and may omit format; keep that default.
        specs[topic] = (fps, codecs[topic] or "hevc")
    return specs


# ---- bitstream copy muxer ----

def _has_annexb_start(data: bytes) -> bool:
    return data.startswith(b"\x00\x00\x00\x01") or data.startswith(b"\x00\x00\x01")


def _remux_annexb_to_mp4(annexb_path: Path, mp4_path: Path, codec: str, fps: float):
    """Wrap original Annex-B H264/HEVC in MP4 without re-encoding.

    Raw elementary streams have no timestamps. We stamp PTS/DTS at ``fps``
    and write ``moov`` first (faststart) so browsers/OSS players can play.
    """
    fmt = "h264" if codec == "h264" else "hevc"
    fps = float(fps) if fps and fps > 0 else 30.0
    fps_int = max(1, int(round(fps)))
    tb = Fraction(1, fps_int)

    inp = av.open(
        str(annexb_path),
        format=fmt,
        options={"framerate": str(fps_int)},
    )
    try:
        in_stream = next((s for s in inp.streams if s.type == "video"), None)
        if in_stream is None:
            raise RuntimeError(f"no video stream parsed from {annexb_path.name}")
        out = av.open(
            str(mp4_path),
            "w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        try:
            out_stream = out.add_stream_from_template(in_stream)
            out_stream.time_base = tb
            frame_i = 0
            for packet in inp.demux(in_stream):
                if packet.size <= 0:
                    continue
                if packet.dts is not None and packet.dts < 0:
                    continue
                packet.stream = out_stream
                packet.pts = frame_i
                packet.dts = frame_i
                packet.duration = 1
                packet.time_base = tb
                out.mux(packet)
                frame_i += 1
            if frame_i == 0:
                raise RuntimeError(f"no packets remuxed from {annexb_path.name}")
        finally:
            out.close()
    finally:
        inp.close()


class StreamCopyMuxer:
    """Write original Annex-B packets, then remux to MP4 (codec copy)."""

    def __init__(self, output_path: Path, fps: float, label: str, codec: str = "hevc"):
        self.output_path = output_path
        self.fps = fps
        self.label = label
        self.codec = codec
        self.n = 0
        suffix = ".h264" if codec == "h264" else ".h265"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        self._annexb_path = Path(tmp.name)
        self._annexb_fp = open(self._annexb_path, "wb")
        self._closed = False

    def feed(self, raw_payload: bytes):
        if not raw_payload:
            return
        if not _has_annexb_start(raw_payload):
            self._annexb_fp.write(b"\x00\x00\x00\x01")
        self._annexb_fp.write(raw_payload)
        self.n += 1
        if self.n % 300 == 0:
            print(f"    [{self.label}] copied {self.n} packets "
                  f"({self.n / self.fps:.1f}s)")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._annexb_fp is not None:
            self._annexb_fp.close()
            self._annexb_fp = None
        try:
            if self.n == 0:
                print(f"  [{self.label}] done: no packets")
                return
            _remux_annexb_to_mp4(self._annexb_path, self.output_path, self.codec, self.fps)
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            print(f"  [{self.label}] done: {self.output_path.name}  "
                  f"(copy {self.codec}, {self.n} packets, "
                  f"{self.n / self.fps:.1f}s, {size_mb:.1f}MB)")
        finally:
            self._annexb_path.unlink(missing_ok=True)

    def abort(self):
        """Close without remuxing (used on failure)."""
        if self._closed:
            return
        self._closed = True
        if self._annexb_fp is not None:
            try:
                self._annexb_fp.close()
            except Exception:
                pass
            self._annexb_fp = None
        self._annexb_path.unlink(missing_ok=True)


# ---- OSS helpers ----

def _get_oss_client(bucket_name=None, endpoint=None):
    """Initialize OSS client from environment variables.

    If bucket_name is provided, it overrides the OSS_BUCKET env var.
    If endpoint is provided, it overrides the OSS_ENDPOINT env var.
    """
    ep = endpoint or os.environ.get("OSS_ENDPOINT", "")
    ak_id = os.environ.get("OSS_ACCESS_KEY_ID", "")
    ak_secret = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    if bucket_name is None:
        bucket_name = os.environ.get("OSS_BUCKET", "")
    if not all([ep, ak_id, ak_secret, bucket_name]):
        sys.exit("OSS_ENDPOINT / OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / OSS_BUCKET must be set")
    auth = oss2.Auth(ak_id, ak_secret)
    return oss2.Bucket(auth, ep, bucket_name)


def _parse_oss_uri(uri: str):
    """Parse oss:// URI into (bucket_name, key, endpoint).

    Supports two formats:
      oss://bucket/key                → endpoint from OSS_ENDPOINT env
      oss://bucket.endpoint/key       → virtual-hosted style, endpoint extracted
    """
    uri = uri[len("oss://"):]
    parts = uri.split("/", 1)
    host = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    dot_oss = host.find(".oss-")
    if dot_oss > 0:
        bucket_name = host[:dot_oss]
        endpoint = host[dot_oss + 1:]
        return bucket_name, key, endpoint
    return host, key, None


def _compute_oss_output_prefix(source_key: str, oss_output_prefix: str) -> str:
    """Compute OSS output prefix by replacing 'original_data' with OSS_OUTPUT_PREFIX.

    source_key: OSS key like 'original_data/ego_human_data/head_cam/260706/001U_device/file.mcap'
    Returns: like 'processed_data/ego_human_data/head_cam/260706/001U_device/file/train_data'
    """
    if not oss_output_prefix:
        return ""
    replaced = source_key.replace("original_data", oss_output_prefix, 1)
    if replaced.endswith(".mcap"):
        stem = PurePosixPath(replaced).stem
        dir_path = str(PurePosixPath(replaced).parent)
        return f"{dir_path}/{stem}/{OUTPUT_SUBDIR}"
    dir_path = replaced.rstrip("/")
    return f"{dir_path}/{OUTPUT_SUBDIR}"


def _object_exists(bucket, oss_key: str) -> bool:
    try:
        return bucket.object_exists(oss_key)
    except Exception:
        return False


class _FrameMappingValidator:
    """Require one decoded display frame per payload, in message order.

    Synthetic packet PTS carry message indices through decoder buffering.
    Reordered, missing or multiple frames are rejected, never guessed.
    This validation is head-camera only; it does not change encoded output.
    """

    def __init__(self, codec: str):
        self.decoder = av.CodecContext.create(codec, "r")
        self.decoder.options = {"err_detect": "explode"}
        self.accepted = 0
        self.decoded = 0

    def _check(self, frames):
        for frame in frames:
            if frame.is_corrupt or frame.pts != self.decoded:
                raise RuntimeError(
                    f"frame mapping invalid: display_index={self.decoded}, "
                    f"message_index={frame.pts}, corrupt={frame.is_corrupt}"
                )
            self.decoded += 1

    def feed(self, payload: bytes):
        if not _has_annexb_start(payload):
            payload = b"\x00\x00\x00\x01" + payload
        packet = av.Packet(payload)
        packet.pts = packet.dts = self.accepted
        packet.time_base = Fraction(1, 1)
        self._check(self.decoder.decode(packet))
        self.accepted += 1

    def finish(self):
        self._check(self.decoder.decode(None))
        if self.decoded != self.accepted:
            raise RuntimeError(
                f"frame mapping count mismatch: payloads={self.accepted}, frames={self.decoded}"
            )


def _count_video_frames(path: Path) -> int:
    """Count decoded frames, not container metadata or duration * FPS."""
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.options = {"err_detect": "explode"}
        count = 0
        for frame in container.decode(video=0):
            if frame.is_corrupt:
                raise RuntimeError(f"corrupt output frame: {path.name}, index={count}")
            count += 1
        return count


# ---- single MCAP processor ----

def process_one(mcap_path: Path, train_dir: Path,
                fps_override: int, bucket=None, oss_key_prefix: str = "") -> bool:
    """Process a single MCAP file -> four full-length original-quality MP4s.

    Returns True on success (including skip-existing), False if no cameras exist.
    Invalid frames, incomplete pairs and publication failures raise an exception.
    """
    cameras = list_camera_topics(mcap_path)
    if not cameras:
        return False

    raw_overwrite = os.environ.get("OVERWRITE", "1").strip().lower()
    overwrite = raw_overwrite not in ("0", "false", "no", "off")
    print(f"  Cameras in file: {', '.join(cameras)}")

    need: dict[str, str] = {}
    for cam, topic in cameras.items():
        if not overwrite:
            names = (f"{cam}.mp4", f"{cam}_timestamp.txt")
            # Propagate OSS errors: a failed existence check is not permission to overwrite.
            exists = [bucket.object_exists(f"{oss_key_prefix}/{name}")
                      if bucket and oss_key_prefix else (train_dir / name).exists()
                      for name in names]
            if all(exists):
                print(f"  SKIP {cam}: MP4 and timestamps already exist (OVERWRITE=0)")
                continue
            if any(exists):
                raise RuntimeError(
                    f"{cam}: incomplete MP4/timestamp pair; use OVERWRITE=1 to rebuild"
                )
        need[cam] = topic

    if not need:
        print("  SKIP: all camera MP4/timestamp pairs already exist")
        return True

    specs = inspect_camera_topics(mcap_path, list(need.values()), fps_override)
    muxers: dict[str, StreamCopyMuxer] = {}
    tmp_paths: dict[str, Path] = {}
    timestamp_paths: dict[str, Path] = {}
    timestamps = {cam: [] for cam in need}
    message_counts = {cam: 0 for cam in need}
    validators = {}
    try:
        for cam, topic in need.items():
            fps, codec = specs[topic]
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            tmp_paths[cam] = tmp_path
            muxers[cam] = StreamCopyMuxer(tmp_path, fps, cam, codec=codec)
            validators[cam] = _FrameMappingValidator(codec)
            print(f"  Copying {cam} ({topic}) @ {fps:.2f} fps, codec={codec}, original bitstream")

        topics = list(need.values())
        print(f"  Streaming {len(need)} camera(s) from {mcap_path.name}...")
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for _sch, ch, msg in reader.iter_messages(topics=topics):
                cam = TOPIC_TO_CAM.get(ch.topic)
                mux = muxers.get(cam)
                if mux is None:
                    continue
                message_counts[cam] += 1
                data_field = extract_field(msg.data, 3)
                if not data_field:
                    continue
                try:
                    validators[cam].feed(data_field)
                except Exception as exc:
                    raise RuntimeError(
                        f"{mcap_path} [{cam}] message={message_counts[cam]}: {exc}"
                    ) from exc
                mux.feed(data_field)
                timestamps[cam].append(int(msg.log_time))

        # Validate every camera before publishing any new output for this MCAP.
        for cam, mux in muxers.items():
            if not timestamps[cam]:
                raise RuntimeError(f"{mcap_path} [{cam}]: no valid frames")
            try:
                validators[cam].finish()
            except Exception as exc:
                raise RuntimeError(f"{mcap_path} [{cam}]: {exc}") from exc
            mux.close()
            muxers[cam] = None
            tmp_path = tmp_paths[cam]
            frame_count = _count_video_frames(tmp_path)
            print(f"  [{cam}] messages={message_counts[cam]}, payloads={mux.n}, "
                  f"skipped={message_counts[cam] - mux.n}, frames={frame_count}, "
                  f"timestamps={len(timestamps[cam])}")
            if frame_count != len(timestamps[cam]):
                raise RuntimeError(f"{mcap_path} [{cam}]: MP4/timestamp count mismatch")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             suffix=".txt", delete=False) as output:
                timestamp_paths[cam] = Path(output.name)
                output.writelines(f"{stamp}\n" for stamp in timestamps[cam])

        for cam in need:
            for name, source in ((f"{cam}.mp4", tmp_paths[cam]),
                                 (f"{cam}_timestamp.txt", timestamp_paths[cam])):
                if bucket and oss_key_prefix:
                    oss_key = f"{oss_key_prefix}/{name}"
                    bucket.put_object_from_file(oss_key, str(source))
                    print(f"    uploaded {oss_key}")
                else:
                    train_dir.mkdir(parents=True, exist_ok=True)
                    # Stage on the destination filesystem for atomic per-file replacement.
                    with tempfile.NamedTemporaryFile(dir=train_dir, delete=False) as staged:
                        staged_path = Path(staged.name)
                    try:
                        shutil.copyfile(source, staged_path)
                        os.replace(staged_path, train_dir / name)
                    finally:
                        staged_path.unlink(missing_ok=True)
                    print(f"    saved {train_dir / name}")
    finally:
        for mux in muxers.values():
            if mux is not None:
                try:
                    mux.abort()
                except Exception:
                    pass
        for tmp_path in tmp_paths.values():
            tmp_path.unlink(missing_ok=True)
        for tmp_path in timestamp_paths.values():
            tmp_path.unlink(missing_ok=True)

    return True


# ---- main ----

def main():
    mcap_path_str = os.environ.get("MCAP_PATH", "")
    if not mcap_path_str:
        sys.exit("MCAP_PATH is not set, exiting.")

    out_dir = Path(os.environ.get("OUT_DIR", "./output"))
    fps_override = int(os.environ.get("FPS", "0"))

    oss_prefix = os.environ.get("OSS_OUTPUT_PREFIX", "")
    bucket = None
    if oss_prefix:
        bucket = _get_oss_client()
        print(f"OSS upload enabled: bucket={os.environ.get('OSS_BUCKET', '')}, "
              f"prefix={oss_prefix}")

    if mcap_path_str.startswith("oss://"):
        src_bucket_name, src_key, src_endpoint = _parse_oss_uri(mcap_path_str)

        if bucket and os.environ.get("OSS_BUCKET", "") == src_bucket_name:
            src_bucket = bucket
        else:
            src_bucket = _get_oss_client(src_bucket_name, src_endpoint)

        is_single = src_key.endswith(".mcap")

        if is_single:
            keys = [src_key]
        else:
            prefix = src_key.rstrip("/") + "/" if src_key and not src_key.endswith("/") else src_key
            keys = []
            for obj in oss2.ObjectIterator(src_bucket, prefix=prefix):
                if obj.key.endswith(".mcap"):
                    keys.append(obj.key)
            keys.sort()
            if not keys:
                sys.exit(f"No .mcap files found at {mcap_path_str}")
            print(f"Found {len(keys)} .mcap file(s) at {mcap_path_str}")

        n_workers = min(int(os.environ.get("N_WORKERS", "2")), os.cpu_count() or 2, len(keys))

        def _process_oss_file(idx_key):
            i, oss_key = idx_key
            mcap_name = Path(oss_key).name
            out_oss_prefix = _compute_oss_output_prefix(oss_key, oss_prefix)
            rel_dir = Path(oss_key).stem

            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, mcap_name)
            try:
                src_bucket.get_object_to_file(oss_key, tmp_path)

                train_dir = out_dir / rel_dir / OUTPUT_SUBDIR
                train_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n[{i}/{len(keys)}] Processing: {oss_key}")
                ok = process_one(Path(tmp_path), train_dir, fps_override,
                                 bucket, out_oss_prefix)
                gc.collect()
                if not ok:
                    raise RuntimeError(f"extract failed: {oss_key}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        n_fail = 0
        if len(keys) == 1:
            try:
                _process_oss_file((1, keys[0]))
            except Exception as e:
                print(f"Error processing {keys[0]}: {e}")
                n_fail = 1
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_oss_file, (i, k)): k
                           for i, k in enumerate(keys, 1)}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        n_fail += 1
                        print(f"Error processing {futures[future]}: {e}")

        print(f"\n=== All done! {len(keys)} file(s) processed, {n_fail} failed. ===")
        if n_fail:
            sys.exit(1)
        return

    mcap_path = Path(mcap_path_str)
    if not mcap_path.exists():
        sys.exit(f"MCAP path not found: {mcap_path}")

    oss_source = os.environ.get("OSS_SOURCE_PATH", "")
    oss_source_key = ""
    if oss_source.startswith("oss://") and oss_prefix:
        _, oss_source_key, _ = _parse_oss_uri(oss_source)

    if mcap_path.is_dir():
        mcap_files = sorted(mcap_path.rglob("*.mcap"))
        if not mcap_files:
            sys.exit(f"No .mcap files found in {mcap_path}")
        n_workers = min(int(os.environ.get("N_WORKERS", "2")), os.cpu_count() or 2, len(mcap_files))
        print(f"Found {len(mcap_files)} .mcap file(s) in {mcap_path}, using {n_workers} workers")

        def _process_file(idx_mf):
            i, mf = idx_mf
            rel = mf.relative_to(mcap_path)
            train_dir = out_dir / rel.parent / rel.stem / OUTPUT_SUBDIR
            train_dir.mkdir(parents=True, exist_ok=True)
            if oss_source_key:
                source_key = f"{oss_source_key.rstrip('/')}/{rel.as_posix()}"
                oss_key_prefix = _compute_oss_output_prefix(source_key, oss_prefix)
            else:
                oss_key_prefix = (
                    f"{oss_prefix}/{rel.with_suffix('').as_posix()}/{OUTPUT_SUBDIR}"
                    if oss_prefix else ""
                )
            print(f"\n[{i}/{len(mcap_files)}] Processing: {rel}")
            ok = process_one(mf, train_dir, fps_override, bucket, oss_key_prefix)
            gc.collect()
            if not ok:
                raise RuntimeError(f"extract failed: {rel}")

        n_fail = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_file, (i, mf)): mf
                       for i, mf in enumerate(mcap_files, 1)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    n_fail += 1
                    print(f"Error processing {futures[future]}: {e}")
            gc.collect()

        print(f"\n=== All done! {len(mcap_files)} file(s) processed, {n_fail} failed. ===")
        if n_fail:
            sys.exit(1)
        return

    if oss_source_key:
        oss_key_prefix = _compute_oss_output_prefix(oss_source_key, oss_prefix)
    else:
        oss_key_prefix = (
            f"{oss_prefix}/{mcap_path.stem}/{OUTPUT_SUBDIR}" if oss_prefix else ""
        )
    print(f"Processing: {mcap_path.name}")
    train_dir = out_dir / mcap_path.stem / OUTPUT_SUBDIR
    train_dir.mkdir(parents=True, exist_ok=True)
    ok = process_one(mcap_path, train_dir, fps_override, bucket, oss_key_prefix)
    gc.collect()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
