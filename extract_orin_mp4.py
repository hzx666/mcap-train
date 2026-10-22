#!/usr/bin/env python3
"""
MCAP -> full-length MP4 for all four teleoperation cameras.

Same operation as extract_head_train_mp4.py:
  - every frame, not the middle 10s preview clip
  - all cameras present in the file, not the first usable one
  - H264/H265 payloads are copied into MP4 without resize or re-encode
  - JPEG/raw image topics are encoded at their original resolution
  - outputs under train_data/, no first-frame JPG

Cameras:
  torso, head, right, left

Output layout (local or OSS)::

    .../<mcap_stem>/train_data/torso.mp4
    .../<mcap_stem>/train_data/head.mp4
    .../<mcap_stem>/train_data/right.mp4
    .../<mcap_stem>/train_data/left.mp4
"""

import os
import struct
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader

import extract_head_train_mp4 as head

# Prefer the compressed image topic. Fall back to the raw image topic.
CAMERA_CANDIDATES = {
    "torso": (
        "/torso_camera/torso_camera/color/image_raw/compressed",
        "/torso_camera/torso_camera/color/image_raw",
    ),
    "head": (
        "/head_camera/head_camera/color/image_raw/compressed",
        "/head_camera/head_camera/color/image_raw",
    ),
    "right": (
        "/right_camera/right_camera/color/image_raw/compressed",
        "/right_camera/right_camera/color/image_raw",
    ),
    "left": (
        "/left_camera/left_camera/color/image_raw/compressed",
        "/left_camera/left_camera/color/image_raw",
    ),
}


def _list_cameras(mcap_path: Path) -> dict[str, str]:
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        topics = {ch.topic for ch in summary.channels.values()}
        available = []
        if summary and summary.statistics:
            for cid, ch in summary.channels.items():
                count = summary.statistics.channel_message_counts.get(cid, 0)
                if count > 0:
                    available.append((count, ch.topic))

    found = {}
    for cam, candidates in CAMERA_CANDIDATES.items():
        for topic in candidates:
            if topic in topics:
                found[cam] = topic
                break
    if not found:
        print(f"  WARNING: none of the teleop camera topics found, skipping: {mcap_path}")
        print(f"    expected: {[t for pair in CAMERA_CANDIDATES.values() for t in pair]}")
        if available:
            print(f"    available ({len(available)}):")
            for count, topic in sorted(available, key=lambda item: item[1]):
                print(f"      [{count:>7} msgs] {topic}")
    return found


def _parse_cdr_image(data: bytes) -> dict | None:
    candidates = [b"torso_camera_color_optical_frame", b"camera_color_optical_frame"]
    idx = -1
    for fid in candidates:
        idx = data.find(fid)
        if idx >= 12:
            break
    if idx < 12:
        return None
    flen = struct.unpack_from("<I", data, idx - 4)[0]
    if flen > 256 or idx + flen > len(data):
        return None
    offset = idx + flen
    if offset % 4 != 0:
        offset += 4 - (offset % 4)
    height = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    width = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    enclen = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    encoding = data[offset:offset + enclen].rstrip(b"\x00").decode()
    bpp_map = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8UC1": 1, "8UC3": 3}
    bpp = bpp_map.get(encoding, 3)
    data_size = height * width * bpp
    if data_size <= 0 or data_size > len(data):
        return None
    return {
        "height": height,
        "width": width,
        "encoding": encoding,
        "data": data[len(data) - data_size:],
    }


def _to_bgr(info: dict) -> np.ndarray | None:
    h, w = info["height"], info["width"]
    raw = info["data"]
    enc = info["encoding"]
    try:
        if enc == "rgb8":
            return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        if enc == "bgr8":
            return np.frombuffer(raw, np.uint8).reshape((h, w, 3))
        if enc in ("mono8", "8UC1"):
            return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape((h, w)), cv2.COLOR_GRAY2BGR)
        return np.frombuffer(raw, np.uint8).reshape((h, w, 3))
    except ValueError:
        return None


def _decode_compressed_image(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    for marker in (b"\xff\xd8", b"\x89PNG"):
        idx = data.find(marker)
        if idx >= 0:
            img = cv2.imdecode(np.frombuffer(data[idx:], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def _is_video_payload(data: bytes) -> bool:
    fmt = head.extract_field(data, 4)
    if fmt and head._normalize_codec(fmt.decode("utf-8", errors="replace")):
        return True
    payload = head.extract_field(data, 3)
    if payload and (head._has_annexb_start(payload) or head._codec_from_annexb(payload)):
        return True
    return head._has_annexb_start(data)


def _video_payload(data: bytes) -> bytes | None:
    payload = head.extract_field(data, 3)
    if payload:
        return payload
    if head._has_annexb_start(data):
        return data
    return None


def _frame_from_msg(data: bytes) -> np.ndarray | None:
    frame = _decode_compressed_image(data)
    if frame is not None:
        return frame
    info = _parse_cdr_image(data)
    if info is None:
        return None
    return _to_bgr(info)


class _ImageEncoder:
    """Encode original-resolution frames to MP4. Used when the topic is not a video bitstream."""

    def __init__(self, output_path: Path, fps: float, label: str):
        self.output_path = output_path
        self.fps = fps if fps and fps > 0 else 30.0
        self.label = label
        self.n = 0
        self._container = None
        self._stream = None
        self._width = 0
        self._height = 0
        self._closed = False

    def feed(self, frame: np.ndarray):
        frame = _even_frame(frame)
        if self._container is None:
            self._height, self._width = frame.shape[:2]
            fps_int = max(1, int(round(self.fps)))
            self._container = av.open(
                str(self.output_path),
                "w",
                format="mp4",
                options={"movflags": "+faststart"},
            )
            self._stream = self._container.add_stream("libx264", rate=fps_int)
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = "yuv420p"
            self._stream.time_base = Fraction(1, fps_int)
            self._stream.options["crf"] = "18"
            self._stream.options["preset"] = "fast"
        elif frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.n
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)
        self.n += 1
        if self.n % 300 == 0:
            print(f"    [{self.label}] encoded {self.n} frames "
                  f"({self.n / self.fps:.1f}s)")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._stream = None
        if self._container is not None:
            self._container.close()
            self._container = None
        if self.n == 0:
            print(f"  [{self.label}] done: no frames")
            return
        size_mb = self.output_path.stat().st_size / (1024 * 1024)
        print(f"  [{self.label}] done: {self.output_path.name}  "
              f"({self.n} frames, {self.n / self.fps:.1f}s, {size_mb:.1f}MB)")

    def abort(self):
        if self._closed:
            return
        self._closed = True
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        self.output_path.unlink(missing_ok=True)


def _even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return frame


def _probe(mcap_path: Path, topics: list[str], fps_override: int, sample: int = 30):
    """Return {topic: (fps, is_video, codec)} from the first messages of each topic."""
    time_needed = sample if fps_override <= 0 else 1
    times = {topic: [] for topic in topics}
    video = {topic: None for topic in topics}
    codecs = {topic: None for topic in topics}
    pending = set(topics)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _sch, ch, msg in reader.iter_messages(topics=topics):
            topic = ch.topic
            if topic not in pending:
                continue
            if video[topic] is None:
                video[topic] = _is_video_payload(msg.data)
                if video[topic]:
                    fmt = head.extract_field(msg.data, 4)
                    fmt_str = fmt.decode("utf-8", errors="replace") if fmt else None
                    codecs[topic] = head._normalize_codec(fmt_str)
                    if codecs[topic] is None:
                        payload = _video_payload(msg.data)
                        if payload:
                            codecs[topic] = head._codec_from_annexb(payload)
            if len(times[topic]) < time_needed:
                times[topic].append(msg.log_time)
            if len(times[topic]) >= time_needed and video[topic] is not None:
                pending.discard(topic)
            if not pending:
                break

    specs = {}
    for topic in topics:
        fps = float(fps_override) if fps_override > 0 else 30.0
        ts = times[topic]
        if fps_override <= 0 and len(ts) >= 2:
            dur_s = (ts[-1] - ts[0]) / 1e9
            if dur_s > 0:
                fps = (len(ts) - 1) / dur_s
        specs[topic] = (fps, bool(video[topic]), codecs[topic] or "hevc")
    return specs


def process_one(mcap_path: Path, train_dir: Path,
                fps_override: int, bucket=None, oss_key_prefix: str = "") -> bool:
    """Write one full-length MP4 per teleop camera that exists in the file."""
    cameras = _list_cameras(mcap_path)
    if not cameras:
        return False

    raw_overwrite = os.environ.get("OVERWRITE", "1").strip().lower()
    overwrite = raw_overwrite not in ("0", "false", "no", "off")
    print(f"  Cameras in file: {', '.join(cameras)}")

    need = {}
    for cam, topic in cameras.items():
        output_name = f"{cam}.mp4"
        if bucket and oss_key_prefix:
            exists = head._object_exists(bucket, f"{oss_key_prefix}/{output_name}")
        else:
            exists = (train_dir / output_name).exists()
        if exists and not overwrite:
            print(f"  SKIP {cam}: {output_name} already exists (OVERWRITE=0)")
            continue
        need[cam] = topic
    if not need:
        print("  SKIP: all camera MP4s already exist")
        return True

    topic_to_cam = {topic: cam for cam, topic in need.items()}
    specs = _probe(mcap_path, list(need.values()), fps_override)
    muxers = {}
    tmp_paths = {}
    zero_cams = []
    try:
        for cam, topic in need.items():
            fps, is_video, codec = specs[topic]
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            tmp_paths[cam] = tmp_path
            if is_video:
                muxers[cam] = head.StreamCopyMuxer(tmp_path, fps, cam, codec=codec)
                print(f"  Copying {cam} ({topic}) @ {fps:.2f} fps, codec={codec}, original bitstream")
            else:
                muxers[cam] = _ImageEncoder(tmp_path, fps, cam)
                print(f"  Encoding {cam} ({topic}) @ {fps:.2f} fps, original resolution")

        print(f"  Streaming {len(need)} camera(s) from {mcap_path.name}...")
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for _sch, ch, msg in reader.iter_messages(topics=list(need.values())):
                cam = topic_to_cam.get(ch.topic)
                mux = muxers.get(cam)
                if mux is None:
                    continue
                if isinstance(mux, head.StreamCopyMuxer):
                    payload = _video_payload(msg.data)
                    if payload:
                        mux.feed(payload)
                    continue
                frame = _frame_from_msg(msg.data)
                if frame is not None:
                    mux.feed(frame)

        for cam, mux in list(muxers.items()):
            mux.close()
            muxers[cam] = None
            tmp_path = tmp_paths[cam]
            output_name = f"{cam}.mp4"
            if mux.n == 0:
                print(f"  [{cam}] no frames, skipping upload")
                zero_cams.append(cam)
                continue
            if bucket and oss_key_prefix:
                oss_key = f"{oss_key_prefix}/{output_name}"
                bucket.put_object_from_file(oss_key, str(tmp_path))
                print(f"    uploaded {oss_key}")
            else:
                dest = train_dir / output_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_path, dest)
                print(f"    saved {dest}")
    finally:
        for mux in muxers.values():
            if mux is not None:
                try:
                    mux.abort()
                except Exception:
                    pass
        for tmp_path in tmp_paths.values():
            tmp_path.unlink(missing_ok=True)

    if zero_cams:
        print(f"  ERROR: no frames for: {', '.join(zero_cams)}")
        return False
    return True


def main():
    head.OUTPUT_SUBDIR = "train_data"
    head.process_one = process_one
    head.main()


if __name__ == "__main__":
    main()
