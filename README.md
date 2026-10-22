# MCAP training video extraction

The default Docker entrypoint (`run_extract.py`) consumes Redis tasks. The
alternative `run_extract_mns.py` consumes MNS tasks and reports completion.
Both map `oss://<bucket>/<key>` to mounted files at `/data/<bucket>/<key>`.

## Head-camera timestamp outputs

`extract_head_train_mp4.py` creates a full-length MP4 and a corresponding
`{CAM}_timestamp.txt` for each available FL/FR/RL/RR camera. Teleoperation output
is unchanged and does not include timestamp files.

Example with `OSS_OUTPUT_PREFIX=processed_data`:

```text
Input:  oss://<bucket>/original_data/.../recording.mcap
Output: oss://<OSS_BUCKET>/processed_data/.../recording/train_data/FL.mp4
        oss://<OSS_BUCKET>/processed_data/.../recording/train_data/FL_timestamp.txt
```

Without OSS output, files are written under `OUT_DIR` (default `./output`).
Each timestamp file is UTF-8 without BOM, with one integer nanosecond timestamp
per line (LF, no header or blank lines). Line 1 corresponds to display frame 0.
Values are the camera's original MCAP `message.log_time`, never MP4 PTS/DTS,
floating-point seconds or FL-aligned values. Duplicate timestamps are preserved.
Empty or missing payloads do not contribute timestamps.

The extractor decodes source payloads for validation, carrying message indices
as packet PTS. It rejects corrupt, missing, multiple or reordered display frames
that cannot satisfy one frame per accepted message in message order. It also
decodes the finished MP4 and requires its frame count to match the timestamp
count. Video output is still copied, not re-encoded. This adds CPU cost; tune
`N_WORKERS` to the available CPU and memory. Representative production MCAPs
still need acceptance testing, particularly for codec compatibility.

All attempted cameras are validated before publication begins. A missing camera
topic is allowed; an existing camera with zero valid frames fails. No timestamps
are invented, truncated or padded to make counts match.

## Existing outputs and failures

| Setting / output state | Behavior |
| --- | --- |
| `OVERWRITE=1` (default) | Rebuild both files |
| `OVERWRITE=0`, both files exist | Skip; existing contents are not revalidated |
| `OVERWRITE=0`, neither exists | Generate both files |
| `OVERWRITE=0`, only one exists | Fail; rerun with `OVERWRITE=1` to rebuild |

Historical MP4-only backfill is not supported. Rebuilding makes the video and
timestamps come from the same extraction. Publication of two files is not
atomic: local replacements are atomic per file, and OSS uploads are sequential.
A failure can leave partial or mixed generations. The task reports success only
after all required files are saved; rerun with `OVERWRITE=1` after a failed upload.
Downstream consumers should use successful task completion, not directory
presence, as the readiness signal. Redis failures require external requeueing;
the existing Redis consumer does not automatically retry failed tasks.
