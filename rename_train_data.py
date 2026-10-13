#!/usr/bin/env python3
"""
从 Excel 读取 oss_path，把对应输出目录下错误的 train-data 重命名为 train_data。

OSS 没有真正的目录 rename：对每个对象 copy 到新 key，再删旧 key。

路径映射与 extract_head_train_mp4.py 一致：

  original_data/.../<stem>.mcap
    -> processed_data/.../<stem>/train-data/<cam>.mp4
    -> processed_data/.../<stem>/train_data/<cam>.mp4

用法:
  python rename_train_data.py --dry-run
  python rename_train_data.py
  python rename_train_data.py <excel.xlsx>
  python rename_train_data.py <excel.xlsx> --force

环境变量（与 extract_head_train_mp4.py 相同）:
  OSS_ENDPOINT
  OSS_ACCESS_KEY_ID
  OSS_ACCESS_KEY_SECRET
  OSS_BUCKET
  OSS_OUTPUT_PREFIX   original_data 替换成的前缀 (default: processed_data)
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath

from push_excel_redis import find_default_excel, read_oss_paths

OLD_SUBDIR = "train-data"
NEW_SUBDIR = "train_data"
COPY_SIMPLE_LIMIT = 1024 * 1024 * 1024  # OSS CopyObject 上限 1GB

OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "")
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET = os.environ.get("OSS_BUCKET", "")
OSS_OUTPUT_PREFIX = os.environ.get("OSS_OUTPUT_PREFIX", "processed_data").strip().strip("/")


def parse_oss_uri(uri: str) -> tuple[str, str]:
    """Return (bucket_or_empty, key) from oss:// URI or bare key."""
    text = str(uri).strip().strip('"').strip("'")
    idx = text.find("oss://")
    if idx < 0:
        return "", text.lstrip("/")
    text = text[idx + len("oss://"):]
    parts = text.split("/", 1)
    host = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    dot_oss = host.find(".oss-")
    bucket = host[:dot_oss] if dot_oss > 0 else host
    return bucket, key


def mcap_key_to_old_prefix(source_key: str) -> str:
    """original_data/.../file.mcap -> processed_data/.../file/train-data/"""
    replaced = source_key.replace("original_data", OSS_OUTPUT_PREFIX, 1)
    if replaced.endswith(".mcap"):
        p = PurePosixPath(replaced)
        return f"{p.parent}/{p.stem}/{OLD_SUBDIR}/"
    return f"{replaced.rstrip('/')}/{OLD_SUBDIR}/"


def dest_key_for(src_key: str) -> str:
    parts = src_key.split("/")
    return "/".join(NEW_SUBDIR if p == OLD_SUBDIR else p for p in parts)


def get_bucket():
    try:
        import oss2
    except ImportError:
        print("[ERROR] oss2 未安装，请执行: pip install oss2")
        sys.exit(1)

    missing = [
        name
        for name, val in (
            ("OSS_ENDPOINT", OSS_ENDPOINT),
            ("OSS_ACCESS_KEY_ID", OSS_ACCESS_KEY_ID),
            ("OSS_ACCESS_KEY_SECRET", OSS_ACCESS_KEY_SECRET),
            ("OSS_BUCKET", OSS_BUCKET),
        )
        if not val
    ]
    if missing:
        print("[ERROR] 缺少环境变量: " + ", ".join(missing))
        sys.exit(1)

    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)


def collect_mcap_keys(bucket, source_key: str) -> list[str]:
    import oss2

    if source_key.endswith(".mcap"):
        return [source_key]
    prefix = source_key.rstrip("/") + "/" if source_key else ""
    keys = [
        obj.key
        for obj in oss2.ObjectIterator(bucket, prefix=prefix)
        if obj.key.endswith(".mcap")
    ]
    keys.sort()
    return keys


def list_old_objects(bucket, prefix: str) -> list[tuple[str, int]]:
    import oss2

    found = []
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        if OLD_SUBDIR not in obj.key.split("/"):
            continue
        found.append((obj.key, int(getattr(obj, "size", 0) or 0)))
    return found


def rename_one(bucket, src_key: str, size: int, dry_run: bool, force: bool) -> str:
    import oss2

    dest = dest_key_for(src_key)
    if dest == src_key:
        return "skip"
    if dry_run:
        print(f"  DRY  {src_key}  ->  {dest}")
        return "dry"

    try:
        if bucket.object_exists(dest) and not force:
            print(f"  SKIP exists: {dest}")
            return "exists"
        if size >= COPY_SIMPLE_LIMIT:
            oss2.resumable_copy(bucket, bucket.bucket_name, src_key, dest)
        else:
            bucket.copy_object(bucket.bucket_name, src_key, dest)
        bucket.delete_object(src_key)
        print(f"  OK   {src_key}  ->  {dest}")
        return "ok"
    except Exception as e:
        print(f"  FAIL {src_key}: {e}")
        return "fail"


def main():
    parser = argparse.ArgumentParser(
        description="把 Excel 对应输出里的 train-data 重命名为 train_data",
    )
    parser.add_argument(
        "excel", nargs="?", default=None,
        help="Excel 文件路径（.xlsx）。省略时自动使用当前目录下的 xlsx",
    )
    parser.add_argument("--sheet", default=None, help="工作表名称（默认第一个 sheet）")
    parser.add_argument("--column", default=None, help="路径列名（默认自动识别 oss_path / ossPath）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要重命名的对象，不 copy、不删除")
    parser.add_argument("--force", action="store_true", help="目标 train_data 已存在时覆盖")
    parser.add_argument("--workers", type=int, default=8, help="并行 copy 数（默认 8）")
    args = parser.parse_args()

    excel = args.excel or find_default_excel()
    if not os.path.isfile(excel):
        print(f"[ERROR] 文件不存在: {excel}")
        sys.exit(1)

    paths = read_oss_paths(excel, args.sheet, args.column)
    unique_paths = list(dict.fromkeys(paths))
    print(f"[INFO] Excel    : {excel}")
    print(f"[INFO] 读取     : {len(paths)} 条 oss_path（去重后 {len(unique_paths)}）")
    print(f"[INFO] OSS      : bucket={OSS_BUCKET}  prefix={OSS_OUTPUT_PREFIX}")
    print(f"[INFO] Rename   : {OLD_SUBDIR}/  ->  {NEW_SUBDIR}/")
    if args.dry_run:
        print("[INFO] Dry run  : 不 copy、不删除")

    if not unique_paths:
        print("[ERROR] 没有可处理的 oss_path")
        sys.exit(1)

    bucket = get_bucket()

    jobs: list[tuple[str, int]] = []
    seen_src: set[str] = set()
    n_mcap = 0
    n_missing = 0

    for raw in unique_paths:
        _, source_key = parse_oss_uri(raw)
        if not source_key:
            print(f"[WARN] 空路径，跳过: {raw!r}")
            continue
        try:
            mcap_keys = collect_mcap_keys(bucket, source_key)
        except Exception as e:
            print(f"[WARN] 列举失败 {source_key}: {e}")
            continue
        if not mcap_keys and source_key.endswith(".mcap"):
            mcap_keys = [source_key]

        for mcap_key in mcap_keys:
            n_mcap += 1
            old_prefix = mcap_key_to_old_prefix(mcap_key)
            try:
                objs = list_old_objects(bucket, old_prefix)
            except Exception as e:
                print(f"[WARN] 列举失败 {old_prefix}: {e}")
                continue
            if not objs:
                n_missing += 1
                continue
            print(f"[INFO] {mcap_key}  ->  {old_prefix}  ({len(objs)} objects)")
            for key, size in objs:
                if key in seen_src:
                    continue
                seen_src.add(key)
                jobs.append((key, size))

    print(
        f"[INFO] MCAP {n_mcap} 个，已是新名/无旧目录 {n_missing} 个，"
        f"待重命名对象 {len(jobs)} 个"
    )
    if not jobs:
        print("OK — 没有需要重命名的 train-data 对象")
        return

    counts = {"ok": 0, "dry": 0, "exists": 0, "fail": 0, "skip": 0}
    workers = max(1, args.workers)
    if args.dry_run or len(jobs) == 1:
        for key, size in jobs:
            counts[rename_one(bucket, key, size, args.dry_run, args.force)] += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(rename_one, bucket, key, size, False, args.force)
                for key, size in jobs
            ]
            for fut in as_completed(futs):
                counts[fut.result()] += 1

    print(
        f"OK — renamed={counts['ok']} dry={counts['dry']} "
        f"exists={counts['exists']} fail={counts['fail']} skip={counts['skip']}"
    )
    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
