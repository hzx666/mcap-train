#!/usr/bin/env python3
"""
从 Excel 读取 oss_path / ossPath，RPUSH 到 Redis 队列。

消费端 run_extract.py 使用 LPOP，因此本脚本用 RPUSH，保证 FIFO。

用法:
  python push_excel_redis.py --dry-run
  python push_excel_redis.py
  python push_excel_redis.py <excel.xlsx>
  python push_excel_redis.py <excel.xlsx> -k mcap:queue

未指定 excel 时，自动使用当前目录下的 .xlsx 文件。

环境变量（与 run_extract.py / push_redis.py 相同）:
  REDIS_HOST        Redis 主机 (default: localhost)
  REDIS_PORT        Redis 端口 (default: 6379)
  REDIS_DB          Redis 数据库 (default: 0)
  REDIS_USERNAME    Redis 用户名 (default: 空)
  REDIS_PASSWORD    Redis 密码   (default: 空)
  REDIS_TIMEOUT     连接超时秒数 (default: 5)
  REDIS_QUEUE_KEY   队列 key    (default: mcap:queue, -k 优先级更高)
"""

import argparse
import os
import sys

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_QUEUE_KEY = os.environ.get("REDIS_QUEUE_KEY", "mcap:queue")
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_TIMEOUT = int(os.environ.get("REDIS_TIMEOUT", "5"))

OSS_COL_ALIASES = {"osspath", "oss_path", "oss path"}
BATCH_SIZE = 200


def _norm_header(value) -> str:
    return str(value).strip().lower() if value is not None else ""


def find_default_excel() -> str:
    """Pick the only .xlsx in cwd (or /app)."""
    search_dirs = [os.getcwd()]
    app_dir = "/app"
    if os.path.isdir(app_dir) and os.path.abspath(app_dir) not in {
        os.path.abspath(d) for d in search_dirs
    }:
        search_dirs.append(app_dir)

    found = []
    for d in search_dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name.startswith("~$"):
                continue
            if name.lower().endswith(".xlsx"):
                found.append(os.path.join(d, name))

    # de-duplicate while preserving order
    uniq = list(dict.fromkeys(os.path.abspath(p) for p in found))
    if len(uniq) == 1:
        return uniq[0]
    if not uniq:
        print("[ERROR] 未指定 Excel，且当前目录没有 .xlsx 文件")
        sys.exit(1)
    print("[ERROR] 未指定 Excel，且找到多个 .xlsx，请显式传入其中一个：")
    for p in uniq:
        print(f"  - {p}")
    sys.exit(1)


def read_oss_paths(xlsx_path: str, sheet: str | None, column: str | None) -> list[str]:
    try:
        from openpyxl import load_workbook
    except ImportError:
        print("[ERROR] openpyxl 未安装，请执行: pip install openpyxl")
        sys.exit(1)

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    try:
        ws = wb[sheet] if sheet else wb.active
        rows = ws.iter_rows(values_only=True)
        header_row = next(rows, None)
        if not header_row:
            print("[ERROR] Excel 为空")
            sys.exit(1)

        headers = [_norm_header(h) for h in header_row]
        if column:
            want = _norm_header(column)
            try:
                col_idx = headers.index(want)
            except ValueError:
                print(f"[ERROR] 找不到列 '{column}'，现有列: {list(header_row)}")
                sys.exit(1)
        else:
            col_idx = next(
                (i for i, h in enumerate(headers) if h in OSS_COL_ALIASES),
                None,
            )
            if col_idx is None:
                print(f"[ERROR] 找不到 oss_path / ossPath 列，现有列: {list(header_row)}")
                sys.exit(1)

        paths = []
        for row in rows:
            if col_idx >= len(row):
                continue
            val = row[col_idx]
            if val is None:
                continue
            s = str(val).strip()
            if s:
                paths.append(s)
        return paths
    finally:
        wb.close()


def connect_redis():
    try:
        import redis
    except ImportError:
        print("[ERROR] redis 包未安装，请执行: pip install redis")
        sys.exit(1)

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        username=REDIS_USERNAME or None,
        password=REDIS_PASSWORD or None,
        socket_connect_timeout=REDIS_TIMEOUT,
        decode_responses=True,
    )
    try:
        client.ping()
    except Exception as e:
        print(f"[ERROR] Redis 连接失败: {e}")
        sys.exit(1)
    return client


def main():
    parser = argparse.ArgumentParser(description="从 Excel 读取 oss_path 并 RPUSH 到 Redis 队列")
    parser.add_argument(
        "excel", nargs="?", default=None,
        help="Excel 文件路径（.xlsx）。省略时自动使用当前目录下的 xlsx",
    )
    parser.add_argument(
        "-k", "--key",
        default=REDIS_QUEUE_KEY,
        help="队列 key（默认: mcap:queue，可通过环境变量 REDIS_QUEUE_KEY 覆盖）",
    )
    parser.add_argument("--sheet", default=None, help="工作表名称（默认第一个 sheet）")
    parser.add_argument("--column", default=None, help="路径列名（默认自动识别 oss_path / ossPath）")
    parser.add_argument("--dry-run", action="store_true", help="只读取并打印，不连接 Redis、不推送")
    args = parser.parse_args()

    excel = args.excel or find_default_excel()
    if not os.path.isfile(excel):
        print(f"[ERROR] 文件不存在: {excel}")
        sys.exit(1)

    paths = read_oss_paths(excel, args.sheet, args.column)
    unique = len(set(paths))
    print(f"[INFO] Excel    : {excel}")
    print(f"[INFO] 读取     : {len(paths)} 条 oss_path（去重后 {unique}）")
    print(f"[INFO] Redis    : {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}  queue={args.key}")

    if not paths:
        print("[ERROR] 没有可推送的 oss_path")
        sys.exit(1)

    if args.dry_run:
        print(f"[INFO] Dry run: 不推送。将 RPUSH {len(paths)} 条到 '{args.key}'")
        for i, p in enumerate(paths[:10], 1):
            print(f"  [{i}] {p}")
        if len(paths) > 10:
            print(f"  ... 其余 {len(paths) - 10} 条省略")
        return

    r = connect_redis()
    try:
        before = r.llen(args.key)
        new_len = before
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            new_len = r.rpush(args.key, *batch)
        print(
            f"OK — 向队列 '{args.key}' RPUSH 了 {len(paths)} 条，"
            f"推送前长度 {before}，当前长度 {new_len}"
        )
    finally:
        try:
            r.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
