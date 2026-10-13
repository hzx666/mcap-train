#!/usr/bin/env python3
"""
取出并清空 Redis 队列中的全部任务。

消费端 run_extract.py 使用 LPOP，本脚本用 LRANGE + DELETE 一次性取出。

用法:
  python drain_redis.py --dry-run
  python drain_redis.py
  python drain_redis.py -o drained.txt
  python drain_redis.py -k mcap:queue

环境变量（与 run_extract.py / push_excel_redis.py 相同）:
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
    parser = argparse.ArgumentParser(description="取出并清空 Redis 队列")
    parser.add_argument(
        "-k", "--key",
        default=REDIS_QUEUE_KEY,
        help="队列 key（默认: mcap:queue，可通过环境变量 REDIS_QUEUE_KEY 覆盖）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="把取出的路径写入文件（一行一条）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只查看队列，不取出、不删除",
    )
    args = parser.parse_args()

    r = connect_redis()
    try:
        before = r.llen(args.key)
        print(f"[INFO] Redis    : {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}  queue={args.key}")
        print(f"[INFO] 当前长度 : {before}")

        if before == 0:
            print("[INFO] 队列为空")
            return

        if args.dry_run:
            preview = min(before, 20)
            items = r.lrange(args.key, 0, preview - 1)
            print(f"[INFO] Dry run: 不取出。队列里有 {before} 条，预览前 {len(items)} 条：")
            for i, item in enumerate(items, 1):
                print(f"  [{i}] {item}")
            if before > preview:
                print(f"  ... 其余 {before - preview} 条省略")
            return

        pipe = r.pipeline()
        pipe.multi()
        pipe.lrange(args.key, 0, -1)
        pipe.delete(args.key)
        items, _deleted = pipe.execute()

        after = r.llen(args.key)
        print(f"[INFO] 已取出   : {len(items)} 条")
        print(f"[INFO] 清空后长度: {after}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(f"{item}\n")
            print(f"[INFO] 已写入   : {args.output}")

        show = min(len(items), 20)
        for i, item in enumerate(items[:show], 1):
            print(f"  [{i}] {item}")
        if len(items) > show:
            print(f"  ... 其余 {len(items) - show} 条省略")
    finally:
        try:
            r.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
