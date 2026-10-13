#!/usr/bin/env python3
"""
查看 Redis 队列里还有多少任务没被取走。

消费端 run_extract.py 用 LPOP 取任务，因此队列长度就是还没开始处理的条数。
已经 LPOP 出去、正在跑的任务不在这个数字里。本脚本只读，不取出、不删除。

用法:
  python queue_pending.py
  python queue_pending.py -k mcap:queue
  python queue_pending.py --preview 20

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
    parser = argparse.ArgumentParser(description="查看 Redis 队列中尚未处理的任务数")
    parser.add_argument(
        "-k", "--key",
        default=REDIS_QUEUE_KEY,
        help="队列 key（默认: mcap:queue，可通过环境变量 REDIS_QUEUE_KEY 覆盖）",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="额外打印队头前 N 条（默认不打印）",
    )
    args = parser.parse_args()

    if args.preview < 0:
        print("[ERROR] --preview 不能为负数")
        sys.exit(1)

    r = connect_redis()
    try:
        key_type = r.type(args.key)
        if key_type not in ("none", "list"):
            print(f"[ERROR] key '{args.key}' 类型是 {key_type}，不是 list 队列")
            sys.exit(1)

        pending = r.llen(args.key)
        print(f"[INFO] Redis    : {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}  queue={args.key}")
        print(f"[INFO] 未处理   : {pending}")

        if pending == 0 or args.preview == 0:
            return

        n = min(pending, args.preview)
        items = r.lrange(args.key, 0, n - 1)
        print(f"[INFO] 队头预览 : {len(items)} / {pending}")
        for i, item in enumerate(items, 1):
            print(f"  [{i}] {item}")
        if pending > n:
            print(f"  ... 其余 {pending - n} 条省略")
    finally:
        try:
            r.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
