FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY extract_head_train_mp4.py extract_orin_mp4.py run_extract.py run_extract_mns.py push_excel_redis.py send_mns.py drain_redis.py queue_pending.py rename_train_data.py ./

# run_extract.py 作为常驻服务运行，轮询 Redis 队列获取任务，
# 头环路径路由到 extract_head_train_mp4.py（全长四路相机 MP4）。
# 无需传 CLI 参数，所有配置通过环境变量注入。
#
# OSS -> 容器内挂载：
#   oss://ss-oss-data-platform/original_data/ego_human_data/head_cam/  ->  /head-data/
#   oss://ss-oss-data-platform/original_data/teleoperation/Orin/       ->  /orin-data/
#
# 从 Excel 推送队列 / 清空队列（容器内）：
#   python3 push_excel_redis.py --dry-run
#   python3 push_excel_redis.py
#   python3 drain_redis.py --dry-run
#   python3 drain_redis.py
#   python3 queue_pending.py
#   python3 queue_pending.py --preview 20
#   python3 rename_train_data.py --dry-run
#   python3 rename_train_data.py
#
# MNS 消费（路径映射与 run_extract.py 相同，默认入口仍是 Redis）:
#   python3 send_mns.py --dry-run --oss-path oss://bucket/a.mcap --trace-id t1
#   python3 send_mns.py --oss-path oss://bucket/a.mcap --trace-id t1
#   python3 run_extract_mns.py
#   docker run --entrypoint python ... mcap-extract-train run_extract_mns.py
#
# 构建：
#   docker build -t mcap-extract-train .
#
# 运行示例：
#   docker run -d --restart unless-stopped \
#     -v /mnt/head_cam:/head-data:ro \
#     -v $(pwd)/output:/app/output \
#     -e POLL_INTERVAL=10 \
#     -e REDIS_HOST=10.0.0.1 \
#     -e OSS_ENDPOINT=oss-cn-shanghai-internal.aliyuncs.com \
#     -e OSS_ACCESS_KEY_ID=xxx \
#     -e OSS_ACCESS_KEY_SECRET=xxx \
#     -e OSS_BUCKET=ss-oss-data-platform \
#     -e OSS_OUTPUT_PREFIX=processed_data \
#     mcap-extract-train

ENTRYPOINT ["python", "run_extract.py"]
