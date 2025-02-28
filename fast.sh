#!/bin/bash

# 检查并安装 lsof
if ! command -v lsof &> /dev/null; then
    echo "正在安装必要的工具 lsof..."
    apt-get update && apt-get install -y lsof
fi

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "请提供参数: --start 或 --stop"
    exit 1
fi


PORT=$(grep "^PORT=" .env 2>/dev/null | cut -d '=' -f2 || echo "8000")

case "$1" in
    --start)
        # 检查端口是否已被占用
        if lsof -i:$PORT > /dev/null; then
            echo "端口 $PORT 已被占用，请先停止现有服务"
            exit 1
        fi
        # 使用 node 直接启动主文件
        nohup python main.py > ./app.log 2>&1 &

        # 获取进程ID并保存到文件
        PID=$!
        echo $PID > app.pid

        echo "正在启动 python 应用 $PORT 端口..."
        
        ;;
    
    --stop)
        # 检查 PID 文件是否存在
        if [ ! -f app.pid ]; then
            echo "找不到 app.pid 文件"
            exit 1
        fi

        # 从文件中读取进程ID
        PID=$(cat app.pid)

        # 使用 kill 命令终止进程
        kill $PID 2>/dev/null

        # 确保使用 pkill 来终止所有相关进程

        # 等待进程完全终止
        sleep 2

        # 再次检查端口是否还在使用
        if lsof -i:$PORT > /dev/null; then
            echo "警告：端口 $PORT 仍在使用，尝试强制终止进程..."
            kill -9 $PID 2>/dev/null
            kill -9 $(lsof -t -i :$PORT)
        fi
        
        # 删除 PID 文件
        rm app.pid
        echo "python 应用已关闭"
        ;;
    
    *)
        echo "无效的参数。请使用 --start 或 --stop"
        exit 1
        ;;
esac