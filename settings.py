import os

from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量中读取PORT，如果没有定义，就使用默认值8000
PORT = int(os.getenv("PORT", default=8000))
