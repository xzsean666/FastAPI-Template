from slowapi import Limiter
from slowapi.util import get_remote_address

# 创建限速器
limiter = Limiter(key_func=get_remote_address)
