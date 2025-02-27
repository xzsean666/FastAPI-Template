import functools
import inspect
import json
import time
from typing import Any, Callable, Optional, TypeVar, cast

from .db_sqlite import KVDatabase

T = TypeVar("T")


def create_cache_decorator(db: KVDatabase, default_ttl: int = 60 * 1000):
    """
    创建一个缓存装饰器工厂函数

    Args:
        db: KVDatabase实例
        default_ttl: 默认缓存时间（毫秒）
    """

    def cache(ttl: int = default_ttl, prefix: str = ""):
        """
        缓存装饰器

        Args:
            ttl: 缓存有效期（毫秒）
            prefix: 缓存键前缀
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    # 创建缓存键
                    args_str = json.dumps(args, default=str)
                    kwargs_str = json.dumps(kwargs, default=str, sort_keys=True)
                    cache_key = f"{prefix}:{func.__name__}:{args_str}:{kwargs_str}"
                    # 限制键长度为255
                    cache_key = cache_key[:255]

                    # 获取缓存
                    cached = await db.get(cache_key)

                    now = int(time.time() * 1000)  # 当前时间（毫秒）

                    # 检查缓存是否有效
                    if cached and now - cached.get("timestamp", 0) < ttl:
                        return cached.get("value")

                    # 执行原始方法
                    if inspect.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    # 缓存结果
                    await db.put(cache_key, {"value": result, "timestamp": now})

                    return result

                except Exception as error:
                    print(f"缓存操作失败: {error}")
                    # 发生错误时直接执行原方法
                    if inspect.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

            return wrapper

        return decorator

    return cache


# 创建数据库和缓存装饰器
# db = KVDatabase("cache.db")
# cache = create_cache_decorator(db, default_ttl=60*1000)  # 默认1分钟

# class UserService:
#     @cache(ttl=5*60*1000, prefix="users")  # 缓存5分钟
#     async def get_user(self, user_id: int):
#         # 模拟数据库查询
#         print(f"从数据库获取用户: {user_id}")
#         return {"id": user_id, "name": f"用户{user_id}"}

#     @cache(ttl=30*60*1000)  # 缓存30分钟
#     async def get_user_stats(self, user_id: int, period: str = "month"):
#         # 模拟耗时计算
#         print(f"计算用户统计数据: {user_id}, 周期: {period}")
#         return {"visits": 100, "actions": 50}
