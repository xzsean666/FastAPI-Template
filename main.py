import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from helper.limiter import limiter
from routers import api
from settings import PORT

app = FastAPI(
    title="AI API",
    description="多个AI服务的接口",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 地址
    redoc_url="/redoc",  # ReDoc 文档地址
    openapi_url="/openapi.json",  # OpenAPI 规范地址
    openapi_tags=[
        {
            "name": "sentence",
            "description": "多个AI服务的接口",
        }
    ],
)

app.include_router(api.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建限速器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# 在路由上应用限速
@app.get("/api/example")
@limiter.limit("5/second")  # 每秒最多5次请求
async def example(request: Request):
    return {"hello": "world"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
