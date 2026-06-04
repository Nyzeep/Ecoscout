from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.pages import build_pages_router
from app.api.routes import build_api_router
from app.bootstrap import bootstrap_application
from app.config import get_settings


settings = get_settings()
started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_app() -> FastAPI:
    bootstrap_application()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "请求参数校验失败", "detail": exc.errors()})

    templates = Jinja2Templates(directory=str(settings.templates_dir))

    app.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir)), name="uploads")
    app.include_router(build_pages_router(templates))
    app.include_router(build_api_router(settings=settings, started_at=started_at))
    return app


app = create_app()
