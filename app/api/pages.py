from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


def build_pages_router(templates: Jinja2Templates) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/", response_class=HTMLResponse)
    async def page_index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="index.html")

    @router.get("/detection", response_class=HTMLResponse)
    async def page_detection(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="detection.html")

    @router.get("/alerts", response_class=HTMLResponse)
    async def page_alerts(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="alerts.html")

    @router.get("/statistics", response_class=HTMLResponse)
    async def page_statistics(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="statistics.html")

    @router.get("/dataset", response_class=HTMLResponse)
    async def page_dataset(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="dataset.html")

    @router.get("/video", response_class=HTMLResponse)
    async def page_video(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="video.html")

    @router.get("/collection", response_class=HTMLResponse)
    async def page_collection(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="collection.html")

    return router

