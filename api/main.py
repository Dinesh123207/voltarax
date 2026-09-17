import asyncio, logging, time, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from config.settings import settings
from db.database import init_db
from api.routers.auth         import router as auth_r
from api.routers.users        import router as users_r
from api.routers.panels       import router as panels_r
from api.routers.mixed_routers import (alerts_router, wo_router, bfci_router,
                                        forecast_router, dashboard_router,
                                        cv_router, relay_router, carbon_router, system_router)
from api.routers.websocket    import router as ws_r, broadcast_loop

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("VoltVision AI v%s starting (%s)", settings.APP_VERSION, settings.APP_ENV)
    init_db()
    asyncio.create_task(broadcast_loop())
    logger.info("Ready → http://%s:%d/docs", settings.APP_HOST, settings.APP_PORT)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="VoltVision AI",
    description="AI-Integrated BioSolar Panel Management — Voltarax Pvt Ltd | Patent 487411-001",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins_list,
                   allow_credentials=True, allow_methods=["*"],
                   allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def timer(request: Request, call_next):
    t0 = time.perf_counter()
    r  = await call_next(request)
    r.headers["X-Process-Time-Ms"] = str(round((time.perf_counter()-t0)*1000, 1))
    return r


@app.exception_handler(404)
async def e404(req: Request, exc):
    return JSONResponse(status_code=404, content={"success":False,"message":"Not found.","path":str(req.url.path)})


@app.exception_handler(500)
async def e500(req: Request, exc):
    logger.error("500: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"success":False,"message":"Internal server error."})


for r in [auth_r, users_r, panels_r, alerts_router, wo_router, bfci_router,
          forecast_router, dashboard_router, cv_router, relay_router,
          carbon_router, system_router, ws_r]:
    app.include_router(r)


@app.get("/", tags=["Health"])
def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION,
            "status": "running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


_dash = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(_dash):
    app.mount("/dashboard", StaticFiles(directory=_dash, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.APP_HOST, port=settings.APP_PORT,
                reload=settings.DEBUG)
