import asyncio, json, random, logging
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config.settings import settings

logger = logging.getLogger(__name__)
router  = APIRouter(tags=["WebSocket"])

_SOLAR = [0,0,0,0,0,0,.1,.3,.8,1.8,3.2,4.4,5.1,5.6,5.7,5.3,4.6,3.8,2.7,1.5,.6,.2,0,0]
_N     = settings.SITE_PANEL_COUNT


def _payload() -> dict:
    h  = datetime.now().hour
    kw = round(max(0, _SOLAR[h]*_N*0.835*0.972*0.912 + random.uniform(-0.3, 0.3)), 2)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "power_kw":          kw,
        "daily_kwh":         round(kw * random.uniform(4.2, 5.1), 1),
        "battery_soc_pct":   round(random.uniform(76, 82), 1),
        "efficiency_pct":    round(random.uniform(90.5, 92.5), 1),
        "bfci_avg_pct":      round(random.uniform(82.8, 84.2), 1),
        "panel_temp_c":      round(random.uniform(42.8, 45.8), 1),
        "co2_offset_kg":     round(kw * 0.82 * 0.0833, 3),
        "uv_irradiance":     round(random.uniform(790, 840), 0),
        "humidity_pct":      round(random.uniform(32, 36), 0),
        "ambient_temp_c":    round(random.uniform(37.5, 39.5), 1),
        "active_alerts":     3,
        "relay": {"battery_on": True, "load_on": True, "grid_export": kw > 4.5},
    }


class _Manager:
    def __init__(self):
        self._conns: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._conns.append(ws)
        logger.info("WS connected. total=%d", len(self._conns))

    def disconnect(self, ws: WebSocket):
        if ws in self._conns: self._conns.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._conns:
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)


manager = _Manager()


async def broadcast_loop():
    while True:
        await asyncio.sleep(3)
        if manager._conns:
            await manager.broadcast(_payload())


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_json(_payload())
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        manager.disconnect(ws)
