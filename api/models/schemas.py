from pydantic import BaseModel, EmailStr, field_validator, Field
from typing import Optional, List, Any
from enum import Enum


class UserRole(str, Enum):
    admin    = "admin"
    operator = "operator"
    viewer   = "viewer"


class AlertSeverity(str, Enum):
    info     = "info"
    warning  = "warning"
    critical = "critical"


class WorkOrderStatus(str, Enum):
    open        = "open"
    in_progress = "in_progress"
    closed      = "closed"


class WorkOrderPriority(str, Enum):
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class CVClass(str, Enum):
    clean        = "clean"
    dusty        = "dusty"
    cracked      = "cracked"
    bio_degraded = "bio_degraded"
    pending      = "pending"


# ── Generic ───────────────────────────────────────────────────────────────────
class APIResponse(BaseModel):
    success: bool = True
    message: str  = "OK"
    data: Optional[Any] = None


class PaginatedResponse(BaseModel):
    success:  bool = True
    total:    int
    page:     int
    per_page: int
    data:     List[Any]


# ── Auth ──────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    EmailStr
    password: str = Field(min_length=6)


class LoginResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user:          "UserOut"


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ── Users ─────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    email:     EmailStr
    password:  str = Field(min_length=8)
    full_name: str = Field(min_length=2, max_length=100)
    role:      UserRole = UserRole.viewer

    @field_validator("password")
    @classmethod
    def strong(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Need at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Need at least one digit")
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str]      = None
    role:      Optional[UserRole] = None
    is_active: Optional[bool]     = None


class UserOut(BaseModel):
    id:         int
    email:      str
    full_name:  str
    role:       UserRole
    is_active:  bool
    created_at: str
    last_login: Optional[str] = None


# ── Panels ────────────────────────────────────────────────────────────────────
class PanelOut(BaseModel):
    id:           int
    label:        str
    row_label:    str
    col_number:   int
    is_active:    bool
    installed_at: str
    notes:        str


class PanelDetail(PanelOut):
    efficiency_pct:  Optional[float] = None
    bfci_score:      Optional[float] = None
    voltage_v:       Optional[float] = None
    current_a:       Optional[float] = None
    power_w:         Optional[float] = None
    temp_c:          Optional[float] = None
    cv_class:        Optional[CVClass] = None
    cv_confidence:   Optional[float] = None
    status:          str = "unknown"


# ── Alerts ────────────────────────────────────────────────────────────────────
class AlertOut(BaseModel):
    id:          int
    panel_id:    Optional[int]
    panel_label: Optional[str]
    alert_type:  str
    severity:    AlertSeverity
    message:     str
    confidence:  Optional[float]
    resolved:    bool
    resolved_at: Optional[str]
    created_at:  str


class AlertResolve(BaseModel):
    resolved: bool = True


# ── Work Orders ───────────────────────────────────────────────────────────────
class WorkOrderCreate(BaseModel):
    panel_id:    Optional[int]           = None
    alert_id:    Optional[int]           = None
    title:       str = Field(min_length=5, max_length=200)
    description: str                     = ""
    priority:    WorkOrderPriority       = WorkOrderPriority.medium
    assigned_to: Optional[int]           = None


class WorkOrderUpdate(BaseModel):
    status:      Optional[WorkOrderStatus]   = None
    description: Optional[str]               = None
    priority:    Optional[WorkOrderPriority] = None
    assigned_to: Optional[int]               = None


class WorkOrderOut(BaseModel):
    id:           int
    panel_id:     Optional[int]
    panel_label:  Optional[str]
    alert_id:     Optional[int]
    title:        str
    description:  str
    status:       WorkOrderStatus
    priority:     WorkOrderPriority
    created_by:   Optional[int]
    assigned_to:  Optional[int]
    created_at:   str
    updated_at:   str


# ── BFCI ─────────────────────────────────────────────────────────────────────
class BFCIOut(BaseModel):
    panel_id:          int
    panel_label:       str
    bfci_score:        float
    uv_absorption_pct: Optional[float]
    days_to_recoat:    Optional[int]
    coat_date:         Optional[str]
    batch_id:          Optional[str]
    recorded_at:       str
    status:            str


# ── CV ────────────────────────────────────────────────────────────────────────
class CVDetectionOut(BaseModel):
    id:            int
    panel_id:      Optional[int]
    panel_label:   Optional[str]
    cv_class:      CVClass
    confidence:    float
    eff_loss_pct:  float
    model_ver:     str
    scanned_at:    str


class CVScanRequest(BaseModel):
    panel_ids: Optional[List[int]] = None


# ── Relay ─────────────────────────────────────────────────────────────────────
class RelayState(BaseModel):
    battery_on:  bool
    load_on:     bool
    grid_export: bool
    spare_on:    bool
    reason:      str           = "manual"
    soc_pct:     Optional[float] = None


class RelayOut(RelayState):
    id:            int
    commanded_by:  str
    commanded_at:  str


# ── Forecast ──────────────────────────────────────────────────────────────────
class ForecastOut(BaseModel):
    forecast_1h_kw:   float
    forecast_6h_kwh:  float
    forecast_24h_kwh: float
    forecast_7d_kwh:  float
    confidence:       List[float]
    model_version:    str
    generated_at:     str
    recommendation:   str


# ── Carbon ────────────────────────────────────────────────────────────────────
class CarbonOut(BaseModel):
    today_kwh:        float
    today_co2_kg:     float
    month_kwh:        float
    month_co2_kg:     float
    year_kwh:         float
    year_co2_kg:      float
    trees_equivalent: float
    emission_factor:  float
    grid_saved_inr:   float


# ── System ────────────────────────────────────────────────────────────────────
class ServiceStatus(BaseModel):
    name:     str
    status:   str
    uptime_s: Optional[int] = None


class SystemHealthOut(BaseModel):
    app_version:    str
    environment:    str
    sqlite_ok:      bool
    influxdb_ok:    bool
    mqtt_ok:        bool
    ai_engine_ok:   bool
    services:       List[ServiceStatus]
    panel_count:    int
    active_alerts:  int
    db_size_mb:     float
    uptime_s:       int


# ── Dashboard ─────────────────────────────────────────────────────────────────
class DashboardSummary(BaseModel):
    current_power_kw:       float
    daily_generation_kwh:   float
    battery_soc_pct:        float
    system_efficiency_pct:  float
    bfci_score_avg:         float
    panel_temp_avg_c:       float
    co2_offset_today_kg:    float
    uv_irradiance_wm2:      float
    panels_optimal:         int
    panels_warning:         int
    panels_critical:        int
    panels_offline:         int
    active_alerts:          int
    forecast_1h_kw:         float
    relay_state:            RelayOut
