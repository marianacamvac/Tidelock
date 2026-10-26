import os
import heapq
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="TideLock-AI", page_icon=":material/water_drop:", layout="wide", initial_sidebar_state="expanded")

STALL_CM = 25.0
LOCK_THRESHOLD_M = 2.50
REDUCTION = 0.55
MAX_ETA_MIN = 45.0
STALL_PENALTY_MIN = 45.0
SETUP_MIN = 12.0
PIN = os.environ.get("TIDELOCK_PIN", "TIDELOCK2026")
CFG = {"displayModeBar": False}

PRESETS = {
    "Light rain + low tide": (8.0, 20.0, 1.2),
    "Moderate rain + mid tide": (30.0, 60.0, 2.1),
    "Extreme compound event (rain + high tide)": (53.0, 105.0, 2.65),
    "Severe storm + spring tide": (85.0, 150.0, 3.3),
}
DEFAULT_PRESET = "Extreme compound event (rain + high tide)"

NODE_ROWS = [
    ("N01", "Ceibos – Emergency Base", "Ceibos Base", -2.1760, -79.9440, .35, 55, .2),
    ("N02", "Av. Principal Ceibos – Roundabout", "Ceibos Roundabout", -2.1700, -79.9350, .50, 50, .3),
    ("N03", "Urdesa Central", "Urdesa Central", -2.1750, -79.9060, .90, 45, .8),
    ("N04", "Av. V. E. Estrada – Urdesa", "Estrada-Urdesa", -2.1700, -79.9040, .60, 48, .5),
    ("N05", "Av. Francisco de Orellana – San Marino / Kennedy Norte", "Orellana-San Marino", -2.1700, -79.8970, 1.0, 42, 1.0),
    ("N06", "Av. Juan Tanca Marengo – Mall del Sol / Gómez Gould", "Tanca Marengo-Mall del Sol", -2.1550, -79.8930, 1.1, 40, 1.0),
    ("N07", "Av. de las Américas – Airport", "Américas-Airport", -2.1560, -79.8860, .40, 55, .3),
    ("N08", "Av. C. J. Arosemena – Km 2", "Arosemena Km 2", -2.1800, -79.9180, .50, 50, .4),
    ("N09", "Av. Francisco de Orellana – Alborada", "Orellana-Alborada", -2.1450, -79.9050, .60, 48, .5),
    ("N10", "Vía a Daule – Mapasingue Oeste", "Vía a Daule-Mapasingue", -2.1480, -79.9390, 1.05, 40, .9),
    ("N11", "Av. Juan Tanca Marengo – Km 5", "Tanca Marengo Km 5", -2.1450, -79.8920, .70, 46, .6),
    ("N12", "Sauces 6 – Topographic sink ('La Olla')", "Sauces 6 'La Olla'", -2.1330, -79.9060, 1.3, 38, .8),
    ("N13", "Av. Narcisa de Jesús – Sauces 9", "Narcisa de Jesús", -2.1250, -79.9000, .50, 50, .3),
    ("N14", "Vía a Daule – Km 6", "Vía a Daule Km 6", -2.1380, -79.9300, .55, 48, .4),
    ("N15", "Av. Benjamín Carrión – Kennedy", "Carrión-Kennedy", -2.1600, -79.9050, .70, 45, .6),
    ("N16", "Reference hospital (destination)", "Hospital", -2.1400, -79.9120, .30, 60, .1),
]
NODES = pd.DataFrame(NODE_ROWS, columns=["id", "name", "short", "lat", "lon", "s", "c", "e"])
N = len(NODES)
ORIGIN, DEST = 0, N - 1

CREWS = pd.DataFrame(
    [("C1", -2.1480, -79.9000), ("C2", -2.1650, -79.9050), ("C3", -2.1450, -79.9300), ("C4", -2.1350, -79.9100)],
    columns=["crew", "lat", "lon"],
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0A1120;--card:#101A2E;--card2:#0D1626;--line:#1E2C47;--txt:#E6EDF7;--mut:#8DA2C0;--cy:#22D3EE;
--red:#F43F5E;--amb:#F59E0B;--grn:#10B981}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,sans-serif}
.stApp{background:radial-gradient(1100px 520px at 12% -8%,#0f3358 0%,var(--bg) 58%);color:var(--txt)}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1440px}
h1,h2,h3,h4,p,span,label,li{color:var(--txt)}
section[data-testid="stSidebar"]{background:#080E1B;border-right:1px solid var(--line)}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border-color:var(--line)!important;border-radius:16px}
.hero{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px 26px;border:1px solid var(--line);
border-radius:18px;background:linear-gradient(120deg,#0e2a4a 0%,#101A2E 65%);margin-bottom:18px}
.brand{display:flex;align-items:center;gap:16px}
.hero h1{margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-.6px;line-height:1.1}
.hero .accent{color:var(--cy)}
.hero p{color:var(--mut);margin:.3rem 0 0 0;font-size:.92rem}
.chip{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.74rem;font-weight:600;border:1px solid var(--line);
color:var(--mut);background:rgba(255,255,255,.03);margin:2px 4px 2px 0}
.chip.role{color:#67E8F9;border-color:rgba(34,211,238,.35);background:rgba(34,211,238,.08)}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;height:100%}
.kpi .t{color:var(--mut);font-size:.74rem;text-transform:uppercase;letter-spacing:.7px;font-weight:600}
.kpi .v{font-size:2rem;font-weight:800;margin:6px 0 8px;letter-spacing:-.5px}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.73rem;font-weight:600}
.pill.r{background:rgba(244,63,94,.15);color:#FB7185}.pill.g{background:rgba(16,185,129,.15);color:#34D399}
.pill.a{background:rgba(245,158,11,.15);color:#FBBF24}.pill.c{background:rgba(34,211,238,.15);color:#67E8F9}
.brief{border:1px solid var(--line);border-left:5px solid var(--cy);border-radius:14px;padding:16px 20px;background:var(--card2);margin:16px 0 8px}
.brief.r{border-left-color:var(--red)}.brief.a{border-left-color:var(--amb)}.brief.g{border-left-color:var(--grn)}
.brief .h{font-size:.74rem;text-transform:uppercase;letter-spacing:.7px;color:var(--mut);font-weight:600;margin-bottom:6px}
.brief p{margin:0;line-height:1.6;font-size:.96rem}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--mut);font-size:.8rem;margin:6px 2px 14px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.step{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;height:100%}
.step .n{width:30px;height:30px;border-radius:9px;background:rgba(34,211,238,.14);color:#67E8F9;font-weight:700;
display:flex;align-items:center;justify-content:center;margin-bottom:10px}
.step h4{margin:0 0 6px;font-size:1rem}.step p{color:var(--mut);font-size:.88rem;line-height:1.55;margin:0}
.route{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;height:100%}
.route.good{border-color:rgba(16,185,129,.5)}.route.bad{border-color:rgba(244,63,94,.35)}
.route .t{color:var(--mut);font-size:.76rem;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.route .v{font-size:1.9rem;font-weight:800;margin:4px 0}
.hop{display:inline-block;padding:3px 9px;margin:3px 2px;border-radius:8px;font-size:.76rem;background:rgba(255,255,255,.05);border:1px solid var(--line)}
.risk{border-radius:18px;padding:22px 26px;border:1px solid var(--line)}
.risk .big{font-size:2.3rem;font-weight:800;margin:2px 0 6px;letter-spacing:-.5px}
.risk.g{background:linear-gradient(120deg,rgba(16,185,129,.22),#101A2E)}
.risk.a{background:linear-gradient(120deg,rgba(245,158,11,.22),#101A2E)}
.risk.r{background:linear-gradient(120deg,rgba(244,63,94,.25),#101A2E)}
.feat{color:var(--mut);font-size:.88rem;line-height:1.8;padding-left:18px;margin:.4rem 0 1rem}
.small{color:var(--mut);font-size:.82rem}
div[data-testid="stTabs"] button{font-weight:600}
.stButton>button{border-radius:10px;font-weight:600}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
if st.session_state.get("role") is None:
    st.markdown("<style>section[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none}</style>",
                unsafe_allow_html=True)

LOGO = ("<svg width='44' height='44' viewBox='0 0 48 48' fill='none'><rect width='48' height='48' rx='12' fill='#0E7490'/>"
        "<path d='M24 8c4 5 7 8 7 12a7 7 0 0 1-14 0c0-4 3-7 7-12z' fill='#E6FBFF'/>"
        "<path d='M8 30c4-5 8-5 12 0s8 5 12 0 6-4 8-2' stroke='#67E8F9' stroke-width='3' stroke-linecap='round'/>"
        "<path d='M8 38c4-5 8-5 12 0s8 5 12 0 6-4 8-2' stroke='#E6FBFF' stroke-width='3' stroke-linecap='round'/></svg>")
ICON_AUTH = ("<svg width='34' height='34' viewBox='0 0 24 24' fill='none' stroke='#22D3EE' stroke-width='1.8' stroke-linecap='round' "
             "stroke-linejoin='round'><path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/><path d='M9 12l2 2 4-4'/></svg>")
ICON_CIT = ("<svg width='34' height='34' viewBox='0 0 24 24' fill='none' stroke='#22D3EE' stroke-width='1.8' stroke-linecap='round' "
            "stroke-linejoin='round'><circle cx='9' cy='8' r='3.2'/><path d='M3 20c0-3.5 2.7-6 6-6s6 2.5 6 6'/>"
            "<circle cx='17' cy='9' r='2.4'/><path d='M16 14.2c3 .2 5 2.2 5 5.2'/></svg>")


def style(fig, h=320):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#E6EDF7", family="Inter, sans-serif", size=12),
                      margin=dict(l=12, r=16, t=16, b=12),
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12)))
    fig.update_xaxes(gridcolor="#1E2C47", zeroline=False, title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#1E2C47", zeroline=False, title_font=dict(size=12))
    return fig


def kpi(title, value, pill, kind):
    st.markdown(f"<div class='kpi'><div class='t'>{title}</div><div class='v'>{value}</div>"
                f"<span class='pill {kind}'>{pill}</span></div>", unsafe_allow_html=True)


def panel(title, caption):
    st.markdown(f"**{title}**")
    st.caption(caption)


def risk_of(d):
    return ("g", "Low") if d <= 10 else (("a", "Moderate") if d <= STALL_CM else ("r", "High"))


def severity(flooded, lock):
    if flooded >= 4:
        return "Emergency", "r"
    if flooded >= 2:
        return "Alert", "a"
    if flooded == 1 or lock >= 30:
        return "Watch", "c"
    return "Normal", "g"


def hops(path):
    return "".join(f"<span class='hop'>{NODES.short[i]}</span>" for i in path)


def legend_html():
    return ("<div class='legend'><span><span class='dot' style='background:#F43F5E'></span>Over 25 cm (stalling risk)</span>"
            "<span><span class='dot' style='background:#F59E0B'></span>10 to 25 cm</span>"
            "<span><span class='dot' style='background:#10B981'></span>Under 10 cm</span></div>")


def lock_frac(H):
    return np.clip((np.asarray(H, dtype=float) - 2.0) / 1.5, 0, 1)


def physics_depth(I, A, H, s, c, e):
    q = I + 0.15 * A
    cap = c * (1 - 0.8 * e * lock_frac(H))
    return s * np.maximum(q - cap, 0)


@st.cache_resource(show_spinner="Training the surrogate model...")
def train_surrogate():
    rng = np.random.default_rng(42)
    n = 9000
    I, A, H = rng.uniform(0, 120, n), rng.uniform(0, 200, n), rng.uniform(0, 4, n)
    s, c, e = rng.uniform(.3, 1.4, n), rng.uniform(35, 62, n), rng.uniform(0, 1, n)
    y = physics_depth(I, A, H, s, c, e) + rng.normal(0, 0.8, n)
    X = np.c_[I, A, H, s, c, e]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, float(r2_score(yte, pred)), float(mean_absolute_error(yte, pred))


def predict_depths(model, I, A, H):
    X = np.c_[np.full(N, I), np.full(N, A), np.full(N, H), NODES[["s", "c", "e"]].values]
    return np.round(np.clip(model.predict(X), 0, None), 1)


def hav(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@st.cache_data
def build_graph():
    D = np.array([[hav(NODES.lat[i], NODES.lon[i], NODES.lat[j], NODES.lon[j]) * 1.3 for j in range(N)] for i in range(N)])
    E = set()
    for i in range(N):
        for j in np.argsort(D[i])[1:4]:
            E.add((min(i, int(j)), max(i, int(j))))
    seen = {0}
    while len(seen) < N:
        best = min(((D[i][j], i, j) for i in seen for j in range(N) if j not in seen))
        E.add((min(best[1], best[2]), max(best[1], best[2])))
        seen.add(best[2])
    adj = {i: [] for i in range(N)}
    for i, j in E:
        adj[i].append((j, D[i][j]))
        adj[j].append((i, D[i][j]))
    return adj, D


def edge_min(km, depth_v):
    return km / 30.0 * 60.0 * (1 + min(depth_v, STALL_CM) / 25.0)


def dijkstra(adj, src, dst, cost):
    pq, best = [(0.0, src, [src])], {}
    while pq:
        c, u, path = heapq.heappop(pq)
        if u == dst:
            return path
        if u in best and best[u] <= c:
            continue
        best[u] = c
        for v, km in adj[u]:
            heapq.heappush(pq, (c + cost(u, v, km), v, path + [v]))
    return [src]


def route_report(adj, path, depth):
    t, crossed = 0.0, 0
    for u, v in zip(path[:-1], path[1:]):
        km = dict(adj[u])[v]
        t += edge_min(km, depth[v])
        if depth[v] > STALL_CM:
            t += STALL_PENALTY_MIN
            crossed += 1
    return t, crossed


def routes(depth_raw, depth_mit, src=ORIGIN):
    adj, _ = build_graph()
    naive = dijkstra(adj, src, DEST, lambda u, v, km: km)
    smart = dijkstra(adj, src, DEST, lambda u, v, km: edge_min(km, depth_mit[v]) + (60 if depth_mit[v] > STALL_CM else 0))
    return naive, smart, route_report(adj, naive, depth_raw), route_report(adj, smart, depth_mit)


def milp_dispatch(depth):
    cand = [j for j in range(N) if depth[j] > 10 and j != DEST]
    if not cand:
        return pd.DataFrame(), {}
    nc, nj = len(CREWS), len(cand)
    eta = np.array([[hav(CREWS.lat[c], CREWS.lon[c], NODES.lat[j], NODES.lon[j]) * 1.3 / 30 * 60 + SETUP_MIN
                     for j in cand] for c in range(nc)])
    benefit = np.array([[depth[j] * REDUCTION + (30 if depth[j] > STALL_CM else 0) - 0.2 * eta[c][k]
                         for k, j in enumerate(cand)] for c in range(nc)])
    cvec = -benefit.flatten()
    rows = []
    for c in range(nc):
        r = np.zeros(nc * nj)
        r[c * nj:(c + 1) * nj] = 1
        rows.append(r)
    for k in range(nj):
        r = np.zeros(nc * nj)
        r[k::nj] = 1
        rows.append(r)
    ub = np.where(eta.flatten() <= MAX_ETA_MIN, 1, 0)
    res = milp(cvec, constraints=LinearConstraint(np.array(rows), -np.inf, 1),
               integrality=np.ones(nc * nj), bounds=Bounds(np.zeros(nc * nj), ub))
    assign, out = {}, []
    if res.x is not None:
        x = np.round(res.x).reshape(nc, nj)
        for c in range(nc):
            for k in range(nj):
                if x[c][k] > 0.5:
                    j = cand[k]
                    assign[j] = c
                    out.append({"Crew": CREWS.crew[c], "Critical point": NODES.name[j],
                                "Predicted depth (cm)": depth[j],
                                "Mitigated depth (cm)": round(depth[j] * (1 - REDUCTION), 1),
                                "ETA + setup (min)": round(float(eta[c][k]), 1)})
    return pd.DataFrame(out), assign


def depth_color(d):
    return "#F43F5E" if d > STALL_CM else ("#F59E0B" if d > 10 else "#10B981")


def make_map(depth, naive=None, smart=None, assign=None, height=520, zoom=12.0):
    new = hasattr(go, "Scattermap")
    Sc = go.Scattermap if new else go.Scattermapbox
    fig = go.Figure()

    def line(path, color, name, w):
        if path and len(path) > 1:
            fig.add_trace(Sc(lat=[NODES.lat[i] for i in path], lon=[NODES.lon[i] for i in path],
                             mode="lines", line=dict(width=w, color=color), name=name, hoverinfo="skip"))

    line(naive, "#F43F5E", "Conventional GPS route", 4)
    line(smart, "#10B981", "TideLock-AI route", 6)
    fig.add_trace(Sc(lat=NODES.lat, lon=NODES.lon, mode="markers+text", name="Critical nodes",
                     marker=dict(size=[14 + min(d, 50) * .25 for d in depth], color=[depth_color(d) for d in depth]),
                     text=NODES.id, textposition="top right", textfont=dict(size=11, color="#0B1220"),
                     customdata=np.c_[NODES.name, depth],
                     hovertemplate="<b>%{customdata[0]}</b><br>Depth: %{customdata[1]} cm<extra></extra>"))
    if assign:
        js = list(assign.keys())
        fig.add_trace(Sc(lat=[NODES.lat[j] for j in js], lon=[NODES.lon[j] for j in js], mode="markers",
                         name="Assigned crew", marker=dict(size=26, color="rgba(168,85,247,.55)"),
                         hovertemplate="Crew %{text}<extra></extra>",
                         text=[CREWS.crew[assign[j]] for j in js]))
    mp = dict(style="open-street-map", center=dict(lat=float(NODES.lat.mean()), lon=float(NODES.lon.mean())), zoom=zoom)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=0.01, yanchor="bottom", x=0.01, bgcolor="rgba(255,255,255,.85)",
                                  font=dict(color="#0B1220")))
    if new:
        fig.update_layout(map=mp)
    else:
        fig.update_layout(mapbox=mp)
    return fig


def header(role_label):
    left, right = st.columns([6, 1])
    with left:
        st.markdown(f"<div class='hero'><div class='brand'>{LOGO}<div><h1>TideLock<span class='accent'>-AI</span></h1>"
                    "<p>Rain and tide flood prediction · Crew dispatch · Emergency routing · Guayaquil</p>"
                    f"</div></div><span class='chip role'>{role_label}</span></div>", unsafe_allow_html=True)
    return right


def gate():
    st.markdown(f"<div class='hero'><div class='brand'>{LOGO}<div><h1>TideLock<span class='accent'>-AI</span></h1>"
                "<p>Intelligent flood alert and response system for rain and tide events in Guayaquil</p></div></div></div>",
                unsafe_allow_html=True)
    st.markdown("#### Select your access profile")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown(f"{ICON_AUTH}<h3 style='margin:.4rem 0'>Authorities and operations center</h3>"
                        "<p class='small'>Decision-support tool to anticipate flooding and coordinate the response.</p>"
                        "<ul class='feat'><li>Rain and tide scenario simulation</li>"
                        "<li>Critical point map and alerts</li><li>Optimal crew dispatch</li>"
                        "<li>Emergency routes to hospitals</li></ul>", unsafe_allow_html=True)
            code = st.text_input("Institutional access code", type="password", key="pin_in", placeholder="Enter your code")
            if "TIDELOCK_PIN" not in os.environ:
                st.caption(f"Demo access code: {PIN}")
            if st.button("Enter operations center", type="primary", icon=":material/shield:", key="b_auth"):
                if code == PIN:
                    st.session_state.role = "auth"
                    st.rerun()
                else:
                    st.error("Incorrect code.")
    with b:
        with st.container(border=True):
            st.markdown(f"{ICON_CIT}<h3 style='margin:.4rem 0'>Citizens</h3>"
                        "<p class='small'>Clear, registration-free information to protect your family and move safely.</p>"
                        "<ul class='feat'><li>Risk level for your area</li><li>Areas you should avoid</li>"
                        "<li>Safest route to the hospital</li><li>Safety recommendations</li></ul>",
                        unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("")
            if st.button("Check the status of my area", type="primary", icon=":material/groups:", key="b_cit"):
                st.session_state.role = "citizen"
                st.rerun()
    st.caption("Demo data and parameters. Operational use requires official forecast feeds and sensor calibration.")


def apply_preset():
    I, A, H = PRESETS[st.session_state.preset]
    st.session_state.rain, st.session_state.ante, st.session_state.tide = I, A, H


def authority_view():
    model, r2, mae = train_surrogate()
    if "rain" not in st.session_state:
        st.session_state.preset = DEFAULT_PRESET
        apply_preset()
    with st.sidebar:
        st.markdown("### Scenario simulator")
        st.caption("Define the expected conditions. The whole dashboard (alerts, map, charts, dispatch and routes) updates instantly.")
        st.selectbox("Preset scenario", list(PRESETS), key="preset", on_change=apply_preset,
                     help="Typical combinations to get started quickly. You can then fine-tune each value.")
        st.slider("Rainfall intensity [mm/h]", 0.0, 120.0, key="rain", step=1.0,
                  help="How much rain is falling right now. Above 50 mm/h is considered very intense rain.")
        st.slider("Rainfall over the last 6 h [mm]", 0.0, 200.0, key="ante", step=1.0,
                  help="Rain from the previous 6 hours. When it is high, streets and drains are already saturated.")
        st.slider("Tide level [m MLWS]", 0.0, 4.0, key="tide", step=0.05,
                  help="Height of the Guayas estuary. At high tide, drainage culverts cannot discharge stormwater.")
        with st.expander("What is this panel for?"):
            st.markdown("It lets you **anticipate** what would happen in the city under different combinations of rain and tide, "
                        "before it happens. Authorities can then decide in advance where to send crews and which "
                        "routes to use. In real operation, these three values are fed automatically from weather "
                        "forecasts and tide sensors.")
        st.divider()
        if st.button("Sign out", icon=":material/logout:"):
            st.session_state.role = None
            st.rerun()

    header("Authority mode")
    I, A, H = st.session_state.rain, st.session_state.ante, st.session_state.tide
    depth = predict_depths(model, I, A, H)
    table, assign = milp_dispatch(depth)
    mit = depth.copy()
    for j in assign:
        mit[j] = round(depth[j] * (1 - REDUCTION), 1)
    naive, smart, (t_n, x_n), (t_s, x_s) = routes(depth, mit)
    lock = float(lock_frac(H)) * 100
    flooded, flooded_m = int((depth > STALL_CM).sum()), int((mit > STALL_CM).sum())
    sev, sk = severity(flooded, lock)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Culvert blockage", f"{lock:.0f}%", "Tide above threshold" if H > LOCK_THRESHOLD_M else "Free drainage",
            "r" if H > LOCK_THRESHOLD_M else "g")
    with k2:
        kpi("Flooded points (> 25 cm)", f"{flooded} of {N}", f"{flooded - flooded_m} resolved by crews", "g")
    with k3:
        kpi("Emergency route time", f"{t_s:.0f} min", f"{max(t_n - t_s, 0):.0f} min faster than conventional GPS", "g")
    with k4:
        kpi("Model reliability", f"R² {r2:.3f}", f"Mean error {mae:.1f} cm", "c")

    st.markdown(
        f"<div class='brief {sk}'><div class='h'>Scenario briefing · Situation level: {sev}</div><p>"
        f"With rainfall of <b>{I:.0f} mm/h</b>, <b>{A:.0f} mm</b> accumulated over 6 hours and a tide of <b>{H:.2f} m</b>, "
        f"drainage culverts are <b>{lock:.0f}%</b> blocked: stormwater cannot flow out to the river. "
        f"<b>{flooded}</b> points are expected to exceed 25 cm of water (vehicles may stall); "
        f"with crews dispatched this drops to <b>{flooded_m}</b>. The recommended emergency route takes "
        f"<b>{t_s:.0f} min</b> versus <b>{t_n:.0f} min</b> on a conventional route.</p></div>", unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs([":material/map: Digital twin", ":material/local_shipping: Crew dispatch",
                                      ":material/route: Emergency routes", ":material/menu_book: How it works"])
    with tab1:
        mcol, icol = st.columns([3, 1], gap="large")
        with mcol:
            st.markdown("**Live map · north-central Guayaquil**")
            st.plotly_chart(make_map(depth, naive, smart, assign, 560), config=CFG, key="map_main")
            st.markdown(legend_html(), unsafe_allow_html=True)
        with icol:
            st.markdown("**Point inspector**")
            name = st.selectbox("Point", list(NODES.name), index=5, key="insp", label_visibility="collapsed")
            j = int(NODES.index[NODES.name == name][0])
            k, lab = risk_of(depth[j])
            crew = "No crew assigned"
            if not table.empty and name in set(table["Critical point"]):
                r = table[table["Critical point"] == name].iloc[0]
                crew = f"Crew {r['Crew']} · arrives in {r['ETA + setup (min)']:.0f} min"
            st.markdown(f"<div class='kpi'><div class='t'>{lab} risk</div>"
                        f"<div class='v'>{depth[j]:.0f} cm</div><span class='pill {k}'>predicted water, unmitigated</span>"
                        f"<p class='small' style='margin-top:12px'>With mitigation: <b>{mit[j]:.0f} cm</b><br>{crew}</p></div>",
                        unsafe_allow_html=True)
            st.write("")
            st.caption("Pick a point to see its details. Points above 25 cm can leave vehicles stalled.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                panel("Tide and culvert blockage",
                      "When the curve rises above the red line, the river prevents culverts from discharging rainwater.")
                t = np.linspace(0, 24, 241)
                h = (H - 1.4) + 1.4 * np.cos(2 * np.pi * t / 12.42)
                f = go.Figure(go.Scatter(x=t, y=h, name="Tide level", line=dict(color="#22D3EE", width=3)))
                f.add_hrect(y0=LOCK_THRESHOLD_M, y1=max(h.max(), 3.0) + .3, fillcolor="rgba(244,63,94,.10)", line_width=0)
                f.add_hline(y=LOCK_THRESHOLD_M, line=dict(color="#F43F5E", dash="dash"),
                            annotation_text="Blockage threshold 2.50 m", annotation_font_color="#FB7185")
                f.update_layout(xaxis_title="Hour of day (24 h cycle)", yaxis_title="Level [m]", showlegend=False)
                st.plotly_chart(style(f, 360), config=CFG, key="ch_tide")
        with c2:
            with st.container(border=True):
                panel("Tide sensitivity",
                      "Flooded points by tide level under the current rainfall. The dotted line marks your scenario.")
                Hs = np.linspace(0, 4, 41)
                Xs = np.vstack([np.c_[np.full(N, I), np.full(N, A), np.full(N, hh), NODES[["s", "c", "e"]].values] for hh in Hs])
                cnt = (model.predict(Xs).reshape(len(Hs), N) > STALL_CM).sum(axis=1)
                g = go.Figure(go.Scatter(x=Hs, y=cnt, line=dict(color="#F59E0B", width=3, shape="hv"),
                                         fill="tozeroy", fillcolor="rgba(245,158,11,.15)", name="Flooded points"))
                g.add_vline(x=H, line=dict(color="#E6EDF7", dash="dot"), annotation_text="Current scenario")
                g.update_layout(xaxis_title="Tide level [m]", yaxis_title="Points above 25 cm", showlegend=False)
                st.plotly_chart(style(g, 360), config=CFG, key="ch_sens")
        with st.container(border=True):
            panel("Water depth before and after mitigation",
                  "The 10 most affected points. The orange line is the 25 cm threshold above which vehicles may stall.")
            top = np.argsort(depth)[::-1][:10][::-1]
            b = go.Figure()
            b.add_trace(go.Bar(y=NODES.short[top], x=depth[top], orientation="h", name="Unmitigated", marker_color="#F43F5E",
                               texttemplate="%{x:.0f} cm", textposition="outside"))
            b.add_trace(go.Bar(y=NODES.short[top], x=mit[top], orientation="h", name="With crews (optimized)",
                               marker_color="#10B981", texttemplate="%{x:.0f} cm", textposition="outside"))
            b.add_vline(x=STALL_CM, line=dict(color="#F59E0B", dash="dot"))
            b.update_layout(barmode="group", xaxis_title="Water depth [cm]", bargap=.25)
            st.plotly_chart(style(b, 520), config=CFG, key="ch_bar")
    with tab2:
        st.markdown("#### Preventive hydro-cleaner dispatch (60 min before the event)")
        if table.empty:
            st.success("No critical points: preventive dispatch is not required.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Crews deployed", f"{len(table)} of {len(CREWS)}")
            m2.metric("Water reduced (sum of points)", f"{(table.iloc[:, 2] - table.iloc[:, 3]).sum():.0f} cm")
            m3.metric("Average arrival", f"{table['ETA + setup (min)'].mean():.0f} min")
            st.dataframe(table, hide_index=True, column_config={
                "Predicted depth (cm)": st.column_config.NumberColumn(format="%.1f cm"),
                "Mitigated depth (cm)": st.column_config.NumberColumn(format="%.1f cm"),
                "ETA + setup (min)": st.column_config.NumberColumn(format="%.0f min")})
            st.download_button("Download dispatch plan (CSV)", table.to_csv(index=False).encode("utf-8"),
                               "tidelock_dispatch.csv", "text/csv", icon=":material/download:")
        st.caption(f"Each crew serves a single point and must arrive and set up within {MAX_ETA_MIN:.0f} min. "
                   "Points with stalling risk are prioritized.")
    with tab3:
        st.markdown("#### Route to the hospital: conventional vs. TideLock-AI")
        ra, rb = st.columns(2, gap="large")
        with ra:
            st.markdown(f"<div class='route bad'><div class='t'>Conventional GPS (shortest route)</div>"
                        f"<div class='v'>{t_n:.0f} min</div><span class='pill r'>{x_n} flooded points on the way</span>"
                        f"<p style='margin-top:10px'>{hops(naive)}</p></div>", unsafe_allow_html=True)
        with rb:
            st.markdown(f"<div class='route good'><div class='t'>TideLock-AI (dynamic route)</div>"
                        f"<div class='v'>{t_s:.0f} min</div><span class='pill g'>{x_s} flooded points on the way</span>"
                        f"<p style='margin-top:10px'>{hops(smart)}</p></div>", unsafe_allow_html=True)
        st.write("")
        st.plotly_chart(make_map(depth, naive, smart, assign, 460), config=CFG, key="map_routes")
        st.markdown(legend_html(), unsafe_allow_html=True)
    with tab4:
        st.markdown("#### How TideLock-AI works")
        st.markdown("When it rains heavily and the Guayas river tide is high, stormwater cannot leave the city: "
                    "the drainage culverts are \"plugged\" by river water. TideLock-AI anticipates **where** flooding will occur and "
                    "**what to do** before it happens.")
        steps = [("1", "Conditions", "Current rainfall, rainfall over the last 6 hours and the tide level are entered. "
                  "In real operation they arrive automatically from forecasts and sensors."),
                 ("2", "Prediction", "For each of the 16 critical intersections, the water depth is estimated by comparing "
                  "the water arriving with the water the culverts can discharge. A machine learning model "
                  "reproduces that calculation in milliseconds."),
                 ("3", "Decision", "Cleaning crews are assigned to the points where their work reduces water the most, "
                  f"making sure they arrive in under {MAX_ETA_MIN:.0f} minutes."),
                 ("4", "Safe routes", "The path to the hospital is calculated avoiding points with more than 25 cm of water, "
                  "something a conventional GPS does not consider.")]
        cols = st.columns(4, gap="medium")
        for c, (n_, ttl, txt) in zip(cols, steps):
            c.markdown(f"<div class='step'><div class='n'>{n_}</div><h4>{ttl}</h4><p>{txt}</p></div>", unsafe_allow_html=True)
        st.write("")
        l, r = st.columns(2, gap="large")
        with l:
            with st.container(border=True):
                st.markdown("**How to read the indicators**")
                st.markdown("- **Culvert blockage:** share of drainage capacity lost because of high tide.\n"
                            "- **Flooded points:** intersections with more than 25 cm of water; beyond that a vehicle may stall.\n"
                            "- **Emergency route time:** estimated minutes to the hospital, including water delays.\n"
                            f"- **Model reliability:** the model reproduces the physical simulation with a mean error of {mae:.1f} cm (R² {r2:.3f}).")
        with r:
            with st.container(border=True):
                st.markdown("**Scope and limitations**")
                st.markdown("- The data and parameters shown are for demonstration.\n"
                            "- Reliability is measured against simulations, not field measurements.\n"
                            "- Operational use requires connecting official rainfall and tide forecasts and calibrating with level sensors.\n"
                            "- The water reduction per crew (55%) is an assumption that must be validated with real data.")


def citizen_view():
    model, _, _ = train_surrogate()
    right = header("Citizen mode")
    with right:
        if st.button("Switch profile", icon=":material/swap_horiz:"):
            st.session_state.role = None
            st.rerun()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        preset = st.selectbox("Current conditions (simulation)", list(PRESETS), index=list(PRESETS).index(DEFAULT_PRESET))
    with c2:
        zone = st.selectbox("Where are you?", list(NODES.name[:-1]), index=5)
    I, A, H = PRESETS[preset]
    depth = predict_depths(model, I, A, H)
    table, assign = milp_dispatch(depth)
    mit = depth.copy()
    for j in assign:
        mit[j] = round(depth[j] * (1 - REDUCTION), 1)
    zi = int(NODES.index[NODES.name == zone][0])
    d = float(depth[zi])
    kind, label, advice = (("g", "LOW RISK", "You can travel normally. Stay informed.") if d <= 10 else
                           ("a", "MODERATE RISK", "Drive slowly and avoid underpasses and streets with standing water.") if d <= STALL_CM else
                           ("r", "HIGH RISK", "Do not drive through this area. Find a safe place and wait for instructions."))
    st.markdown(f"<div class='risk {kind}'><div class='small'>Your area: {zone}</div><p class='big'>{label}</p>"
                f"<div>Predicted water: <b>{d:.0f} cm</b> · {advice}</div></div>", unsafe_allow_html=True)
    st.write("")
    _, smart, _, (t_s, x_s) = routes(depth, mit, src=zi)
    left, side = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Risk map and safe route to the hospital**")
        st.plotly_chart(make_map(depth, None, smart, None, 500, zoom=11.9), config=CFG, key="map_cit")
        st.markdown(legend_html(), unsafe_allow_html=True)
    with side:
        with st.container(border=True):
            st.markdown("**Areas to avoid**")
            bad = [j for j in range(N) if depth[j] > STALL_CM]
            if not bad:
                st.success("No critical areas right now.")
            for j in bad:
                st.markdown(f"<span class='pill r'>{depth[j]:.0f} cm</span> &nbsp;{NODES.name[j]}", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Suggested safe route to the hospital**")
            st.markdown(hops(smart), unsafe_allow_html=True)
            st.caption(f"Estimated time: {t_s:.0f} min.")
        with st.container(border=True):
            st.markdown("**What to do**")
            st.markdown("- Do not drive through water if you cannot see the road surface.\n- Avoid underpasses and low-lying areas.\n"
                        "- Do not park in channels or near drains.\n- In an emergency, call **ECU 911**.")


if "role" not in st.session_state:
    st.session_state.role = None
role = st.session_state.role
if role == "auth":
    authority_view()
elif role == "citizen":
    citizen_view()
else:
    gate()
