
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import io
import datetime

# =============================
# Page config (sidebar collapsed)
# =============================
st.set_page_config(
    page_title="SlamPlan - baeredygtigheds- og oekonomiberegner",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("SlamPlan")

intro = (
    "### Vaerktoejets formaal\n"
    "Dette vaerktoej hjaelper med at vurdere **klima (Scope 1-3 + total)**, **oekonomi (NPV)**, "
    "**naeringsstoffer (P/N/K og goedningsvaerdi)** og **biodiversitet** i slamhaandteringsstrategier.\n\n"
    "Vaelg opgoerelsesaar og sammenlign referencescenarierne side om side - eller aabn **Indstillinger** i venstre side for at tilpasse antagelserne."
)
st.markdown(intro)

# =============================
# Danish number helpers
# =============================
def format_da(x, decimals=1):
    try:
        s = f"{x:,.{decimals}f}"
    except Exception:
        return str(x)
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def da_formatter(decimals=0, suffix=""):
    def _fmt(x, pos):
        return (format_da(x, decimals) + (suffix if suffix else ""))
    return FuncFormatter(_fmt)

# =============================
# Defaults & constants
# =============================
current_year = datetime.datetime.now().year
MAX_YEARS = 50

# DK2 forecast (gCO2/kWh) 2025..2035 then linear to 10 g by horizon
embedded_dk2 = {
    2025: 40.0, 2026: 38.0, 2027: 35.0, 2028: 33.0, 2029: 30.0,
    2030: 25.0, 2031: 22.0, 2032: 20.0, 2033: 18.0, 2034: 16.0, 2035: 15.0
}
FLOOR_G_CO2_PER_KWH = 10.0

# Scope 1 defaults (gCO2 per wet tonne)
default_treatment_scope1 = {
    "Anaerob forraadnelse": -5000.0,
    "Forbraending": 2000.0,
    "Pyrolyse": -1000.0,
    "Landudbringning": 5000.0
}

# Transport
default_truck = {
    "Diesel": {"type":"fuel", "ef_g_per_tkm": 60.0},
    "HVO": {"type":"fuel", "ef_g_per_tkm": 20.0},
    "El (grid)": {"type":"electric", "kwh_per_tkm": 0.3}
}

# Economy (DKK per wet tonne)
default_econ = {
    "Anaerob forraadnelse": {"capex_per_t": 200.0, "opex_per_t_per_yr": 25.0, "rev_energy_per_t": 15.0, "rev_product_per_t": 10.0},
    "Forbraending": {"capex_per_t": 150.0, "opex_per_t_per_yr": 25.0, "rev_energy_per_t": 10.0, "rev_product_per_t": 0.0},
    "Pyrolyse": {"capex_per_t": 300.0, "opex_per_t_per_yr": 30.0, "rev_energy_per_t": 20.0, "rev_product_per_t": 30.0},
    "Landudbringning": {"capex_per_t": 50.0, "opex_per_t_per_yr": 10.0, "rev_energy_per_t": 0.0, "rev_product_per_t": 0.0}
}

# Nutrients
DEFAULT_TS_PCT = 25.0
nutrient_defaults = {"P_kg_per_t_TS": 20.0, "N_kg_per_t_TS": 35.0, "K_kg_per_t_TS": 4.0}
nutrient_retention = {
    "Anaerob forraadnelse": {"P_ret": 0.90, "P_avail": 0.70, "N_ret": 0.60, "K_ret": 0.80},
    "Pyrolyse": {"P_ret": 0.95, "P_avail": 0.25, "N_ret": 0.10, "K_ret": 0.60},
    "Forbraending": {"P_ret": 0.10, "P_avail": 0.10, "N_ret": 0.05, "K_ret": 0.20},
    "Landudbringning": {"P_ret": 0.90, "P_avail": 0.70, "N_ret": 0.60, "K_ret": 0.80}
}
fertilizer_prices = {"N": 10.0, "P": 25.0, "K": 7.0}

# Biodiversity (simple indicator)
biodiv_params = {
    "Landudbringning": {"area":0.8, "metals":0.5, "pfas":0.6, "p_benefit":0.8},
    "Pyrolyse": {"area":0.1, "metals":0.1, "pfas":0.1, "p_benefit":0.7},
    "Forbraending": {"area":0.05, "metals":0.2, "pfas":0.3, "p_benefit":0.1},
    "Anaerob forraadnelse": {"area":0.2, "metals":0.2, "pfas":0.2, "p_benefit":0.6},
}
def biodiversity_score(area, metals, pfas, p_benefit):
    return (1 - area) * (1 - metals) * (1 - pfas) + 0.2 * p_benefit

# Colors
COLOR_SCOPE1 = "#D7191C"
COLOR_SCOPE2 = "#FDAE61"
COLOR_SCOPE3 = "#FEE08B"  # gul som oensket
COLOR_TOTAL  = "#808080"  # graa som oensket

# =============================
# Sidebar (collapsed)
# =============================
with st.sidebar:
    st.header("Indstillinger")
    years_horizon = st.slider("Tidshorisont (aar)", 1, MAX_YEARS, 30, help="Vaelg 1-50 aar frem.")
    start_year = st.number_input("Startaar", value=max(2025, current_year), step=1,
                                 help="DK2-prognose 2025-2035, derefter ekstrapolation.")
    view_year = st.slider("Vaelg opgoerelsesaar", min_value=int(start_year),
                          max_value=int(start_year + years_horizon - 1), value=int(start_year),
                          help="Totaler og grafer opgoeres for dette aar.")

    annual_sludge_wet_t = st.number_input("Aarlig slam (vaad ton/aar)", min_value=0.0, value=10000.0, step=500.0,
                                          help="Typisk 1.000–50.000 t/aar.")
    ts_pct = st.number_input("TS i slam (%)", 1.0, 100.0, value=DEFAULT_TS_PCT, step=1.0,
                             help="Typisk 20–30 % efter afvanding.")
    distance_km = st.number_input("Transportafstand (km, en vej)", min_value=0.0, value=100.0, step=10.0,
                                  help="Typisk 10–150 km.")
    roundtrip = st.checkbox("Tur/retur", value=True, help="Returtransport (typisk ja).")

    st.markdown("---")
    st.header("El-mix (DK2)")
    use_el_forecast = st.checkbox("Brug DK2-elprognose (aar-for-aar)", value=True,
                                  help="Hvis fra, bruges fast el-faktor for alle aar.")
    fixed_el_factor = st.number_input("Fast el-emissionsfaktor (gCO2/kWh)", value=50.0, step=1.0,
                                      help="Typisk 40–80 gCO2/kWh i 2025-niveau.")
    uploaded_el = st.file_uploader("Upload el-CSV (year,gCO2_per_kWh) (valgfrit)", type=["csv"],
                                   help="Hvis uploadet, overskriver indlejret DK2-serie.")

    st.markdown("---")
    st.header("Proces og transport")
    el_kwh_per_t = st.number_input("Elforbrug pr. ton behandlet (kWh/t)", value=50.0, step=1.0,
                                   help="Typisk 30–80 kWh/t.")
    truck_mode = st.selectbox("Transporttype", options=list(default_truck.keys()),
                              help="Diesel/HVO bruger gCO2/tkm; el-lastbil bruger kWh/tkm * el-mix.")
    truck_val = default_truck[truck_mode]
    if truck_val["type"] == "fuel":
        truck_ef_g_per_tkm = st.number_input("Lastbil EF (gCO2/ton-km)", value=float(truck_val["ef_g_per_tkm"]), step=1.0)
        truck_kwh_per_tkm = None
    else:
        truck_kwh_per_tkm = st.number_input("Elforbrug lastbil (kWh/ton-km)", value=float(truck_val["kwh_per_tkm"]), step=0.01)
        truck_ef_g_per_tkm = None

    st.markdown("---")
    st.header("Oekonomi og finans")
    discount_rate = st.number_input("Diskonteringsrente (%)", value=4.0, min_value=0.0, max_value=20.0, step=0.1)
    fert_N = st.number_input("Goedningspris N (DKK/kg N)", value=fertilizer_prices["N"], step=0.5)
    fert_P = st.number_input("Goedningspris P (DKK/kg P)", value=fertilizer_prices["P"], step=0.5)
    fert_K = st.number_input("Goedningspris K (DKK/kg K)", value=fertilizer_prices["K"], step=0.5)

    st.markdown("---")
    st.header("Naeringsstoffer (kg/ton TS)")
    P_per_t_TS = st.number_input("P (kg/ton TS)", value=nutrient_defaults["P_kg_per_t_TS"], step=1.0)
    N_per_t_TS = st.number_input("N (kg/ton TS)", value=nutrient_defaults["N_kg_per_t_TS"], step=1.0)
    K_per_t_TS = st.number_input("K (kg/ton TS)", value=nutrient_defaults["K_kg_per_t_TS"], step=0.5)

    st.markdown("---")
    st.header("Scenarier")
    all_scenarios = list(default_treatment_scope1.keys())
    sel_scenarios = st.multiselect("Vaelg scenarier", options=all_scenarios, default=all_scenarios)

    st.markdown("**Oekonomi pr. scenarie (DKK/ton vaad)**")
    econ_inputs = {}
    for s in sel_scenarios:
        col1, col2, col3, col4 = st.columns(4)
        econ_inputs[s] = {
            "capex_per_t": col1.number_input(f"CAPEX {s}", value=default_econ[s]["capex_per_t"], step=10.0),
            "opex_per_t_per_yr": col2.number_input(f"OPEX {s}", value=default_econ[s]["opex_per_t_per_yr"], step=1.0),
            "rev_energy_per_t": col3.number_input(f"Energiindt. {s}", value=default_econ[s]["rev_energy_per_t"], step=1.0),
            "rev_product_per_t": col4.number_input(f"Produktindt. {s}", value=default_econ[s]["rev_product_per_t"], step=1.0),
        }

    st.markdown("**Proces-udledning (Scope 1) (gCO2/ton vaad)**")
    treatment_scope1_inputs = {}
    for s in sel_scenarios:
        treatment_scope1_inputs[s] = st.number_input(f"Scope1 EF {s}", value=default_treatment_scope1[s], step=100.0)

# =============================
# Build year series (forecast)
# =============================
years = np.arange(int(start_year), int(start_year) + int(years_horizon))
view_idx = int(np.where(years == view_year)[0][0]) if view_year in years else 0

# Electricity EF series
if uploaded_el is not None:
    try:
        df_el = pd.read_csv(uploaded_el)
        ef_map = dict(zip(df_el['year'].astype(int), df_el['gCO2_per_kWh'].astype(float)))
    except Exception as e:
        st.error(f"Kunne ikke laese uploadet el-CSV: {e}. Bruger indlejret DK2 + ekstrapolation.")
        ef_map = {}
else:
    ef_map = embedded_dk2.copy()

el_series = np.full_like(years, np.nan, dtype=float)
for i, y in enumerate(years):
    if y in ef_map:
        el_series[i] = ef_map[y]
ser = pd.Series(el_series, index=years).interpolate(limit_direction='both')
el_series = ser.values

last_known_idx = np.where(~np.isnan(el_series))[0]
if len(last_known_idx) > 0:
    last_i = last_known_idx[-1]; last_val = el_series[last_i]
    end_val = FLOOR_G_CO2_PER_KWH
    if last_i < len(el_series) - 1:
        steps = len(el_series) - 1 - last_i
        slope = (end_val - last_val) / steps if steps>0 else 0.0
        for j in range(last_i+1, len(el_series)):
            el_series[j] = last_val + slope * (j - last_i)
else:
    el_series[:] = FLOOR_G_CO2_PER_KWH

if not use_el_forecast:
    el_series[:] = fixed_el_factor

# =============================
# Core calculations
# =============================
def npv(cashflows, rate_pct):
    r = rate_pct/100.0
    years_idx = np.arange(1, len(cashflows)+1)
    return np.sum(cashflows / ((1+r)**years_idx))

def compute_transport_emissions(tonnes_per_year, distance_oneway_km, roundtrip, truck_mode, truck_ef_g_per_tkm, truck_kwh_per_tkm, el_g_per_kwh_series):
    trips = 2 if roundtrip else 1
    if truck_mode == "El (grid)":
        per_year_g = tonnes_per_year * distance_oneway_km * trips * truck_kwh_per_tkm * el_g_per_kwh_series
        return per_year_g
    else:
        per_year_g = np.full(len(years), tonnes_per_year * distance_oneway_km * trips * truck_ef_g_per_tkm)
        return per_year_g

def compute_electricity_emissions(kwh_per_tonne, tonnes_per_year, el_g_per_kwh_series):
    return kwh_per_tonne * tonnes_per_year * el_g_per_kwh_series

def tonnes_wet_to_TS(tonnes_wet, ts_pct):
    return tonnes_wet * (ts_pct/100.0)

def compute_nutrients(tonnes_wet, ts_pct, scen_name):
    t_TS = tonnes_wet_to_TS(tonnes_wet, ts_pct)
    P_in = t_TS * P_per_t_TS
    N_in = t_TS * N_per_t_TS
    K_in = t_TS * K_per_t_TS
    pars = nutrient_retention.get(scen_name, {"P_ret":1.0,"P_avail":1.0,"N_ret":1.0,"K_ret":1.0})
    P_out = P_in * pars["P_ret"] * pars["P_avail"]
    N_out = N_in * pars["N_ret"]
    K_out = K_in * pars["K_ret"]
    value_dkk = P_out * fert_P + N_out * fert_N + K_out * fert_K
    return {"P_kg": P_out, "N_kg": N_out, "K_kg": K_out, "value_DKK": value_dkk}

# grab values defined in sidebar scope
P_per_t_TS = locals().get("P_per_t_TS", nutrient_defaults["P_kg_per_t_TS"])
N_per_t_TS = locals().get("N_per_t_TS", nutrient_defaults["N_kg_per_t_TS"])
K_per_t_TS = locals().get("K_per_t_TS", nutrient_defaults["K_kg_per_t_TS"])
fert_N = locals().get("fert_N", fertilizer_prices["N"])
fert_P = locals().get("fert_P", fertilizer_prices["P"])
fert_K = locals().get("fert_K", fertilizer_prices["K"])

results = {}
for s in sel_scenarios:
    scope1_ef = treatment_scope1_inputs[s]
    scope1 = np.full(len(years), annual_sludge_wet_t * scope1_ef)
    scope2 = compute_electricity_emissions(el_kwh_per_t, annual_sludge_wet_t, el_series)
    scope3 = compute_transport_emissions(annual_sludge_wet_t, distance_km, roundtrip, truck_mode, truck_ef_g_per_tkm, truck_kwh_per_tkm, el_series)
    total = scope1 + scope2 + scope3

    econ = econ_inputs[s]
    investment = -econ["capex_per_t"] * annual_sludge_wet_t
    annual_net = -econ["opex_per_t_per_yr"] * annual_sludge_wet_t + (econ["rev_energy_per_t"] + econ["rev_product_per_t"]) * annual_sludge_wet_t
    cashflows = np.concatenate(([investment], np.full(len(years), annual_net)))
    npv_val = investment + npv(cashflows[1:], discount_rate)

    pars = biodiv_params.get(s, {"area":0.5,"metals":0.5,"pfas":0.5,"p_benefit":0.5})
    bio_score = biodiversity_score(pars["area"], pars["metals"], pars["pfas"], pars["p_benefit"])

    nutrients = compute_nutrients(annual_sludge_wet_t, ts_pct, s)

    results[s] = {
        "scope1_g": scope1, "scope2_g": scope2, "scope3_g": scope3, "total_g": total,
        "npv": npv_val, "investment": investment, "annual_net": annual_net,
        "biodiversity": bio_score, "nutrients": nutrients
    }

# =============================
# UI: Tabs
# =============================
tab_res, tab_sens, tab_info = st.tabs(["Resultater", "Foelsomhed", "Baggrund"])

with tab_res:
    st.subheader(f"Aarsresultater - {int(view_year)}")
    rows = []
    for s, d in results.items():
        rows.append({
            "Scenarie": s,
            "Scope 1 (tCO2e/aar)": d["scope1_g"][view_idx]/1e6,
            "Scope 2 (tCO2e/aar)": d["scope2_g"][view_idx]/1e6,
            "Scope 3 (tCO2e/aar)": d["scope3_g"][view_idx]/1e6,
            "Total (tCO2e/aar)": d["total_g"][view_idx]/1e6,
            "NPV (DKK)": d["npv"],
            "Investering (DKK)": d["investment"],
            "Aarlig net (DKK/aar)": d["annual_net"],
            "P (kg/aar)": d["nutrients"]["P_kg"],
            "N (kg/aar)": d["nutrients"]["N_kg"],
            "K (kg/aar)": d["nutrients"]["K_kg"],
            "Goedningsvaerdi (DKK/aar)": d["nutrients"]["value_DKK"],
            "Biodiversitet (0-1)": d["biodiversity"]
        })
    df = pd.DataFrame(rows)
    def apply_fmt(col, dec=1):
        return df[col].apply(lambda v: format_da(v,dec))
    for c in ["Scope 1 (tCO2e/aar)","Scope 2 (tCO2e/aar)","Scope 3 (tCO2e/aar)","Total (tCO2e/aar)"]:
        df[c] = apply_fmt(c,1)
    for c in ["NPV (DKK)","Investering (DKK)","Aarlig net (DKK/aar)","Goedningsvaerdi (DKK/aar)"]:
        df[c] = apply_fmt(c,0)
    for c in ["P (kg/aar)","N (kg/aar)","K (kg/aar)"]:
        df[c] = apply_fmt(c,0)
    df["Biodiversitet (0-1)"] = apply_fmt("Biodiversitet (0-1)",2)
    st.dataframe(df, use_container_width=True)
    st.download_button("Eksport: aarsresultater (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="slamplan_arsresultater.csv", mime="text/csv")

    # Scopes stacked
    labels = list(results.keys())
    s1 = [results[s]["scope1_g"][view_idx]/1e6 for s in labels]
    s2 = [results[s]["scope2_g"][view_idx]/1e6 for s in labels]
    s3 = [results[s]["scope3_g"][view_idx]/1e6 for s in labels]
    totals = [results[s]["total_g"][view_idx]/1e6 for s in labels]
    max_y = max(totals) if totals else 1.0

    st.markdown("**Scopes fordelt pr. scenarie (tCO2e/aar)**")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, s1, label="Scope 1", color=COLOR_SCOPE1)
    ax.bar(labels, s2, bottom=s1, label="Scope 2", color=COLOR_SCOPE2)
    bottom_s12 = [a+b for a,b in zip(s1,s2)]
    ax.bar(labels, s3, bottom=bottom_s12, label="Scope 3", color=COLOR_SCOPE3)
    ax.set_ylim(0, max_y*1.15 if max_y>0 else 1)
    ax.set_ylabel("tCO2e/aar"); ax.yaxis.set_major_formatter(da_formatter(1))
    plt.xticks(rotation=15); ax.legend(loc="upper right"); plt.tight_layout()
    st.pyplot(fig); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200)
    st.download_button("Download PNG (scopes)", data=buf.getvalue(), file_name="slamplan_scopes.png", mime="image/png")

    st.markdown("**Total tCO2e pr. scenarie (valgt aar)**")
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.bar(labels, totals, color=COLOR_TOTAL)
    ax2.set_ylim(0, max_y*1.15 if max_y>0 else 1); ax2.set_ylabel("tCO2e/aar"); ax2.yaxis.set_major_formatter(da_formatter(1))
    plt.xticks(rotation=15); plt.tight_layout()
    st.pyplot(fig2); buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=200)
    st.download_button("Download PNG (total)", data=buf2.getvalue(), file_name="slamplan_total.png", mime="image/png")

    st.markdown("**Naeringsstof-tilfoersel og vaerdi (valgt aar)**")
    P_vals = [results[s]["nutrients"]["P_kg"] for s in labels]
    N_vals = [results[s]["nutrients"]["N_kg"] for s in labels]
    K_vals = [results[s]["nutrients"]["K_kg"] for s in labels]
    V_vals = [results[s]["nutrients"]["value_DKK"] for s in labels]
    fig3, ax3 = plt.subplots(figsize=(8,4))
    x = np.arange(len(labels)); width = 0.2
    ax3.bar(x - width, P_vals, width, label="P (kg/aar)")
    ax3.bar(x, N_vals, width, label="N (kg/aar)")
    ax3.bar(x + width, K_vals, width, label="K (kg/aar)")
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=15)
    ax3.yaxis.set_major_formatter(da_formatter(0)); ax3.legend(); plt.tight_layout()
    st.pyplot(fig3); buf3 = io.BytesIO(); fig3.savefig(buf3, format="png", dpi=200)
    st.download_button("Download PNG (naeringsstoffer)", data=buf3.getvalue(), file_name="slamplan_naeringsstoffer.png", mime="image/png")

    nut_rows = [{"Scenarie":lab, "P (kg/aar)":P_vals[i], "N (kg/aar)":N_vals[i], "K (kg/aar)":K_vals[i], "Goedningsvaerdi (DKK/aar)":V_vals[i]} for i, lab in enumerate(labels)]
    nut_df = pd.DataFrame(nut_rows)
    for c in ["P (kg/aar)","N (kg/aar)","K (kg/aar)"]:
        nut_df[c] = nut_df[c].apply(lambda v: format_da(v,0))
    nut_df["Goedningsvaerdi (DKK/aar)"] = nut_df["Goedningsvaerdi (DKK/aar)"].apply(lambda v: format_da(v,0))
    st.dataframe(nut_df, use_container_width=True)
    st.download_button("Eksport: naeringsstoffer (CSV)", data=nut_df.to_csv(index=False).encode("utf-8"), file_name="slamplan_naeringsstoffer.csv", mime="text/csv")

    st.markdown("**Indikativ biodiversitet (valgt aar) - score 0-1**")
    scores = [results[s]["biodiversity"] for s in labels]
    colors = []
    for sc in scores:
        if sc < 0.4: colors.append("#D7191C")
        elif sc < 0.7: colors.append("#FEE08B")
        else: colors.append("#1A9641")
    fig4, ax4 = plt.subplots(figsize=(6,2.5))
    ax4.barh(labels, scores, color=colors)
    ax4.set_xlim(0,1); ax4.grid(axis="x", linestyle="--", alpha=0.4)
    ax4.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{x:,.2f}".replace(".",",")))
    plt.tight_layout(); st.pyplot(fig4); buf4 = io.BytesIO(); fig4.savefig(buf4, format="png", dpi=200)
    st.download_button("Download PNG (biodiversitet)", data=buf4.getvalue(), file_name="slamplan_biodiversitet.png", mime="image/png")

with tab_sens:
    st.subheader("Foelsomhedsanalyse (+/-10%) - vaelg scenarie")
    scen = st.selectbox("Scenarie", options=list(results.keys()))
    def npv_func(cashflows, rate_pct):
        r = rate_pct/100.0
        years_idx = np.arange(1, len(cashflows)+1)
        return np.sum(cashflows / ((1+r)**years_idx))
    def recompute_for(varname, factor):
        _annual = annual_sludge_wet_t
        _dist = distance_km * (factor if varname=="distance" else 1.0)
        _el_kwh = el_kwh_per_t * (factor if varname=="eluse" else 1.0)
        _el_series = el_series * (factor if varname=="elef" else 1.0)
        e = econ_inputs[scen].copy()
        if varname=="capex": e["capex_per_t"] *= factor
        if varname=="opex": e["opex_per_t_per_yr"] *= factor
        if varname=="re_ene": e["rev_energy_per_t"] *= factor
        if varname=="re_prod": e["rev_product_per_t"] *= factor
        if truck_mode=="El (grid)":
            _truck_kwh = truck_kwh_per_tkm * (factor if varname=="truck_energy" else 1.0)
            _truck_ef = None
        else:
            _truck_ef = (truck_ef_g_per_tkm * (factor if varname=="truck_ef" else 1.0))
            _truck_kwh = None
        s1 = np.full(len(years), _annual * treatment_scope1_inputs[scen])
        s2 = compute_electricity_emissions(_el_kwh, _annual, _el_series)
        s3 = compute_transport_emissions(_annual, _dist, roundtrip, truck_mode, _truck_ef, _truck_kwh, _el_series)
        total = s1 + s2 + s3
        inv = -e["capex_per_t"] * _annual
        annual_net = -e["opex_per_t_per_yr"] * _annual + (e["rev_energy_per_t"] + e["rev_product_per_t"]) * _annual
        cf = np.concatenate(([inv], np.full(len(years), annual_net)))
        npv_val = inv + npv_func(cf[1:], discount_rate)
        return total[view_idx]/1e6, npv_val

    base_total_t, base_npv = recompute_for("none", 1.0)

    params = [("distance","Transportafstand"),("eluse","Elforbrug pr. ton"),("elef","El-udledning (fast/prognose)"),
              ("capex","CAPEX"),("opex","OPEX"),("re_ene","Energiindtægt"),("re_prod","Produktindtægt")]
    if truck_mode=="El (grid)":
        params.insert(0, ("truck_energy","Elforbrug lastbil (kWh/tkm)"))
    else:
        params.insert(0, ("truck_ef","Lastbil EF (gCO2/tkm)"))

    sens_rows = []
    for key, name in params:
        up_total, up_npv = recompute_for(key, 1.10)
        dn_total, dn_npv = recompute_for(key, 0.90)
        delta_t = max(abs(up_total - base_total_t), abs(dn_total - base_total_t))
        delta_npv = max(abs(up_npv - base_npv), abs(dn_npv - base_npv))
        sens_rows.append({"Parameter": name, "Delta tCO2e/aar": delta_t, "Delta NPV (DKK)": delta_npv})

    sens_df = pd.DataFrame(sens_rows).sort_values("Delta tCO2e/aar", ascending=True)
    st.markdown("**Tornado - Klimapavirking (tCO2e/aar)**")
    fig5, ax5 = plt.subplots(figsize=(8,4))
    y = np.arange(len(sens_df))
    ax5.barh(y, sens_df["Delta tCO2e/aar"], color=COLOR_TOTAL)
    ax5.set_yticks(y); ax5.set_yticklabels(sens_df["Parameter"])
    ax5.xaxis.set_major_formatter(da_formatter(1)); ax5.set_xlabel("AEndring i tCO2e/aar ved +/-10%")
    plt.tight_layout(); st.pyplot(fig5); buf5 = io.BytesIO(); fig5.savefig(buf5, format="png", dpi=200)
    st.download_button("Download PNG (tornado CO2)", data=buf5.getvalue(), file_name="slamplan_tornado_co2.png", mime="image/png")

    sens_df2 = pd.DataFrame(sens_rows).sort_values("Delta NPV (DKK)", ascending=True)
    st.markdown("**Tornado - Oekonomi (NPV, DKK)**")
    fig6, ax6 = plt.subplots(figsize=(8,4))
    y2 = np.arange(len(sens_df2))
    ax6.barh(y2, sens_df2["Delta NPV (DKK)"], color=COLOR_TOTAL)
    ax6.set_yticks(y2); ax6.set_yticklabels(sens_df2["Parameter"])
    ax6.xaxis.set_major_formatter(da_formatter(0)); ax6.set_xlabel("AEndring i NPV (DKK) ved +/-10%")
    plt.tight_layout(); st.pyplot(fig6); buf6 = io.BytesIO(); fig6.savefig(buf6, format="png", dpi=200)
    st.download_button("Download PNG (tornado NPV)", data=buf6.getvalue(), file_name="slamplan_tornado_npv.png", mime="image/png")

    edf = pd.DataFrame(sens_rows)
    st.download_button("Eksport: foelsomhed (CSV)", data=edf.to_csv(index=False).encode("utf-8"),
                       file_name="slamplan_tornado_sensitivity.csv", mime="text/csv")

with tab_info:
    st.markdown("Baggrund og kilder:")
    st.markdown("- El-mix (DK2): Energinet / Energistyrelsen, 2025-2035 (indlejret), derefter lineaer ekstrapolation mod 10 gCO2/kWh.")
    st.markdown("- Defaultvaerdier for behandling, transport og oekonomi er vejledende og boer erstattes med lokale data, hvor muligt.")
    st.markdown("- Scopes: 1=proces (inddata pr. scenarie), 2=el (fast/prognose), 3=transport (diesel/HVO EF eller el-kWh/tkm via DK2).")
    st.markdown("- Biodiversitet: Forenklet indeks baseret paa areal, metaller, PFAS og fosforfordel. Vises som separat indikator.")
