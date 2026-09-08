import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Page Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Virtual Chemistry Lab | BTech First Year",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Modern Header */
.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 22px 28px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.main-header h1 { margin: 0; font-size: 30px; font-weight: 700; color: #ffffff; }
.main-header p { margin: 6px 0 0; opacity: 0.9; font-size: 15px; }

/* Visual Card Callouts */
.concept-card {
    background-color: #f8fa0c10;
    border-left: 5px solid #2a5298;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 15px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants & Helper Functions
# -----------------------------
C_LIGHT = 3.0e8              # m/s
H_PLANCK = 6.62607015e-34    # J s
E_CHARGE = 1.602176634e-19   # C
R_HYD = 1.0973731568539e7    # m^-1 (Rydberg Constant)

SEMICONDUCTORS = {
    "ZnO (Zinc Oxide)": {"Eg": 3.28, "alpha0": 1.0e4, "use": "UV absorbers, sunscreens, transparent electronics"},
    "GaAs (Gallium Arsenide)": {"Eg": 1.42, "alpha0": 1.5e4, "use": "High-efficiency solar cells, optoelectronics"},
    "Si (Silicon)": {"Eg": 1.12, "alpha0": 1.0e4, "use": "Microchips, commercial photovoltaic solar cells"},
    "CdS (Cadmium Sulfide)": {"Eg": 2.42, "alpha0": 1.2e4, "use": "Quantum dots, photoresistors"},
    "TiO₂ (Titanium Dioxide)": {"Eg": 3.20, "alpha0": 1.0e4, "use": "Photocatalysis, self-cleaning coatings"},
}

CATHODE_MATERIALS = {
    "Cesium (Cs)": {"work_function": 2.14, "color": "#e2e8f0"},
    "Potassium (K)": {"work_function": 2.29, "color": "#cbd5e1"},
    "Sodium (Na)": {"work_function": 2.36, "color": "#94a3b8"},
    "Calcium (Ca)": {"work_function": 2.87, "color": "#64748b"},
    "Copper (Cu)": {"work_function": 4.70, "color": "#b45309"},
}

LAMPS_DATA = {
    "Hydrogen (H₂) Lamp": {
        "lines": [
            {"name": "H-alpha (n=3→2)", "lambda_nm": 656.28, "color": "#ff0000", "intensity": 1.0},
            {"name": "H-beta (n=4→2)", "lambda_nm": 486.13, "color": "#00ffff", "intensity": 0.75},
            {"name": "H-gamma (n=5→2)", "lambda_nm": 434.05, "color": "#4b0082", "intensity": 0.50},
            {"name": "H-delta (n=6→2)", "lambda_nm": 410.17, "color": "#8a2be2", "intensity": 0.35},
        ]
    },
    "Mercury (Hg) Lamp": {
        "lines": [
            {"name": "Violet 1", "lambda_nm": 404.66, "color": "#8b00ff", "intensity": 0.6},
            {"name": "Violet 2", "lambda_nm": 435.83, "color": "#0000ff", "intensity": 0.9},
            {"name": "Blue-Green", "lambda_nm": 491.60, "color": "#00f5d4", "intensity": 0.3},
            {"name": "Green", "lambda_nm": 546.07, "color": "#00ff00", "intensity": 1.0},
            {"name": "Yellow Line 1", "lambda_nm": 576.96, "color": "#ffff00", "intensity": 0.7},
            {"name": "Yellow Line 2", "lambda_nm": 579.07, "color": "#ffee00", "intensity": 0.7},
        ]
    }
}

def nm_to_rgb_color(wavelength):
    """Maps wavelength (nm) to representative CSS color hex."""
    wl = float(wavelength)
    if wl < 420:
        return "#7b1fa2"
    elif wl < 450:
        return "#3f51b5"
    elif wl < 495:
        return "#00bcd4"
    elif wl < 570:
        return "#4caf50"
    elif wl < 590:
        return "#ffeb3b"
    elif wl < 620:
        return "#ff9800"
    elif wl <= 750:
        return "#f44336"
    return "#9e9e9e"

# -----------------------------
# Simulation Models
# -----------------------------
def battery_sim(cell_type, nominal_v, capacity_ah, capacitance_f,
                r_internal, charge_current, discharge_current, cutoff_v, time_step):
    if cell_type == "Lithium-ion Battery":
        def ocv(soc):
            soc = np.clip(soc, 0, 1)
            return 3.0 + 0.75 * soc + 0.10 * np.log10(np.maximum(soc, 1e-5) / np.maximum(1-soc, 1e-5)) * 0.05

        soc0 = 0.15
        charge_seconds = max((1.0 - soc0) * capacity_ah / max(charge_current, 1e-9) * 3600, 1)
        n_charge = int(np.ceil(charge_seconds / time_step)) + 1
        t_charge = np.arange(n_charge) * time_step
        soc_charge = np.minimum(soc0 + charge_current * t_charge / 3600 / capacity_ah, 1.0)
        v_charge = np.clip(ocv(soc_charge) + r_internal * charge_current, 2.8, 4.25)

        soc = 1.0
        rows = []
        t = 0.0
        while t <= capacity_ah / max(discharge_current, 1e-9) * 3600 * 1.15:
            v = float(np.clip(ocv(soc) - r_internal * discharge_current, 2.5, 4.25))
            if v <= cutoff_v:
                break
            rows.append((t, v, discharge_current, (1.0 - soc) * capacity_ah))
            soc -= discharge_current * time_step / 3600 / capacity_ah
            soc = max(soc, 0)
            t += time_step

        d = np.array(rows)
        t_dis = d[:, 0] if len(d) else np.array([0.0])
        v_dis = d[:, 1] if len(d) else np.array([cutoff_v])
        i_dis = d[:, 2] if len(d) else np.array([discharge_current])
        cap_dis = d[:, 3] if len(d) else np.array([0.0])

        charge_df = pd.DataFrame({
            "Time (s)": t_charge,
            "Voltage (V)": v_charge,
            "Current (A)": np.full_like(t_charge, charge_current),
            "Capacity (Ah)": charge_current * t_charge / 3600,
            "SOC (%)": soc_charge * 100,
            "Phase": "Charging",
        })
        dis_df = pd.DataFrame({
            "Time (s)": t_dis + t_charge[-1],
            "Voltage (V)": v_dis,
            "Current (A)": -i_dis,
            "Capacity (Ah)": cap_dis,
            "SOC (%)": (1 - cap_dis / capacity_ah) * 100,
            "Phase": "Discharging",
        })
        df = pd.concat([charge_df, dis_df], ignore_index=True)
        
        t_charge_h = t_charge[-1] / 3600
        t_dis_h = t_dis[-1] / 3600
        energy_wh = np.trapezoid(v_dis * i_dis, t_dis) / 3600 if len(t_dis) > 1 else 0
        charge_ah = np.trapezoid(np.full_like(t_charge, charge_current), t_charge) / 3600
        discharge_ah = np.trapezoid(i_dis, t_dis) / 3600 if len(t_dis) > 1 else 0
        efficiency = (discharge_ah / charge_ah * 100) if charge_ah > 0 else 0

        return df, {
            "Charge Time": f"{t_charge_h:.2f} hrs",
            "Discharge Time": f"{t_dis_h:.2f} hrs",
            "Energy Delivered": f"{energy_wh:.2f} Wh",
            "Coulombic Efficiency": f"{efficiency:.1f}%",
        }

    # Supercapacitor
    v_min = max(cutoff_v, 0.05)
    v_max = nominal_v
    q_max = capacitance_f * v_max
    charge_seconds = q_max / max(charge_current, 1e-9)
    n_charge = int(np.ceil(charge_seconds / time_step)) + 1
    t_charge = np.arange(n_charge) * time_step
    v_charge = np.clip(np.minimum(v_min + charge_current * t_charge / capacitance_f, v_max) + r_internal * charge_current, 0, v_max * 1.05)

    t_max = max(capacitance_f * max(v_max - v_min, 0) / max(discharge_current, 1e-9), time_step)
    n_dis = int(np.ceil(t_max / time_step)) + 1
    t_dis = np.arange(n_dis) * time_step
    v_dis = v_max - discharge_current * t_dis / capacitance_f - r_internal * discharge_current
    keep = v_dis > v_min
    t_dis, v_dis = t_dis[keep], v_dis[keep]
    if len(t_dis) == 0:
        t_dis, v_dis = np.array([0.0]), np.array([v_min])

    charge_df = pd.DataFrame({
        "Time (s)": t_charge,
        "Voltage (V)": v_charge,
        "Current (A)": np.full_like(t_charge, charge_current),
        "Capacity (Ah)": charge_current * t_charge / 3600,
        "SOC (%)": np.clip((v_charge - v_min) / (v_max - v_min) * 100, 0, 100),
        "Phase": "Charging",
    })
    dis_df = pd.DataFrame({
        "Time (s)": t_dis + t_charge[-1],
        "Voltage (V)": v_dis,
        "Current (A)": -np.full_like(t_dis, discharge_current),
        "Capacity (Ah)": discharge_current * t_dis / 3600,
        "SOC (%)": np.clip((v_dis - v_min) / (v_max - v_min) * 100, 0, 100),
        "Phase": "Discharging",
    })
    df = pd.concat([charge_df, dis_df], ignore_index=True)

    t_charge_h = t_charge[-1] / 3600
    t_dis_h = t_dis[-1] / 3600
    energy_wh = np.trapezoid(v_dis * discharge_current, t_dis) / 3600 if len(t_dis) > 1 else 0
    charge_ah = np.trapezoid(np.full_like(t_charge, charge_current), t_charge) / 3600
    discharge_ah = np.trapezoid(np.full_like(t_dis, discharge_current), t_dis) / 3600
    efficiency = (discharge_ah / charge_ah * 100) if charge_ah > 0 else 0

    return df, {
        "Charge Time": f"{t_charge_h * 3600:.1f} sec",
        "Discharge Time": f"{t_dis_h * 3600:.1f} sec",
        "Energy Delivered": f"{energy_wh:.4f} Wh",
        "Coulombic Efficiency": f"{efficiency:.1f}%",
    }


def bandgap_sim(material_key, thickness_mm, wl_min, wl_max, step_nm, noise_pct):
    p = SEMICONDUCTORS[material_key]
    eg = p["Eg"]
    wl = np.arange(wl_min, wl_max + step_nm, step_nm)
    hv = 1239.841984 / wl  # eV

    excess = np.maximum(hv - eg, 0)
    alpha = p["alpha0"] * np.sqrt(excess) / np.maximum(hv, 1e-9) + p["alpha0"] * 0.002
    absorbance = alpha * (thickness_mm / 10.0) / 2.302585
    
    rng = np.random.default_rng(42)
    if noise_pct > 0:
        absorbance *= 1 + rng.normal(0, noise_pct / 100, len(absorbance))
    absorbance = np.clip(absorbance, 1e-5, None)
    
    alpha_calc = 2.302585 * absorbance / (thickness_mm / 10.0)
    tauc = (alpha_calc * hv) ** 2

    mask = (hv >= eg * 0.92) & (hv <= eg * 1.15) & (alpha_calc > p["alpha0"] * 0.01)
    x_fit, y_fit = hv[mask], tauc[mask]
    
    if len(x_fit) >= 4:
        m, c = np.polyfit(x_fit, y_fit, 1)
        eg_fit = -c / m if m != 0 else eg
    else:
        m, c, eg_fit = 1.0, -eg, eg

    edge_nm = 1239.841984 / eg_fit

    df = pd.DataFrame({
        "Wavelength (nm)": wl,
        "Photon Energy hν (eV)": hv,
        "Absorbance (A)": absorbance,
        "Absorption Coeff α (cm⁻¹)": alpha_calc,
        "(αhν)² (eV²cm⁻²)": tauc,
    })

    return df, {
        "Theoretical Eg": f"{eg:.2f} eV",
        "Calculated Eg": f"{eg_fit:.2f} eV",
        "Absorption Edge": f"{edge_nm:.1f} nm",
        "Primary Application": p["use"],
    }, (m, c)


def photoelectric_sim(material_key, lambda_nm, intensity_percent, voltage_range):
    work_func_ev = CATHODE_MATERIALS[material_key]["work_function"]
    
    freq = C_LIGHT / (lambda_nm * 1e-9)            # Hz
    e_photon_ev = (H_PLANCK * freq) / E_CHARGE      # eV
    stopping_v = max(0.0, e_photon_ev - work_func_ev)
    
    volts = np.linspace(voltage_range[0], voltage_range[1], 150)
    
    # Saturation photocurrent is proportional to light intensity
    i_sat = intensity_percent * 0.1  # μA
    
    # Photocurrent vs Voltage curve model
    currents = []
    for v in volts:
        if v <= -stopping_v:
            currents.append(0.0)
        else:
            # Smooth transition to saturation current above stopping potential
            i_val = i_sat * (1 - np.exp(-1.8 * (v + stopping_v)))
            currents.append(max(0.0, float(i_val)))

    df_iv = pd.DataFrame({
        "Applied Voltage V (Volts)": volts,
        "Photocurrent I (μA)": currents
    })

    # Frequency vs Stopping Potential across visible-UV spectrum (250nm - 650nm)
    wl_test = np.linspace(250, 650, 20)
    freq_test = C_LIGHT / (wl_test * 1e-9)
    ev_test = (H_PLANCK * freq_test) / E_CHARGE
    vs_test = np.maximum(0.0, ev_test - work_func_ev)
    
    # Fit line: Vs = (h/e)*nu - (Phi/e)
    active_mask = vs_test > 0
    if np.sum(active_mask) >= 3:
        slope, intercept = np.polyfit(freq_test[active_mask], vs_test[active_mask], 1)
        h_estimated = slope * E_CHARGE
        phi_estimated = -intercept
    else:
        h_estimated = H_PLANCK
        phi_estimated = work_func_ev
        slope = H_PLANCK / E_CHARGE

    df_h = pd.DataFrame({
        "Frequency ν (10¹⁴ Hz)": freq_test / 1e14,
        "Stopping Potential Vs (V)": vs_test,
        "Wavelength λ (nm)": wl_test
    })

    return df_iv, df_h, {
        "Photon Energy": f"{e_photon_ev:.2f} eV",
        "Work Function (Φ)": f"{work_func_ev:.2f} eV",
        "Stopping Voltage (Vs)": f"{stopping_v:.2f} V",
        "Estimated Planck's h": f"{h_estimated:.3e} J·s",
        "True Planck's h": f"{H_PLANCK:.3e} J·s",
        "Percentage Error": f"{abs(h_estimated - H_PLANCK)/H_PLANCK * 100:.2f}%"
    }, stopping_v, slope


def spectrometer_sim(lamp_key, grating_lines_per_mm, order_m=1):
    d_grating_m = (1.0 / grating_lines_per_mm) * 1e-3  # grating pitch in meters
    lamp_lines = LAMPS_DATA[lamp_key]["lines"]
    
    results = []
    for line in lamp_lines:
        wl_m = line["lambda_nm"] * 1e-9
        sin_theta = (order_m * wl_m) / d_grating_m
        
        if sin_theta <= 1.0:
            theta_rad = np.arcsin(sin_theta)
            theta_deg = np.degrees(theta_rad)
        else:
            theta_deg = np.nan
            
        # For Balmer Rydberg calculation (if Hydrogen)
        ryd_val = np.nan
        if "H-alpha" in line["name"]:
            ryd_val = 1.0 / (wl_m * (1/4 - 1/9))
        elif "H-beta" in line["name"]:
            ryd_val = 1.0 / (wl_m * (1/4 - 1/16))
        elif "H-gamma" in line["name"]:
            ryd_val = 1.0 / (wl_m * (1/4 - 1/25))
        elif "H-delta" in line["name"]:
            ryd_val = 1.0 / (wl_m * (1/4 - 1/36))

        results.append({
            "Spectral Line": line["name"],
            "Wavelength λ (nm)": line["lambda_nm"],
            "Diffraction Angle θ (°)": theta_deg,
            "Line Color": line["color"],
            "Relative Intensity": line["intensity"],
            "Rydberg Constant (m⁻¹)": ryd_val
        })
        
    df = pd.DataFrame(results)
    
    calc_rydberg = df["Rydberg Constant (m⁻¹)"].dropna().mean() if "Hydrogen" in lamp_key else np.nan
    
    return df, {
        "Grating Pitch (d)": f"{d_grating_m*1e6:.2f} µm",
        "Lines / mm": f"{grating_lines_per_mm}",
        "Calculated Rydberg R_H": f"{calc_rydberg:.4e} m⁻¹" if not np.isnan(calc_rydberg) else "N/A (Hg Lamp)",
        "Theoretical R_H": f"{R_HYD:.4e} m⁻¹"
    }

# -----------------------------
# App Layout & Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🧪 Virtual Engineering Chemistry Laboratory</h1>
    <p>Interactive Simulations & Animated Models for First-Year BTech Students • Department of Chemistry</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🔬 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Lab Overview",
        "🔋 Exp 1: Battery & Supercap Testing",
        "💡 Exp 2: Semiconductor Band Gap",
        "⚡ Exp 3: Photoelectric Effect & Planck's Constant",
        "🌈 Exp 4: Emission Spectrum (H₂ / Hg Lamp)",
        "📖 Theory & Formulations",
        "❓ Viva Voce Quiz",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Student Tip:** 
Adjust parameters on the left controls, observe real-time animations, and export generated data tables directly for your lab reports.
""")

# -----------------------------
# Page 1: Overview
# -----------------------------
if page == "🏠 Lab Overview":
    st.subheader("Welcome to the Virtual Chemistry & Applied Physics Laboratory")
    st.write("This interactive platform designed for **1st Year BTech Students** brings core experiments to life with dynamic physics engines and visual animations.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔋 Experiment 1: Energy Storage
        - Compare **Lithium-ion Batteries** vs **Supercapacitors**.
        - Measure charging/discharging time curves.
        - Observe **Internal Resistance ($R_i$ / ESR)** Ohmic loss drops.
        
        ### 💡 Experiment 2: Semiconductor Optics
        - Determine bandgap ($E_g$) using **UV-Vis Spectrometry**.
        - Plot and analyze **Tauc Plots** $((\\alpha h\\nu)^2 \\text{ vs } h\\nu)$.
        - Find absorption edge cut-offs.
        """)
    with col2:
        st.markdown("""
        ### ⚡ Experiment 3: Photoelectric Effect
        - Verify Einstein's Photoelectric equation ($E = h\\nu - \\Phi$).
        - Estimate **Planck's constant ($h$)** from stopping potential $V_s$.
        - **Interactive Particle Animation:** Watch photoelectron velocity and flux in real-time.

        ### 🌈 Experiment 4: Atomic Emission Spectroscopy
        - Measure atomic spectrum lines of **Hydrogen** & **Mercury** discharge lamps.
        - Calculate **Rydberg Constant ($R_H$)** via diffraction grating.
        - **Virtual Spectrometer Eyepiece:** Telescope angle sweep animation.
        """)

    st.markdown("---")
    st.success("👈 Choose an experiment from the sidebar to start your virtual experiment!")

# -----------------------------
# Page 2: Experiment 1
# -----------------------------
elif page == "🔋 Exp 1: Battery & Supercap Testing":
    st.header("Experiment 1: Charge-Discharge Characteristics")
    
    st.markdown("""
    <div class="concept-card">
    <b>💡 What are you testing?</b> Energy storage devices deliver stored chemical or electrostatic energy. 
    Notice how voltage instantly jumps up during charge or drops down during discharge—this instantaneous change is due to internal resistance (IR drop / ESR).
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        st.subheader("🛠️ Control Panel")
        cell_type = st.selectbox("Device Type", ["Lithium-ion Battery", "Supercapacitor"])

        if cell_type == "Lithium-ion Battery":
            nominal_v = st.slider("Nominal Voltage (V)", 3.0, 4.2, 3.7, 0.1)
            capacity = st.number_input("Rated Capacity (Ah)", 0.5, 10.0, 2.2, 0.1)
            capacitance = 0.0
            r_internal = st.slider("Internal Resistance R_i (Ω)", 0.01, 0.50, 0.08, 0.01)
            cutoff_v = st.slider("Cut-off Voltage (V)", 2.5, 3.2, 2.8, 0.1)
        else:
            nominal_v = st.slider("Max Voltage (V)", 1.5, 5.0, 2.7, 0.1)
            capacitance = st.number_input("Capacitance (Farads)", 10.0, 5000.0, 500.0, 50.0)
            capacity = 0.0
            r_internal = st.slider("Equivalent Series Resistance ESR (Ω)", 0.005, 0.200, 0.030, 0.005)
            cutoff_v = st.slider("Cut-off Voltage (V)", 0.1, 1.5, 0.5, 0.1)

        charge_curr = st.number_input("Charge Current (A)", 0.1, 20.0, 1.5, 0.1)
        dis_curr = st.number_input("Discharge Current (A)", 0.1, 20.0, 2.0, 0.1)

    df, summary = battery_sim(cell_type, nominal_v, capacity, capacitance, r_internal, charge_curr, dis_curr, cutoff_v, 2.0)

    with col_chart:
        st.subheader("📊 Live Characteristic Curves")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Terminal Voltage vs Time", "Current vs Time"))

        chg_data = df[df["Phase"] == "Charging"]
        dis_data = df[df["Phase"] == "Discharging"]

        fig.add_trace(go.Scatter(x=chg_data["Time (s)"], y=chg_data["Voltage (V)"], 
                                 mode='lines', name='Charging Phase', line=dict(color='#2b6cb0', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dis_data["Time (s)"], y=dis_data["Voltage (V)"], 
                                 mode='lines', name='Discharging Phase', line=dict(color='#e53e3e', width=3)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df["Time (s)"], y=df["Current (A)"], 
                                 mode='lines', name='Current (A)', line=dict(color='#319795', width=2)), row=2, col=1)

        fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", template="plotly_white")
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_yaxes(title_text="Current (A)", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Experimental Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Charging Duration", summary["Charge Time"])
    m2.metric("Discharging Duration", summary["Discharge Time"])
    m3.metric("Energy Delivered", summary["Energy Delivered"])
    m4.metric("Coulombic Efficiency", summary["Coulombic Efficiency"])

    with st.expander("📄 View Generated Data Table"):
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Experiment Data (CSV)", df.to_csv(index=False), "battery_discharge_data.csv")

# -----------------------------
# Page 3: Experiment 2
# -----------------------------
elif page == "💡 Exp 2: Semiconductor Band Gap":
    st.header("Experiment 2: Band Gap Energy Determination (UV-Vis)")
    
    st.markdown("""
    <div class="concept-card">
    <b>💡 What are you testing?</b> When light shines on a semiconductor, photons with energy greater than the bandgap (<i>hν</i> ≥ <i>E<sub>g</sub></i>) get absorbed, kicking electrons from the valence band to the conduction band. 
    By plotting a <b>Tauc Plot</b> ((α<i>hν</i>)<sup>2</sup> vs <i>hν</i>), we can linearly extrapolate the absorption edge to find the exact energy gap <i>E<sub>g</sub></i>.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        st.subheader("🛠️ Control Panel")
        selected_mat = st.selectbox("Select Semiconductor", list(SEMICONDUCTORS.keys()))
        thickness = st.slider("Sample Thickness (mm)", 0.1, 5.0, 1.0, 0.1)
        noise_level = st.slider("Simulated Sensor Noise (%)", 0.0, 10.0, 2.0, 0.5)

        st.markdown("---")
        st.markdown("**Fitting Mode:**")
        fit_mode = st.radio("Extrapolation Method", ["Auto Fit", "Manual Tangent Slider"])
        
    df, summary, (auto_m, auto_c) = bandgap_sim(selected_mat, thickness, 250, 900, 2, noise_level)

    if fit_mode == "Manual Tangent Slider":
        st.sidebar.markdown("### Manual Tangent Adjuster")
        manual_eg = st.sidebar.slider("Set Tangent Intercept Eg (eV)", 0.5, 4.5, float(summary["Theoretical Eg"].split()[0]), 0.02)
        manual_slope = st.sidebar.slider("Set Tangent Slope", 0.1, 10.0, 2.5, 0.1) * 1e8
        m, c = manual_slope, -manual_slope * manual_eg
        calc_eg = manual_eg
    else:
        m, c = auto_m, auto_c
        calc_eg = float(summary["Calculated Eg"].split()[0])

    with col_chart:
        tab_spec, tab_tauc = st.tabs(["1. UV-Vis Absorbance Spectrum", "2. Tauc Plot ((αhν)² vs hν)"])

        with tab_spec:
            fig_abs = go.Figure()
            fig_abs.add_trace(go.Scatter(x=df["Wavelength (nm)"], y=df["Absorbance (A)"],
                                         mode='lines', name='Absorbance', line=dict(color='#805ad5', width=3)))
            fig_abs.update_layout(title="Absorbance (A) vs Wavelength (nm)", xaxis_title="Wavelength λ (nm)",
                                  yaxis_title="Absorbance A", template="plotly_white", height=400)
            st.plotly_chart(fig_abs, use_container_width=True)

        with tab_tauc:
            fig_tauc = go.Figure()
            fig_tauc.add_trace(go.Scatter(x=df["Photon Energy hν (eV)"], y=df["(αhν)² (eV²cm⁻²)"],
                                          mode='markers', name='Data Points', marker=dict(color='#3182ce', size=5)))

            x_line = np.linspace(calc_eg - 0.3, calc_eg + 0.8, 50)
            y_line = m * x_line + c
            fig_tauc.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Linear Extrapolation',
                                          line=dict(color='#e53e3e', width=2, dash='dash')))

            fig_tauc.add_vline(x=calc_eg, line_width=2, line_dash="dot", line_color="green",
                               annotation_text=f"Eg = {calc_eg:.2f} eV", annotation_position="top left")

            fig_tauc.update_layout(title="Tauc Plot: (αhν)² vs Photon Energy hν", xaxis_title="Photon Energy hν (eV)",
                                   yaxis_title="(αhν)² [eV² cm⁻²]", template="plotly_white", height=400,
                                   yaxis_range=[0, df["(αhν)² (eV²cm⁻²)"].max() * 1.05])
            st.plotly_chart(fig_tauc, use_container_width=True)

    st.subheader("📋 Derived Band Gap Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Theoretical Eg", summary["Theoretical Eg"])
    m2.metric("Extrapolated Eg", f"{calc_eg:.2f} eV")
    m3.metric("Absorption Cut-off Edge", summary["Absorption Edge"])
    m4.metric("Real-World Use", summary["Primary Application"])

    with st.expander("📄 View Derived Optical Parameters Table"):
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Optical Data (CSV)", df.to_csv(index=False), "semiconductor_bandgap_data.csv")

# -----------------------------
# Page 4: Experiment 3 (Photoelectric)
# -----------------------------
elif page == "⚡ Exp 3: Photoelectric Effect & Planck's Constant":
    st.header("Experiment 3: Photoelectric Effect & Planck's Constant")

    st.markdown("""
    <div class="concept-card">
    <b>💡 What are you testing?</b> When light shines on a cathode metal plate, photons transfer energy to bound electrons. 
    If photon energy <i>hν</i> exceeds the metal's work function <i>Φ</i>, photoelectrons fly to the anode with kinetic energy <i>K<sub>max</sub> = e V<sub>s</sub></i>.
    By plotting Stopping Potential <i>V<sub>s</sub></i> vs Light Frequency <i>ν</i>, the slope gives <b>Planck's Constant (<i>h/e</i>)</b>!
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        st.subheader("🛠️ Control Panel")
        selected_metal = st.selectbox("Cathode Material", list(CATHODE_MATERIALS.keys()))
        lambda_nm = st.slider("Monochromatic Light Wavelength λ (nm)", 200, 700, 380, 5)
        intensity = st.slider("Light Intensity (%)", 10, 100, 80, 5)
        
        v_min, v_max = st.slider("Applied Voltage Range V (Volts)", -3.0, 3.0, (-2.5, 2.0), 0.1)

    df_iv, df_h, summary, stopping_v, slope = photoelectric_sim(selected_metal, lambda_nm, intensity, (v_min, v_max))
    light_color = nm_to_rgb_color(lambda_nm)

    with col_chart:
        tab_anim, tab_iv, tab_planck = st.tabs(["🎬 Animated Photocell", "📊 Photocurrent (I vs V)", "📈 Planck's Constant (Vs vs ν)"])

        with tab_anim:
            st.markdown(f"**Live Visual Photocell Simulation** (Light: **{lambda_nm} nm**, Cathode: **{selected_metal}**)")
            
            # Interactive Animation of Photoelectrons moving Cathode -> Anode
            fig_anim = go.Figure()

            # Plates
            fig_anim.add_shape(type="rect", x0=0.5, y0=0, x1=1.0, y1=10, fillcolor="#94a3b8", line=dict(color="#475569", width=2))
            fig_anim.add_shape(type="rect", x0=9.0, y0=0, x1=9.5, y1=10, fillcolor="#cbd5e1", line=dict(color="#475569", width=2))

            # Light Beam
            fig_anim.add_shape(type="path", path=f"M -2 12 L 0.75 5 L -2 -2 Z", fillcolor=light_color, opacity=intensity/200.0, line=dict(width=0))

            # Photoelectrons particles animation logic
            work_func = CATHODE_MATERIALS[selected_metal]["work_function"]
            photon_ev = 1239.84 / lambda_nm

            if photon_ev >= work_func:
                num_e = int((intensity / 10) * 8)
                np.random.seed(42)
                e_x = np.random.uniform(1.1, 8.8, num_e)
                e_y = np.random.uniform(1.0, 9.0, num_e)
                
                fig_anim.add_trace(go.Scatter(
                    x=e_x, y=e_y, mode='markers',
                    marker=dict(size=10, color='#e11d48', symbol='circle'),
                    name='Photoelectrons (e⁻)'
                ))
                annot_text = f"Emission Active! K_max = {photon_ev - work_func:.2f} eV"
                annot_color = "#15803d"
            else:
                annot_text = f"No Emission (Photon Energy {photon_ev:.2f} eV < Work Function {work_func:.2f} eV)"
                annot_color = "#b91c1c"

            fig_anim.add_annotation(x=5, y=11, text=annot_text, showarrow=False, font=dict(size=14, color=annot_color, family="sans-serif"))
            fig_anim.add_annotation(x=0.75, y=-1, text="Cathode (-)", showarrow=False, font=dict(size=12, color="black"))
            fig_anim.add_annotation(x=9.25, y=-1, text="Anode (+)", showarrow=False, font=dict(size=12, color="black"))

            fig_anim.update_layout(
                xaxis=dict(range=[-3, 11], visible=False),
                yaxis=dict(range=[-2, 12.5], visible=False),
                height=350, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white"
            )
            st.plotly_chart(fig_anim, use_container_width=True)

        with tab_iv:
            fig_iv = go.Figure()
            fig_iv.add_trace(go.Scatter(x=df_iv["Applied Voltage V (Volts)"], y=df_iv["Photocurrent I (μA)"],
                                        mode='lines', name='I-V Curve', line=dict(color='#0284c7', width=3)))
            fig_iv.add_vline(x=-stopping_v, line_dash="dash", line_color="#ef4444",
                             annotation_text=f"Vs = {stopping_v:.2f} V", annotation_position="top left")
            fig_iv.update_layout(title="Photocurrent (I) vs Retarding/Accelerating Voltage (V)",
                                  xaxis_title="Voltage V (Volts)", yaxis_title="Photocurrent I (μA)",
                                  template="plotly_white", height=380)
            st.plotly_chart(fig_iv, use_container_width=True)

        with tab_planck:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=df_h["Frequency ν (10¹⁴ Hz)"], y=df_h["Stopping Potential Vs (V)"],
                                       mode='markers', name='Data Points', marker=dict(size=8, color='#7c3aed')))
            
            # Linear Fit Line
            x_fit = df_h["Frequency ν (10¹⁴ Hz)"]
            y_fit = slope * (x_fit * 1e14) - (CATHODE_MATERIALS[selected_metal]["work_function"])
            fig_h.add_trace(go.Scatter(x=x_fit, y=np.maximum(0, y_fit), mode='lines', name='Linear Fit (Slope = h/e)',
                                       line=dict(color='#22c55e', width=2, dash='dash')))

            fig_h.update_layout(title="Stopping Potential Vs vs Light Frequency ν",
                                 xaxis_title="Frequency ν (× 10¹⁴ Hz)", yaxis_title="Stopping Voltage Vs (Volts)",
                                 template="plotly_white", height=380)
            st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("📋 Derived Experimental Values")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Photon Energy", summary["Photon Energy"])
    m2.metric("Stopping Potential", summary["Stopping Voltage (Vs)"])
    m3.metric("Estimated Planck's h", summary["Estimated Planck's h"])
    m4.metric("Fit Accuracy Error", summary["Percentage Error"])

    with st.expander("📄 View Photoelectric Data Table"):
        st.dataframe(df_iv, use_container_width=True)
        st.download_button("📥 Download I-V Data (CSV)", df_iv.to_csv(index=False), "photoelectric_data.csv")

# -----------------------------
# Page 5: Experiment 4 (Spectrometer)
# -----------------------------
elif page == "🌈 Exp 4: Emission Spectrum (H₂ / Hg Lamp)":
    st.header("Experiment 4: Emission Spectrum & Rydberg Constant")

    st.markdown("""
    <div class="concept-card">
    <b>💡 What are you testing?</b> Discrete spectral lines correspond to electron transitions between atomic energy levels. 
    By passing light through a diffraction grating ($d \\sin\\theta = m\\lambda$), we measure the angular position $\\theta$ of each spectral line to determine wavelengths and calculate the <b>Rydberg Constant ($R_H$)</b>.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        st.subheader("🛠️ Control Panel")
        lamp_choice = st.selectbox("Gas Discharge Lamp", list(LAMPS_DATA.keys()))
        grating_density = st.select_slider("Grating Lines / mm", options=[300, 500, 600, 1000], value=600)
        
        st.markdown("---")
        st.markdown("**Virtual Spectrometer Telescope Sweep:**")
        telescope_angle = st.slider("Telescope Angle θ (°)", 0.0, 50.0, 16.0, 0.1)

    df_spec, summary = spectrometer_sim(lamp_choice, grating_density)

    with col_chart:
        tab_eyepiece, tab_bars = st.tabs(["🔭 Virtual Eyepiece View", "🌈 Full Diffraction Spectrum"])

        with tab_eyepiece:
            st.markdown(f"**Spectrometer Crosshair View** (Telescope set to **{telescope_angle:.1f}°**)")
            
            # Interactive Visual Eyepiece Circle
            fig_eye = go.Figure()
            
            # Dark background field of view
            fig_eye.add_shape(type="circle", x0=-10, y0=-10, x1=10, y1=10, fillcolor="#0f172a", line=dict(color="#334155", width=4))
            
            # Reticle Crosshairs
            fig_eye.add_line(x0=-10, y0=0, x1=10, y1=0, line=dict(color="#475569", width=1, dash="dot"))
            fig_eye.add_line(x0=0, y0=-10, x1=0, y1=10, line=dict(color="#ef4444", width=1.5)) # Vertical red crosshair line

            # Check if any line is near telescope angle
            visible_line_found = False
            for _, row in df_spec.iterrows():
                line_theta = row["Diffraction Angle θ (°)"]
                if not np.isnan(line_theta):
                    delta_angle = line_theta - telescope_angle
                    if abs(delta_angle) <= 3.0: # Visible in field of view
                        x_pos = delta_angle * 3.0 # scale to crosshair view
                        fig_eye.add_line(x0=x_pos, y0=-8, x1=x_pos, y1=8, line=dict(color=row["Line Color"], width=4))
                        fig_eye.add_annotation(x=x_pos, y=8.5, text=f"{row['Spectral Line']}<br>{row['Wavelength λ (nm)']} nm",
                                               showarrow=False, font=dict(size=10, color="white"))
                        visible_line_found = True

            if not visible_line_found:
                fig_eye.add_annotation(x=0, y=-8.5, text="Rotate telescope angle slider to align spectral lines with crosshair",
                                       showarrow=False, font=dict(size=10, color="#94a3b8"))

            fig_eye.update_layout(
                xaxis=dict(range=[-11, 11], visible=False),
                yaxis=dict(range=[-11, 11], visible=False),
                height=350, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white"
            )
            st.plotly_chart(fig_eye, use_container_width=True)

        with tab_bars:
            fig_bars = go.Figure()
            for _, row in df_spec.iterrows():
                if not np.isnan(row["Diffraction Angle θ (°)"]):
                    fig_bars.add_trace(go.Bar(
                        x=[row["Wavelength λ (nm)"]], y=[row["Relative Intensity"]],
                        width=3.0, marker_color=row["Line Color"], name=row["Spectral Line"],
                        hovertemplate=f"<b>{row['Spectral Line']}</b><br>Wavelength: {row['Wavelength λ (nm)']} nm<br>Angle θ: {row['Diffraction Angle θ (°)']:.2f}°"
                    ))

            fig_bars.update_layout(title="Emission Spectrum Wavelength Distribution",
                                   xaxis_title="Wavelength λ (nm)", yaxis_title="Relative Line Intensity",
                                   template="plotly_dark", height=380, xaxis_range=[380, 700])
            st.plotly_chart(fig_bars, use_container_width=True)

    st.subheader("📋 Spectral Line Calculations")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grating Pitch (d)", summary["Grating Pitch (d)"])
    m2.metric("Grating Specification", f"{grating_density} lines/mm")
    m3.metric("Calculated Rydberg R_H", summary["Calculated Rydberg R_H"])
    m4.metric("Theoretical R_H", summary["Theoretical R_H"])

    with st.expander("📄 View Spectral Line Measurements Table"):
        st.dataframe(df_spec, use_container_width=True)
        st.download_button("📥 Download Spectrum Data (CSV)", df_spec.to_csv(index=False), "spectrometer_data.csv")

# -----------------------------
# Page 6: Theory
# -----------------------------
elif page == "📖 Theory & Formulations":
    st.header("Fundamental Equations & Concepts")

    st.markdown(r"""
    ### 1. Energy Storage (Li-ion & Supercapacitors)
    * **Terminal Voltage:** $$V_{\text{terminal}} = V_{\text{ocv}} \pm I \cdot R_i$$
    * **Supercapacitor Discharge:** $$V(t) = V_0 - \frac{I}{C}t$$
    * **Coulombic Efficiency:** $$\eta = \frac{Q_{\text{discharge}}}{Q_{\text{charge}}} \times 100\%$$

    ---

    ### 2. Semiconductor Optical Bandgap (UV-Vis)
    * **Photon Energy:** $$h\nu = \frac{hc}{\lambda} \approx \frac{1239.84}{\lambda \text{ (in nm)}} \text{ eV}$$
    * **Direct Allowed Tauc Relation:** $$(\alpha h\nu)^2 = B(h\nu - E_g)$$

    ---

    ### 3. Einstein's Photoelectric Effect
    * **Photoelectric Equation:** $$K_{\max} = e V_s = h\nu - \Phi$$
    * **Linear Fit for Planck's Constant:** $$V_s = \left(\frac{h}{e}\right)\nu - \frac{\Phi}{e}$$

    ---

    ### 4. Atomic Emission Spectroscopy & Grating Equation
    * **Diffraction Grating Formula:** $$d \sin\theta = m \lambda \quad \implies \quad \theta = \arcsin\left(\frac{m\lambda}{d}\right)$$
    * **Balmer Series Rydberg Formula (Hydrogen):** $$\frac{1}{\lambda} = R_H \left( \frac{1}{2^2} - \frac{1}{n^2} \right) \quad \text{for } n = 3, 4, 5, 6$$
    """)

# -----------------------------
# Page 7: Viva Quiz
# -----------------------------
elif page == "❓ Viva Voce Quiz":
    st.header("🧪 Self-Assessment & Viva Practice")
    st.write("Test your understanding across all four virtual experiments!")

    q1 = st.radio("1. What does the slope of the Stopping Potential ($V_s$) vs Frequency ($\nu$) plot represent?",
                  ["Work function (Φ)", "Planck's constant divided by electron charge (h/e)", "Threshold frequency", "Speed of light"])
    if q1 == "Planck's constant divided by electron charge (h/e)":
        st.success("Correct! The slope equals $h/e$.")

    q2 = st.radio("2. In the Balmer series of Hydrogen emission, which spectral transition gives the red H-alpha line?",
                  ["n = 3 → n = 2", "n = 4 → n = 2", "n = 2 → n = 1", "n = 5 → n = 2"])
    if q2 == "n = 3 → n = 2":
        st.success("Correct! $n=3 \rightarrow n=2$ emits $H_\\alpha$ at $\\approx 656.3\\text{ nm}$.")

    q3 = st.radio("3. Why does terminal voltage jump or drop immediately when switching between charging and discharging a battery?",
                  ["Chemical degradation", "Ohmic drop across internal resistance (I × R_i)", "Capacitance loss", "Temperature shift"])
    if q3 == "Ohmic drop across internal resistance (I × R_i)":
        st.success("Correct! $IR_i$ drop causes the instantaneous step change.")

st.markdown("---")
st.caption("Virtual Chemistry & Physics Laboratory • Developed for First-Year BTech Engineering Courses")
