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
/* Modern Glassmorphic Header */
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
# Helper Simulation Functions
# -----------------------------
SEMICONDUCTORS = {
    "ZnO (Zinc Oxide)": {"Eg": 3.28, "alpha0": 1.0e4, "use": "UV absorbers, sunscreens, transparent electronics"},
    "GaAs (Gallium Arsenide)": {"Eg": 1.42, "alpha0": 1.5e4, "use": "High-efficiency solar cells, optoelectronics"},
    "Si (Silicon)": {"Eg": 1.12, "alpha0": 1.0e4, "use": "Microchips, commercial photovoltaic solar cells"},
    "CdS (Cadmium Sulfide)": {"Eg": 2.42, "alpha0": 1.2e4, "use": "Quantum dots, photoresistors"},
    "TiO₂ (Titanium Dioxide)": {"Eg": 3.20, "alpha0": 1.0e4, "use": "Photocatalysis, self-cleaning coatings"},
}

def battery_sim(cell_type, nominal_v, capacity_ah, capacitance_f,
                r_internal, charge_current, discharge_current,
                cutoff_v, time_step):
    """Simulates charging and discharging curves with internal resistance effects."""
    if cell_type == "Lithium-ion Battery":
        def ocv(soc):
            soc = np.clip(soc, 0, 1)
            return 3.0 + 0.75 * soc + 0.10 * np.log10(np.maximum(soc, 1e-5) / np.maximum(1-soc, 1e-5)) * 0.05

        soc0 = 0.15
        charge_seconds = max((1.0 - soc0) * capacity_ah / max(charge_current, 1e-9) * 3600, 1)
        n_charge = int(np.ceil(charge_seconds / time_step)) + 1
        t_charge = np.arange(n_charge) * time_step
        soc_charge = np.minimum(soc0 + charge_current * t_charge / 3600 / capacity_ah, 1.0)
        v_charge = ocv(soc_charge) + r_internal * charge_current
        v_charge = np.clip(v_charge, 2.8, 4.25)

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

    # Supercapacitor Simulation
    v_min = max(cutoff_v, 0.05)
    v_max = nominal_v
    q_max = capacitance_f * v_max
    charge_seconds = q_max / max(charge_current, 1e-9)
    n_charge = int(np.ceil(charge_seconds / time_step)) + 1
    t_charge = np.arange(n_charge) * time_step
    v_charge = np.minimum(v_min + charge_current * t_charge / capacitance_f, v_max) + r_internal * charge_current
    v_charge = np.clip(v_charge, 0, v_max * 1.05)

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
    hv = 1239.841984 / wl  # photon energy in eV

    excess = np.maximum(hv - eg, 0)
    alpha = p["alpha0"] * np.sqrt(excess) / np.maximum(hv, 1e-9) + p["alpha0"] * 0.002
    absorbance = alpha * (thickness_mm / 10.0) / 2.302585
    
    rng = np.random.default_rng(42)
    if noise_pct > 0:
        absorbance *= 1 + rng.normal(0, noise_pct / 100, len(absorbance))
    absorbance = np.clip(absorbance, 1e-5, None)
    
    alpha_calc = 2.302585 * absorbance / (thickness_mm / 10.0)
    tauc = (alpha_calc * hv) ** 2

    # Automatic linear fitting on steep region
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


# -----------------------------
# App Layout & Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🧪 Virtual Engineering Chemistry Laboratory</h1>
    <p>Interactive Simulations for First-Year BTech Students • Department of Chemistry</p>
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
        "📖 Theory & Formulations",
        "❓ Viva Voce Quiz",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Student Tip:** 
Adjust parameters on the left controls, observe changes in real-time on interactive charts, and export data for your lab record reports.
""")

# -----------------------------
# Page 1: Overview
# -----------------------------
if page == "🏠 Lab Overview":
    st.subheader("Welcome to the Virtual Chemistry Laboratory")
    st.write("This interactive platform designed for **1st Year BTech Chemistry** helps you perform, visualize, and analyze core experiments virtually.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔋 Experiment 1
        **Energy Storage Systems**
        - Compare **Lithium-ion Batteries** vs **Supercapacitors**.
        - Measure charging/discharging time curves.
        - Observe **Internal Resistance ($R_i$ / ESR)** and Ohmic loss drops.
        - Calculate **Coulombic Efficiency** & Energy Density.
        """)
    with col2:
        st.markdown("""
        ### 💡 Experiment 2
        **Semiconductor Photophysics**
        - Determine optical bandgap ($E_g$) using **UV-Vis Absorption Spectrometry**.
        - Plot and analyze **Tauc Plots** $((\alpha h\nu)^2 \text{ vs } h\nu)$.
        - Convert optical wavelengths to photon energy ($eV$).
        - Extrapolate linear regions to find absorption cut-offs.
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
        
        # Interactive Plotly Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Terminal Voltage vs Time", "Current vs Time"))

        # Voltage curve
        chg_data = df[df["Phase"] == "Charging"]
        dis_data = df[df["Phase"] == "Discharging"]

        fig.add_trace(go.Scatter(x=chg_data["Time (s)"], y=chg_data["Voltage (V)"], 
                                 mode='lines', name='Charging Phase', line=dict(color='#2b6cb0', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dis_data["Time (s)"], y=dis_data["Voltage (V)"], 
                                 mode='lines', name='Discharging Phase', line=dict(color='#e53e3e', width=3)), row=1, col=1)

        # Current curve
        fig.add_trace(go.Scatter(x=df["Time (s)"], y=df["Current (A)"], 
                                 mode='lines', name='Current (A)', line=dict(color='#319795', width=2)), row=2, col=1)

        fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", template="plotly_white")
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_yaxes(title_text="Current (A)", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # Metric Cards
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
    <b>💡 What are you testing?</b> When light shines on a semiconductor, photons with energy greater than the bandgap ($h\\nu \\ge E_g$) get absorbed, kicking electrons from the valence band to the conduction band. 
    By plotting a <b>Tauc Plot</b> $((\\alpha h\\nu)^2 \\text{ vs } h\\nu)$, we can linearly extrapolate the absorption edge to find the exact energy gap $E_g$.
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
            # Raw scatter data
            fig_tauc.add_trace(go.Scatter(x=df["Photon Energy hν (eV)"], y=df["(αhν)² (eV²cm⁻²)"],
                                          mode='markers', name='Data Points', marker=dict(color='#3182ce', size=5)))

            # Extrapolation line
            x_line = np.linspace(calc_eg - 0.3, calc_eg + 0.8, 50)
            y_line = m * x_line + c
            fig_tauc.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Linear Extrapolation',
                                          line=dict(color='#e53e3e', width=2, dash='dash')))

            # Vertical Band gap marker
            fig_tauc.add_vline(x=calc_eg, line_width=2, line_dash="dot", line_color="green",
                               annotation_text=f"Eg = {calc_eg:.2f} eV", annotation_position="top left")

            fig_tauc.update_layout(title="Tauc Plot: (αhν)² vs Photon Energy hν", xaxis_title="Photon Energy hν (eV)",
                                   yaxis_title="(αhν)² [eV² cm⁻²]", template="plotly_white", height=400,
                                   yaxis_range=[0, df["(αhν)² (eV²cm⁻²)"].max() * 1.05])
            st.plotly_chart(fig_tauc, use_container_width=True)

    # Result Summary
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
# Page 4: Theory
# -----------------------------
elif page == "📖 Theory & Formulations":
    st.header("Fundamental Equations & Concepts")

    st.markdown("""
    ### 1. Lithium-Ion & Supercapacitor Characterization
    * **Ohmic Resistance Drop (IR Drop):**
      $$V_{\text{terminal}} = V_{\text{ocv}} \\pm I \\cdot R_i$$
    * **Supercapacitor Capacitance Formula:**
      $$Q = C \\cdot V \\implies V(t) = V_0 - \\frac{I}{C}t$$
    * **Coulombic Efficiency ($\eta$):**
      $$\eta = \\frac{Q_{\text{discharge}}}{Q_{\text{charge}}} \\times 100\\% = \\frac{\int I_{\text{dis}} dt}{\int I_{\text{chg}} dt} \\times 100\\%$$

    ---

    ### 2. Optical Absorption & Semiconductor Tauc Plot
    * **Photon Energy ($h\\nu$):**
      $$h\\nu = \\frac{h c}{\\lambda} \\approx \\frac{1239.84}{\\lambda \\text{ (in nm)}} \\text{ eV}$$
    * **Beer-Lambert Law Absorption Coefficient ($\alpha$):**
      $$\\alpha = 2.303 \\times \\frac{A}{d} \\quad \\text{(where } A = \\text{absorbance, } d = \\text{sample thickness in cm)}$$
    * **Direct Allowed Tauc Relation:**
      $$(\\alpha h\\nu)^2 = B(h\\nu - E_g)$$
      *Plotting $(\\alpha h\\nu)^2$ vs $h\\nu$ and extending the straight line to the x-axis ($y=0$) gives the **Bandgap Energy ($E_g$)**.*
    """)

# -----------------------------
# Page 5: Viva Quiz
# -----------------------------
elif page == "❓ Viva Voce Quiz":
    st.header("🧪 Self-Assessment & Viva Practice")
    st.write("Test your understanding for your upcoming practical viva examinations!")

    q1 = st.radio("1. What unit is used to express the bandgap energy ($E_g$) of a semiconductor?",
                  ["Joules (J)", "Electron-Volts (eV)", "Nanometers (nm)", "Farads (F)"])
    if q1 == "Electron-Volts (eV)":
        st.success("Correct! Band gaps are conventionally measured in electron-volts (eV).")

    q2 = st.radio("2. In a Tauc plot for a direct band gap material, what is plotted on the Y-axis?",
                  ["Absorbance (A)", "Absorption Coefficient (α)", "(αhν)²", "Photon Energy (hν)"])
    if q2 == "(αhν)²":
        st.success("Correct! $(\\alpha h\\nu)^2$ is plotted on the Y-axis against photon energy $h\\nu$ on the X-axis.")

    q3 = st.radio("3. Why does terminal voltage suddenly drop the moment a battery starts discharging?",
                  ["The battery runs out of charges", "Ohmic voltage drop across internal resistance (I × R_i)", "The temperature drops", "Capacitance decreases"])
    if q3 == "Ohmic voltage drop across internal resistance (I × R_i)":
        st.success("Correct! This is caused by internal resistance $R_i$.")

st.markdown("---")
st.caption("Virtual Chemistry Lab • Developed for BTech Engineering Chemistry Courses")
