import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page Setup
st.set_page_config(
    page_title="Virtual Chemistry & Physics Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helper Functions
# -----------------------------
def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def battery_sim(cell_type, nominal_v, capacity_ah, capacitance_f,
                r_internal, charge_current, discharge_current,
                cutoff_v, time_step, temperature):
    """Simulates battery charging/discharging using an equivalent circuit model."""
    if cell_type == "Lithium-ion Battery":
        def ocv(soc):
            soc = np.clip(soc, 0, 1)
            return 3.0 + 0.75 * soc + 0.10 * np.log10(np.maximum(soc, 1e-5) / np.maximum(1-soc, 1e-5)) * 0.05

        soc0 = 0.20
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

        energy_wh = np.trapezoid(v_dis * i_dis, t_dis) / 3600 if len(t_dis) > 1 else 0
        charge_ah = np.trapezoid(np.full_like(t_charge, charge_current), t_charge) / 3600
        discharge_ah = np.trapezoid(i_dis, t_dis) / 3600 if len(t_dis) > 1 else 0
        efficiency = (discharge_ah / charge_ah * 100) if charge_ah > 0 else 0

        df = pd.concat([
            pd.DataFrame({"Time (s)": t_charge, "Voltage (V)": v_charge, "Current (A)": np.full_like(t_charge, charge_current), "Capacity (Ah)": charge_current * t_charge / 3600, "SOC": soc_charge * 100, "Phase": "Charge"}),
            pd.DataFrame({"Time (s)": t_dis + t_charge[-1], "Voltage (V)": v_dis, "Current (A)": -i_dis, "Capacity (Ah)": cap_dis, "SOC": (1 - cap_dis / capacity_ah) * 100, "Phase": "Discharge"})
        ], ignore_index=True)

        return df, {"Charge Time (h)": t_charge[-1] / 3600, "Discharge Time (h)": t_dis[-1] / 3600, "Energy Discharged (Wh)": energy_wh, "Coulombic Efficiency (%)": efficiency}

    # Supercapacitor Model
    v_min, v_max = max(cutoff_v, 0.05), nominal_v
    q_max = capacitance_f * v_max
    n_charge = int(np.ceil((q_max / max(charge_current, 1e-9)) / time_step)) + 1
    t_charge = np.arange(n_charge) * time_step
    v_charge = np.clip(v_min + charge_current * t_charge / capacitance_f + r_internal * charge_current, 0, v_max * 1.02)

    t_max = max(capacitance_f * max(v_max - v_min, 0) / max(discharge_current, 1e-9), time_step)
    t_dis = np.arange(int(np.ceil(t_max / time_step)) + 1) * time_step
    v_dis = v_max - discharge_current * t_dis / capacitance_f - r_internal * discharge_current
    keep = v_dis > v_min
    t_dis, v_dis = t_dis[keep], v_dis[keep]
    if len(t_dis) == 0:
        t_dis, v_dis = np.array([0.0]), np.array([v_min])

    df = pd.concat([
        pd.DataFrame({"Time (s)": t_charge, "Voltage (V)": v_charge, "Current (A)": np.full_like(t_charge, charge_current), "Capacity (Ah)": charge_current * t_charge / 3600, "SOC": np.clip((v_charge - v_min) / (v_max - v_min) * 100, 0, 100), "Phase": "Charge"}),
        pd.DataFrame({"Time (s)": t_dis + t_charge[-1], "Voltage (V)": v_dis, "Current (A)": -np.full_like(t_dis, discharge_current), "Capacity (Ah)": discharge_current * t_dis / 3600, "SOC": np.clip((v_dis - v_min) / (v_max - v_min) * 100, 0, 100), "Phase": "Discharge"})
    ], ignore_index=True)

    energy_wh = np.trapezoid(v_dis * discharge_current, t_dis) / 3600 if len(t_dis) > 1 else 0
    charge_ah = np.trapezoid(np.full_like(t_charge, charge_current), t_charge) / 3600
    discharge_ah = np.trapezoid(np.full_like(t_dis, discharge_current), t_dis) / 3600

    return df, {"Charge Time (h)": t_charge[-1] / 3600, "Discharge Time (h)": t_dis[-1] / 3600, "Energy Discharged (Wh)": energy_wh, "Coulombic Efficiency (%)": (discharge_ah / charge_ah * 100) if charge_ah > 0 else 0}

SEMICONDUCTORS = {
    "ZnO": {"Eg": 3.28, "alpha0": 1.0e4},
    "GaAs": {"Eg": 1.42, "alpha0": 1.5e4},
    "Si": {"Eg": 1.12, "alpha0": 1.0e4},
    "CdS": {"Eg": 2.42, "alpha0": 1.2e4},
    "TiO₂": {"Eg": 3.20, "alpha0": 1.0e4},
}

def bandgap_sim(material, thickness_mm, wl_min, wl_max, step_nm, noise_pct):
    p = SEMICONDUCTORS[material]
    eg = p["Eg"]
    wl = np.arange(wl_min, wl_max + step_nm, step_nm)
    hv = 1239.841984 / wl
    alpha = p["alpha0"] * np.sqrt(np.maximum(hv - eg, 0)) / np.maximum(hv, 1e-9) + p["alpha0"] * 0.002
    absorbance = alpha * (thickness_mm / 10.0) / 2.302585
    
    if noise_pct > 0:
        absorbance *= 1 + np.random.default_rng(42).normal(0, noise_pct / 100, len(absorbance))
    
    absorbance = np.clip(absorbance, 1e-5, None)
    alpha_calc = 2.302585 * absorbance / (thickness_mm / 10.0)
    tauc = (alpha_calc * hv) ** 2

    mask = (hv >= eg * 0.9) & (hv <= eg * 1.1) & (alpha_calc > p["alpha0"] * 0.01)
    x, y = hv[mask], tauc[mask]
    
    candidates = []
    if len(x) >= 5:
        for f1 in np.linspace(0.00, 0.35, 8):
            for f2 in np.linspace(0.65, 1.00, 8):
                a, b = int(f1 * len(x)), max(int(f1 * len(x)) + 5, int(f2 * len(x)))
                xx, yy = x[a:b], y[a:b]
                if len(xx) >= 5:
                    m_fit, c_fit = np.polyfit(xx, yy, 1)
                    r2 = 1 - np.sum((yy - (m_fit * xx + c_fit)) ** 2) / np.sum((yy - np.mean(yy)) ** 2) if np.sum((yy - np.mean(yy)) ** 2) > 0 else 0
                    intercept = -c_fit / m_fit if m_fit != 0 else np.nan
                    if 0.5 * eg < intercept < 1.5 * eg and m_fit > 0:
                        candidates.append((r2, intercept, xx.min(), xx.max(), m_fit, c_fit))

    if candidates:
        r2, eg_fit, fit_lo, fit_hi, m, c = max(candidates, key=lambda z: z[0])
    else:
        eg_fit, r2, fit_lo, fit_hi, m, c = eg, 0.0, eg * 0.9, eg * 1.1, 1, -eg

    df = pd.DataFrame({"Wavelength (nm)": wl, "Absorbance (A)": absorbance, "Absorption coefficient α (cm⁻¹)": alpha_calc, "Photon energy hν (eV)": hv, "(αhν)² (eV²cm⁻²)": tauc})
    return df, {"Band Gap Eg (eV)": eg_fit, "R²": r2, "Fit Range (eV)": f"{fit_lo:.2f} – {fit_hi:.2f}", "Absorption Edge (nm)": 1239.841984 / eg_fit, "Reference Eg (eV)": eg}, (m, c)

# -----------------------------
# UI Header & Navigation
# -----------------------------
st.title("🔬 Interactive Virtual Science Lab")
st.markdown("Welcome! Choose an experiment from the sidebar to explore energy systems and semiconductor physics.")

page = st.sidebar.radio("Navigate Experiments", ["🏠 Home", "🔋 Charge–Discharge Lab", "💡 Band Gap Lab", "📖 Student Theory Guide"])

if page == "🏠 Home":
    st.subheader("Welcome to the Student Simulation Lab")
    st.write("This platform allows you to run virtual laboratory experiments, observe physical principles in real-time, and download simulated datasets.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experiment 1: Energy Storage**\n\nStudy how Lithium-ion batteries and Supercapacitors charge and discharge under custom current loads.")
    with col2:
        st.info("**Experiment 2: Band Gap Analysis**\n\nAnalyze optical absorbance spectra to determine semiconductor band gaps using Tauc plots.")

elif page == "🔋 Charge–Discharge Lab":
    st.header("🔋 Battery & Supercapacitor Characteristics")
    
    col_params, col_plots = st.columns([1, 2])
    
    with col_params:
        st.subheader("Control Panel")
        cell_type = st.selectbox("Device Type", ["Lithium-ion Battery", "Supercapacitor"])
        
        if cell_type == "Lithium-ion Battery":
            nom_v = st.number_input("Nominal Voltage (V)", 2.0, 5.0, 3.7, 0.1)
            cap = st.number_input("Capacity (Ah)", 0.05, 50.0, 2.2, 0.1)
            capacitance = 0.0
        else:
            nom_v = st.number_input("Max Voltage (V)", 0.5, 100.0, 2.7, 0.1)
            capacitance = st.number_input("Capacitance (F)", 1.0, 100000.0, 1000.0, 50.0)
            cap = 0.0

        r_int = st.number_input("Internal Resistance (Ω)", 0.001, 5.0, 0.05, 0.01)
        i_charge = st.number_input("Charge Current (A)", 0.1, 50.0, 1.0, 0.1)
        i_dis = st.number_input("Discharge Current (A)", 0.1, 50.0, 1.0, 0.1)
        v_cut = st.number_input("Cut-off Voltage (V)", 0.1, 5.0, 2.75 if cell_type == "Lithium-ion Battery" else 1.0, 0.05)

    df, metrics = battery_sim(cell_type, nom_v, cap, capacitance, r_int, i_charge, i_dis, v_cut, 2.0, 25.0)

    with col_plots:
        st.subheader("Simulation Results")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for phase, grp in df.groupby("Phase"):
            ax.plot(grp["Time (s)"], grp["Voltage (V)"], label=phase, linewidth=2)
        style_plot(ax, f"Voltage Profile — {cell_type}", "Time (s)", "Voltage (V)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Charge Time", f"{metrics['Charge Time (h)']:.2f} h")
    m2.metric("Discharge Time", f"{metrics['Discharge Time (h)']:.2f} h")
    m3.metric("Energy Delivered", f"{metrics['Energy Discharged (Wh)']:.2f} Wh")
    m4.metric("Coulombic Efficiency", f"{metrics['Coulombic Efficiency (%)']:.1f}%")

    with st.expander("📚 Student Guidance: What to observe?"):
        st.markdown("""
        * **Internal Resistance Drop:** Notice the immediate vertical shift in voltage when switching between charge and discharge.
        * **Curve Shape:** Supercapacitors display a linear voltage response ($V \\propto t$), while batteries exhibit non-linear plateau regions due to chemical phase changes.
        """)

elif page == "💡 Band Gap Lab":
    st.header("💡 Optical Band Gap Determination")
    
    col_params, col_plots = st.columns([1, 2])
    
    with col_params:
        st.subheader("Control Panel")
        mat = st.selectbox("Semiconductor", list(SEMICONDUCTORS.keys()))
        thick = st.number_input("Sample Thickness (mm)", 0.1, 10.0, 1.0, 0.1)
        noise = st.slider("Measurement Noise (%)", 0.0, 10.0, 2.0, 0.5)

    df, bres, (m, c) = bandgap_sim(mat, thick, 200, 900, 2, noise)

    with col_plots:
        st.subheader("Tauc Plot Analysis")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        x, y = df["Photon energy hν (eV)"].to_numpy(), df["(αhν)² (eV²cm⁻²)"].to_numpy()
        ax.scatter(x, y, s=10, alpha=0.5, label="Optical Data")
        
        fit_lo, fit_hi = float(bres["Fit Range (eV)"].split("–")[0]), float(bres["Fit Range (eV)"].split("–")[1])
        xx = np.linspace(fit_lo, fit_hi, 50)
        ax.plot(xx, m * xx + c, 'r--', label=f"Linear Fit Extrapolation")
        ax.axvline(bres["Band Gap Eg (eV)"], color='g', linestyle=':', label=f"Extrapolated $E_g$ = {bres['Band Gap Eg (eV)']:.2f} eV")
        
        style_plot(ax, f"Tauc Plot — {mat}", "Photon Energy $h\\nu$ (eV)", "$(a h\\nu)^2$")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Measured Band Gap ($E_g$)", f"{bres['Band Gap Eg (eV)']:.2f} eV")
    m2.metric("Reference $E_g$", f"{bres['Reference Eg (eV)']:.2f} eV")
    m3.metric("Absorption Edge", f"≈ {bres['Absorption Edge (nm)']:.0f} nm")

    with st.expander("📚 Student Guidance: How the Tauc Plot works"):
        st.markdown("""
        * **Linear Region:** The linear region of $(a h\\nu)^2$ versus energy represents direct band-edge transitions.
        * **$X$-Intercept:** Extrapolating the linear slope to $y = 0$ provides the experimental band gap energy ($E_g$).
        """)

elif page == "📖 Student Theory Guide":
    st.header("📖 Educational Theory Reference")
    st.markdown("""
    ### 1. Energy Storage Equations
    * **Capacitor Discharge Voltage:** $V(t) = V_0 - \\frac{I}{C}t$
    * **Stored Energy:** $E = \\frac{1}{2} C V^2$

    ### 2. Semiconductor Optical Absorption
    * **Photon Energy:** $h\\nu = \\frac{1240}{\\lambda \\text{ (nm)}} \\text{ eV}$
    * **Tauc Relation (Direct Gap):** $(\\alpha h\\nu)^2 = B(h\\nu - E_g)$
    """)
