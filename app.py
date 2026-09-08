import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Virtual Chemistry Lab Simulations",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helpers
# -----------------------------
def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def battery_sim(cell_type, nominal_v, capacity_ah, capacitance_f,
                r_internal, charge_current, discharge_current,
                cutoff_v, time_step, temperature):
    """
    Educational equivalent-circuit simulation.
    Li-ion: capacity is Ah, charge/discharge follows a SOC-dependent OCV model.
    Supercapacitor: capacitance is F and V = Q/C with ESR.
    """
    if cell_type == "Lithium-ion Battery":
        # SOC-dependent open-circuit voltage model.
        def ocv(soc):
            soc = np.clip(soc, 0, 1)
            return 3.0 + 0.75 * soc + 0.10 * np.log10(np.maximum(soc, 1e-5) / np.maximum(1-soc, 1e-5)) * 0.05

        # Start around 20% SOC and charge to ~100%, then discharge.
        soc0 = 0.20
        charge_seconds = max((1.0 - soc0) * capacity_ah / max(charge_current, 1e-9) * 3600, 1)
        n_charge = int(np.ceil(charge_seconds / time_step)) + 1
        t_charge = np.arange(n_charge) * time_step
        soc_charge = np.minimum(soc0 + charge_current * t_charge / 3600 / capacity_ah, 1.0)
        v_charge = ocv(soc_charge) + r_internal * charge_current
        v_charge = np.clip(v_charge, 2.8, 4.25)

        # Discharge from 100% until cutoff voltage.
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

        t_charge_h = t_charge[-1] / 3600
        t_dis_h = t_dis[-1] / 3600
        energy_wh = np.trapezoid(v_dis * i_dis, t_dis) / 3600 if len(t_dis) > 1 else 0
        charge_ah = np.trapezoid(np.full_like(t_charge, charge_current), t_charge) / 3600
        discharge_ah = np.trapezoid(i_dis, t_dis) / 3600 if len(t_dis) > 1 else 0
        efficiency = (discharge_ah / charge_ah * 100) if charge_ah > 0 else 0

        charge_df = pd.DataFrame({
            "Time (s)": t_charge,
            "Voltage (V)": v_charge,
            "Current (A)": np.full_like(t_charge, charge_current),
            "Capacity (Ah)": charge_current * t_charge / 3600,
            "SOC": soc_charge * 100,
            "Phase": "Charge",
        })
        dis_df = pd.DataFrame({
            "Time (s)": t_dis + t_charge[-1],
            "Voltage (V)": v_dis,
            "Current (A)": -i_dis,
            "Capacity (Ah)": cap_dis,
            "SOC": (1 - cap_dis / capacity_ah) * 100,
            "Phase": "Discharge",
        })
        df = pd.concat([charge_df, dis_df], ignore_index=True)
        return df, {
            "Charge Time (h)": t_charge_h,
            "Discharge Time (h)": t_dis_h,
            "Energy Discharged (Wh)": energy_wh,
            "Coulombic Efficiency (%)": efficiency,
        }

    # Supercapacitor
    v_min = max(cutoff_v, 0.05)
    v_max = nominal_v
    q_max = capacitance_f * v_max
    charge_seconds = q_max / max(charge_current, 1e-9)
    n_charge = int(np.ceil(charge_seconds / time_step)) + 1
    t_charge = np.arange(n_charge) * time_step
    v_charge = np.minimum(v_min + charge_current * t_charge / capacitance_f, v_max)
    v_charge += r_internal * charge_current
    v_charge = np.clip(v_charge, 0, v_max * 1.02)

    # Discharge: ideal capacitor V = V0 - It/C with ESR drop.
    t_max = max(capacitance_f * max(v_max - v_min, 0) / max(discharge_current, 1e-9), time_step)
    n_dis = int(np.ceil(t_max / time_step)) + 1
    t_dis = np.arange(n_dis) * time_step
    v_dis = v_max - discharge_current * t_dis / capacitance_f - r_internal * discharge_current
    keep = v_dis > v_min
    t_dis, v_dis = t_dis[keep], v_dis[keep]
    if len(t_dis) == 0:
        t_dis = np.array([0.0])
        v_dis = np.array([v_min])
    cap_removed = discharge_current * t_dis / 3600

    charge_df = pd.DataFrame({
        "Time (s)": t_charge,
        "Voltage (V)": v_charge,
        "Current (A)": np.full_like(t_charge, charge_current),
        "Capacity (Ah)": charge_current * t_charge / 3600,
        "SOC": np.clip((v_charge - v_min) / (v_max - v_min) * 100, 0, 100),
        "Phase": "Charge",
    })
    dis_df = pd.DataFrame({
        "Time (s)": t_dis + t_charge[-1],
        "Voltage (V)": v_dis,
        "Current (A)": -np.full_like(t_dis, discharge_current),
        "Capacity (Ah)": cap_removed,
        "SOC": np.clip((v_dis - v_min) / (v_max - v_min) * 100, 0, 100),
        "Phase": "Discharge",
    })
    df = pd.concat([charge_df, dis_df], ignore_index=True)

    t_charge_h = t_charge[-1] / 3600
    t_dis_h = t_dis[-1] / 3600
    energy_wh = np.trapezoid(v_dis * discharge_current, t_dis) / 3600 if len(t_dis) > 1 else 0
    charge_ah = np.trapezoid(np.full_like(t_charge, charge_current), t_charge) / 3600
    discharge_ah = np.trapezoid(np.full_like(t_dis, discharge_current), t_dis) / 3600
    efficiency = discharge_ah / charge_ah * 100 if charge_ah > 0 else 0

    return df, {
        "Charge Time (h)": t_charge_h,
        "Discharge Time (h)": t_dis_h,
        "Energy Discharged (Wh)": energy_wh,
        "Coulombic Efficiency (%)": efficiency,
    }


SEMICONDUCTORS = {
    "ZnO": {"Eg": 3.28, "alpha0": 1.0e4},
    "GaAs": {"Eg": 1.42, "alpha0": 1.5e4},
    "Si": {"Eg": 1.12, "alpha0": 1.0e4},
    "CdS": {"Eg": 2.42, "alpha0": 1.2e4},
    "TiO₂": {"Eg": 3.20, "alpha0": 1.0e4},
}


def bandgap_sim(material, thickness_mm, wl_min, wl_max, step_nm, noise_pct, direct_allowed=True):
    p = SEMICONDUCTORS[material]
    eg = p["Eg"]
    wl = np.arange(wl_min, wl_max + step_nm, step_nm)
    hv = 1239.841984 / wl  # eV

    # Synthetic direct-allowed semiconductor absorption:
    # alpha = alpha0 * sqrt(hv-Eg)/hv above Eg.
    excess = np.maximum(hv - eg, 0)
    alpha = p["alpha0"] * np.sqrt(excess) / np.maximum(hv, 1e-9)
    # Add a small sub-gap baseline and measurement-like variation.
    alpha += p["alpha0"] * 0.002
    absorbance = alpha * (thickness_mm / 10.0) / 2.302585
    rng = np.random.default_rng(42)
    if noise_pct > 0:
        absorbance *= 1 + rng.normal(0, noise_pct / 100, len(absorbance))
    absorbance = np.clip(absorbance, 1e-5, None)
    alpha_calc = 2.302585 * absorbance / (thickness_mm / 10.0)
    tauc = (alpha_calc * hv) ** 2

    # Automatic fit around the transition. Search windows and choose the best R².
    candidates = []
    lo = eg * 0.90
    hi = eg * 1.10
    mask = (hv >= lo) & (hv <= hi) & (alpha_calc > p["alpha0"] * 0.01)
    x = hv[mask]
    y = tauc[mask]
    if len(x) >= 5:
        for frac_lo in np.linspace(0.00, 0.35, 8):
            for frac_hi in np.linspace(0.65, 1.00, 8):
                a = int(frac_lo * len(x))
                b = max(a + 5, int(frac_hi * len(x)))
                xx, yy = x[a:b], y[a:b]
                if len(xx) < 5:
                    continue
                m, c = np.polyfit(xx, yy, 1)
                pred = m * xx + c
                ss_res = np.sum((yy - pred) ** 2)
                ss_tot = np.sum((yy - np.mean(yy)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                intercept = -c / m if m != 0 else np.nan
                if 0.5 * eg < intercept < 1.5 * eg and m > 0:
                    candidates.append((r2, intercept, xx.min(), xx.max(), m, c))

    if candidates:
        r2, eg_fit, fit_lo, fit_hi, m, c = max(candidates, key=lambda z: z[0])
    else:
        eg_fit, r2, fit_lo, fit_hi, m, c = eg, 0.0, eg * 0.9, eg * 1.1, 1, -eg

    edge_nm = 1239.841984 / eg_fit
    df = pd.DataFrame({
        "Wavelength (nm)": wl,
        "Absorbance (A)": absorbance,
        "Absorption coefficient α (cm⁻¹)": alpha_calc,
        "Photon energy hν (eV)": hv,
        "(αhν)² (eV²cm⁻²)": tauc,
    })
    return df, {
        "Band Gap Eg (eV)": eg_fit,
        "R²": r2,
        "Fit Range (eV)": f"{fit_lo:.2f} – {fit_hi:.2f}",
        "Absorption Edge (nm)": edge_nm,
        "Reference Eg (eV)": eg,
    }, (m, c)


# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg,#0b2947,#123f70);
    padding: 16px 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 18px;
}
.main-header h1 { margin: 0; font-size: 28px; }
.main-header p { margin: 4px 0 0; opacity: .85; }
.card {
    border: 1px solid #d8e1eb;
    border-radius: 12px;
    padding: 16px;
    background: white;
}
.result {
    border: 1px solid #d8e1eb;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.small-note { font-size: 13px; color: #536273; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
<h1>🔬 Virtual Lab Simulations</h1>
<p>Engineering Chemistry / Applied Chemistry Laboratory</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Virtual Lab")
page = st.sidebar.radio(
    "Select Experiment",
    [
        "🏠 Home",
        "🔋 1. Charge–Discharge Characteristics",
        "💡 2. Semiconductor Band Gap",
        "📖 Theory",
        "🧮 Calculator",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use**
1. Select an experiment.
2. Set the parameters.
3. Click **Run Simulation**.
4. Analyze graphs and extracted results.
5. Download the simulated data.
""")
st.sidebar.caption("Educational simulation — not a substitute for laboratory measurements.")

# -----------------------------
# Home
# -----------------------------
if page == "🏠 Home":
    st.subheader("Virtual Chemistry Laboratory")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### 🔋 Experiment 1\n**Determination of the charge–discharge characteristics of a lithium-ion battery / supercapacitor cell.**\n\nStudy voltage, current, capacity, charge time, discharge time, energy and efficiency.")
    with c2:
        st.info("### 💡 Experiment 2\n**Determination of the band-gap energy of a semiconductor sample from its optical absorption edge.**\n\nStudy absorbance, absorption coefficient, photon energy and the Tauc plot.")
    st.markdown("### Learning outcomes")
    st.markdown("""
    - Interpret charge and discharge curves of energy-storage devices.
    - Understand the effect of internal resistance / ESR.
    - Determine approximate energy and Coulombic efficiency from simulated data.
    - Convert optical wavelength to photon energy.
    - Calculate the absorption coefficient from Beer–Lambert law.
    - Determine semiconductor band gap using a Tauc plot.
    """)
    st.success("Tip: Run both experiments, inspect the plots, then download the generated CSV data for student records or assignments.")

# -----------------------------
# Experiment 1
# -----------------------------
elif page == "🔋 1. Charge–Discharge Characteristics":
    st.header("1. Determination of Charge–Discharge Characteristics")
    st.caption("Educational equivalent-circuit simulation for a lithium-ion battery or supercapacitor cell.")

    left, right = st.columns([1, 1.55])

    with left:
        st.subheader("1. Setup & Parameters")
        cell = st.selectbox("Select Cell Type", ["Lithium-ion Battery", "Supercapacitor"])

        if cell == "Lithium-ion Battery":
            nominal_v = st.number_input("Nominal Voltage (V)", 2.0, 5.0, 3.7, 0.1)
            capacity = st.number_input("Capacity (Ah)", 0.05, 50.0, 2.2, 0.05)
            capacitance = 0.0
        else:
            nominal_v = st.number_input("Maximum Voltage (V)", 0.5, 100.0, 2.7, 0.1)
            capacitance = st.number_input("Capacitance (F)", 1.0, 100000.0, 1000.0, 10.0)
            capacity = 0.0

        r_internal = st.number_input("Internal Resistance / ESR (Ω)", 0.001, 10.0, 0.08 if cell == "Lithium-ion Battery" else 0.03, 0.001)
        charge_current = st.number_input("Charge Current (A)", 0.01, 100.0, 1.0, 0.05)
        discharge_current = st.number_input("Discharge Current (A)", 0.01, 100.0, 1.0, 0.05)
        cutoff_v = st.number_input("Cut-off Voltage (V)", 0.1, 5.0, 2.75 if cell == "Lithium-ion Battery" else 1.0, 0.05)
        temperature = st.number_input("Temperature (°C)", -20.0, 80.0, 25.0, 1.0)
        time_step = st.number_input("Time Step (s)", 0.1, 60.0, 2.0, 0.1)

        run = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run or "battery_df" not in st.session_state:
        df, results = battery_sim(
            cell, nominal_v, capacity, capacitance, r_internal,
            charge_current, discharge_current, cutoff_v, time_step, temperature
        )
        st.session_state.battery_df = df
        st.session_state.battery_results = results

    df = st.session_state.battery_df
    results = st.session_state.battery_results

    with right:
        st.subheader("2. Results")
        tab1, tab2, tab3 = st.tabs(["Voltage vs Time", "Current vs Time", "Capacity vs Time"])

        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            for phase, grp in df.groupby("Phase"):
                ax.plot(grp["Time (s)"], grp["Voltage (V)"], label=phase)
            style_plot(ax, "Charge–Discharge Voltage Characteristic", "Time (s)", "Voltage (V)")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(df["Time (s)"], df["Current (A)"])
            style_plot(ax, "Current vs Time", "Time (s)", "Current (A)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            for phase, grp in df.groupby("Phase"):
                ax.plot(grp["Time (s)"], grp["Capacity (Ah)"], label=phase)
            style_plot(ax, "Capacity vs Time", "Time (s)", "Capacity (Ah)")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Extracted Parameters")
    a, b, c, d = st.columns(4)
    a.metric("Charge Time", f"{results['Charge Time (h)']:.2f} h")
    b.metric("Discharge Time", f"{results['Discharge Time (h)']:.2f} h")
    c.metric("Energy Discharged", f"{results['Energy Discharged (Wh)']:.2f} Wh")
    d.metric("Coulombic Efficiency", f"{results['Coulombic Efficiency (%)']:.1f}%")

    with st.expander("Show simulated data table"):
        st.dataframe(df, use_container_width=True, height=300)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Charge–Discharge CSV", csv, "charge_discharge_simulation.csv", "text/csv")

    st.markdown("**Chemistry model:** Li-ion simulation uses a SOC-dependent open-circuit-voltage model plus internal-resistance polarization. The supercapacitor uses \(V=Q/C\) with ESR. These are educational models and do not represent a particular commercial cell.")

# -----------------------------
# Experiment 2
# -----------------------------
elif page == "💡 2. Semiconductor Band Gap":
    st.header("2. Determination of Band-Gap Energy from Optical Absorption Edge")
    st.caption("Synthetic UV–Vis absorption simulation with automatic Tauc-plot analysis.")

    left, right = st.columns([1, 1.55])

    with left:
        st.subheader("1. Setup & Parameters")
        material = st.selectbox("Select Semiconductor", list(SEMICONDUCTORS.keys()))
        thickness = st.number_input("Sample Thickness (mm)", 0.01, 10.0, 1.0, 0.01)
        wl_min, wl_max = st.slider("Wavelength Range (nm)", 200, 1200, (300, 800), 10)
        step = st.number_input("Wavelength Step (nm)", 1, 20, 2, 1)
        noise = st.number_input("Add Measurement Noise (%)", 0.0, 20.0, 2.0, 0.5)
        show_points = st.checkbox("Show Raw Data Points", True)
        run2 = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run2 or "band_df" not in st.session_state or st.session_state.get("band_material") != material:
        bdf, bres, fit = bandgap_sim(material, thickness, wl_min, wl_max, step, noise)
        st.session_state.band_df = bdf
        st.session_state.band_results = bres
        st.session_state.band_fit = fit
        st.session_state.band_material = material

    bdf = st.session_state.band_df
    bres = st.session_state.band_results
    m, c = st.session_state.band_fit

    with right:
        st.subheader("2. Results")
        tab1, tab2 = st.tabs(["Absorbance Spectrum", "Tauc Plot ((αhν)² vs hν)"])

        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(bdf["Wavelength (nm)"], bdf["Absorbance (A)"])
            ax.invert_xaxis()
            style_plot(ax, f"Optical Absorption Spectrum — {material}", "Wavelength (nm)", "Absorbance (A)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            x = bdf["Photon energy hν (eV)"].to_numpy()
            y = bdf["(αhν)² (eV²cm⁻²)"].to_numpy()
            if show_points:
                ax.scatter(x, y, s=12, label="Data")
            fit_mask = (x >= float(bres["Fit Range (eV)"].split("–")[0])) & (x <= float(bres["Fit Range (eV)"].split("–")[1]))
            xx = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 100) if fit_mask.any() else np.linspace(m * 0 + 1, 4, 100)
            ax.plot(xx, m * xx + c, label="Linear fit")
            ax.axvline(bres["Band Gap Eg (eV)"], linestyle="--", label=f"Eg = {bres['Band Gap Eg (eV)']:.2f} eV")
            style_plot(ax, "Tauc Plot — Direct Allowed Transition", "Photon energy hν (eV)", "(αhν)² (eV² cm⁻²)")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Extracted Parameters")
    a, b, c1, d = st.columns(4)
    a.metric("Calculated Band Gap", f"{bres['Band Gap Eg (eV)']:.2f} eV")
    b.metric("Fit Range", bres["Fit Range (eV)"])
    c1.metric("R²", f"{bres['R²']:.4f}")
    d.metric("Absorption Edge", f"≈ {bres['Absorption Edge (nm)']:.0f} nm")

    with st.expander("Show simulated optical data"):
        st.dataframe(bdf, use_container_width=True, height=300)

    csv = bdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Band-Gap CSV", csv, "semiconductor_bandgap_simulation.csv", "text/csv")

    st.info("Analysis uses the direct-allowed Tauc relation: (αhν)² ∝ (hν − Eg). The x-intercept of the fitted linear region gives Eg. For an actual sample, use experimentally measured absorbance and select the correct transition model.")

# -----------------------------
# Theory
# -----------------------------
elif page == "📖 Theory":
    st.header("Theory")

    st.subheader("Experiment 1 — Charge–Discharge Characteristics")
    st.markdown("""
    **Lithium-ion battery:** During charging, electrical energy is stored chemically and the terminal voltage rises.
    During discharge, the stored energy is delivered to the external load. Internal resistance produces an
    instantaneous voltage drop/rise approximately described by **V = OCV ± IR**.

    **Supercapacitor:** Charge and voltage are related by:

    **Q = CV**

    and, for constant-current discharge,

    **V(t) = V₀ − (I/C)t**

    The energy stored in an ideal capacitor is:

    **E = ½CV²**

    Coulombic efficiency can be estimated from:

    **η = (Q_discharge / Q_charge) × 100%**
    """)

    st.subheader("Experiment 2 — Semiconductor Band Gap")
    st.markdown("""
    The photon energy corresponding to wavelength λ is:

    **hν = 1240 / λ(nm)  eV**

    From Beer–Lambert law, the absorption coefficient can be obtained approximately from:

    **α = 2.303 A / d**

    where **A** is absorbance and **d** is sample thickness in cm.

    For a direct allowed transition, the Tauc relation is:

    **(αhν)² = B(hν − Eg)**

    Plotting **(αhν)²** against **hν** and extrapolating the linear absorption region to the energy axis gives the
    optical band gap **Eg**.
    """)

# -----------------------------
# Calculator
# -----------------------------
elif page == "🧮 Calculator":
    st.header("Calculator")

    tab1, tab2, tab3 = st.tabs(["Photon Energy", "Capacitor Energy", "Absorption Coefficient"])

    with tab1:
        wl = st.number_input("Wavelength (nm)", 200.0, 2000.0, 378.0, 1.0)
        st.success(f"Photon energy hν = **{1239.841984 / wl:.4f} eV**")

    with tab2:
        C = st.number_input("Capacitance (F)", 0.1, 1_000_000.0, 1000.0, 10.0)
        V = st.number_input("Voltage (V)", 0.01, 1000.0, 2.7, 0.1)
        st.success(f"Stored energy E = **{0.5*C*V*V:.3f} J** = **{0.5*C*V*V/3600:.6f} Wh**")

    with tab3:
        A = st.number_input("Absorbance", 0.0001, 10.0, 0.50, 0.01)
        d_mm = st.number_input("Thickness (mm)", 0.01, 100.0, 1.0, 0.01)
        alpha = 2.302585 * A / (d_mm / 10)
        st.success(f"Absorption coefficient α = **{alpha:.3f} cm⁻¹**")

st.markdown("---")
st.caption("Virtual Lab Simulations • Engineering Chemistry / Applied Chemistry • Values are generated for educational purposes.")
