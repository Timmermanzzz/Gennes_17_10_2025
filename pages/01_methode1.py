"""
Method 1: Compute Droplet Shape
"""

import streamlit as st
from auth import require_password
import pandas as pd
import numpy as np
from solver import generate_droplet_shape, get_physical_parameters
import solver as solver_mod  # module import zodat we kunnen reloaden
import importlib
from utils import (
    shift_x_coordinates,
    get_droplet_metrics,
    find_height_for_diameter,
    solve_gamma_for_volume,
    solve_gamma_for_height,
    calculate_diameter_at_height,
    find_collar_tube_diameter_for_volume,
    find_collar_tube_diameter_with_displacement,
    init_streamlit_logger,
    get_console_logs,
    clear_console_logs,
)
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf, export_to_3dm
from pdf_export import export_to_pdf
import tempfile

st.set_page_config(
    page_title="Method 1 - Compute Droplet Shape",
    page_icon="📊",
    layout="wide"
)

require_password()

# Console logger voor debug
logger = init_streamlit_logger(name='method1', level=20)

st.title("📊 Method 1 — Compute Droplet Shape")
st.markdown("Compute droplet shapes for given parameters and choose your cut options.")

# Uitleg/Help
with st.expander("ℹ️ Help — How does Method 1 work?", expanded=False):
    st.markdown(
        """
        - **Goal**: Compute the droplet (Young–Laplace) and design a collar that restores the removed water volume.
        - **Key principle**: **Collar volume = Cut volume** (pure volume matching, no pressure/Δh logic).
        - **Inputs**:
          - **γₛ (N/m)**, **ρ (kg/m³)**, **g (m/s²)**
          - **Cut**: by percentage or by opening diameter → we determine the cut height and flat top.
          - **Sloshing height (m)**: optional freeboard added above the water level inside the collar.
        - **Definitions**:
          - **Cut volume** = full droplet volume − truncated droplet volume.
          - **Collar volume (net)** = extra capacity added by the collar around the opening.
          - **Opening area** A = π·R² with R = opening_diameter/2.
        - **Physics (always enabled)**:
          - We solve the tube diameter D from the exact balance:
            \(\;\text{Cut} = A·(D − s) − \pi^2 R (D/2)^2\;\) with sloshing s.
          - De tweede term is de waterverdringing door de torus (fysisch correct).
        - **Outputs**: droplet volume, cut volume, collar volume, collar tube diameter, opening diameter, heights/diameters.
        - **Units**: m and m³.
        """
    )

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'physical_params' not in st.session_state:
    st.session_state.physical_params = None

st.markdown("---")

st.header("⚙️ Parameters")

# Physics model selector
physics_model = st.selectbox(
    "Physics model",
    ["Young–Laplace (de Gennes)", "Timoshenko (membrane)"]
)

# Row 1: Physical properties
col1, col2, col3 = st.columns(3)

with col1:
    gamma_s = st.number_input(
        "γₛ - Surface tension (N/m)",
        min_value=100.0,
        max_value=1000000.0,
        value=35000.0,
        step=1000.0,
        help="Oppervlaktespanningsparameter van het materiaal"
    )

with col2:
    rho = st.number_input(
        "ρ - Density (kg/m³)",
        min_value=1.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Dichtheid van de vloeistof"
    )

with col3:
    g = st.number_input(
        "g - Gravity (m/s²)",
        min_value=0.1,
        max_value=20.0,
        value=9.8,
        step=0.1,
        help="Gravitatieversnelling (standaard 9.8 op aarde)"
    )

# Extra parameters voor Timoshenko
if physics_model == "Timoshenko (membrane)":
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        N_timo = st.number_input(
            "N - Membrane force (N/m)",
            min_value=1000.0,
            max_value=1_000_000.0,
            value=27500.0,
            step=500.0,
            help="Membraankracht per eenheid omtrek"
        )
    with col_t2:
        top_pressure = st.number_input(
            "P₀ - Top pressure (Pa)",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            step=10.0,
            help="Druk bovenin; bepaalt head d = P₀/(ρg)"
        )
    with col_t3:
        phi_max_deg = st.slider(
            "φ max (deg)",
            min_value=60,
            max_value=170,
            value=120,
            step=5,
            help="Stopcriterium voor integratie (veilig weg van u→1)"
        )

# Row 2: Shape adjustment
col4, col5 = st.columns([1, 1])

with col4:
    st.subheader("Shape adjustment")
    
    cut_method = st.selectbox(
        "Cut method:",
        ["No cut", "Cut percentage", "Cut diameter"],
        help="Choose how you want to adjust the droplet"
    )
    
    use_diameter_mode = False
    use_percentage_mode = False
    
    if cut_method == "Cut percentage":
        cut_percentage = st.slider(
            "Cut percentage (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Percentage om van de bovenkant af te knippen"
        )
        cut_diameter = None
        use_percentage_mode = cut_percentage > 0
    elif cut_method == "Cut diameter":
        cut_diameter = st.number_input(
            "Cut diameter (m)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
            help="Diameter van de opening"
        )
        cut_percentage = 0
        use_diameter_mode = cut_diameter > 0
    else:
        cut_percentage = 0
        cut_diameter = 0
    
    st.markdown("")
    st.subheader("Constraints (optional)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_volume_constraint = st.toggle("Enforce target volume", value=False)
        target_volume = 0.0
        if use_volume_constraint:
            target_volume = st.number_input("Target volume (m³)", min_value=0.0, value=1000.0, step=10.0)
    with col_c2:
        use_height_constraint = st.toggle("Enforce target height", value=False)
        target_height = 0.0
        if use_height_constraint:
            target_height = st.number_input("Target height (m)", min_value=0.0, value=3.3, step=0.01)
    
    st.markdown("")
    st.subheader("Collar / torus (optional)")
    sloshing_height = st.number_input(
        "Sloshing height (m)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Extra unfilled height in collar tube to prevent splashing (freeboard)"
    )

with col5:
    st.subheader("Action")
    if st.button("🔬 Compute Droplet", type="primary", use_container_width=True):
        with st.spinner("Computing..."):
            try:
                logger.info("Start compute | model=%s", physics_model)
                logger.info("rho=%.3f kg/m^3, g=%.3f m/s^2", float(rho), float(g))
                if physics_model == "Timoshenko (membrane)":
                    logger.info("N=%.2f N/m, P0=%.2f Pa, phi_max=%d deg", float(N_timo), float(top_pressure), int(phi_max_deg))
                else:
                    logger.info("gamma_s=%.1f N/m", float(gamma_s))
                # Generate full shape (per selected model)
                physical_params = None
                if physics_model == "Young–Laplace (de Gennes)":
                    df_full = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                    full_metrics = get_droplet_metrics(df_full)
                    full_basis_diameter = full_metrics['bottom_diameter']
                    full_max_diameter = full_metrics['max_diameter']
                    physical_params = get_physical_parameters(df_full, gamma_s, rho, g)
                else:
                    # Timoshenko membrane (forceer module-reload zodat edits direct gelden)
                    try:
                        importlib.reload(solver_mod)
                    except Exception:
                        pass
                    df_timo, info_timo = solver_mod.solve_timoshenko_membrane(
                        rho=float(rho), g=float(g), N=float(N_timo), top_pressure=float(top_pressure), phi_max_deg=float(phi_max_deg)
                    )
                    logger.info("Timo solved | steps=%s, steps_z=%s, reason=%s, head_d=%.4f, r1_apex=%.2f",
                                info_timo.get('steps'), info_timo.get('steps_z'), info_timo.get('stopped_reason'), info_timo.get('head_d'), info_timo.get('r1_apex'))
                    import numpy as _np
                    z_arr = _np.asarray(df_timo['z'], dtype=float)
                    x_arr = _np.asarray(df_timo['x'], dtype=float)
                    H_max = float(_np.max(z_arr)) if len(z_arr) else 0.0
                    h_arr = H_max - z_arr
                    df_full = pd.DataFrame({'h': h_arr, 'x-x_0': x_arr})
                    # Belangrijk: Timoshenko x is al de straal t.o.v. de as.
                    # Voor correcte volumes/plotten markeren we dit expliciet als 'x_shifted'
                    # met de rechterrand op 0 (dus links negatief): x_shifted = -x.
                    try:
                        df_full['x_shifted'] = -pd.Series(x_arr, dtype=float)
                    except Exception:
                        pass
                    if len(df_full) > 1:
                        logger.info("Timo profile | H_max=%.4f m, R_max=%.4f m", float(df_full['h'].max()), float(abs(df_full['x-x_0']).max()))
                    full_metrics = get_droplet_metrics(df_full)
                    full_basis_diameter = full_metrics['bottom_diameter']
                    full_max_diameter = full_metrics['max_diameter']
                    physical_params = {
                        'model': 'timoshenko',
                        'rho': float(rho),
                        'g': float(g),
                        'N': float(N_timo),
                        'top_pressure': float(top_pressure),
                        'head_d': float(info_timo.get('head_d', 0.0)),
                        'r1_apex': float(info_timo.get('r1_apex', 0.0))
                    }
                
                df = df_full.copy()
                actual_cut_diameter = None
                
                df_before_top = None  # Bewaar voor volume berekening
                if use_diameter_mode and cut_diameter > 0:
                    cut_at_height = find_height_for_diameter(df, cut_diameter)
                    if not np.isnan(cut_at_height):
                        logger.info("Cut by diameter | D=%.3f m → h_cut=%.4f m", float(cut_diameter), float(cut_at_height))
                        df = df[df['h'] <= cut_at_height].copy()
                        # Bewaar profiel VOOR top-toevoeging voor volume berekening
                        df_before_top = df.copy()
                        target_radius = cut_diameter / 2.0
                        n_points = 30
                        # Plaats vlakke top aan de rechterkant [0, R] zodat deze naar rechts wijst
                        x_shifted_vals = np.linspace(-target_radius, 0.0, n_points)
                        top_points_data = []
                        x_max_current = df['x-x_0'].max() if 'x-x_0' in df.columns else 0.0
                        for x_sh in x_shifted_vals:
                            top_points_data.append({
                                'B': 1.0, 'C': 1.0, 'z': 0,
                                'x-x_0': x_sh + x_max_current,
                                'x_shifted': x_sh,
                                'h': cut_at_height
                            })
                        top_points = pd.DataFrame(top_points_data)
                        # Dedupliceer op ruwe x-kolom en hoogte zodat vlakke top een enkele lijn vormt
                        subset_cols = ['x-x_0', 'h'] if 'x-x_0' in df.columns else ['x_shifted', 'h']
                        df = pd.concat([df, top_points], ignore_index=True).drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
                        actual_cut_diameter = cut_diameter
                
                if use_percentage_mode and cut_percentage > 0:
                    if physics_model == "Young–Laplace (de Gennes)":
                        df = generate_droplet_shape(gamma_s, rho, g, cut_percentage=int(cut_percentage))
                    else:
                        # Cut Timoshenko at relative height and add flat top
                        h_max_local = float(df['h'].max()) if not df.empty else 0.0
                        cut_at_height = h_max_local * (1.0 - int(cut_percentage) / 100.0)
                        logger.info("Cut by %% | pct=%d → h_cut=%.4f m", int(cut_percentage), float(cut_at_height))
                        df = df[df['h'] <= cut_at_height].copy()
                        df_before_top = df.copy()
                        if not df.empty:
                            x_max_current = df['x-x_0'].max() if 'x-x_0' in df.columns else 0.0
                            # schat target radius uit doorsnede op hoogste h
                            try:
                                target_radius = abs(float(df[df['h'] == df['h'].max()]['x-x_0'].min()))
                            except Exception:
                                target_radius = 0.0
                            n_points = 30
                            x_shifted_vals = np.linspace(-target_radius, 0.0, n_points)
                            top_points = pd.DataFrame({
                                'B': 1.0,
                                'C': 1.0,
                                'z': 0.0,
                                'x-x_0': x_shifted_vals + x_max_current,
                                'x_shifted': x_shifted_vals,
                                'h': cut_at_height
                            })
                            subset_cols = ['x-x_0', 'h'] if 'x-x_0' in df.columns else ['x_shifted', 'h']
                            df = pd.concat([df, top_points], ignore_index=True).drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
                
                if physics_model == "Young–Laplace (de Gennes)" and use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    cut_pct = int(cut_percentage) if use_percentage_mode else 0
                    cut_diam = float(actual_cut_diameter or 0.0) if use_diameter_mode else 0.0
                    gamma_opt, df_opt, vol_opt = solve_gamma_for_volume(
                        target_volume=target_volume, rho=rho, g=g,
                        cut_percentage=cut_pct, cut_diameter=cut_diam,
                    )
                    df = df_opt
                    gamma_s = gamma_opt
                
                if physics_model == "Young–Laplace (de Gennes)" and use_height_constraint and target_height > 0:
                    cut_pct = int(cut_percentage) if use_percentage_mode else 0
                    cut_diam = float(actual_cut_diameter or 0.0) if use_diameter_mode else 0.0
                    gamma_opt, df_opt, h_opt = solve_gamma_for_height(
                        target_height=target_height, rho=rho, g=g,
                        cut_percentage=cut_pct, cut_diameter=cut_diam,
                    )
                    df = df_opt
                    gamma_s = gamma_opt
                    # Voor enforce height: df_opt bevat al een correct afgekapte vorm, geen extra top nodig
                    if use_diameter_mode and cut_diam > 0:
                        actual_cut_diameter = cut_diam
                
                if physics_model == "Young–Laplace (de Gennes)":
                    df_full_final = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
                else:
                    df_full_final = df_full.copy()
                full_metrics_final = get_droplet_metrics(df_full_final)
                full_basis_diameter_final = full_metrics_final['bottom_diameter']
                full_max_diameter_final = full_metrics_final['max_diameter']
                full_max_height_final = full_metrics_final['max_height']
                
                metrics = get_droplet_metrics(df)
                if physics_model == "Young–Laplace (de Gennes)":
                    physical_params = get_physical_parameters(df, gamma_s, rho, g)
                else:
                    # Voor Timoshenko: gebruik volume uit de ODE (π∫x^2 dz) voor precisie
                    try:
                        metrics['volume'] = float(info_timo.get('volume', metrics.get('volume', 0.0)))
                    except Exception:
                        pass
                    # Base diameter blijft wat de metriekfunctie berekent: diameter op laagste h
                logger.info("Metrics | V=%.3f m^3, H=%.3f m, Dmax=%.3f m", float(metrics.get('volume', 0.0)), float(metrics.get('max_height', 0.0)), float(metrics.get('max_diameter', 0.0)))
                
                if actual_cut_diameter is not None:
                    metrics['top_diameter'] = actual_cut_diameter
                
                metrics['bottom_diameter'] = full_basis_diameter_final
                metrics['max_diameter'] = full_max_diameter_final
                
                # Bewaar afkaphoogte voor visualisatie
                cut_h_final = None
                if (use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0)) or (use_percentage_mode and cut_percentage > 0):
                    if use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0):
                        cut_h_final = find_height_for_diameter(df_full_final, float(actual_cut_diameter))
                    else:
                        cut_h_final = float(df['h'].max()) if not df.empty else np.nan
                
                # Bewaar afkaphoogte voor duidelijke visual (cut-plane)
                if cut_h_final is not None and not np.isnan(cut_h_final):
                    metrics['h_cut'] = float(cut_h_final)
                else:
                    metrics['h_cut'] = 0.0
                
                opening_diam_for_torus = None
                if use_diameter_mode and (actual_cut_diameter is not None and actual_cut_diameter > 0):
                    opening_diam_for_torus = float(actual_cut_diameter)
                elif use_percentage_mode and cut_percentage > 0:
                    if 'x_shifted' in df_full_final.columns and cut_h_final is not None and not np.isnan(cut_h_final):
                        opening_diam_for_torus = float(calculate_diameter_at_height(df_full_final, cut_h_final))
                
                # VOLLEDIG NIEUWE LOGICA: Puur volume-gebaseerd, GEEN druk
                if opening_diam_for_torus is not None and opening_diam_for_torus > 0:
                    # Stap 1: Bereken volume_afgekapt
                    volume_full = full_metrics_final.get('volume', 0.0)
                    if df_before_top is not None:
                        metrics_before_top = get_droplet_metrics(df_before_top)
                        volume_cut = metrics_before_top.get('volume', 0.0)
                    else:
                        volume_cut = metrics.get('volume', 0.0)
                    volume_afgekapt = volume_full - volume_cut
                    
                    # Stap 2: Auto-bereken tube diameter zodat volume_kraag = volume_afgekapt
                    from utils import find_collar_tube_diameter_with_displacement
                    tube_result = find_collar_tube_diameter_with_displacement(
                        target_volume=volume_afgekapt,
                        opening_diameter=opening_diam_for_torus,
                        sloshing_height=sloshing_height,
                        center_offset=0.0,
                    )
                    if tube_result['converged']:
                        tube_diameter_final = tube_result['tube_diameter']
                    else:
                        # Fallback indien niet converged
                        tube_diameter_final = 0.5
                    
                    # Stap 3: Bereken kraagvolume met simpel model
                    # Collar Volume = π × r_ring² × water_height
                    # Water height = tube diameter - sloshing height
                    import math
                    R_major = opening_diam_for_torus / 2.0
                    r_ring = R_major
                    r_tube = max(0.0, float(tube_diameter_final) / 2.0)
                    water_height = tube_diameter_final - sloshing_height  # Water fills to top of tube (no sloshing in this calc)
                    # Physically exact: subtract torus displacement
                    volume_kraag_calc = math.pi * (r_ring ** 2) * water_height - (math.pi ** 2) * R_major * (r_tube ** 2)
                    
                    # Opslaan metrics
                    metrics['volume_afgekapt'] = volume_afgekapt
                    metrics['volume_kraag'] = volume_kraag_calc
                    metrics['collar_tube_diameter'] = float(tube_diameter_final)
                    metrics['torus_R_major'] = float(R_major)
                    metrics['torus_r_top'] = float(r_tube)
                    metrics['torus_r_water'] = float(r_tube)
                    metrics['torus_water_volume'] = float(volume_kraag_calc)
                    
                    # Voor visualisatie: h_seam_eff = cut hoogte
                    if cut_h_final is not None and not np.isnan(cut_h_final):
                        metrics['h_seam_eff'] = float(cut_h_final)
                    else:
                        metrics['h_seam_eff'] = 0.0
                    # Water level inside collar: seam + (tube_diameter - sloshing)
                    try:
                        h_seam = float(metrics.get('h_seam_eff', 0.0))
                        d_tube = float(metrics.get('collar_tube_diameter', 0.0))
                        s_free = float(sloshing_height)
                        metrics['h_waterline'] = float(h_seam + max(0.0, d_tube - s_free))
                    except Exception:
                        metrics['h_waterline'] = float(metrics.get('h_seam_eff', 0.0))
                
                st.session_state.df = df
                st.session_state.metrics = metrics
                st.session_state.physical_params = physical_params
                
                if use_diameter_mode and cut_diameter > 0:
                    st.success(f"✅ Reservoir with {cut_diameter:.1f} m opening computed!")
                elif use_percentage_mode and cut_percentage > 0:
                    st.success(f"✅ Reservoir with {int(cut_percentage)}% cut computed!")
                elif use_volume_constraint and target_volume > 0 and not use_height_constraint:
                    st.success(f"✅ Target volume ≈ {target_volume:.1f} m³ achieved!")
                elif use_height_constraint and target_height > 0:
                    st.success(f"✅ Target height ≈ {target_height:.2f} m achieved!")
                else:
                    st.success("✅ Computation successful!")
                
            except Exception as e:
                import traceback
                logger.exception("Computation error: %s", str(e))
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# Results
if st.session_state.df is not None:
    st.header("📊 Specifications")
    
    # Volumes section
    st.subheader("💧 Volumes")
    col_vol1, col_vol2, col_vol3 = st.columns(3)
    
    droplet_vol = st.session_state.metrics.get('volume', 0)
    collar_vol = st.session_state.metrics.get('volume_kraag', 0)
    total_vol = droplet_vol + collar_vol
    
    with col_vol1:
        st.metric("Droplet Volume (m³)", f"{droplet_vol:.2f}")
    with col_vol2:
        st.metric("Collar Volume (m³)", f"{collar_vol:.2f}")
    with col_vol3:
        st.metric("Total Volume (m³)", f"{total_vol:.2f}")
    
    st.markdown("")
    
    # Geometry section
    st.subheader("📐 Geometry")
    col1, col2 = st.columns(2)
    
    with col1:
        droplet_height = float(st.session_state.metrics.get('max_height', 0) or 0.0)
        tube_d_for_total = float(st.session_state.metrics.get('collar_tube_diameter', 0) or 0.0) if st.session_state.metrics.get('volume_afgekapt', 0) > 0 else 0.0
        total_height = droplet_height + tube_d_for_total
        st.metric("Droplet height (m)", f"{droplet_height:.2f}")
        st.metric("Total height (m)", f"{total_height:.2f}")
        st.metric("Max diameter (m)", f"{st.session_state.metrics.get('max_diameter', 0):.2f}")
        st.metric("Base diameter (m)", f"{st.session_state.metrics.get('bottom_diameter', 0):.2f}")
    
    with col2:
        if cut_method != "No cut":
            st.metric("Opening diameter (m)", f"{st.session_state.metrics.get('top_diameter', 0):.2f}")
        else:
            st.metric("Opening diameter (m)", "-")
        if st.session_state.metrics.get('volume_afgekapt', 0) > 0:
            st.metric("Collar tube diameter (m)", f"{st.session_state.metrics.get('collar_tube_diameter', 0):.2f}")
        else:
            st.metric("Collar tube diameter (m)", "-")
    
    st.markdown("")
    
    # Material section
    st.subheader("⚙️ Material")
    col3, col4 = st.columns(2)
    
    if st.session_state.physical_params and 'N' in st.session_state.physical_params:
        with col3:
            st.metric("N (N/m)", f"{st.session_state.physical_params.get('N', 0):.0f}")
            st.metric("head d (m)", f"{st.session_state.physical_params.get('head_d', 0.0):.3f}")
    else:
        with col3:
            st.metric("γₛ (N/m)", f"{st.session_state.physical_params.get('gamma_s', 0):.0f}")
    with col4:
        if st.session_state.metrics.get('volume_afgekapt', 0) > 0:
            st.metric("Cut volume (m³)", f"{st.session_state.metrics.get('volume_afgekapt', 0):.2f}")
    
    st.markdown("---")
    st.header("📈 Visualisation")
    st.subheader("2D Cross-section")
    fig_2d = create_2d_plot(
        st.session_state.df,
        metrics=st.session_state.metrics,
        view="full",
        show_seam=False,
        show_cut_plane=True,
        cut_plane_h=st.session_state.metrics.get('h_cut', None)
    )
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.subheader("3D Model")
    fig_3d = create_3d_plot(
        st.session_state.df,
        metrics=st.session_state.metrics,
        title="Droplet 3D Model"
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    st.header("💾 Export")
    
    col_exp1, col_exp2, col_exp3, col_exp5 = st.columns(4)
    
    with col_exp1:
        def gen_stl():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                if export_to_stl(st.session_state.df, tmp.name, metrics=st.session_state.metrics):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        stl_data = gen_stl()
        if stl_data:
            st.download_button("📥 Download STL", data=stl_data, file_name="droplet.stl", mime="application/octet-stream", use_container_width=True)
    
    with col_exp2:
        def gen_dxf():
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                if export_to_dxf(st.session_state.df, tmp.name, metrics=st.session_state.metrics):
                    with open(tmp.name, 'rb') as f:
                        return f.read()
            return None
        
        dxf_data = gen_dxf()
        if dxf_data:
            st.download_button("📥 Download DXF", data=dxf_data, file_name="droplet.dxf", mime="application/dxf", use_container_width=True)

    with col_exp3:
        if st.button("📄 Generate PDF (A3)", use_container_width=True):
            pdf_bytes = None
            err = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    export_to_pdf(st.session_state.df, st.session_state.metrics, tmp.name, physical_params=st.session_state.physical_params)
                    with open(tmp.name, 'rb') as f:
                        pdf_bytes = f.read()
            except Exception as e:
                err = str(e)
            if pdf_bytes:
                st.download_button("📥 Download PDF (A3)", data=pdf_bytes, file_name="droplet_a3.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.error(f"PDF genereren mislukt. Controleer of reportlab is geïnstalleerd. Fout: {err}")

    with col_exp5:
        def gen_3dm():
            import os, tempfile
            fd, tmp_path = tempfile.mkstemp(suffix='.3dm')
            os.close(fd)  # voorkom Windows-lock tijdens schrijven door rhino3dm
            ok, err = export_to_3dm(st.session_state.df, tmp_path, metrics=st.session_state.metrics)
            if ok and os.path.exists(tmp_path):
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return data, None
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return None, err
        if st.button("🦏 Download 3DM", use_container_width=True):
            data3dm, err = gen_3dm()
            if data3dm:
                st.download_button("📥 3DM", data=data3dm, file_name="droplet.3dm", mime="application/octet-stream", use_container_width=True)
            else:
                st.error(f"3DM export mislukt. {err or 'Controleer of rhino3dm is geïnstalleerd.'}")

else:
    st.info("👆 Set parameters and click 'Compute Droplet' to begin.")

st.markdown("---")
st.header("🖥️ Console")
col_log1, col_log2 = st.columns([1, 0.2])
with col_log2:
    if st.button("Clear logs", use_container_width=True):
        clear_console_logs()
logs = "\n".join(get_console_logs())
st.code(logs or "(no logs yet)")
