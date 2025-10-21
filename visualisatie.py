"""
Visualisatie functies voor druppelvormen met Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np


def create_2d_plot(df: pd.DataFrame, metrics: dict | None = None, title: str = "Druppelvorm 2D Doorsnede") -> go.Figure:
    """
    Maak interactieve 2D doorsnede plot met Plotly.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
        title: Titel voor de plot
    
    Returns:
        Plotly Figure object
    """
    # Zorg dat x_shifted bestaat
    if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
        # Centreer rond het midden: (min + max) / 2
        x_min = df['x-x_0'].min()
        x_max = df['x-x_0'].max()
        center = (x_min + x_max) / 2
        df['x_shifted'] = df['x-x_0'] - center
    
    # Filter geldige data
    df_valid = df.dropna(subset=['x_shifted', 'h'])
    
    if df_valid.empty:
        # Lege plot als er geen data is
        fig = go.Figure()
        fig.add_annotation(
            text="Geen geldige data om te visualiseren",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Sorteer op hoogte voor mooie lijnen
    df_sorted = df_valid.sort_values('h')
    
    # Maak de plot
    fig = go.Figure()
    
    # Voeg lijn toe voor profiel
    fig.add_trace(go.Scatter(
        x=df_sorted['x_shifted'],
        y=df_sorted['h'],
        mode='lines+markers',
        name='Druppelprofiel',
        line=dict(color='red', width=3),
        marker=dict(size=6, color='darkred', opacity=0.8),
        hovertemplate='<b>Radius:</b> %{x:.4f} m<br><b>Hoogte:</b> %{y:.4f} m<extra></extra>'
    ))

    # Optionele seam- en kraag/donut-weergave op basis van metrics
    if metrics is None:
        metrics = {}
    try:
        seam_h = float(metrics.get('h_seam_eff', 0.0))
        if seam_h > 0:
            fig.add_hline(y=seam_h, line=dict(color='#60a5fa', width=1.5), annotation_text='Seam/top', annotation_position='top left')
    except Exception:
        pass

    # Donut/torus in 2D (twee cirkels links/rechts)
    try:
        R_major = float(metrics.get('torus_R_major', 0.0))
        r_top = float(metrics.get('torus_r_top', 0.0))
        delta_h = float(metrics.get('delta_h_water', 0.0))
        if R_major > 0 and r_top > 0 and seam_h > 0:
            ring_z = seam_h - delta_h
            zc_top = ring_z + r_top  # donut net boven de ring
            th = np.linspace(0, 2*np.pi, 200)
            # Links en rechts
            x_left = -R_major + r_top * np.cos(th)
            z_left = zc_top + r_top * np.sin(th)
            x_right = R_major + r_top * np.cos(th)
            z_right = zc_top + r_top * np.sin(th)
            fig.add_trace(go.Scatter(x=x_left, y=z_left, mode='lines', name='Kraag (links)', line=dict(color='purple', width=1.5)))
            fig.add_trace(go.Scatter(x=x_right, y=z_right, mode='lines', name='Kraag (rechts)', line=dict(color='purple', width=1.5)))
    except Exception:
        pass
    
    # Layout configuratie
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="x (m) - Radius vanaf centrum",
        yaxis_title="h (m) - Hoogte",
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Gelijke aspect ratio voor realistische weergave
    x_range = df_sorted['x_shifted'].max() - df_sorted['x_shifted'].min()
    y_range = df_sorted['h'].max() - df_sorted['h'].min()
    
    # Zorg voor wat padding
    padding = 0.1
    x_center = (df_sorted['x_shifted'].max() + df_sorted['x_shifted'].min()) / 2
    y_center = (df_sorted['h'].max() + df_sorted['h'].min()) / 2
    
    max_range = max(x_range, y_range)
    
    fig.update_xaxes(
        range=[x_center - max_range * (0.5 + padding), x_center + max_range * (0.5 + padding)],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    
    fig.update_yaxes(
        range=[y_center - max_range * (0.5 + padding), y_center + max_range * (0.5 + padding)],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        scaleanchor="x",
        scaleratio=1
    )
    
    return fig


def create_3d_plot(df: pd.DataFrame, metrics: dict | None = None, title: str = "Druppelvorm 3D Model", 
                   n_theta: int = 60) -> go.Figure:
    """
    Maak interactieve 3D rotatiemodel met Plotly.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
        title: Titel voor de plot
        n_theta: Aantal hoeken voor rotatie (hogere waarde = gladdere vorm)
    
    Returns:
        Plotly Figure object
    """
    # Zorg dat x_shifted bestaat
    if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
        x_max = df['x-x_0'].max()
        df['x_shifted'] = df['x-x_0'] - x_max
    
    # Filter geldige data
    df_valid = df.dropna(subset=['x_shifted', 'h'])
    
    if df_valid.empty:
        # Lege plot als er geen data is
        fig = go.Figure()
        fig.add_annotation(
            text="Geen geldige data om te visualiseren",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Sorteer op hoogte
    df_sorted = df_valid.sort_values('h')
    
    # Zorg dat x_shifted gecentreerd is rond 0 (symmetrisch)
    x_shifted_vals = df_sorted['x_shifted'].values
    x_center = (x_shifted_vals.max() + x_shifted_vals.min()) / 2
    x_shifted_centered = x_shifted_vals - x_center
    
    # Maak rotatie-oppervlak
    r = np.abs(x_shifted_centered)
    z = df_sorted['h'].to_numpy()
    theta = np.linspace(0, 2 * np.pi, n_theta)
    
    # Maak meshgrid
    Z = np.tile(z[:, None], (1, n_theta))
    R = np.tile(r[:, None], (1, n_theta))
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    
    # Maak 3D surface plot, sluit top af met een dunne extra schijf indien vlakke top aanwezig is
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Blues',
        showscale=True,
        hovertemplate='<b>X:</b> %{x:.4f} m<br><b>Y:</b> %{y:.4f} m<br><b>Z:</b> %{z:.4f} m<extra></extra>'
    )])

    # Indien bovenaan een vlak niveau bestaat, teken extra "deksel" exact op de naad
    # Detectie: groepeer op afgeronde hoogte en kies het hoogste niveau met >= 10 punten
    z_rounded = np.round(z, 6)
    unique_vals, counts = np.unique(z_rounded, return_counts=True)
    candidate_levels = unique_vals[counts >= 10]
    if candidate_levels.size > 0:
        h_top = float(np.max(candidate_levels))
        sel = np.isclose(z, h_top)
        if np.any(sel):
            # Radius op dit niveau uit gecentreerde r-vector
            r_top = float(np.max(r[sel]))
            if r_top > 0:
                rr = np.linspace(0, r_top, 2)
                TT, RR = np.meshgrid(theta, rr, indexing='xy')
                Xcap = RR * np.cos(TT)
                Ycap = RR * np.sin(TT)
                # Plaats de deksel exact op de naad (minus kleine epsilon om z-fighting/zweven te voorkomen)
                Zcap = np.full_like(Xcap, h_top - 1e-6)
                fig.add_surface(x=Xcap, y=Ycap, z=Zcap, colorscale='Blues', showscale=False, opacity=0.95)
    
    # Voeg 3D torus toe indien aanwezig
    if metrics is None:
        metrics = {}
    try:
        R_major = float(metrics.get('torus_R_major', 0.0))
        r_top = float(metrics.get('torus_r_top', 0.0))
        delta_h = float(metrics.get('delta_h_water', 0.0))
        seam_h = float(metrics.get('h_seam_eff', 0.0))
        if R_major > 0 and r_top > 0 and seam_h > 0:
            ring_z = seam_h - delta_h
            zc_top = ring_z + r_top
            u = np.linspace(0, 2*np.pi, 50)
            v = np.linspace(0, 2*np.pi, 30)
            U, V = np.meshgrid(u, v)
            X_t = (R_major + r_top*np.cos(V)) * np.cos(U)
            Y_t = (R_major + r_top*np.cos(V)) * np.sin(U)
            Z_t = zc_top + r_top*np.sin(V)
            fig.add_surface(x=X_t, y=Y_t, z=Z_t, colorscale='Purples', showscale=False, opacity=0.6)
    except Exception:
        pass

    # Layout configuratie
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m) - Hoogte",
            # Forceer gelijke schaalverdeling op X/Y/Z om vertekening te voorkomen
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=700,
        template='plotly_white'
    )
    
    return fig


def create_metrics_table(metrics: dict, physical_params: dict) -> str:
    """
    Creëer een geformatteerde HTML tabel met metrieken.
    
    Parameters:
        metrics: Dictionary met druppel metrieken
        physical_params: Dictionary met fysische parameters
    
    Returns:
        HTML string voor tabel weergave
    """
    # Zorg voor veilige defaults
    if metrics is None:
        metrics = {}
    if physical_params is None:
        physical_params = {}
    
    # Helper functie voor veilige waarden
    def safe_get(d, key, default=0):
        try:
            if d is None:
                return default
            value = d.get(key, default)
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError, AttributeError):
            return default
    
    # Haal alle waarden op met veilige defaults
    gamma_s = safe_get(physical_params, 'gamma_s', 0)
    rho = safe_get(physical_params, 'rho', 0)
    g = safe_get(physical_params, 'g', 0)
    kappa = safe_get(physical_params, 'kappa', 0)
    H = safe_get(physical_params, 'H', 0)
    volume = safe_get(metrics, 'volume', 0)
    max_height = safe_get(metrics, 'max_height', 0)
    max_diameter = safe_get(metrics, 'max_diameter', 0)
    bottom_diameter = safe_get(metrics, 'bottom_diameter', 0)
    top_diameter = safe_get(metrics, 'top_diameter', 0)
    
    # Bouw HTML direct zonder .format() om KeyError te voorkomen
    html = f"""
    <style>
        .metrics-table {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        .metrics-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table td:first-child {{
            font-weight: bold;
            color: #2c3e50;
            width: 40%;
        }}
        .metrics-table td:last-child {{
            color: #34495e;
            text-align: right;
        }}
        .section-header {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
            padding: 10px;
        }}
    </style>
    <table class="metrics-table">
        <tr><td colspan="2" class="section-header">Fysische Parameters</td></tr>
        <tr><td>γₛ (Oppervlaktespanning)</td><td>{gamma_s:.1f} N/m</td></tr>
        <tr><td>ρ (Dichtheid)</td><td>{rho:.1f} kg/m³</td></tr>
        <tr><td>g (Zwaartekracht)</td><td>{g:.2f} m/s²</td></tr>
        <tr><td>κ (Kappa)</td><td>{kappa:.6f} m⁻¹</td></tr>
        <tr><td>H (Karakteristieke hoogte)</td><td>{H:.6f} m</td></tr>
        
        <tr><td colspan="2" class="section-header">Druppel Metrieken</td></tr>
        <tr><td>Volume</td><td>{volume:.6f} m³</td></tr>
        <tr><td>Maximale hoogte</td><td>{max_height:.6f} m</td></tr>
        <tr><td>Maximale diameter</td><td>{max_diameter:.6f} m</td></tr>
        <tr><td>Basis diameter (bodem)</td><td>{bottom_diameter:.6f} m</td></tr>
        <tr><td>Afkap diameter</td><td>{top_diameter:.6f} m</td></tr>
    </table>
    """
    
    return html

