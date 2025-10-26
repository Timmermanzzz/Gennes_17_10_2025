"""
Visualisation utilities for droplet shapes with Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np


def create_2d_plot(
    df: pd.DataFrame,
    metrics: dict | None = None,
    title: str = "Droplet 2D Cross-section",
    view: str = "full",
    show_seam: bool = False,
    show_cut_plane: bool = False,
    cut_plane_h: float | None = None,
    show_torus_right: bool = False,
) -> go.Figure:
    """
    Create an interactive 2D cross-section plot with Plotly.
    
    Parameters:
        df: DataFrame containing droplet data (must include 'x_shifted' and 'h')
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    # Gebruik altijd een lokale gecentreerde as voor plotten om
    # inconsistenties tussen 'x-x_0' en vooraf berekende 'x_shifted' te vermijden.
    if 'x-x_0' in df.columns:
        # Plaats rechterrand (max x) op 0: x_plot = x - x_max
        x_max = df['x-x_0'].max()
        x_plot = df['x-x_0'] - x_max
    elif 'x_shifted' in df.columns:
        # Fallback: verschuif bestaande kolom zodat rechterrand op 0 ligt
        x_max = df['x_shifted'].max()
        x_plot = df['x_shifted'] - x_max
    else:
        # Geen bruikbare x-informatie
        fig = go.Figure()
        fig.add_annotation(
            text="No x-coordinates available for plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Filter geldige data
    df_all = pd.DataFrame({'x_plot': x_plot, 'h': df['h']})
    df_valid = df_all.dropna(subset=['x_plot', 'h'])
    
    if df_valid.empty:
        # Lege plot als er geen data is
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data to visualise",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Sorteer op hoogte voor mooie lijnen
    # Filter half-view indien gevraagd
    if view == "half-left":
        df_valid = df_valid[df_valid['x_plot'] <= 0]
    elif view == "half-right":
        df_valid = df_valid[df_valid['x_plot'] >= 0]

    # Als er een expliciete afkaphoogte is, splits de top uit zodat de polyline niet
    # via verticale segmenten naar de vlakke top gaat
    cp_h_try = None
    try:
        cp_h_try = float(cut_plane_h if cut_plane_h is not None else metrics.get('h_cut', 0.0))
    except Exception:
        cp_h_try = None

    # Gebruik x_plot direct; door de definitie x_plot = x - x_max ligt de rechterrand standaard op 0.

    df_body = df_valid
    df_top = pd.DataFrame(columns=df_valid.columns)
    if show_cut_plane and cp_h_try and cp_h_try > 0:
        eps = 1e-4
        # Body: strikt onder de afkaphoogte
        df_body = df_valid[df_valid['h'] < (cp_h_try - eps)].copy()
        # Bepaal de topradius R_top uit metrics of uit punten op de afkaphoogte
        R_top = 0.0
        try:
            R_top = float(metrics.get('top_diameter', 0.0)) / 2.0 if metrics else 0.0
        except Exception:
            R_top = 0.0
        # Fallback: haal radius en rechterrand uit data rond afkaphoogte
        band = df_valid[(df_valid['h'] >= cp_h_try - eps) & (df_valid['h'] <= cp_h_try + eps)]
        if not band.empty:
            try:
                x_right_edge_band = float(band['x_plot'].max())
                x_left_edge_band = float(band['x_plot'].min())
                # Radius = halve afstand (bandbreedte is diameter)
                R_est = max(0.0, 0.5 * (x_right_edge_band - x_left_edge_band))
                R_top = max(R_top, R_est)
            except Exception:
                pass
        # Construeer de vlakke top als aparte trace verankerd aan de rechterrand
        if R_top > 0:
            try:
                x_right_edge = float(band['x_plot'].max()) if not band.empty else 0.0
            except Exception:
                x_right_edge = 0.0
            # Teken de top links van de rechterrand: [x_right - R_top, x_right]
            x_top = np.linspace(x_right_edge - R_top, x_right_edge, 60)
            df_top = pd.DataFrame({'x_plot': x_top, 'h': np.full_like(x_top, cp_h_try)})

    # Teken body in oplopende hoogte (voorkomt horizontale overshoots)
    df_sorted = df_body.sort_values('h')
    df_top_sorted = df_top.sort_values('x_plot') if not df_top.empty else df_top
    
    # Maak de plot
    fig = go.Figure()
    
    # Add profile line
    fig.add_trace(go.Scatter(
        x=df_sorted['x_plot'],
        y=df_sorted['h'],
        mode='lines',
        name='Droplet profile',
        line=dict(color='red', width=3),
        hovertemplate='<b>Radius:</b> %{x:.4f} m<br><b>Height:</b> %{y:.4f} m<extra></extra>'
    ))

    if df_top_sorted is not None and not df_top_sorted.empty:
        fig.add_trace(go.Scatter(
            x=df_top_sorted['x_plot'],
            y=df_top_sorted['h'],
            mode='lines',
            name='Droplet profile (top)',
            line=dict(color='red', width=3),
            hovertemplate='<b>Radius:</b> %{x:.4f} m<br><b>Height:</b> %{y:.4f} m<extra></extra>'
        ))

    # Optionele seam- en kraag/donut-weergave op basis van metrics
    if metrics is None:
        metrics = {}
    try:
        seam_h = float(metrics.get('h_seam_eff', 0.0))
        if show_seam and seam_h > 0:
            fig.add_hline(y=seam_h, line=dict(color='#60a5fa', width=1.5), annotation_text='Opening', annotation_position='top left')
    except Exception:
        seam_h = 0.0

    # Optionele cut-plane (afkaphoogte) weergave
    try:
        cp_h = float(cut_plane_h if cut_plane_h is not None else metrics.get('h_cut', 0.0))
        if show_cut_plane and cp_h > 0:
            fig.add_hline(y=cp_h, line=dict(color='#94a3b8', width=1, dash='dot'), annotation_text='Cut height', annotation_position='top left')
    except Exception:
        pass

    # Waterline (blauwe streepjeslijn)
    try:
        h_water = float(metrics.get('h_waterline', metrics.get('h_seam_eff', metrics.get('h_cut', 0.0))))
        if h_water and h_water > 0:
            fig.add_hline(y=h_water, line=dict(color='#3b82f6', width=1.5, dash='dash'), annotation_text='Water level', annotation_position='bottom left')
    except Exception:
        pass

    # Donut/torus in 2D (twee cirkels links/rechts)
    try:
        R_major = float(metrics.get('torus_R_major', 0.0))
        r_top = float(metrics.get('torus_r_top', 0.0))
        if R_major > 0 and r_top > 0 and seam_h > 0:
            # Kraag op opening hoogte
            zc_top = seam_h + r_top  # donut net boven de opening
            th = np.linspace(0, 2*np.pi, 200)
            # Links en/of rechts afhankelijk van view; standaard alleen links tonen
            if view in ("full", "half-left"):
                x_left = -R_major + r_top * np.cos(th)
                z_left = zc_top + r_top * np.sin(th)
                fig.add_trace(go.Scatter(x=x_left, y=z_left, mode='lines', name='Collar (left)', line=dict(color='purple', width=1.5)))
            if show_torus_right and view in ("full", "half-right"):
                x_right = R_major + r_top * np.cos(th)
                z_right = zc_top + r_top * np.sin(th)
                fig.add_trace(go.Scatter(x=x_right, y=z_right, mode='lines', name='Collar (right)', line=dict(color='purple', width=1.5)))
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
        xaxis_title="x (m) - 0 at right edge",
        yaxis_title="h (m) - Height",
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
    x_range = df_sorted['x_plot'].max() - df_sorted['x_plot'].min()
    y_range = max(df_sorted['h'].max(), df_top_sorted['h'].max() if not df_top_sorted.empty else df_sorted['h'].max()) - \
              min(df_sorted['h'].min(), df_top_sorted['h'].min() if not df_top_sorted.empty else df_sorted['h'].min())
    
    # Zorg voor wat padding
    padding = 0.1
    x_center = (df_sorted['x_plot'].max() + df_sorted['x_plot'].min()) / 2
    y_max_all = max(df_sorted['h'].max(), df_top_sorted['h'].max() if not df_top_sorted.empty else df_sorted['h'].max())
    y_min_all = min(df_sorted['h'].min(), df_top_sorted['h'].min() if not df_top_sorted.empty else df_sorted['h'].min())
    y_center = (y_max_all + y_min_all) / 2
    
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
    Create an interactive 3D surface of revolution with Plotly.
    
    Parameters:
        df: DataFrame containing droplet data (must include 'x_shifted' and 'h')
        title: Plot title
        n_theta: Number of rotation angles (higher = smoother)
    
    Returns:
        Plotly Figure object
    """
    # Hergebruik dezelfde x-constructie als bij 2D: plaats rechterrand (max x) op 0
    if 'x-x_0' in df.columns:
        x_max = df['x-x_0'].max()
        x_plot = df['x-x_0'] - x_max
    elif 'x_shifted' in df.columns:
        x_max = df['x_shifted'].max()
        x_plot = df['x_shifted'] - x_max
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="No x-coordinates available for 3D plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

    df_all = pd.DataFrame({'x_plot': x_plot, 'h': df['h']})
    df_valid = df_all.dropna(subset=['x_plot', 'h'])

    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data to visualise",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

    # Sorteer op hoogte
    df_sorted = df_valid.sort_values('h')

    # Straal is de afstand tot de rotatie-as op x=0
    r = np.abs(df_sorted['x_plot'].to_numpy())
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
        hovertemplate='<b>X:</b> %{x:.4f} m<br><b>Y:</b> %{y:.4f} m<br><b>Z (height):</b> %{z:.4f} m<extra></extra>'
    )])

    # Opmerking: de dataset bevat al expliciete top-punten; het oppervlak sluit daardoor vanzelf.
    # We voegen daarom GEEN extra deksel meer toe, om dubbele schijven te voorkomen.
    
    # Voeg 3D torus toe indien aanwezig
    if metrics is None:
        metrics = {}
    try:
        R_major = float(metrics.get('torus_R_major', 0.0))
        r_top = float(metrics.get('torus_r_top', 0.0))
        seam_h = float(metrics.get('h_seam_eff', 0.0))
        if R_major > 0 and r_top > 0 and seam_h > 0:
            # Kraag op opening hoogte
            zc_top = seam_h + r_top
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
            zaxis_title="Z (m) - Height",
            # Forceer gelijke schaal per eenheid op X/Y/Z om vertekening te voorkomen
            aspectmode='data',
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
    Create a formatted HTML table with metrics.
    
    Parameters:
        metrics: Dictionary with droplet metrics
        physical_params: Dictionary with physical parameters
    
    Returns:
        HTML string for table rendering
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
    
    # Build HTML directly without .format() to avoid KeyError
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
        <tr><td colspan="2" class="section-header">Physical Parameters</td></tr>
        <tr><td>γₛ (Surface tension)</td><td>{gamma_s:.1f} N/m</td></tr>
        <tr><td>ρ (Density)</td><td>{rho:.1f} kg/m³</td></tr>
        <tr><td>g (Gravity)</td><td>{g:.2f} m/s²</td></tr>
        <tr><td>κ (Kappa)</td><td>{kappa:.6f} m⁻¹</td></tr>
        <tr><td>H (Characteristic height)</td><td>{H:.6f} m</td></tr>
        
        <tr><td colspan="2" class="section-header">Droplet Metrics</td></tr>
        <tr><td>Volume</td><td>{volume:.6f} m³</td></tr>
        <tr><td>Max height</td><td>{max_height:.6f} m</td></tr>
        <tr><td>Max diameter</td><td>{max_diameter:.6f} m</td></tr>
        <tr><td>Base diameter</td><td>{bottom_diameter:.6f} m</td></tr>
        <tr><td>Opening diameter</td><td>{top_diameter:.6f} m</td></tr>
    </table>
    """
    
    return html

