"""
Visualisatie functies voor druppelvormen met Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np


def create_2d_plot(df: pd.DataFrame, title: str = "Druppelvorm 2D Doorsnede") -> go.Figure:
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
        line=dict(color='royalblue', width=2),
        marker=dict(size=4, color='darkblue', opacity=0.6),
        hovertemplate='<b>Radius:</b> %{x:.4f} m<br><b>Hoogte:</b> %{y:.4f} m<extra></extra>'
    ))
    
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


def create_3d_plot(df: pd.DataFrame, title: str = "Druppelvorm 3D Model", 
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
    
    # Maak rotatie-oppervlak
    r = np.abs(df_sorted['x_shifted'].to_numpy())
    z = df_sorted['h'].to_numpy()
    theta = np.linspace(0, 2 * np.pi, n_theta)
    
    # Maak meshgrid
    Z = np.tile(z[:, None], (1, n_theta))
    R = np.tile(r[:, None], (1, n_theta))
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    
    # Maak 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Blues',
        showscale=True,
        hovertemplate='<b>X:</b> %{x:.4f} m<br><b>Y:</b> %{y:.4f} m<br><b>Z:</b> %{z:.4f} m<extra></extra>'
    )])
    
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
        <tr><td>Top diameter</td><td>{top_diameter:.6f} m</td></tr>
    </table>
    """
    
    return html

