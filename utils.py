"""
Hulpfuncties voor druppelvorm berekeningen.
"""

import numpy as np
import pandas as pd
import bisect


def shift_x_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verschuif x-coördinaten zodat de rechterkant op x=0 ligt.
    
    Parameters:
        df: DataFrame met 'x-x_0' kolom
    
    Returns:
        DataFrame met toegevoegde 'x_shifted' kolom
    """
    if 'x-x_0' not in df.columns:
        return df
    
    # x_shifted is gewoon de negatieve waarde van x-x_0 (spiegelen om y-as)
    # Dit zorgt ervoor dat de linkerkant negatief is en rechterkant positief
    df['x_shifted'] = -df['x-x_0']

    return df


def calculate_diameter_at_height(df: pd.DataFrame, height: float, tolerance: float = 1e-6) -> float:
    """
    Bereken de diameter op een gegeven hoogte met lineaire interpolatie.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
        height: Hoogte waarop diameter berekend moet worden (m)
        tolerance: Tolerantie voor hoogte matching
    
    Returns:
        Diameter op de gespecificeerde hoogte (m)
    """
    # Zorg dat x_shifted bestaat
    if 'x_shifted' not in df.columns:
        df = shift_x_coordinates(df)
    
    df_valid = df.dropna(subset=['x_shifted', 'h']).copy()
    if df_valid.empty:
        return 0.0
    
    # Sorteer op hoogte
    df_valid = df_valid.sort_values('h')
    h_vals = df_valid['h'].values
    x_vals = df_valid['x_shifted'].values
    
    # Directe match binnen tolerantie: neem grootste radius (meest negatieve x_shifted)
    height_points = df_valid[np.abs(df_valid['h'] - height) < tolerance]
    if not height_points.empty:
        xmin = height_points['x_shifted'].min()
        return 2.0 * abs(xmin)
    
    min_h = h_vals.min()
    max_h = h_vals.max()
    
    # Randgevallen: onder of boven bereik
    if height <= min_h + 1e-12:
        xmin = df_valid[df_valid['h'] == min_h]['x_shifted'].min()
        return 2.0 * abs(xmin)
    
    if height >= max_h - 1e-12:
        xmax_row = df_valid[df_valid['h'] == max_h]
        xmin = xmax_row['x_shifted'].min() if not xmax_row.empty else df_valid['x_shifted'].min()
        return 2.0 * abs(xmin)
    
    # Zoek indices rond de doelhoogte
    idx = bisect.bisect_left(h_vals, height)
    
    # Bepaal exacte onderste en bovenste unieke hoogtes
    h_lower = h_vals[idx - 1]
    h_upper = h_vals[idx]
    
    if np.isclose(h_upper, h_lower):
        xmin = df_valid[df_valid['h'] == h_lower]['x_shifted'].min()
        return 2.0 * abs(xmin)
    
    # Radius op h_lower en h_upper: neem de grootste radius (meest negatieve x_shifted)
    xmin_lower = df_valid[df_valid['h'] == h_lower]['x_shifted'].min()
    xmin_upper = df_valid[df_valid['h'] == h_upper]['x_shifted'].min()
    r_lower = abs(xmin_lower)
    r_upper = abs(xmin_upper)
    
    # Lineaire interpolatie in r(h)
    t = (height - h_lower) / (h_upper - h_lower)
    r = r_lower + t * (r_upper - r_lower)
    
    return 2.0 * abs(r)


def calculate_volume(df: pd.DataFrame) -> float:
    """
    Bereken volume van een druppelvorm op basis van coördinaatpunten.
    Gebruikt numerieke integratie (trapeziumregel) op cirkelvormige segmenten.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
    
    Returns:
        Volume in kubieke meters (m³)
    """
    try:
        # Zorg dat x_shifted bestaat
        if 'x_shifted' not in df.columns:
            df = shift_x_coordinates(df)
        
        df_valid = df.dropna(subset=['x_shifted', 'h'])
        if len(df_valid) < 2:
            return np.nan
        
        # Sorteer op hoogte
        df_valid = df_valid.sort_values('h')
        
        x_vals = df_valid['x_shifted'].values
        h_vals = df_valid['h'].values
        
        # Bereken stralen (absolute waarde van x_shifted)
        r_vals = np.abs(x_vals)
        
        # Bereken oppervlaktes van cirkels op elke hoogte
        area_vals = np.pi * r_vals ** 2
        
        # Gebruik trapeziumregel voor integratie over hoogte
        volume = np.trapz(area_vals, h_vals)
        
        return abs(volume)
    except:
        return np.nan


def get_droplet_metrics(df: pd.DataFrame) -> dict:
    """
    Bereken alle belangrijke metrieken van een druppelvorm.
    
    Parameters:
        df: DataFrame met druppelvorm data
    
    Returns:
        Dictionary met metrieken: volume, max_height, max_diameter, 
        bottom_diameter, top_diameter
    """
    # Zorg dat x_shifted bestaat
    if 'x_shifted' not in df.columns:
        df = shift_x_coordinates(df)
    
    df_valid = df.dropna(subset=['x_shifted', 'h'])
    
    if df_valid.empty:
        return {
            'volume': 0.0,
            'max_height': 0.0,
            'max_diameter': 0.0,
            'bottom_diameter': 0.0,
            'top_diameter': 0.0
        }
    
    # Volume
    volume = calculate_volume(df)
    
    # Hoogtes
    min_height = df_valid['h'].min()
    max_height = df_valid['h'].max()
    
    # Maximum diameter (grootste radius punt)
    max_diameter = 2.0 * abs(df_valid['x_shifted'].min())
    
    # Bottom diameter (op minimale hoogte)
    bottom_diameter = calculate_diameter_at_height(df, min_height)
    
    # Top diameter (op maximale hoogte)
    top_diameter = calculate_diameter_at_height(df, max_height)
    
    return {
        'volume': volume,
        'max_height': max_height,
        'min_height': min_height,
        'max_diameter': max_diameter,
        'bottom_diameter': bottom_diameter,
        'top_diameter': top_diameter
    }

