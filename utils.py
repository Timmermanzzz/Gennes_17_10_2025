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
    
    # Zet rechterkant op 0 (zoals in het vergelijkbare project)
    x_max = df['x-x_0'].max()
    df['x_shifted'] = df['x-x_0'] - x_max

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
    
    # Maximale diameter (volledige vorm): 2 * grootste radius over alle hoogtes
    # Neem de grootste absolute waarde van x_shifted
    widest_radius = float(np.max(np.abs(df_valid['x_shifted'])))
    max_diameter = 2.0 * widest_radius

    # Basis diameter (volledige vorm): diameter op de laagste hoogte (voet)
    bottom_diameter = calculate_diameter_at_height(df_valid, min_height)

    # Top diameter: diameter op de hoogste hoogte (opening na eventuele afkap)
    top_diameter = calculate_diameter_at_height(df_valid, max_height)
    
    return {
        'volume': volume,
        'max_height': max_height,
        'min_height': min_height,
        'max_diameter': max_diameter,
        'bottom_diameter': bottom_diameter,  # = breedste plek (basis)
        'top_diameter': top_diameter  # = opening (afkap)
    }


def find_height_for_diameter(df: pd.DataFrame, target_diameter: float) -> float:
    """
    Vind de hoogte van het EERSTE/BOVENSTE punt waar een bepaalde diameter voorkomt.
    Dit is belangrijk voor afkappen - we willen de opening op een specifieke diameter instellen.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
        target_diameter: Gewenste diameter in meters
    
    Returns:
        Hoogte van het eerste/bovenste punt waar deze diameter voorkomt
    """
    # Zorg dat x_shifted bestaat
    if 'x_shifted' not in df.columns:
        df = shift_x_coordinates(df)
    
    df_valid = df.dropna(subset=['x_shifted', 'h']).copy()
    if df_valid.empty:
        return np.nan
    
    # Sorteer op hoogte
    df_valid = df_valid.sort_values('h')
    
    # Bereken diameters op alle hoogtes
    df_valid['diameter'] = 2.0 * np.abs(df_valid['x_shifted'])
    
    # Tolerance voor matching
    tolerance = target_diameter * 0.02  # 2% tolerantie
    
    # Vind ALLE punten die dicht bij de target diameter liggen
    points_near_diameter = df_valid[np.abs(df_valid['diameter'] - target_diameter) < tolerance]
    
    if points_near_diameter.empty:
        return np.nan
    
    # Pak het EERSTE/BOVENSTE punt (laagste h-waarde, want h loopt van hoog naar laag)
    return points_near_diameter['h'].min()


def _make_df_with_cuts(gamma_s: float, rho: float, g: float,
                       cut_percentage: int = 0,
                       cut_diameter: float = 0.0) -> pd.DataFrame:
    """
    Maak een druppel DataFrame met eventuele afkap (percentage of diameter).
    """
    from solver import generate_droplet_shape

    if cut_diameter and cut_diameter > 0:
        df_full = generate_droplet_shape(gamma_s, rho, g, cut_percentage=0)
        h_cut = find_height_for_diameter(df_full, float(cut_diameter))
        if np.isnan(h_cut):
            return df_full
        df = df_full[df_full['h'] <= h_cut].copy()
        # Voeg vlakke top toe (alleen voor visualisatie/export)
        target_radius = float(cut_diameter) / 2.0
        n_points = 30
        x_shifted_vals = np.linspace(-target_radius, target_radius, n_points)
        x_max_current = df['x-x_0'].max() if 'x-x_0' in df.columns else 0.0
        top_points = pd.DataFrame({
            'B': 1.0,
            'C': 1.0,
            'z': 0.0,
            'x-x_0': x_shifted_vals + x_max_current,
            'x_shifted': x_shifted_vals,
            'h': h_cut
        })
        df = pd.concat([df, top_points], ignore_index=True)
        return df

    # Percentage-afkap: ondersteund in solver
    return generate_droplet_shape(gamma_s, rho, g, cut_percentage=int(cut_percentage or 0))


def solve_gamma_for_volume(target_volume: float,
                           rho: float,
                           g: float,
                           cut_percentage: int = 0,
                           cut_diameter: float = 0.0,
                           gamma_min: float = 100.0,
                           gamma_max: float = 1_000_000.0,
                           max_iter: int = 30,
                           rel_tol: float = 1e-3) -> tuple:
    """
    Zoek gamma_s zodat het volume van de (eventueel afgekapte) druppel
    gelijk wordt aan target_volume. Retourneert (gamma_opt, df_opt, volume_opt).
    """
    def volume_for_gamma(gamma_val: float):
        df_local = _make_df_with_cuts(gamma_val, rho, g, cut_percentage, cut_diameter)
        vol = calculate_volume(df_local)
        return vol, df_local

    vol_min, df_min = volume_for_gamma(gamma_min)
    vol_max, df_max = volume_for_gamma(gamma_max)

    # Bracketing uitbreiden indien nodig
    expand = 0
    while not (vol_min <= target_volume <= vol_max) and expand < 10:
        if target_volume < vol_min:
            gamma_min = max(1.0, gamma_min * 0.1)
            vol_min, df_min = volume_for_gamma(gamma_min)
        else:
            gamma_max = gamma_max * 10.0
            vol_max, df_max = volume_for_gamma(gamma_max)
        expand += 1

    if not (vol_min <= target_volume <= vol_max):
        # kies dichtstbijzijnde
        if abs(vol_min - target_volume) <= abs(vol_max - target_volume):
            return gamma_min, df_min, vol_min
        return gamma_max, df_max, vol_max

    left, right = gamma_min, gamma_max
    best_df, best_vol, best_gamma = df_min, vol_min, gamma_min
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        vol_mid, df_mid = volume_for_gamma(mid)
        if abs(vol_mid - target_volume) < abs(best_vol - target_volume):
            best_df, best_vol, best_gamma = df_mid, vol_mid, mid
        if target_volume > 0 and abs(vol_mid - target_volume)/target_volume < rel_tol:
            return mid, df_mid, vol_mid
        if vol_mid < target_volume:
            left = mid
        else:
            right = mid

    # eindresultaat
    vol_opt, df_opt = volume_for_gamma(best_gamma)
    return best_gamma, df_opt, vol_opt


def solve_gamma_for_height(target_height: float,
                           rho: float,
                           g: float,
                           cut_percentage: int = 0,
                           cut_diameter: float = 0.0,
                           gamma_min: float = 100.0,
                           gamma_max: float = 1_000_000.0,
                           max_iter: int = 30,
                           rel_tol: float = 1e-3) -> tuple:
    """
    Zoek gamma_s zodat de maximale hoogte (na afkap) gelijk wordt aan target_height.
    Retourneert (gamma_opt, df_opt, height_opt).
    """
    def height_for_gamma(gamma_val: float):
        df_local = _make_df_with_cuts(gamma_val, rho, g, cut_percentage, cut_diameter)
        if df_local is None or df_local.empty:
            return np.nan, df_local
        h = float(df_local['h'].max())
        return h, df_local

    h_min, df_min = height_for_gamma(gamma_min)
    h_max, df_max = height_for_gamma(gamma_max)

    # Bracketing uitbreiden indien nodig
    expand = 0
    while not (h_min <= target_height <= h_max) and expand < 10:
        if target_height < h_min:
            gamma_min = max(1.0, gamma_min * 0.1)
            h_min, df_min = height_for_gamma(gamma_min)
        else:
            gamma_max = gamma_max * 10.0
            h_max, df_max = height_for_gamma(gamma_max)
        expand += 1

    if not (h_min <= target_height <= h_max):
        # kies dichtstbijzijnde
        if abs(h_min - target_height) <= abs(h_max - target_height):
            return gamma_min, df_min, h_min
        return gamma_max, df_max, h_max

    left, right = gamma_min, gamma_max
    best_df, best_h, best_gamma = df_min, h_min, gamma_min
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        h_mid, df_mid = height_for_gamma(mid)
        if abs(h_mid - target_height) < abs(best_h - target_height):
            best_df, best_h, best_gamma = df_mid, h_mid, mid
        if target_height > 0 and abs(h_mid - target_height)/target_height < rel_tol:
            return mid, df_mid, h_mid
        # hoogte stijgt met gamma
        if h_mid < target_height:
            left = mid
        else:
            right = mid

    h_opt, df_opt = height_for_gamma(best_gamma)
    return best_gamma, df_opt, h_opt


def compute_torus_from_head(opening_diameter: float,
                            head_total: float,
                            wall_thickness: float = 0.2,
                            safety_freeboard: float = 0.0) -> dict:
    """
    Bepaal een eenvoudige torusgeometrie voor de kraag op basis van opening en gewenste waterhoogte.
    Aannames: torus buiten op ring, kleine straal r_top; waterkanaal r_water = max(0, r_top - wall_thickness).
    head_total = delta_h_water + safety_freeboard.
    Retourneert: R_major, r_top, r_water, water_volume (m^3) als grove schatting (ringvormig kanaal).
    """
    R_major = opening_diameter / 2.0
    # Kies r_top zodat er voldoende hoogte is voor head_total boven ring; eenvoudige keuze:
    r_top = max(head_total, wall_thickness * 2.0) / 2.0 + wall_thickness  # marge
    r_water = max(0.0, r_top - wall_thickness)
    # Schatting waterkanaal-volume als dunne ring: V ≈ 2π^2 R_major r_water^2
    water_volume = 2.0 * (np.pi ** 2) * R_major * (r_water ** 2)
    return {
        'R_major': R_major,
        'r_top': r_top,
        'r_water': r_water,
        'head_total': head_total,
        'water_volume': water_volume
    }



# =============================
# Methode 2 helpers (krommingsmatching)
# =============================

def _fit_local_quadratic(h_vals: np.ndarray, r_vals: np.ndarray, h0: float) -> tuple:
    """
    Pas een lokale kwadratische fit r(h) = a h^2 + b h + c rond h0 toe en
    retourneer (r, r', r'') geëvalueerd op h0.
    """
    if len(h_vals) >= 3:
        # Center de data numeriek voor betere conditie
        h_shift = h_vals - h0
        coeffs = np.polyfit(h_shift, r_vals, 2)  # a, b, c in verschoven coördinaten
        a, b, c = coeffs
        r0 = a * 0.0**2 + b * 0.0 + c
        r1 = 2.0 * a * 0.0 + b
        r2 = 2.0 * a
        return float(r0), float(r1), float(r2)
    # Fallbacks
    if len(h_vals) == 2:
        # Lineaire schatting voor r, afgeleide; r'' ≈ 0
        (h1, h2), (r1v, r2v) = h_vals, r_vals
        if np.isclose(h2, h1):
            return float(r1v), 0.0, 0.0
        slope = (r2v - r1v) / (h2 - h1)
        r_at_h0 = r1v + slope * (h0 - h1)
        return float(r_at_h0), float(slope), 0.0
    # Onvoldoende data
    return float(r_vals[0]) if len(r_vals) else 0.0, 0.0, 0.0


def estimate_mean_curvature_at_height(df: pd.DataFrame, height: float, k_neighbors: int = 7) -> float:
    """
    Schat de gemiddelde kromming H = (k1 + k2)/2 van de as-ronde vorm r(h)
    op een specifieke hoogte met lokale polynoom-fit.

    - k1 (meridionaal): r'' / (1 + r'^2)^(3/2)
    - k2 (circulair):  1 / (r * sqrt(1 + r'^2))

    Returns H (1/m). Absolute waarde wordt teruggegeven.
    """
    if df is None or df.empty:
        return np.nan
    if 'x_shifted' not in df.columns:
        df = shift_x_coordinates(df.copy())

    # Sorteer en kies k dichtstbijzijnde punten rond gewenste hoogte
    df_sorted = df.dropna(subset=['x_shifted', 'h']).sort_values('h')
    if df_sorted.empty:
        return np.nan
    df_sorted['dist'] = np.abs(df_sorted['h'] - float(height))
    df_local = df_sorted.nsmallest(max(3, min(k_neighbors, len(df_sorted))), 'dist')

    h_vals = df_local['h'].to_numpy(dtype=float)
    r_vals = np.abs(df_local['x_shifted'].to_numpy(dtype=float))
    r0, r1, r2 = _fit_local_quadratic(h_vals, r_vals, float(height))

    denom = np.sqrt(1.0 + r1 * r1)
    if denom <= 0.0 or r0 <= 1e-12:
        return np.nan
    k1 = r2 / (denom ** 3)
    k2 = 1.0 / (r0 * denom)
    H = 0.5 * (k1 + k2)
    return float(abs(H))


def curvature_from_head(delta_h: float, rho: float, g: float, gamma_s: float) -> float:
    """
    Equivalentie tussen drukhoofd en mean curvature via Young–Laplace.
    Δp = ρ g Δh = 2 γₛ H ⇒ H = ρ g Δh / (2 γₛ)
    """
    if gamma_s <= 0:
        return np.nan
    return float((rho * g * max(0.0, delta_h)) / (2.0 * gamma_s))


def delta_h_from_curvature(H_target: float, rho: float, g: float, gamma_s: float) -> float:
    """
    Omgekeerde relatie: Δh = 2 γₛ H_target / (ρ g)
    """
    if rho <= 0 or g <= 0:
        return np.nan
    return float((2.0 * gamma_s * max(0.0, H_target)) / (rho * g))
