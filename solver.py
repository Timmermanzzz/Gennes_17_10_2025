"""
Young-Laplace solver voor druppelvorm berekeningen.
Gebaseerd op De Gennes natuurkundige principes.
"""

import numpy as np
import pandas as pd
import math


def calculate_kappa(rho: float, g: float, gamma_s: float) -> float:
    """
    Bereken de kappa parameter.
    
    Parameters:
        rho: Dichtheid (kg/m³)
        g: Gravitatieversnelling (m/s²)
        gamma_s: Oppervlaktespanningsparameter (N/m)
    
    Returns:
        kappa: Vormparameter (m⁻¹)
    """
    return np.sqrt(rho * g / gamma_s)


def calculate_H(kappa: float) -> float:
    """
    Bereken de karakteristieke hoogte H.
    
    Parameters:
        kappa: Vormparameter (m⁻¹)
    
    Returns:
        H: Karakteristieke hoogte (m)
    """
    return 2.0 / kappa


def _calculate_formula_local(z: float, kappa: float) -> float:
    """
    Implementeer de De Gennes formule voor druppelvorm.
    
    Parameters:
        z: Hoogteparameter
        kappa: Vormparameter
    
    Returns:
        x-coördinaat waarde
    """
    try:
        if kappa <= 0 or z <= 0:
            return np.nan
        
        acosh_input = 2.0 / (kappa * z)
        if acosh_input < 1.0:
            return np.nan
        
        sqrt_input = 1.0 - (z ** 2) / (4.0 / (kappa ** 2))
        if sqrt_input < 0:
            return np.nan
        
        term1 = (1.0 / kappa) * math.acosh(acosh_input)
        term2 = (2.0 / kappa) * math.sqrt(sqrt_input)
        
        return term1 - term2
    except:
        return np.nan


def generate_droplet_shape(gamma_s: float, rho: float, g: float, 
                           cut_percentage: float = 0.0, res_factor: float = 3.0) -> pd.DataFrame:
    """
    Genereer druppelvorm op basis van Young-Laplace vergelijking.
    
    Parameters:
        gamma_s: Oppervlaktespanningsparameter (N/m)
        rho: Dichtheid (kg/m³)
        g: Gravitatieversnelling (m/s²)
        cut_percentage: Percentage om van de top af te knippen (0-100)
        res_factor: Resolutie factor voor aantal punten (hogere waarde = meer punten)
    
    Returns:
        DataFrame met kolommen: ['B', 'C', 'z', 'x-x_0', 'h']
    """
    # Bereken fysische parameters
    kappa = calculate_kappa(rho, g, gamma_s)
    H = calculate_H(kappa)
    
    # Genereer input waarden (B parameter van 0.001 tot 1.0)
    input_b_waarden = [0.001]
    current_b = 0.001
    
    # Dynamische stap gebaseerd op resolutie factor
    base_step_1 = 0.009
    base_step_2 = 0.01
    
    step_1 = base_step_1 / res_factor
    step_2 = base_step_2 / res_factor
    
    while current_b <= 1.0:
        if current_b == 0.001:
            current_b += step_1
        else:
            current_b += step_2
        
        if current_b > 1.0:
            break
        input_b_waarden.append(round(current_b, 6))
    
    # Bereken druppelvorm punten
    data = []
    for input_b in input_b_waarden:
        column_c = np.sin(input_b * 0.5 * np.pi)
        z_coord = column_c * H
        
        # Bereken x-coördinaat met De Gennes formule
        x_min_x0_coord = _calculate_formula_local(z_coord, kappa)
        
        # Bereken hoogte (h = H - z)
        height_f = H - z_coord
        
        data.append([input_b, column_c, z_coord, x_min_x0_coord, height_f])
    
    # Maak DataFrame
    df = pd.DataFrame(data, columns=['B', 'C', 'z', 'x-x_0', 'h'])
    
    # Pas afkapping toe indien gevraagd
    if cut_percentage > 0:
        max_height = df['h'].max()
        cut_at_height = max_height * (1.0 - cut_percentage / 100.0)
        df_cut = df[df['h'] <= cut_at_height].copy()
        
        # Voeg punten toe op de afkap-hoogte voor vlakke bovenkant
        x_values_at_cut = df[df['h'] >= cut_at_height]['x-x_0'].values
        if len(x_values_at_cut) > 0:
            min_x_at_cut = np.min(x_values_at_cut)
            max_x_at_cut = np.max(x_values_at_cut)
            n_points = 10
            x_top = np.linspace(min_x_at_cut, max_x_at_cut, n_points)
            
            top_points = pd.DataFrame([{
                'B': 1.0,
                'C': 1.0,
                'z': H - cut_at_height,
                'x-x_0': x,
                'h': cut_at_height
            } for x in x_top])
            
            df_cut = pd.concat([df_cut, top_points], ignore_index=True)
        
        return df_cut
    
    return df


def get_physical_parameters(df: pd.DataFrame, gamma_s: float, rho: float, g: float) -> dict:
    """
    Bereken fysische parameters uit een druppelvorm DataFrame.
    
    Parameters:
        df: DataFrame met druppelvorm data
        gamma_s: Oppervlaktespanningsparameter (N/m)
        rho: Dichtheid (kg/m³)
        g: Gravitatieversnelling (m/s²)
    
    Returns:
        Dictionary met kappa en H waarden
    """
    kappa = calculate_kappa(rho, g, gamma_s)
    H = calculate_H(kappa)
    
    return {
        'kappa': kappa,
        'H': H,
        'gamma_s': gamma_s,
        'rho': rho,
        'g': g
    }

