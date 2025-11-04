"""
Young-Laplace solver voor druppelvorm berekeningen.
Gebaseerd op De Gennes natuurkundige principes.
"""

import numpy as np
import pandas as pd
import math
from typing import Tuple, Dict

try:
    # SciPy is in requirements.txt
    from scipy.optimize import minimize
except Exception:
    minimize = None


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
    
    # Voeg x_shifted toe (rechterkant op 0, zoals in vergelijkbaar project)
    x_max = df['x-x_0'].max()
    df['x_shifted'] = df['x-x_0'] - x_max
    
    # Pas afkapping toe indien gevraagd
    if cut_percentage > 0:
        # Afkapping op basis van percentage (bestaande logica)
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
                'x_shifted': x - x_max,
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


def solve_variational_droplet_fixed_cut(
    gamma_s: float,
    rho: float,
    g: float,
    opening_diameter: float,
    cut_height: float,
    num_points: int = 400,
    regularization_weight: float = 1e-6,
    max_iter: int = 500,
    tol: float = 1e-6
) -> Tuple[pd.DataFrame, Dict]:
    """
    Variationale methode: minimaliseer totale energie bij vaste snijhoogte en openingdiameter.

    Doel: E[r] = gamma_s * A[r] + rho * g * W[r]
    met A = 2π ∫ r * sqrt(1 + (dr/dz)^2) dz, W = π ∫ z * r^2 dz (zwaartepuntterm).

    Randvoorwaarden:
        r(0) = 0 (apex), r(cut_height) = opening_diameter / 2.

    Parameters:
        gamma_s: Oppervlaktespanning (N/m)
        rho: Dichtheid (kg/m³)
        g: Zwaartekracht (m/s²)
        opening_diameter: Openingdiameter (m)
        cut_height: Snijhoogte h_cut (m)
        num_points: Discretisatie (>= 3)
        regularization_weight: Gewicht voor tweede-differentie-regularisatie
        max_iter: Maximaal aantal iteraties voor optimizer
        tol: Convergentietolerantie

    Returns:
        (df, info):
            df: DataFrame met kolommen ['h', 'x-x_0', 'x_shifted']
            info: dict met extra metrics (energie, oppervlakte, volume, status)
    """
    # Validatie van invoer
    if minimize is None:
        raise RuntimeError("scipy.optimize.minimize niet beschikbaar")
    if opening_diameter <= 0 or cut_height <= 0:
        raise ValueError("opening_diameter en cut_height moeten > 0 zijn")
    if num_points < 3:
        num_points = 3

    R_open = opening_diameter / 2.0

    # Discretisatie in z (h in onze conventie)
    z = np.linspace(0.0, float(cut_height), int(num_points))
    dz = z[1] - z[0]

    # Vaste eindpunten (niet in de variabele vector): r0=0, rN=R_open
    r_fixed_start = 0.0
    r_fixed_end = R_open

    # Startprofiel: gladde sinus die aan randvoorwaarden voldoet
    # r(z) = R_open * sin( (pi/2) * (z/h_cut) )
    with np.errstate(invalid='ignore'):
        r0 = R_open * np.sin(0.5 * np.pi * (z / float(cut_height)))
    r0[0] = r_fixed_start
    r0[-1] = r_fixed_end

    # Optimaliseer alleen de binnenste punten
    def pack(r_full: np.ndarray) -> np.ndarray:
        return r_full[1:-1]

    def unpack(x_inner: np.ndarray) -> np.ndarray:
        r_full = np.zeros_like(z)
        r_full[0] = r_fixed_start
        r_full[-1] = r_fixed_end
        r_full[1:-1] = x_inner
        return r_full

    x0 = pack(r0)

    # Energie functioneel
    def energy(x_inner: np.ndarray) -> float:
        r = unpack(x_inner)

        # Forceer niet-negativiteit via zachte barrière (penalty), bounds doen de rest
        penalty_neg = np.sum(np.square(np.minimum(r, 0.0))) * 1e6

        dr = np.diff(r)
        # Oppervlakte (revolutie): 2π Σ r_bar * sqrt(1 + (dr/dz)^2) * dz
        r_bar = 0.5 * (r[:-1] + r[1:])
        slope = dr / dz
        seg_len = np.sqrt(1.0 + slope * slope)
        A = 2.0 * math.pi * np.sum(r_bar * seg_len) * dz

        # Zwaartepuntterm: π Σ z_bar * (r_i^2 + r_{i+1}^2)/2 * dz
        z_bar = 0.5 * (z[:-1] + z[1:])
        r_sq_bar = 0.5 * (r[:-1] * r[:-1] + r[1:] * r[1:])
        W = math.pi * np.sum(z_bar * r_sq_bar) * dz

        E = gamma_s * A + rho * g * W

        # Lichte gladheids-regularisatie (tweede verschil)
        if regularization_weight > 0.0:
            d2 = r[:-2] - 2.0 * r[1:-1] + r[2:]
            reg = regularization_weight * float(np.sum(d2 * d2))
        else:
            reg = 0.0

        return float(E + reg + penalty_neg)

    # Bounds: r_i >= 0 voor binnenpunten
    bounds = [(0.0, None) for _ in range(len(x0))]

    res = minimize(
        energy,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": float(tol)}
    )

    r_opt = unpack(res.x)

    # Afgeleide grootheden
    dr = np.diff(r_opt)
    r_bar = 0.5 * (r_opt[:-1] + r_opt[1:])
    slope = dr / dz
    seg_len = np.sqrt(1.0 + slope * slope)
    A = 2.0 * math.pi * np.sum(r_bar * seg_len) * dz
    z_bar = 0.5 * (z[:-1] + z[1:])
    r_sq_bar = 0.5 * (r_opt[:-1] * r_opt[:-1] + r_opt[1:] * r_opt[1:])
    W = math.pi * np.sum(z_bar * r_sq_bar) * dz
    E_total = gamma_s * A + rho * g * W
    # Volume: π ∫ r^2 dz
    V = math.pi * np.sum(r_sq_bar) * dz

    # Output-DF in bestaande conventie
    df = pd.DataFrame({
        'h': z,
        'x-x_0': r_opt
    })
    x_max = df['x-x_0'].max()
    df['x_shifted'] = df['x-x_0'] - x_max

    info = {
        'success': bool(res.success),
        'message': str(res.message),
        'nit': int(getattr(res, 'nit', -1)),
        'energy_total': float(E_total),
        'surface_area': float(A),
        'gravitational_term': float(rho * g * W),
        'volume': float(V),
        'opening_radius': float(R_open),
        'cut_height': float(cut_height)
    }

    return df, info


def solve_timoshenko_membrane(
    rho: float,
    g: float,
    N: float,
    top_pressure: float = None,
    head_d: float = None,
    phi_max_deg: float = 120.0,
    dx_initial: float = 0.01,
    dz_initial: float = 0.01,
    max_steps: int = 20000,
    adaptive: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Integreer de Timoshenko-membraanvergelijkingen (constant strength) voor een
    axiaal-symmetrische schaal onder hydrostatische druk.

    ODE-set (met u = sin(phi)):
        FASE 1 (x als onafhankelijke variabele, tot ~phi=90°):
            du/dx + u/x = (gamma * (d + z)) / N
            dz/dx = u / sqrt(1 - u^2)

        FASE 2 (z als onafhankelijke variabele, voorbij ~phi=90°):
            dx/dz = sqrt(1 - u^2) / u
            du/dz = [ (gamma * (d + z)) / N - u/x ] * sqrt(1 - u^2) / u

    Startcondities rond apex x -> 0 via lokale benadering met
        r1(0) = 2 N / (gamma * d)
        u ~ x / r1,  z ~ x^2 / (2 r1)

    Parameters:
        rho: dichtheid (kg/m^3)
        g: zwaartekracht (m/s^2)
        N: membraankracht per eenheid omtrek (N/m)
        top_pressure: druk aan de top (Pa); optioneel, alternatief voor head_d
        head_d: drukhoogte d aan de top (m); als None en top_pressure is gezet,
                wordt d = top_pressure / (rho * g)
        phi_max_deg: stop-increment wanneer phi deze waarde bereikt (veiligheidsmarge t.o.v. singulariteit u -> 1)
        dx_initial: startstap in x (m)
        max_steps: maximaal aantal integratiestappen
        adaptive: eenvoudige adaptieve stapgrootteschakeling

    Returns:
        (df, info):
            df kolommen: ['x', 'z', 'u', 'phi', 'dzdx', 'dudx', 'r1', 'r2']
            info: {'gamma':..., 'r1_apex':..., 'volume':..., 'stopped_reason':..., 'steps':...}
    """
    gamma = float(rho) * float(g)  # soortelijk gewicht (N/m^3)

    if head_d is None and top_pressure is None:
        raise ValueError("Geef 'head_d' (m) of 'top_pressure' (Pa) op.")
    if head_d is None and top_pressure is not None:
        head_d = float(top_pressure) / gamma
    if head_d is not None and head_d <= 0:
        raise ValueError("head_d moet > 0 zijn")
    if N <= 0:
        raise ValueError("N moet > 0 zijn")

    # Initiële kromtestraal aan de top en kleine-startbenadering
    r1_apex = (2.0 * float(N)) / (gamma * float(head_d))

    # Start iets weg van de singulariteit x=0
    x = float(max(dx_initial * 0.1, 1e-6))
    u = x / r1_apex
    z = (x * x) / (2.0 * r1_apex)

    phi_max = math.radians(float(phi_max_deg))
    u_switch = 1.0 - 1e-3  # schakel over rond 89.94°

    # Opslag
    xs: list = [0.0, x]
    zs: list = [0.0, z]
    us: list = [0.0, u]

    # RK4 helper voor du/dx, dz/dx
    def rhs(x_loc: float, z_loc: float, u_loc: float) -> Tuple[float, float]:
        # Bescherm tegen u_loc buiten (-1, 1)
        u_clamped = max(min(u_loc, 1.0 - 1e-9), -1.0 + 1e-9)
        # du/dx
        # Hydrostatic pressure head: p = gamma * (d + z)
        term = (gamma * (float(head_d) + z_loc)) / float(N)
        inv_x = 0.0 if x_loc <= 0.0 else (1.0 / x_loc)
        dudx = term - u_clamped * inv_x
        # dz/dx
        denom = math.sqrt(max(1.0 - u_clamped * u_clamped, 1e-16))
        dzdx = u_clamped / denom
        return dudx, dzdx

    dx = float(dx_initial)
    steps = 1
    stopped_reason = "switch_to_z"

    while steps < int(max_steps):
        # Huidige hoek
        u_clamped = max(min(u, 1.0 - 1e-9), -1.0 + 1e-9)
        phi = math.asin(u_clamped)
        if phi >= phi_max:
            break

        # Eenvoudige adaptieve stap: verklein indien dicht bij u -> 1
        if adaptive:
            safety = max(1e-3, 1.0 - abs(u_clamped))
            dx = min(dx_initial, 0.1 * safety)

        # Schakel naar fase 2 zodra we heel dicht bij verticaal zijn
        if abs(u_clamped) >= u_switch:
            stopped_reason = "switch_to_z"
            break

        # RK4 stap
        k1u, k1z = rhs(x, z, u)
        k2u, k2z = rhs(x + 0.5 * dx, z + 0.5 * dx * k1z, u + 0.5 * dx * k1u)
        k3u, k3z = rhs(x + 0.5 * dx, z + 0.5 * dx * k2z, u + 0.5 * dx * k2u)
        k4u, k4z = rhs(x + dx, z + dx * k3z, u + dx * k3u)

        u_next = u + (dx / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        z_next = z + (dx / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        x_next = x + dx

        # Guardrails
        if not (math.isfinite(u_next) and math.isfinite(z_next)):
            stopped_reason = "non_finite"
            break
        if abs(u_next) >= 1.0:
            stopped_reason = "switch_to_z"
            break
        if z_next < 0:
            stopped_reason = "z_negative"
            break

        x, z, u = x_next, z_next, u_next
        xs.append(x)
        zs.append(z)
        us.append(u)
        steps += 1

    # FASE 2: Integreer voorbij de equator met z als onafhankelijke variabele
    steps_z = 0
    if stopped_reason == "switch_to_z" and steps < int(max_steps):
        # Startwaarden iets weg van u=1
        u = max(min(us[-1], 1.0 - 1e-6), 1e-4)
        x = max(xs[-1], 1e-9)
        z = zs[-1]

        def rhs_z(z_loc: float, x_loc: float, u_loc: float) -> Tuple[float, float]:
            u_c = max(min(u_loc, 1.0 - 1e-9), 1e-9)
            s = math.sqrt(max(1.0 - u_c * u_c, 1e-16))
            # Voor fase 2 lopen we in z omlaag terwijl x afneemt ⇒ dx/dz < 0
            dxdz = - s / u_c
            # du/dz = (du/dx) * (dx/dz)
            term = (gamma * (float(head_d) + z_loc)) / float(N) - (u_c / max(x_loc, 1e-12))
            dudz = term * dxdz
            return dxdz, dudz

        dz = float(dz_initial)
        while (steps + steps_z) < int(max_steps):
            phi = math.asin(max(min(u, 1.0 - 1e-9), 0.0))
            if phi >= phi_max:
                stopped_reason = "phi_max"
                break

            # Stop als we de as bereiken
            if x <= 1e-6:
                stopped_reason = "x_to_0"
                break

            # adaptieve stap: maak kleiner als u klein wordt (dx/dz groot)
            if adaptive:
                dz = min(dz_initial, 0.05 * max(u, 1e-3))

            # RK4 over z
            k1x, k1u = rhs_z(z, x, u)
            k2x, k2u = rhs_z(z + 0.5 * dz, x + 0.5 * dz * k1x, u + 0.5 * dz * k1u)
            k3x, k3u = rhs_z(z + 0.5 * dz, x + 0.5 * dz * k2x, u + 0.5 * dz * k2u)
            k4x, k4u = rhs_z(z + dz, x + dz * k3x, u + dz * k3u)

            x_next = x + (dz / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            u_next = u + (dz / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            z_next = z + dz

            if not (math.isfinite(x_next) and math.isfinite(u_next)):
                stopped_reason = "non_finite"
                break
            if x_next < 0:
                stopped_reason = "x_to_0"
                break
            if u_next <= 1e-9 or x_next <= 1e-6:
                # We hebben de onder-apex (phi -> 0) bereikt
                u_next = 0.0
                x = x_next
                z = z_next
                xs.append(x)
                zs.append(z)
                us.append(u_next)
                steps_z += 1
                stopped_reason = "completed_full"
                break

            x, z, u = x_next, z_next, u_next
            xs.append(x)
            zs.append(z)
            us.append(u)
            steps_z += 1

    # Afgeleiden en kromtestralen voor rapportage
    dudx_list = []
    dzdx_list = []
    r1_list = []
    r2_list = []
    phi_list = []
    for i in range(len(xs)):
        xi = xs[i]
        zi = zs[i]
        ui = max(min(us[i], 1.0 - 1e-9), -1.0 + 1e-9)
        dudx_i, dzdx_i = rhs(xi if xi > 0 else 1e-9, zi, ui)
        phi_i = math.asin(ui)
        # r1 = cos(phi) / (dphi/dx) met dphi/dx = dudx / cos(phi)
        cos_phi = math.sqrt(max(1.0 - ui * ui, 0.0))
        dphidx = dudx_i / max(cos_phi, 1e-12)
        r1_i = (cos_phi / max(dphidx, 1e-12)) if dphidx != 0 else float("inf")
        r2_i = (xi / ui) if abs(ui) > 1e-12 else float("inf")

        dudx_list.append(dudx_i)
        dzdx_list.append(dzdx_i)
        r1_list.append(r1_i)
        r2_list.append(r2_i)
        phi_list.append(phi_i)

    df = pd.DataFrame({
        'x': np.array(xs, dtype=float),
        'z': np.array(zs, dtype=float),
        'u': np.array(us, dtype=float),
        'phi': np.array(phi_list, dtype=float),
        'dzdx': np.array(dzdx_list, dtype=float),
        'dudx': np.array(dudx_list, dtype=float),
        'r1': np.array(r1_list, dtype=float),
        'r2': np.array(r2_list, dtype=float),
    })

    # Volume als omwentelingslichaam: V = π ∫ x^2 dz (trapeziumregel)
    if len(df) >= 2:
        x2 = df['x'].to_numpy() ** 2
        z_arr = df['z'].to_numpy()
        dz = np.diff(z_arr)
        x2_bar = 0.5 * (x2[:-1] + x2[1:])
        volume = float(math.pi * np.sum(x2_bar * dz))
    else:
        volume = 0.0

    info = {
        'gamma': float(gamma),
        'r1_apex': float(r1_apex),
        'head_d': float(head_d),
        'phi_max_deg': float(phi_max_deg),
        'steps': int(steps),
        'steps_z': int(steps_z),
        'stopped_reason': str(stopped_reason),
        'volume': float(volume),
    }

    return df, info