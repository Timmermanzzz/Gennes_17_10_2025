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
    dx_initial: float = 0.003,
    dz_initial: float = 0.004,
    max_steps: int = 2000000,
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
    u_switch = 1.0 - 1e-6  # nog dichter bij verticaal switchen

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
            safety = max(1e-4, 1.0 - abs(u_clamped))
            dx = min(dx_initial, 0.05 * safety)

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
        u = max(min(us[-1], 1.0 - 1e-8), 1e-6)
        x = max(xs[-1], 1e-12)
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
            if x <= 1e-10:
                stopped_reason = "x_to_0"
                break

            # adaptieve stap: maak kleiner als u klein wordt (dx/dz groot)
            if adaptive:
                dz = min(dz_initial, 0.01 * max(u, 1e-6))

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
            # Event detection: u -> 0 binnen deze stap (lineaire interpolatie in u)
            u_tol = 1e-10
            if (u > u_tol and u_next <= u_tol):
                denom = (u - u_next)
                t_ev = 0.0 if denom == 0.0 else max(0.0, min(1.0, float(u / denom)))
                x_ev = x + t_ev * (x_next - x)
                z_ev = z + t_ev * (z_next - z)
                x = max(0.0, x_ev)
                z = z_ev
                u = 0.0
                xs.append(x)
                zs.append(z)
                us.append(u)
                steps_z += 1
                stopped_reason = "completed_full"
                break

            if x_next <= 0.0:
                # Lineaire interpolatie naar x=0 om z_bottom accurater te schatten
                dx_step = x_next - x
                if dx_step != 0.0:
                    t = float(x / (-dx_step))  # in (0,1]
                    t = max(0.0, min(1.0, t))
                    z_zero = z + t * dz
                    x = 0.0
                    z = z_zero
                    u = 0.0
                    xs.append(x)
                    zs.append(z)
                    us.append(u)
                    steps_z += 1
                    stopped_reason = "completed_full"
                    break
                stopped_reason = "x_to_0"
                break
            if u_next <= 1e-14 or x_next <= 1e-10:
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

    # FASE 3 (optioneel): wanneer u klein is maar x nog niet 0 is, integreer verder met x als onafhankelijke variabele
    if (steps + steps_z) < int(max_steps) and xs and us and zs and xs[-1] > 0.0 and 0.0 < us[-1] < 0.2:
        x = float(xs[-1])
        z = float(zs[-1])
        u = float(us[-1])
        dx3 = - float(max(dx_initial * 0.5, 1e-4))  # kleine negatieve stap
        while (steps + steps_z) < int(max_steps):
            # adaptieve verkleining naarmate u afneemt
            if adaptive:
                dx3 = - min(abs(dx3), max(1e-5, 0.02 * max(u, 1e-6)))

            # RK4 in x (negatieve richting)
            k1u, k1z = rhs(x, z, u)
            k2u, k2z = rhs(x + 0.5 * dx3, z + 0.5 * dx3 * k1z, u + 0.5 * dx3 * k1u)
            k3u, k3z = rhs(x + 0.5 * dx3, z + 0.5 * dx3 * k2z, u + 0.5 * dx3 * k2u)
            k4u, k4z = rhs(x + dx3,       z + dx3 * k3z,       u + dx3 * k3u)

            u_next = u + (dx3 / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            z_next = z + (dx3 / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
            x_next = x + dx3

            if not (math.isfinite(u_next) and math.isfinite(z_next)):
                stopped_reason = "non_finite"
                break

            if x_next <= 0.0 or u_next <= 0.0:
                # Interpoleer naar x=0 of u=0
                t_num = 1.0
                if x_next <= 0.0 and x != x_next:
                    t_num = min(t_num, float(x / (x - x_next)))
                if u_next <= 0.0 and u != u_next:
                    t_num = min(t_num, float(u / (u - u_next)))
                t_num = max(0.0, min(1.0, t_num))
                x = x + t_num * (x_next - x)
                z = z + t_num * (z_next - z)
                u = 0.0
                xs.append(x)
                zs.append(z)
                us.append(u)
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


def solve_open_rim_membrane(
    D_open: float,
    N: float,
    rho: float,
    g: float,
    h0: float = 0.0,
    h_cap: float = 0.0,
    phi_top_deg: float = 2.0,
    dx_initial: float = 0.003,
    dz_initial: float = 0.004,
    phi_max_deg: float = 179.0,
    z_seed: float = 0.02,
    max_steps: int = 2000000,
    adaptive: bool = True,
    debug: bool = False,
    debug_max: int = 300,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Open-rim (always-open) solver with two phases:
      1) x-phase near the rim (u small): integrate du/dx + u/x = (γ z)/N, dz/dx = u/√(1-u²)
      2) z-phase past near-vertical:     integrate dx/dz = -√(1-u²)/u,  du/dz = [(γ z)/N - u/x] √(1-u²)/u

    Start at rim (z=0, x=R_open, u=sin(φ_top)). This avoids the u→0 singularity and yields
    realistic depth for small φ_top.
    """
    gamma = float(rho) * float(g)
    h0 = max(0.0, float(h0))
    p0 = gamma * h0  # uniform overpressure from collar depth
    h_cap = max(0.0, float(h_cap))

    # Helper: pressure over N as function of local depth
    def p_over_N(z_loc: float) -> float:
        if h_cap > 0.0:
            # constant cap zone to depth h_cap, then hydrostatic beyond
            if z_loc <= h_cap:
                p_eff = p0
            else:
                p_eff = p0 + gamma * (z_loc - h_cap)
        else:
            p_eff = p0 + gamma * z_loc
        return p_eff / float(N)

    if D_open <= 0:
        raise ValueError("D_open must be > 0")
    if N <= 0:
        raise ValueError("N must be > 0")
    if rho <= 0 or g <= 0:
        raise ValueError("rho and g must be > 0")

    R_open = float(D_open) / 2.0
    phi0 = math.radians(max(1e-3, float(phi_top_deg)))
    u0 = max(1e-9, math.sin(phi0))

    # Initial state at rim
    x = float(R_open)
    # Small seed depth to kick hydrostatic term (numerical regularisation)
    z = max(0.0, float(z_seed))
    u = float(u0)

    xs: list = [x]
    zs: list = [z]
    us: list = [u]

    # Switch to z-phase only when we are almost vertical
    u_switch = 0.985
    phi_max = math.radians(float(phi_max_deg))

    # ---- Phase 1: x as independent variable (move inward: dx < 0) ----
    def rhs_x(x_loc: float, z_loc: float, u_loc: float) -> Tuple[float, float]:
        u_c = max(min(u_loc, 1.0 - 1e-9), 1e-9)
        inv_x = 0.0 if x_loc <= 0.0 else (1.0 / x_loc)
        dudx = p_over_N(z_loc) - u_c * inv_x
        # Gebruik dz/dx < 0 zodat met dx<0 de diepte toeneemt: Δz = dzdx*dx > 0
        dzdx = - u_c / math.sqrt(max(1.0 - u_c * u_c, 1e-16))
        return dudx, dzdx

    dx = -float(max(dx_initial, 1e-5))
    # Adaptive control to ensure sufficient depth growth in x-phase
    dz_target_x_base = 0.02   # baseline ~2 cm per x-step
    dx_min_abs = 1e-4
    dx_max_abs = 0.5

    debug_rows: list = []
    steps = 0
    stopped_reason = "switch_to_z"
    u_switch_value = float('nan')

    while steps < int(max_steps):
        u_c = max(min(u, 1.0 - 1e-9), 1e-9)
        if math.asin(u_c) >= phi_max:
            break
        if adaptive:
            # Aggressive depth growth until we are nearly vertical
            if u_c < 0.7:
                dz_target_x = dz_target_x_base * 3.0
            elif u_c < 0.95:
                dz_target_x = dz_target_x_base * 1.5
            else:
                dz_target_x = dz_target_x_base

            # choose dx so that |Δz| ≈ dz_target_x; Δz ≈ |dzdx| * |dx|
            dzdx_mag = u_c / math.sqrt(max(1.0 - u_c * u_c, 1e-16))
            dx_abs = dz_target_x / dzdx_mag if dzdx_mag > 0.0 else abs(dx_initial)
            # prevent overstepping radius shrink near small x
            dx_abs = min(dx_abs, 0.2 * max(x, 1e-6))
            dx_abs = max(dx_min_abs, min(dx_max_abs, dx_abs))
            dx = -dx_abs

        # RK4 in x met negatieve dx (naar binnen)
        k1u, k1z = rhs_x(x, z, u)
        k2u, k2z = rhs_x(x + 0.5 * dx, z + 0.5 * dx * k1z, u + 0.5 * dx * k1u)
        k3u, k3z = rhs_x(x + 0.5 * dx, z + 0.5 * dx * k2z, u + 0.5 * dx * k2u)
        k4u, k4z = rhs_x(x + dx,       z + dx * k3z,       u + dx * k3u)

        # u groeit bij dx<0 en dudx>0 via minusteken
        u_next = u - (dx / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        z_next = z + (dx / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        x_next = x + dx
        # backoff if we overshoot the axis within this step
        if x_next < 0.0:
            # reduce step to hit x=0 boundary more carefully
            frac = max(0.1, min(0.9, float(x / (x - x_next))))
            dx *= frac
            k1u, k1z = rhs_x(x, z, u)
            k2u, k2z = rhs_x(x + 0.5 * dx, z + 0.5 * dx * k1z, u + 0.5 * dx * k1u)
            k3u, k3z = rhs_x(x + 0.5 * dx, z + 0.5 * dx * k2z, u + 0.5 * dx * k2u)
            k4u, k4z = rhs_x(x + dx,       z + dx * k3z,       u + dx * k3u)
            u_next = u - (dx / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            z_next = z + (dx / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
            x_next = x + dx

        if not (math.isfinite(u_next) and math.isfinite(z_next)):
            stopped_reason = "non_finite"
            break
        # debug capture
        if debug and len(debug_rows) < int(debug_max):
            inv_x = 0.0 if x <= 0.0 else (1.0 / x)
            dudx_now = (gamma * z) / float(N) - max(min(u, 1.0 - 1e-9), 1e-9) * inv_x
            dzdx_now = - max(min(u, 1.0 - 1e-9), 1e-9) / math.sqrt(max(1.0 - u * u, 1e-16))
            debug_rows.append({
                'phase': 'x', 'step': steps, 'x': float(x), 'z': float(z), 'u': float(u),
                'dudx': float(dudx_now), 'dzdx': float(dzdx_now), 'dx': float(dx),
                'x_next': float(x_next), 'z_next': float(z_next), 'u_next': float(u_next),
            })

        # switch condition: only when almost vertical (u close to 1)
        if abs(u_next) >= u_switch:
            # ready for z-phase
            x, z, u = x_next, max(z_next, 0.0), max(min(u_next, 1.0 - 1e-8), 1e-8)
            xs.append(x); zs.append(z); us.append(u)
            u_switch_value = float(u_next)
            break
        if x_next <= 1e-12:
            # hit the axis already
            x, z, u = 0.0, max(z_next, 0.0), 0.0
            xs.append(x); zs.append(z); us.append(u)
            stopped_reason = "x_to_0_in_x_phase"
            phi_max = 0.0
            u_switch_value = float(us[-2] if len(us) >= 2 else u_next)
            break

        x, z, u = x_next, max(z_next, 0.0), max(min(u_next, 1.0 - 1e-9), 1e-9)
        xs.append(x); zs.append(z); us.append(u)
        steps += 1

    # ---- Phase 2: z as independent variable until axis ----
    steps_z = 0
    if xs[-1] > 0.0 and us[-1] > 0.0 and steps < int(max_steps):
        def rhs_z(z_loc: float, x_loc: float, u_loc: float) -> Tuple[float, float]:
            u_c = max(min(u_loc, 1.0 - 1e-9), 1e-9)
            s = math.sqrt(max(1.0 - u_c * u_c, 1e-16))
            dxdz = - s / u_c
            term = (p_over_N(z_loc) - (u_c / max(x_loc, 1e-12)))
            # Consistent with chain rule: du/dz = (du/dx) * (dx/dz) = term * dxdz
            dudz = term * dxdz
            return dxdz, dudz

        dz = float(max(dz_initial, 1e-5))
        while (steps + steps_z) < int(max_steps):
            if adaptive:
                dz = min(dz_initial, 0.01 * max(us[-1], 1e-6))

            x0, z0, u0_ = xs[-1], zs[-1], us[-1]
            k1x, k1u = rhs_z(z0, x0, u0_)
            k2x, k2u = rhs_z(z0 + 0.5 * dz, x0 + 0.5 * dz * k1x, u0_ + 0.5 * dz * k1u)
            k3x, k3u = rhs_z(z0 + 0.5 * dz, x0 + 0.5 * dz * k2x, u0_ + 0.5 * dz * k2u)
            k4x, k4u = rhs_z(z0 + dz,       x0 + dz * k3x,       u0_ + dz * k3u)

            x_next = x0 + (dz / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            u_next = u0_ + (dz / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            z_next = z0 + dz

            if not (math.isfinite(x_next) and math.isfinite(u_next)):
                stopped_reason = "non_finite"
                break

            if x_next <= 0.0:
                # interpolate to axis
                dx_step = x_next - x0
                t = 1.0 if dx_step == 0.0 else max(0.0, min(1.0, float(x0 / (x0 - x_next))))
                z_axis = z0 + t * dz
                xs.append(0.0); zs.append(z_axis); us.append(0.0)
                stopped_reason = "x_to_0"
                break

            xs.append(x_next)
            zs.append(z_next)
            us.append(max(min(u_next, 1.0 - 1e-9), 1e-9))
            steps_z += 1

    H_total = float(zs[-1]) if zs else 0.0

    # Volume V = π ∫ x^2 dz
    if len(zs) >= 2:
        z_arr = np.asarray(zs, dtype=float)
        x2 = (np.asarray(xs, dtype=float) ** 2)
        dz_arr = np.diff(z_arr)
        x2_bar = 0.5 * (x2[:-1] + x2[1:])
        volume = float(math.pi * np.sum(x2_bar * dz_arr))
    else:
        volume = 0.0

    # Convert to app convention
    df_solver = pd.DataFrame({'z': np.asarray(zs, dtype=float), 'x': np.asarray(xs, dtype=float), 'u': np.asarray(us, dtype=float)})
    try:
        df_solver['phi'] = df_solver['u'].apply(lambda uu: math.asin(max(min(uu, 1.0), -1.0)))
    except Exception:
        df_solver['phi'] = 0.0

    h = H_total - df_solver['z'].to_numpy(dtype=float)
    x_rad = df_solver['x'].to_numpy(dtype=float)
    df = pd.DataFrame({'h': h, 'x-x_0': x_rad})
    try:
        df['x_shifted'] = -pd.Series(x_rad, dtype=float)
    except Exception:
        pass

    info = {
        'H_total': float(H_total),
        'volume': float(volume),
        'R_open': float(R_open),
        'D_open': float(D_open),
        'h0': float(h0),
        'p0': float(p0),
        'h_cap': float(h_cap),
        'N': float(N),
        'rho': float(rho),
        'g': float(g),
        'phi_top_deg': float(phi_top_deg),
        'steps_x': int(steps),
        'steps_z': int(steps_z),
        'stopped_reason': str(stopped_reason),
        'u_switch': float(u_switch_value),
    }

    if debug:
        info['debug_rows'] = debug_rows

    return df, info

def shoot_open_rim_membrane(
    D_open: float,
    N: float,
    rho: float,
    g: float,
    h0: float = 0.0,
    h_cap: float = 0.0,
    phi_min_deg: float = 1.0,
    phi_max_deg: float = 30.0,
    z_seed: float = 0.0,
    target_u: float = 0.985,
    max_iter: int = 12,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Shooting wrapper: find φ_top so that the x-phase reaches near-vertical (u≈target_u)
    before switching to z-phase, then integrate to the axis.
    Returns (df, info) from the best φ_top.
    """
    def eval_phi(phi_deg: float):
        # Use debug=False for performance; store u_switch in info
        df, info = solve_open_rim_membrane(
            D_open=D_open, N=N, rho=rho, g=g,
            h0=float(h0),
            h_cap=float(h_cap),
            phi_top_deg=float(phi_deg), z_seed=float(z_seed),
            debug=False,
        )
        u_sw = float(info.get('u_switch', 0.0))
        reason = str(info.get('stopped_reason', ''))
        H = float(info.get('H_total', 0.0))
        # Cost prefers u_switch close to target and penalizes early axis
        penalty = 0.0
        if 'x_to_0_in_x_phase' in reason or H <= 0.0:
            penalty += 10.0
        cost = abs(u_sw - target_u) + penalty
        return cost, df, info

    # Coarse scan
    grid = np.linspace(float(phi_min_deg), float(phi_max_deg), num=8)
    best = None
    for phi in grid:
        c, df, info = eval_phi(phi)
        if best is None or c < best[0]:
            best = (c, phi, df, info)

    # Local refine around best
    lo = max(phi_min_deg, best[1] - 5.0)
    hi = min(phi_max_deg, best[1] + 5.0)
    for _ in range(max(3, int(max_iter))):
        phis = np.linspace(lo, hi, num=5)
        for phi in phis:
            c, df, info = eval_phi(phi)
            if c < best[0]:
                best = (c, phi, df, info)
        span = hi - lo
        lo = max(phi_min_deg, best[1] - 0.3 * span)
        hi = min(phi_max_deg, best[1] + 0.3 * span)

    # Attach chosen phi to info
    best_info = best[3]
    best_info['phi_top_deg_chosen'] = float(best[1])
    best_info['shoot_cost'] = float(best[0])
    best_info['h0'] = float(h0)
    best_info['h_cap'] = float(h_cap)
    try:
        gamma = float(rho) * float(g)
        best_info['p0'] = float(gamma * float(h0))
    except Exception:
        pass
    return best[2], best_info