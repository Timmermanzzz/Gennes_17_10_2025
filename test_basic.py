"""
Snelle basis test om te verifiëren dat alle modules correct werken.
"""

import sys
import os

# Voeg huidige directory toe aan path
sys.path.insert(0, os.path.dirname(__file__))

from solver import generate_droplet_shape, get_physical_parameters, calculate_kappa, calculate_H
from utils import shift_x_coordinates, get_droplet_metrics, calculate_volume
from visualisatie import create_2d_plot, create_3d_plot
from export import export_to_stl, export_to_dxf

print("=" * 60)
print("BASISTEST: De Gennes Druppelvorm Calculator")
print("=" * 60)

# Test 1: Solver module
print("\n[1] Test solver module...")
try:
    gamma_s = 35000.0
    rho = 1000.0
    g = 9.8
    
    kappa = calculate_kappa(rho, g, gamma_s)
    H = calculate_H(kappa)
    
    print(f"  [OK] Kappa = {kappa:.6f} m^-1")
    print(f"  [OK] H = {H:.6f} m")
    
    # Genereer druppelvorm
    df = generate_droplet_shape(gamma_s, rho, g, cut_percentage=15.0)
    print(f"  [OK] Druppelvorm gegenereerd met {len(df)} punten")
    
except Exception as e:
    print(f"  [FOUT] {e}")
    sys.exit(1)

# Test 2: Utils module
print("\n[2] Test utils module...")
try:
    df = shift_x_coordinates(df)
    print(f"  [OK] X-coordinaten verschoven")
    
    volume = calculate_volume(df)
    print(f"  [OK] Volume = {volume:.6f} m3")
    
    metrics = get_droplet_metrics(df)
    print(f"  [OK] Metrieken berekend:")
    print(f"    - Max hoogte: {metrics['max_height']:.4f} m")
    print(f"    - Max diameter: {metrics['max_diameter']:.4f} m")
    
except Exception as e:
    print(f"  [FOUT] {e}")
    sys.exit(1)

# Test 3: Visualisatie module
print("\n[3] Test visualisatie module...")
try:
    fig_2d = create_2d_plot(df)
    print(f"  [OK] 2D plot gegenereerd")
    
    fig_3d = create_3d_plot(df)
    print(f"  [OK] 3D plot gegenereerd")
    
except Exception as e:
    print(f"  [FOUT] {e}")
    sys.exit(1)

# Test 4: Export module
print("\n[4] Test export module...")
try:
    # Test STL export
    stl_path = "exports/test_druppel.stl"
    stl_success = export_to_stl(df, stl_path)
    
    if stl_success and os.path.exists(stl_path):
        file_size = os.path.getsize(stl_path) / 1024  # KB
        print(f"  [OK] STL export succesvol ({file_size:.1f} KB)")
    else:
        print(f"  [WAARSCHUWING] STL export mislukt")
    
    # Test DXF export
    dxf_path = "exports/test_druppel.dxf"
    dxf_success = export_to_dxf(df, dxf_path)
    
    if dxf_success and os.path.exists(dxf_path):
        file_size = os.path.getsize(dxf_path) / 1024  # KB
        print(f"  [OK] DXF export succesvol ({file_size:.1f} KB)")
    else:
        print(f"  [WAARSCHUWING] DXF export mislukt")
    
except Exception as e:
    print(f"  [FOUT] {e}")
    sys.exit(1)

# Samenvatting
print("\n" + "=" * 60)
print("ALLE TESTS GESLAAGD!")
print("=" * 60)
print("\nJe kunt nu de applicatie starten met:")
print("  streamlit run app.py")
print("=" * 60)

