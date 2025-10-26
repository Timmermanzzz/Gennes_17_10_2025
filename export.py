"""
Export functions for STL and DXF files.
"""

import numpy as np
import pandas as pd
from stl import mesh
import ezdxf
import os


def export_to_stl(df: pd.DataFrame, filepath: str, metrics: dict | None = None) -> bool:
    """
    Export droplet shape as STL file for 3D printing.
    
    Parameters:
        df: DataFrame with droplet data (must include 'x_shifted' and 'h')
        filepath: Output STL file path
    
    Returns:
        True if successful, False on error
    """
    try:
        # Zorg dat x_shifted bestaat
        if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
            x_max = df['x-x_0'].max()
            df['x_shifted'] = df['x-x_0'] - x_max
        
        # Filter en sorteer data
        df_valid = df.dropna(subset=['x_shifted', 'h'])
        if df_valid.empty:
            raise ValueError("No valid data to export")
        
        df_sorted = df_valid.sort_values('h')
        
        # Haal coördinaten op
        x_data = df_sorted['x_shifted'].values
        h_data = df_sorted['h'].values
        
        # Genereer rotatie mesh
        vertices = []
        faces = []
        vertex_count = 0
        
        # Aantal segmenten voor rotatie
        n_theta = 100
        theta = np.linspace(0, 2 * np.pi, n_theta)
        
        # Bouw mesh door het profiel te roteren
        for i in range(len(x_data) - 1):
            for t in range(n_theta - 1):
                # Vier hoekpunten van een quad
                x1 = x_data[i] * np.cos(theta[t])
                y1 = x_data[i] * np.sin(theta[t])
                z1 = h_data[i]
                
                x2 = x_data[i] * np.cos(theta[t + 1])
                y2 = x_data[i] * np.sin(theta[t + 1])
                z2 = h_data[i]
                
                x3 = x_data[i + 1] * np.cos(theta[t])
                y3 = x_data[i + 1] * np.sin(theta[t])
                z3 = h_data[i + 1]
                
                x4 = x_data[i + 1] * np.cos(theta[t + 1])
                y4 = x_data[i + 1] * np.sin(theta[t + 1])
                z4 = h_data[i + 1]
                
                # Voeg vertices toe
                vertices.extend([
                    [x1, y1, z1],
                    [x2, y2, z2],
                    [x3, y3, z3],
                    [x4, y4, z4]
                ])
                
                # Maak twee driehoeken van de quad
                faces.extend([
                    [vertex_count, vertex_count + 1, vertex_count + 2],
                    [vertex_count + 1, vertex_count + 3, vertex_count + 2]
                ])
                
                vertex_count += 4
        
        # Converteer naar numpy arrays
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Maak STL mesh voor druppel
        droplet_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                droplet_mesh.vectors[i][j] = vertices[face[j], :]

        # Optioneel: voeg torus (kraag) toe als aparte mesh en combineer
        combined_mesh = droplet_mesh
        try:
            if metrics is not None:
                R_major = float(metrics.get('torus_R_major', 0.0) or 0.0)
                r_top = float(metrics.get('torus_r_top', 0.0) or 0.0)
                seam_h = float(metrics.get('h_seam_eff', 0.0) or 0.0)
                if R_major > 0.0 and r_top > 0.0 and seam_h > 0.0:
                    zc = seam_h + r_top
                    nu = 120
                    nv = 60
                    us = np.linspace(0.0, 2.0 * np.pi, nu)
                    vs = np.linspace(0.0, 2.0 * np.pi, nv)
                    # aantal driehoeken: (nu-1)*(nv-1)*2
                    tri_count = (nu - 1) * (nv - 1) * 2
                    torus_mesh = mesh.Mesh(np.zeros(tri_count, dtype=mesh.Mesh.dtype))
                    t_idx = 0
                    for i_u in range(nu - 1):
                        u0 = us[i_u]
                        u1 = us[i_u + 1]
                        cu0, su0 = np.cos(u0), np.sin(u0)
                        cu1, su1 = np.cos(u1), np.sin(u1)
                        for i_v in range(nv - 1):
                            v0 = vs[i_v]
                            v1 = vs[i_v + 1]
                            cv0, sv0 = np.cos(v0), np.sin(v0)
                            cv1, sv1 = np.cos(v1), np.sin(v1)
                            # vier punten van de quad
                            x00 = (R_major + r_top * cv0) * cu0
                            y00 = (R_major + r_top * cv0) * su0
                            z00 = zc + r_top * sv0

                            x10 = (R_major + r_top * cv0) * cu1
                            y10 = (R_major + r_top * cv0) * su1
                            z10 = zc + r_top * sv0

                            x01 = (R_major + r_top * cv1) * cu0
                            y01 = (R_major + r_top * cv1) * su0
                            z01 = zc + r_top * sv1

                            x11 = (R_major + r_top * cv1) * cu1
                            y11 = (R_major + r_top * cv1) * su1
                            z11 = zc + r_top * sv1

                            # twee driehoeken
                            torus_mesh.vectors[t_idx][0] = [x00, y00, z00]
                            torus_mesh.vectors[t_idx][1] = [x10, y10, z10]
                            torus_mesh.vectors[t_idx][2] = [x01, y01, z01]
                            t_idx += 1
                            torus_mesh.vectors[t_idx][0] = [x10, y10, z10]
                            torus_mesh.vectors[t_idx][1] = [x11, y11, z11]
                            torus_mesh.vectors[t_idx][2] = [x01, y01, z01]
                            t_idx += 1

                    combined_mesh = mesh.Mesh(np.concatenate([droplet_mesh.data, torus_mesh.data]))
        except Exception as _:
            combined_mesh = droplet_mesh

        # Zorg dat de output directory bestaat
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Sla op
        combined_mesh.save(filepath)
        
        return True
    
    except Exception as e:
        print(f"STL export error: {e}")
        return False


def export_to_dxf(df: pd.DataFrame, filepath: str, metrics: dict | None = None) -> bool:
    """
    Export droplet shape as DXF file for CAD.
    
    Parameters:
        df: DataFrame with droplet data (must include 'x_shifted' and 'h')
        filepath: Output DXF file path
    
    Returns:
        True if successful, False on error
    """
    try:
        # Zorg dat x_shifted bestaat
        if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
            x_max = df['x-x_0'].max()
            df['x_shifted'] = df['x-x_0'] - x_max
        
        # Filter en sorteer data
        df_valid = df.dropna(subset=['x_shifted', 'h'])
        if df_valid.empty:
            raise ValueError("No valid data to export")
        
        df_sorted = df_valid.sort_values('h')
        
        # Maak DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        # Filter dubbele top-lijnen: verwijder bodypunten op of net boven de seam, en teken één vlakke top
        try:
            seam_h = float(metrics.get('h_seam_eff', float(df_sorted['h'].max()))) if metrics else float(df_sorted['h'].max())
            eps = 1e-6
            df_body = df_sorted[df_sorted['h'] < seam_h - eps]
            # Polyline: body
            points_body = [(row['x_shifted'], row['h']) for _, row in df_body.iterrows()]
            if points_body:
                msp.add_lwpolyline(points_body, close=False)
            # Vlakke top: van 0..R aan rechterzijde
            # Meet R_used uit data (max |x_shifted| op seam), fallback naar metrics
            R_used = 0.0
            try:
                band = df_sorted[(df_sorted['h'] >= seam_h - eps) & (df_sorted['h'] <= seam_h + eps)]
                if not band.empty:
                    R_used = float(np.max(np.abs(band['x_shifted'].to_numpy(dtype=float))))
            except Exception:
                R_used = 0.0
            if R_used <= 0.0 and metrics:
                R_used = float(metrics.get('torus_R_major', 0.0) or 0.0)
                if R_used <= 0.0 and metrics.get('top_diameter'):
                    R_used = float(metrics.get('top_diameter', 0.0)) / 2.0
            if R_used > 0.0:
                import numpy as np
                x_top = np.linspace(-R_used, 0.0, 60)
                points_top = [(float(x), float(seam_h)) for x in x_top]
                msp.add_lwpolyline(points_top, close=False)
            else:
                # fallback: teken hele polyline zoals binnenkwam
                points = [(row['x_shifted'], row['h']) for _, row in df_sorted.iterrows()]
                msp.add_lwpolyline(points, close=False)
        except Exception:
            # fallback: teken hele polyline zoals binnenkwam
            points = [(row['x_shifted'], row['h']) for _, row in df_sorted.iterrows()]
            msp.add_lwpolyline(points, close=False)

        # Teken torus doorsnede als cirkel (links) voor referentie
        try:
            if metrics is not None:
                R_major = float(metrics.get('torus_R_major', 0.0) or 0.0)
                r_top = float(metrics.get('torus_r_top', 0.0) or 0.0)
                seam_h = float(metrics.get('h_seam_eff', 0.0) or 0.0)
                if R_major > 0.0 and r_top > 0.0 and seam_h > 0.0:
                    center_left = (-R_major, seam_h + r_top)
                    msp.add_circle(center=center_left, radius=r_top)
        except Exception:
            pass
        
        # Zorg dat de output directory bestaat
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Sla op
        doc.saveas(filepath)
        
        return True
    
    except Exception as e:
        print(f"DXF export error: {e}")
        return False


def get_export_filename(base_name: str, extension: str, output_dir: str = "exports") -> str:
    """
    Generate a unique filename for export.
    
    Parameters:
        base_name: Base name for the file
        extension: File extension (.stl or .dxf)
        output_dir: Output directory
    
    Returns:
        Full file path
    """
    from datetime import datetime
    
    # Maak timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Maak bestandsnaam
    filename = f"{base_name}_{timestamp}{extension}"
    
    # Volledig pad
    filepath = os.path.join(output_dir, filename)
    
    return filepath


def export_both_formats(df: pd.DataFrame, base_name: str = "druppel", 
                       output_dir: str = "exports", metrics: dict | None = None) -> tuple:
    """
    Export droplet shape in both formats (STL and DXF).
    
    Parameters:
        df: DataFrame with droplet data
        base_name: Base name for the files
        output_dir: Output directory
    
    Returns:
        Tuple of (stl_filepath, dxf_filepath, success)
    """
    # Genereer bestandsnamen
    stl_filepath = get_export_filename(base_name, ".stl", output_dir)
    dxf_filepath = get_export_filename(base_name, ".dxf", output_dir)
    
    # Exporteer beide
    stl_success = export_to_stl(df, stl_filepath, metrics=metrics)
    dxf_success = export_to_dxf(df, dxf_filepath, metrics=metrics)
    
    success = stl_success and dxf_success
    
    return stl_filepath, dxf_filepath, success

