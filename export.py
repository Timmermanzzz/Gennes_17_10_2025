"""
Export functies voor STL en DXF bestanden.
"""

import numpy as np
import pandas as pd
from stl import mesh
import ezdxf
import os


def export_to_stl(df: pd.DataFrame, filepath: str) -> bool:
    """
    Exporteer druppelvorm als STL bestand voor 3D-printen.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
        filepath: Pad naar output STL bestand
    
    Returns:
        True als succesvol, False bij fout
    """
    try:
        # Zorg dat x_shifted bestaat
        if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
            x_max = df['x-x_0'].max()
            df['x_shifted'] = df['x-x_0'] - x_max
        
        # Filter en sorteer data
        df_valid = df.dropna(subset=['x_shifted', 'h'])
        if df_valid.empty:
            raise ValueError("Geen geldige data om te exporteren")
        
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
        
        # Maak STL mesh
        droplet_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                droplet_mesh.vectors[i][j] = vertices[face[j], :]
        
        # Zorg dat de output directory bestaat
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Sla op
        droplet_mesh.save(filepath)
        
        return True
    
    except Exception as e:
        print(f"STL export fout: {e}")
        return False


def export_to_dxf(df: pd.DataFrame, filepath: str) -> bool:
    """
    Exporteer druppelvorm als DXF bestand voor CAD software.
    
    Parameters:
        df: DataFrame met druppelvorm data (moet 'x_shifted' en 'h' kolommen hebben)
        filepath: Pad naar output DXF bestand
    
    Returns:
        True als succesvol, False bij fout
    """
    try:
        # Zorg dat x_shifted bestaat
        if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
            x_max = df['x-x_0'].max()
            df['x_shifted'] = df['x-x_0'] - x_max
        
        # Filter en sorteer data
        df_valid = df.dropna(subset=['x_shifted', 'h'])
        if df_valid.empty:
            raise ValueError("Geen geldige data om te exporteren")
        
        df_sorted = df_valid.sort_values('h')
        
        # Maak DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Maak punten lijst voor polyline
        points = [(row['x_shifted'], row['h']) for _, row in df_sorted.iterrows()]
        
        # Voeg polyline toe
        msp.add_lwpolyline(points, close=False)
        
        # Zorg dat de output directory bestaat
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Sla op
        doc.saveas(filepath)
        
        return True
    
    except Exception as e:
        print(f"DXF export fout: {e}")
        return False


def get_export_filename(base_name: str, extension: str, output_dir: str = "exports") -> str:
    """
    Genereer een unieke bestandsnaam voor export.
    
    Parameters:
        base_name: Basis naam voor het bestand
        extension: Bestandsextensie (.stl of .dxf)
        output_dir: Output directory
    
    Returns:
        Volledig bestandspad
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
                       output_dir: str = "exports") -> tuple:
    """
    Exporteer druppelvorm in beide formaten (STL en DXF).
    
    Parameters:
        df: DataFrame met druppelvorm data
        base_name: Basis naam voor de bestanden
        output_dir: Output directory
    
    Returns:
        Tuple van (stl_filepath, dxf_filepath, success)
    """
    # Genereer bestandsnamen
    stl_filepath = get_export_filename(base_name, ".stl", output_dir)
    dxf_filepath = get_export_filename(base_name, ".dxf", output_dir)
    
    # Exporteer beide
    stl_success = export_to_stl(df, stl_filepath)
    dxf_success = export_to_dxf(df, dxf_filepath)
    
    success = stl_success and dxf_success
    
    return stl_filepath, dxf_filepath, success

