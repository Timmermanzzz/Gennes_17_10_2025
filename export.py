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


# =============== STEP EXPORT (via pythonocc) ===============
def export_to_step(df: pd.DataFrame, filepath: str, metrics: dict | None = None) -> bool:
    """
    Export 3D solid to STEP using pythonocc-core.
    Builds a surface of revolution from the 2D profile and (optionally) adds a torus,
    then writes a STEP file. Falls back gracefully if OCC is unavailable.
    """
    try:
        from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Ax2
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeTorus
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.Interface import Interface_Static_SetCVal
        from OCC.Core.ShapeFix import ShapeFix_Shape
    except Exception as e:
        print(f"STEP export unavailable (pythonocc-core not installed?): {e}")
        return False

    try:
        # Prepare profile points (x >= 0 for revolution around Z)
        if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
            x_max = df['x-x_0'].max()
            df = df.copy()
            df['x_shifted'] = df['x-x_0'] - x_max
        df_valid = df.dropna(subset=['x_shifted', 'h']).sort_values('h')
        if df_valid.empty:
            return False
        # Build OCC points along (r = -x_shifted, z = h), we rotate around Z
        pts = []
        for _, row in df_valid.iterrows():
            r = float(-row['x_shifted'])  # right edge at 0
            z = float(row['h'])
            if r < 0:
                continue
            pts.append(gp_Pnt(r, 0.0, z))
        if len(pts) < 2:
            return False
        # BSpline through points
        bs = GeomAPI_PointsToBSpline(pts).Curve()
        # Wire
        edges = [BRepBuilderAPI_MakeEdge(bs).Edge()]
        wire_mk = BRepBuilderAPI_MakeWire()
        for e in edges:
            wire_mk.Add(e)
        wire = wire_mk.Wire()
        # Revolve wire around Z axis
        axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        solid = BRepPrimAPI_MakeRevol(wire, axis, 2.0 * 3.141592653589793).Shape()

        # Optional torus
        shape = solid
        try:
            if metrics is not None:
                R_major = float(metrics.get('torus_R_major', 0.0) or 0.0)
                r_tube = float(metrics.get('collar_tube_diameter', 0.0) or 0.0) / 2.0
                h_seam = float(metrics.get('h_seam_eff', 0.0) or 0.0)
                if R_major > 0.0 and r_tube > 0.0 and h_seam > 0.0:
                    torus = BRepPrimAPI_MakeTorus(R_major, r_tube).Shape()
                    # Move torus center up by h_seam (center at z = h_seam)
                    # Simple trick: make a second revolution axis at shifted origin
                    # Better: use gp_Trsf translation, but keep it minimal here
                    from OCC.Core.gp import gp_Trsf, gp_Vec
                    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
                    tr = gp_Trsf()
                    tr.SetTranslation(gp_Vec(0, 0, h_seam))
                    torus = BRepBuilderAPI_Transform(torus, tr, True).Shape()
                    shape = BRepAlgoAPI_Fuse(solid, torus).Shape()
        except Exception:
            pass

        # Fix shape and write STEP
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape_fixed = fixer.Shape()

        Interface_Static_SetCVal("write.step.schema", "AP203")
        writer = STEPControl_Writer()
        writer.Transfer(shape_fixed, STEPControl_AsIs)
        status = writer.Write(filepath)
        return bool(status == 1)
    except Exception as e:
        print(f"STEP export error: {e}")
        return False


# =============== 3DM EXPORT (via rhino3dm) ===============
def export_to_3dm(df: pd.DataFrame, filepath: str, metrics: dict | None = None) -> tuple[bool, str | None]:
    """
    Export to Rhino 3DM using rhino3dm. Builds revolve surfaces for the profile
    and adds a torus surface. Not a boolean fused solid, but opens cleanly in Rhino.
    """
    try:
        import rhino3dm as r3d
        import numpy as np
        import math
    except Exception as e:
        msg = f"3DM export unavailable (rhino3dm not installed?): {e}"
        print(msg)
        return False, msg

    try:
        # Ensure x_shifted exists
        if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
            x_max = df['x-x_0'].max()
            df = df.copy()
            df['x_shifted'] = df['x-x_0'] - x_max
        df_valid = df.dropna(subset=['x_shifted', 'h']).sort_values('h')
        if df_valid.empty:
            return False, "No valid profile data (x_shifted,h) to export"

        # Make a polycurve for profile in (r = -x_shifted, z = h) plane (XZ)
        pts = []
        for _, row in df_valid.iterrows():
            r = float(-row['x_shifted'])
            z = float(row['h'])
            if r < 0:
                continue
            pts.append(r3d.Point3d(r, 0.0, z))
        if len(pts) < 2:
            return False, f"Not enough profile points after filtering (count={len(pts)})"

        # Maak een curve uit punten (PolylineCurve is voldoende voor revolve)
        curve = r3d.PolylineCurve(pts)

        # Revolve around Z axis to make a surface
        axis = r3d.Line(r3d.Point3d(0, 0, 0), r3d.Point3d(0, 0, 1))
        # Create expects startAngle/endAngle in radians
        rev = r3d.RevSurface.Create(curve, axis, 0.0, 2.0 * math.pi)

        model = r3d.File3dm()
        if rev:
            try:
                breps = r3d.Brep.CreateFromRevSurface(rev, True, True)
                if breps:
                    for b in breps:
                        model.Objects.AddBrep(b)
                else:
                    model.Objects.AddCurve(curve)
            except Exception:
                model.Objects.AddCurve(curve)
        else:
            # Fallback: schrijf alleen de curve zodat gebruiker kan revolven in Rhino
            model.Objects.AddCurve(curve)

        # Torus (if present)
        try:
            if metrics is not None:
                R_major = float(metrics.get('torus_R_major', 0.0) or 0.0)
                r_tube = float(metrics.get('collar_tube_diameter', 0.0) or 0.0) / 2.0
                h_seam = float(metrics.get('h_seam_eff', 0.0) or 0.0)
                if R_major > 0.0 and r_tube > 0.0 and h_seam > 0.0:
                    center = r3d.Point3d(0, 0, h_seam)
                    axis_torus = r3d.Vector3d(0, 0, 1)
                    torus = r3d.Torus(r3d.Plane(center, axis_torus), R_major, r_tube)
                    torus_brep = torus.ToBrep()
                    if torus_brep:
                        model.Objects.AddBrep(torus_brep)
        except Exception:
            pass

        ok_write = model.Write(filepath, 5)
        if not ok_write:
            return False, "File3dm.Write returned False (file not written)."
        try:
            # Extra safety: ensure non-empty file
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return True, None
            return False, "3DM file size is 0 after write."
        except Exception as _:
            return True, None
    except Exception as e:
        msg = f"3DM export error: {e}"
        print(msg)
        return False, msg

