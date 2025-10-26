"""
PDF export (A3) voor 2D doorsnede met maatvoering.

Gebruik: export_to_pdf(df, metrics, filepath)

Vereist: reportlab
"""

from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import mm
import numpy as np
import pandas as pd


def _compute_scale_and_offset(df_xy: np.ndarray, page_w_mm: float, page_h_mm: float, margin_mm: float = 20.0):
    x = df_xy[:, 0]
    y = df_xy[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    w_m = max(1e-9, xmax - xmin)
    h_m = max(1e-9, ymax - ymin)
    usable_w_mm = page_w_mm - 2 * margin_mm
    usable_h_mm = page_h_mm - 2 * margin_mm
    s = min(usable_w_mm / w_m, usable_h_mm / h_m)
    ox = margin_mm - s * xmin
    oy = margin_mm - s * ymin
    return s, ox, oy, (xmin, xmax, ymin, ymax)


def export_to_pdf(df: pd.DataFrame, metrics: dict, filepath: str, physical_params: dict | None = None):
    # Zorg dat x_shifted bestaat
    if 'x_shifted' not in df.columns and 'x-x_0' in df.columns:
        x_max = df['x-x_0'].max()
        df = df.copy()
        df['x_shifted'] = df['x-x_0'] - x_max

    df_valid = df.dropna(subset=['x_shifted', 'h']).sort_values('h')
    if df_valid.empty:
        raise ValueError("No valid data to export to PDF")

    # Basispagina
    page_w_mm, page_h_mm = landscape(A3)[0] / mm, landscape(A3)[1] / mm
    c = canvas.Canvas(filepath, pagesize=landscape(A3))

    # Zet profielpunten in (x,y)
    body_xy = df_valid[['x_shifted', 'h']].to_numpy(dtype=float)

    # Vlakke top (linkerhelft): reconstrueer net als visual/DXF
    h_seam = float(metrics.get('h_seam_eff', float(df_valid['h'].max())))
    # Meet naadstraal R_used uit data (max |x| rond seam)
    band = df_valid[(df_valid['h'] >= h_seam - 1e-6) & (df_valid['h'] <= h_seam + 1e-6)]
    if not band.empty:
        R_used = float(np.max(np.abs(band['x_shifted'].to_numpy(dtype=float))))
    else:
        R_used = float(metrics.get('torus_R_major', 0.0) or 0.0)
        if R_used <= 0.0 and 'top_diameter' in metrics:
            R_used = float(metrics.get('top_diameter', 0.0)) / 2.0
    x_top = np.linspace(-max(0.0, R_used), 0.0, 120)
    top_xy = np.column_stack([x_top, np.full_like(x_top, h_seam)])

    # Spiegelen voor volledige doorsnede
    body_xy_right = body_xy.copy()
    body_xy_right[:, 0] = -body_xy_right[:, 0]
    body_xy_right = body_xy_right[::-1]  # volgorde om te verbinden

    top_xy_right = np.column_stack([np.linspace(0.0, max(0.0, R_used), 120), np.full(120, h_seam)])

    # Combineer voor schaalbepaling (links + rechts)
    all_xy = np.vstack([body_xy, top_xy, body_xy_right, top_xy_right])
    # Iets ruimere marges en een lichte schaalreductie voor meer witruimte
    # Extra witruimte: grotere marges en kleinere schaal
    s, ox, oy, _ = _compute_scale_and_offset(all_xy, page_w_mm, page_h_mm, margin_mm=40.0)
    s *= 0.75

    def to_mm(pt):
        return (ox + s * pt[0]) * mm, (oy + s * pt[1]) * mm

    # Titel en parameters
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, (page_h_mm - 15) * mm, "Reservoir cross-section (A3)")
    c.setFont("Helvetica", 10)
    opening_title = float(metrics.get('top_diameter', 0) or (2.0 * float(metrics.get('torus_R_major', 0) or 0.0)))
    gamma_val = metrics.get('γₛ match (N/m)', None)
    if gamma_val is None and physical_params is not None:
        gamma_val = physical_params.get('gamma_s', None)
    if gamma_val is None:
        gamma_val = metrics.get('gamma_s', None)
    if gamma_val is None:
        gamma_str = "n/a"
    else:
        try:
            gamma_str = f"{float(gamma_val):.1f}"
        except Exception:
            gamma_str = str(gamma_val)
    c.drawString(20 * mm, (page_h_mm - 22) * mm, f"Opening: {opening_title:.2f} m  |  Tube D: {metrics.get('collar_tube_diameter', 0):.3f} m  |  γs: {gamma_str}")

    # Teken body-polyline links
    c.setLineWidth(1)
    c.setStrokeColor(colors.black)
    x0, y0 = to_mm(body_xy[0])
    for p in body_xy[1:]:
        x, y = to_mm(p)
        c.line(x0, y0, x, y)
        x0, y0 = x, y

    # Teken body-polyline rechts
    x0, y0 = to_mm(body_xy_right[0])
    for p in body_xy_right[1:]:
        x, y = to_mm(p)
        c.line(x0, y0, x, y)
        x0, y0 = x, y

    # Teken vlakke top (links)
    x0, y0 = to_mm(top_xy[0])
    for p in top_xy[1:]:
        x, y = to_mm(p)
        c.line(x0, y0, x, y)
        x0, y0 = x, y

    # Teken vlakke top (rechts)
    x0, y0 = to_mm(top_xy_right[0])
    for p in top_xy_right[1:]:
        x, y = to_mm(p)
        c.line(x0, y0, x, y)
        x0, y0 = x, y

    # Teken kraag (cirkel, links)
    R_major = float(metrics.get('torus_R_major', 0.0) or 0.0)
    r_tube = float(metrics.get('collar_tube_diameter', 0.0) or 0.0) / 2.0
    if R_major > 0.0 and r_tube > 0.0 and h_seam > 0.0:
        cx_mm, cy_mm = to_mm((-R_major, h_seam + r_tube))
        c.setStrokeColor(colors.purple)
        c.circle(cx_mm, cy_mm, s * r_tube * mm, stroke=1, fill=0)
        # rechter kraag
        cx_mm_r, cy_mm_r = to_mm((+R_major, h_seam + r_tube))
        c.circle(cx_mm_r, cy_mm_r, s * r_tube * mm, stroke=1, fill=0)
        c.setStrokeColor(colors.black)

    # Simpele maatlijnen (opening & tube diameter)
    def _label_with_bg(c, x_mm, y_mm, text):
        """Draw label with small white background to improve legibility."""
        pad = 1.5 * mm
        w = pdfmetrics.stringWidth(text, c._fontname, c._fontsize)
        htxt = 3.5 * mm
        c.setFillColor(colors.white)
        c.rect(x_mm - pad, y_mm - htxt + 1 * mm, w + 2 * pad, htxt, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(x_mm, y_mm, text)

    def dim_h(x_left, x_right, h, text, align="center"):
        x1, y1 = to_mm((x_left, h))
        x2, y2 = to_mm((x_right, h))
        c.line(x1, y1, x2, y2)
        c.line(x1, y1, x1, y1 + 3 * mm)
        c.line(x2, y2, x2, y2 + 3 * mm)
        if align == "left":
            _label_with_bg(c, x1 + 3 * mm, y1 + 5 * mm, text)
        elif align == "right":
            w = pdfmetrics.stringWidth(text, c._fontname, c._fontsize)
            _label_with_bg(c, x2 - w - 3 * mm, y1 + 5 * mm, text)
        else:
            _label_with_bg(c, (x1 + x2) / 2 - pdfmetrics.stringWidth(text, c._fontname, c._fontsize) / 2, y1 + 5 * mm, text)

    def dim_v(x, h1, h2, text, label_side="right"):
        x_mm, y1 = to_mm((x, h1))
        _, y2 = to_mm((x, h2))
        c.line(x_mm, y1, x_mm, y2)
        c.line(x_mm, y1, x_mm + 3 * mm, y1)
        c.line(x_mm, y2, x_mm + 3 * mm, y2)
        if label_side == "left":
            w = pdfmetrics.stringWidth(text, c._fontname, c._fontsize)
            _label_with_bg(c, x_mm - w - 5 * mm, (y1 + y2) / 2, text)
        else:
            _label_with_bg(c, x_mm + 5 * mm, (y1 + y2) / 2, text)

    # Opening diameter over volledige breedte
    if R_used > 0:
        dim_h(-R_used, +R_used, h_seam + 0.05, f"Opening {2*R_used:.2f} m", align="center")
    # Tube diameter
    if R_major > 0 and r_tube > 0:
        dim_v(-R_major - r_tube - 0.1, h_seam, h_seam + 2 * r_tube, f"Tube D {2*r_tube:.3f} m")

    # Droplet/Total heights
    droplet_h = float(metrics.get('Droplet height (m)', metrics.get('max_height', 0)))
    # Definitie: Total height = cut height + tube diameter (indien kraag actief),
    # anders gelijk aan droplet height
    tube_d = float(metrics.get('collar_tube_diameter', 0.0) or 0.0)
    if tube_d > 0 and h_seam > 0:
        total_h = float(h_seam + tube_d)
    else:
        total_h = droplet_h
    # Plaats links van de vorm (ruim buiten het profiel voor leesbaarheid)
    c.setFont("Helvetica", 9)
    x_dim_left = float(np.min(all_xy[:, 0])) - 1.2
    x_dim_left2 = x_dim_left - 0.6
    # Droplet height maatlijn weggelaten op verzoek; alleen Total height blijft staan
    if total_h > droplet_h:
        dim_v(x_dim_left2, 0.0, total_h, f"Total height {total_h:.2f} m", label_side="left")

    # Water level en Cut height (gestippelde hulplijnen)
    def hline(h_val_mm, dash=True, label=None, align="left"):
        x1, y = to_mm((float(np.min(all_xy[:, 0])) - 0.5, h_val_mm))
        x2, _ = to_mm((float(np.max(all_xy[:, 0])) + 0.5, h_val_mm))
        if dash:
            c.setDash(3, 2)
        c.line(x1, y, x2, y)
        if dash:
            c.setDash()
        if label:
            c.setFont("Helvetica", 9)
            if align == "right":
                w = pdfmetrics.stringWidth(label, c._fontname, c._fontsize)
                _label_with_bg(c, x2 - w - 3 * mm, y + 2 * mm, label)
            else:
                _label_with_bg(c, x1 + 3 * mm, y + 2 * mm, label)

    # Cut
    hline(h_seam, dash=True, label="Cut height", align="left")
    # Water level
    h_water = float(metrics.get('h_waterline', h_seam))
    if h_water > 0:
        hline(h_water, dash=True, label="Water level", align="right")

    # Tabel
    c.setFont("Helvetica-Bold", 12)
    c.drawString((page_w_mm - 120) * mm, (page_h_mm - 30) * mm, "Key metrics")
    c.setFont("Helvetica", 10)
    lines = [
        f"Droplet Vol: {metrics.get('Reservoir volume (m³)', metrics.get('volume', 0)):.2f} m³",
        f"Collar Vol: {metrics.get('Collar volume (m³)', metrics.get('volume_kraag', 0)):.2f} m³",
        f"Total Vol: {(metrics.get('Reservoir volume (m³)', metrics.get('volume', 0)) + metrics.get('Collar volume (m³)', metrics.get('volume_kraag', 0))):.2f} m³",
        f"Cut Vol: {metrics.get('Cut volume (m³)', metrics.get('volume_afgekapt', 0)):.2f} m³",
        f"Droplet height: {metrics.get('Droplet height (m)', metrics.get('max_height', 0)):.2f} m",
        f"Total height: {total_h:.2f} m",
        f"Max diameter: {metrics.get('Max diameter (m)', metrics.get('max_diameter', 0)):.2f} m",
    ]
    y = (page_h_mm - 38) * mm
    for sline in lines:
        c.drawString((page_w_mm - 120) * mm, y, sline)
        y -= 6 * mm

    # Aslabels
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, 10 * mm, "x (m) – 0 at right edge")
    c.saveState()
    c.translate(10 * mm, 40 * mm)
    c.rotate(90)
    c.drawString(0, 0, "h (m) – Height")
    c.restoreState()

    c.showPage()
    c.save()


