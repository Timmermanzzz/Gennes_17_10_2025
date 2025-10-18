
# 🚀 SNELSTART - De Gennes Druppelvorm Calculator

## ✅ Wat is klaar?

Het complete project is gebouwd en getest! Alle modules werken correct.

### 📁 Projectstructuur
```
Gennes_17_10_2025/
├── app.py              ✓ Streamlit UI (200 regels)
├── solver.py           ✓ Young-Laplace berekeningen
├── utils.py            ✓ Hulpfuncties
├── visualisatie.py     ✓ Plotly 2D/3D plots
├── export.py           ✓ STL & DXF export
├── requirements.txt    ✓ Dependencies
├── README.md           ✓ Volledige documentatie
├── test_basic.py       ✓ Basistest (alle tests geslaagd!)
└── exports/            ✓ Output directory
```

## 🎯 Test Resultaten

**Laatste test run:**
- ✅ Solver module: Kappa & H berekening
- ✅ Druppelvorm generatie: 281 punten
- ✅ Volume berekening: 1064.82 m³
- ✅ 2D & 3D visualisatie
- ✅ STL export: 2707 KB
- ✅ DXF export: 32 KB

## 🏃 Hoe te starten?

### Optie 1: Direct starten (als dependencies al geïnstalleerd zijn)
```bash
cd Gennes_17_10_2025
streamlit run app.py
```

### Optie 2: Vanaf nul (eerste keer)
```bash
cd Gennes_17_10_2025
pip install -r requirements.txt
streamlit run app.py
```

### Optie 3: Test eerst (aanbevolen)
```bash
cd Gennes_17_10_2025
python test_basic.py
streamlit run app.py
```

## 📱 Gebruik

1. **Browser opent automatisch** op `http://localhost:8501`
2. **Stel parameters in** in de zijbalk:
   - γₛ = 35000 N/m (standaard voor water)
   - ρ = 1000 kg/m³ (dichtheid water)
   - g = 9.8 m/s² (zwaartekracht)
   - Afkap = 15% (of 0% voor hele druppel)
3. **Klik "Bereken Druppel"**
4. **Bekijk resultaten** in de tabs:
   - Tab 1: Interactieve 2D visualisatie (Plotly)
   - Tab 2: Alle metrieken in tabel
   - Tab 3: Download STL/DXF

## 🎨 Features

### ✓ Wat werkt al:
- Enkele druppel generatie
- Young-Laplace berekeningen
- 2D visualisatie (Plotly - interactief!)
- 3D visualisatie (optioneel)
- Volume & diameter berekeningen
- STL export (3D-printen)
- DXF export (CAD software)
- Afkap percentage voor open reservoirs
- Responsive UI
- Session state management

### ⏳ Wat nog niet inbegrepen is (zoals gepland):
- Gestapelde druppels
- Donut modellering
- Optimalisatie op volume/diameter
- Meerdere configuraties vergelijken

**Dat is precies de bedoeling - een schone, simpele versie!**

## 💡 Tips

- **Hover over plots** voor gedetailleerde metingen
- **3D model** is interactief - sleep om te roteren
- **Export bestanden** worden opgeslagen in `exports/` map
- **Test script** kan opnieuw gebruikt worden na wijzigingen

## 🔧 Dependencies

Alle packages staan in `requirements.txt`:
- streamlit (UI framework)
- numpy (berekeningen)
- pandas (data structuren)
- plotly (visualisaties)
- scipy (numerieke integratie)
- ezdxf (DXF export)
- numpy-stl (STL export)

## 📚 Meer Info

Zie `README.md` voor:
- Volledige documentatie
- Technische details
- Natuurkundige achtergrond
- Troubleshooting
- Voorbeeldwaarden

---

**Veel succes! 💧**

