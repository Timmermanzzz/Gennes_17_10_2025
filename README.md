# De Gennes Druppelvorm Calculator

Een simpele, gebruiksvriendelijke Streamlit applicatie voor het berekenen en visualiseren van druppelvormen op basis van de Young-Laplace vergelijking en de natuurkundige principes van Pierre-Gilles de Gennes.

## 🎯 Wat doet deze applicatie?

Deze tool berekent de precieze vorm van vloeistofdruppels op basis van:
- **Oppervlaktespanning** (γₛ)
- **Dichtheid** van de vloeistof (ρ)
- **Zwaartekracht** (g)

Perfect voor:
- Ontwerpen van textiele waterreservoirs
- Educatieve demonstraties van vloeistoffysica
- 3D-print voorbereidingen
- CAD-integratie

## 🚀 Snelstart

### Installatie

1. Zorg dat je Python 3.8+ hebt geïnstalleerd
2. Installeer de dependencies:

```bash
pip install -r requirements.txt
```

### De applicatie starten

```bash
streamlit run app.py
```

De applicatie opent automatisch in je browser op `http://localhost:8501`

## 📖 Gebruiksaanwijzing

### Stap 1: Parameters instellen
In de zijbalk zie je drie belangrijke parameters:

- **γₛ (gamma_s)** - Oppervlaktespanningsparameter (N/m)
  - Standaard: 35000 N/m (typisch voor water)
  - Hogere waarde = stijvere oppervlak
  
- **ρ (rho)** - Dichtheid (kg/m³)
  - Standaard: 1000 kg/m³ (water)
  - Bepaalt het gewicht van de vloeistof
  
- **g** - Zwaartekracht (m/s²)
  - Standaard: 9.8 m/s² (op aarde)

### Stap 2: Afkap percentage (optioneel)
Met de slider kun je een percentage van de bovenkant afknippen voor een vlakke top:
- 0% = Volledige druppel
- 15% = Bovenkant afgesneden voor open reservoir
- 50% = Helft afgesneden

### Stap 3: Bereken
Klik op **"Bereken Druppel"** en de vorm wordt gegenereerd.

### Stap 4: Bekijk resultaten
Drie tabs beschikbaar:
- **2D Visualisatie**: Interactieve doorsnede plot (Plotly)
- **Specificaties**: Tabel met alle metrieken (volume, diameters, etc.)
- **Export**: Download STL of DXF bestanden

## 📊 Voorbeeldwaarden

### Waterbuffer (standaard)
```
γₛ = 35000 N/m
ρ = 1000 kg/m³
g = 9.8 m/s²
Afkap = 15%
```

### Dichtere vloeistof (bijv. zoutwater)
```
γₛ = 40000 N/m
ρ = 1200 kg/m³
g = 9.8 m/s²
Afkap = 10%
```

### Lage zwaartekracht (bijv. maan)
```
γₛ = 35000 N/m
ρ = 1000 kg/m³
g = 1.6 m/s²
Afkap = 0%
```

## 📁 Projectstructuur

```
Gennes_17_10_2025/
├── app.py              # Streamlit UI (hoofdapplicatie)
├── solver.py           # Young-Laplace solver
├── utils.py            # Hulpfuncties (volume, diameter)
├── visualisatie.py     # Plotly visualisaties
├── export.py           # STL en DXF export
├── requirements.txt    # Python dependencies
├── README.md           # Deze file
└── exports/            # Output directory voor bestanden
```

## 🔧 Technische Details

### Natuurkundige Basis
De applicatie implementeert de **Young-Laplace vergelijking** die de vorm van een druppel beschrijft onder invloed van:
- Oppervlaktespanning (γₛ)
- Hydrostatische druk (ρ × g × h)

### Belangrijke Parameters
- **κ (kappa)** = √(ρg/γₛ) - Vormparameter (m⁻¹)
- **H** = 2/κ - Karakteristieke hoogte (m)

### Berekende Metrieken
- Volume (m³)
- Maximale hoogte (m)
- Maximale diameter (m)
- Basis diameter (bodem)
- Top diameter (na afkapping)

## 💾 Export Formaten

### STL (Stereolithography)
- Voor 3D-printen
- Mesh van druppelvorm door rotatie van 2D profiel
- Binair formaat voor compactheid

### DXF (Drawing Exchange Format)
- Voor CAD software (AutoCAD, LibreCAD, etc.)
- 2D profiel als polyline
- Makkelijk te bewerken in CAD programma's

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Streamlit start niet
Controleer of je in de juiste directory bent:
```bash
cd Gennes_17_10_2025
streamlit run app.py
```

### Export bestanden niet gevonden
De `exports/` map wordt automatisch aangemaakt. Check of je schrijfrechten hebt.

## 📚 Referenties

Gebaseerd op het werk van:
- **Pierre-Gilles de Gennes** - Nobelprijswinnaar Natuurkunde (1991)
- **Young-Laplace vergelijking** - Fundamentele capillariteit theorie

## 📝 Licentie

Deze software is ontwikkeld voor technische en educatieve toepassingen.

## 🤝 Contact & Support

Voor vragen of problemen, raadpleeg de documentatie of neem contact op met de ontwikkelaar.

---

**Veel succes met je druppelontwerpen!** 💧

