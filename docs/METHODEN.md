## Kraag methoden voor herstel van druppelkromming

Context: een met water gevuld membraanreservoir (druppelvorm) wordt aan de bovenzijde open gemaakt (afkap). We willen de optimale Young–Laplace kromming terugbrengen met een kraag (donut) die met een waterkolom de weggevallen hydrostatische druk compenseert.

### Terminologie en symbolen
- `γₛ` (gamma_s): voorgespannen membraanparameter (N/m). Eigenschap/instelling van het zeildoek; géén gevolg van de druk.
- `ρ`: massadichtheid water (kg/m³), `g`: zwaartekracht (m/s²).
- `h_free`: hoogste waterniveau van de ongesneden druppel; `h_cut`: hoogte van het afkapvlak; `Δh = h_free − h_cut`.
- `H = (1/R₁ + 1/R₂)/2`: gemiddelde kromming van het membraan; Young–Laplace: `Δp = γₛ (1/R₁ + 1/R₂) = 2 γₛ H`.
- Opening/afkapdiameter: vaste diameter van de stijve rand (ring/koord); de ring kan niet rekken/krimpen.

---

## Methode 1 — Directe waterkolomcompensatie (huidige implementatie)
Doel: herstel dezelfde hydrostatische druk op de afkaprand met een kraag bovenop de stijve rand.

Stapplan
1) Genereer de ongesneden druppel (`γₛ, ρ, g`). Bepaal `h_free` en de beoogde afkaphoogte `h_cut` (via percentage of gewenste opening/diameter).
2) Bepaal de weggehaalde waterkolomhoogte: `Δh = h_free − h_cut`.
3) Stel de kraag (donut) zó in dat de effectieve waterkolomhoogte in de kraag `= Δh`. Optioneel extra vrije boord voor klotsen: `head_total = Δh + extra`.
4) Donut-geometrie: grote straal `R_major = opening_diameter/2`; kleine straal `r_top` gekozen om `head_total` te accommoderen; waterkanaalstraal `r_water ≈ r_top` (wanddikte verwaarloosd). Benaderd watervolume: `V_torus ≈ 2π² R_major r_water²`.
5) Bevestiging: donut bovenop de stijve rand, waterdichte las; water in donut staat in open verbinding met het reservoir.

Fysische rationale
- Voor afkap was de druk op de rand: `p_atm + ρ g (h_free − h_cut)`.
- Na afkap zonder kraag is dat `p_atm` lager met `ρ g Δh`.
- Met kraaghoogte `Δh` wordt de druk precies hersteld: `p_atm + ρ g (h_cut − z) + ρ g Δh = p_atm + ρ g (h_free − z)`.
- Omdat `γₛ` niet verandert, herstelt ook de Young–Laplace-balans en dus de oorspronkelijke kromming.

Eigenschappen
- Snel, robuust, direct gekoppeld aan het fysische beeld van een ontbrekende waterkolom.
- Geeft een eenduidige `Δh` zonder iteratie.

---

## Methode 2 — Iteratief krommingsmatchen (alternatieve aanpak)
Doel: vind de kraaghoogte `Δh*` waarbij de randkromming van de afgekapte vorm mét kraag gelijk is aan die van de ongesneden referentie op de afkaphoogte.

Definities
- `H_target`: gemiddelde kromming van de ongesneden druppel ter hoogte `h_cut` (uit de referentieprofielpunten, numeriek bepaald).
- `H_rim(Δh)`: gemiddelde kromming aan de afkaprand wanneer de kraaghoogte `Δh` wordt aangebracht.

Zoekprobleem
- Vind `Δh*` zodat `F(Δh) = H_rim(Δh) − H_target = 0`.
- Praktisch te vinden met bisection/secant: kies `Δh_low, Δh_high` en zoek de wortel.

Procedure
1) Referentie: maak de ongesneden druppel, bepaal `H_target` rond `h_cut` (fit/twee-zijdige differentie).
2) Voor een kandidaat `Δh` simuleer de afgekapte vorm met kraag (drukaanvulling `ρ g Δh`) en bepaal `H_rim(Δh)`.
3) Itereer op `Δh` tot |F(Δh)| < tolerantie. Resultaat `Δh*` en bijbehorende vorm/metrics.

Opmerkingen
- In het ideale Young–Laplace-model valt `Δh*` samen met de directe methode: `Δh* = h_free − h_cut`. De iteratieve aanpak is nuttig als je `H_target` numeriek uit ruwe data wilt halen, of wanneer extra effecten/meetinaccuraatheid spelen.
- De methode levert meteen gevoeligheden (helling van `H_rim` vs. `Δh`).

---

## Verschillen en toepassingsgebied
- M1: expliciet, snel, direct drukherstel; geschikt voor ontwerp en UI-ervaring.
- M2: valideert en kalibreert; handig voor studies, meetdata of alternatieve solvers; levert gevoeligheidsinformatie.

## Visualisaties (voorstel)
- `Δh` vs. krommingsfout `F(Δh)`; nulpunt = oplossing.
- `Δh` vs. `H_rim(Δh)` met horizontale lijn `H_target`.
- `Δh` vs. ontwerp-uitvoer: max_diameter, max_hoogte, volume, torus_water.
- Profiel-overlays: referentie (ongesneden) + afgekapte vormen voor `Δh` {onder, optimaal, boven}.

## UI-koppeling (indicatief)
- Toggle "Kraag methoden": Methode 1 (direct) en Methode 2 (iteratief).
- Bij Methode 2: slider/bereik voor `Δh`, knop "Zoek optimale Δh", en bovengenoemde grafieken.

## Aannames en beperkingen
- Membraan wordt beschreven met voorgespannen `γₛ` (constante per scenario); materiaalrek en niet-lineaire constitutieve effecten zijn niet gemodelleerd.
- Water is in hydrostatisch evenwicht; dynamiek/klotsen wordt uitsluitend als extra vrije boord verwerkt.
- Donutwanddikte verwaarloosd; `r_water ≈ r_top`.

---

Laatste update: automatisch gegenereerd document voor ontwerpbesluiten en fysische rationale. Gebruik dit als naslag bij toekomstige uitbreidingen.


