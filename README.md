# Die Mitte hängt vom Strafmaß ab – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-lp-mitte-demo.streamlit.app/)** -->

Stück 3 der Reihe **Mittelwerte und Lagemaße** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Kante im Abhängigkeitsgraphen: Stück 2 (Median einer Zahlenreihe) → 3 als Kontrast (dieselbe Frage, jetzt mit einem Regler für das Strafmaß) und Stück 3 → 5 als Fix (in der Ebene gibt es für p = 1 keine Formel: `weiszfeld-demo`). Die Demo setzt Mittel und Median als bekannt voraus (`mittelwert-median-demo`) und wiederholt sie nur in einem Satz.

Vehikel: **ein Lager für Kunden in der Ebene** (gleiche Gewichte, Einheit 1 km, Luftlinie). Gesucht ist die **Lp-Mitte**, der Punkt c, der die Summe der Wege hoch p klein macht. Die Demo zeigt (a) den Pfad der Mitte über p auf der Achse und in der Ebene, (b) die **Regret-Matrix** „gewählt nach p, bewertet nach q“, (c) das **Kompromiss-p**, das vom Layout abhängt, (d) die **leere Zone** zwischen zwei Häufungen, (e) p < 1 und den Sprung zur dichtesten Stelle, (f) die **Gewichtsiteration** mit ihrer Grenze.

**Der Buchstabe p.** In dieser Demo ist **p der Exponent der Strecke**. In `p-center-demo` bedeutet p die Zahl der Mittelpunkte, in Stück 7 der Reihe (Potenzmittel) heißt der Exponent anders. Hier gibt es genau eine Mitte; das Maximum-Ende heißt **1-Center** (kleinster umschließender Kreis), nie „p-Center“. Das Bewertungskriterium trägt den Buchstaben q.

Abgrenzung: `p-center-demo` rechnet den kleinsten umschließenden Kreis (Welzl) nur als Radiusvergleich und die Summe gegen das Maximum diskret („Preis der Fairness“); `kmeans-demo` vergleicht Mittelwert und Medoid; `standortplanung-demo` und Verwandte arbeiten mit diskreten Kandidaten. Keine von ihnen rechnet das Strafmaß als stetigen Regler, zeigt Summe, Quadratsumme und Maximum auf derselben Wolke, nennt p < 1 oder die Gewichtsiteration (Stichwortsuche in 455 lokalen und den per `gh api` gelesenen READMEs, Stand 2026-10-24).

## Warum dieses Problem

„Die Mitte“ ist keine Zahl, sondern eine Entscheidung darüber, wie ein langer Weg bestraft wird. Der Hook der Demo sind gemessene Befunde statt eines Lehrbuchbilds: (1) die Wahl nach p ist **asymmetrisch teuer** (mit einem Außenposten kostet die Wahl nach p = ∞, bewertet nach der Quadratsumme, 214,2 % Mehrkosten, die Wahl nach p = 2, bewertet nach dem längsten Weg, 55,1 %); (2) es gibt **kein p, das für alle Kriterien gut ist**, und das beste feste p hängt vom Layout ab; (3) der Pfad der Mitte ist **stetig, aber nicht monoton**; (4) für p < 1 springt die Mitte zur **dichtesten** Stelle, auch gegen die Mehrheit; (5) die **Gewichtsiteration** konvergiert ab p = 4 nicht mehr.

Kernfrage: *Wenn das Strafmaß von „nur zählen, wer nicht dort ist“ (p gegen 0) über Summe (1), Quadratsumme (2) bis zum längsten Weg (∞) wandert: wohin läuft die Mitte, was kostet die Wahl in den anderen Kriterien, und wo hört die Mitte auf, eine einzige Zahl zu sein?*

## Befunde und Korrekturen gegenüber dem Plan

Die Hypothesen des Einzelplans (`mittelwerte-planung/STUECK-3-lp-mitte.md`) wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen; die Vormessung des Plans (numpy-Zufall, Wegwerfskripte) liefert keine README-Zahl. Alle Zahlen: vier feste Zufalls-Seeds je Zelle, Mittel mit Streuung zwischen den Seeds.

| | Hypothese | Ergebnis | Zahl (diese Demo) |
|---|---|---|---|
| H1 | Der Pfad c(p) ist für p ≥ 1 stetig | **Ja** | Größter Schritt zwischen benachbarten p (Achse, 51 Werte, Normal) von p = 1,1 bis 3: 0,42 % / 0,11 % / 0,02 % der Spannweite bei Schrittweite 0,1 / 0,02 / 0,004 |
| H2 | Der Pfad läuft glatt (monoton) von der Medianmitte zur Mitte der Spannweite | **Nein** | Achse, 51 Werte: nur 30 % (Normal), 47 % (zwei Häufungen), 98 % (lognormal) der Pfade laufen ohne Rückschritt; Ebene, n = 30: Summe der Wege wächst nur in 52 % (Normal) der Instanzen strikt mit p |
| H3 | Außenposten bei (6; 6): Schwerpunkt +1,4 % Summe, längster Weg +55 % (Schwerpunkt) und +60 % (Median), 1-Center 2,38-fache Summe | **Ja** | 1,4 % / 55,5 % / 59,6 % / 137,5 % (n = 31, 150 Instanzen je Seed) |
| H4 | Die Regret-Matrix ist stark asymmetrisch: Außenposten ∞→2 214 % gegen 2→∞ 55 % | **Ja** | 214,2 % gegen 55,1 % (n = 30) |
| H5 | Es gibt ein Kompromiss-p, und es hängt vom Layout ab (8 / 16 / 3 / 3) | **Ja, mit anderen Zahlen für Normal** | n = 30, Raster von neun Werten: Normal p = 6 (nicht 8; 6,8 % gegen 6,9 % bei p = 8), Quadrat 16, Außenposten 3, zwei Häufungen 3 |
| H6 | p muss groß sein, um das 1-Center zu erreichen | **Ja** | Abstand in Radien bei p = 8: 5,7 bis 11,7 %, bei p = 32: 2,3 bis 2,6 % |
| H7 | Das Mittel liegt bei zwei fast gleichen Häufungen in der leeren Zone, der Median nie | **Ja** | Abstand 10 km, 60 % in der großen Häufung: Mittel in 80 % der Instanzen, Median in 0 % |
| H8 | Für p < 1 ist der Minimierer ein Kunde; die Mitte springt zur dichtesten Stelle | **Ja** (Achse) | Achse, s = 3: Mitte in der dichten kleinen Häufung in 0 % (p = 1), 2 % (0,7), 84 % (0,5), 100 % (0,4); Ebene: Optimum auf einem Kunden in 98 bis 100 % bei p = 0,5 |
| H9 | Eine Gewichtsiteration löst alle p | **Nein** | Ohne Dämpfung konvergiert sie bei p = 4 / 6 / 8 in 0 % der Läufe (Grenze 5.000 Schritte), mit Dämpfung 1/(p − 1) in 100 % |
| H10 | Zufallsstarts finden das Minimum für p < 1 nur teilweise | **Ja**, mit anderen Zahlen | Achse, 200 Starts: innerhalb 0,2 des besten Minimums enden 61 % (p = 0,7), 7 % (0,5), 3 % (0,2); Ebene: 100 % / 99 % / 37 % |
| H11 | Ab einem Außenposten-Abstand a von 3 kippt das 1-Center | **Ja** | Regret p = 1 nach längstem Weg: 14,1 % (a = 2), 31,3 % (a = 3), 59,6 % (a = 6); feinere Bestimmung nicht untersucht |

### Korrekturen gegenüber dem Plan

- **p = 1 bei zehn Werten ist ein Intervall.** Der Plan nannte 5,51; bei den Werten 1 bis 9 und 40 ist die Mitte für p = 1 das Intervall 5 bis 6. Die App zeigt das Intervall und legt für Bilder die Mitte in die Intervallmitte.
- **Stetig heißt nicht monoton.** Der Plan sagte „wandert stetig“; stetig stimmt (H1), eine glatte Wanderung nicht (H2). Aussagen „von Median zu Spannweitenmitte“ gelten nur für die Endpunkte, nicht für den Verlauf. In der Ebene sind die Rückschritte größer als in der Vormessung des Plans (dort höchstens 1,0 %): bis 6,1 % des längsten Wegs (Quadrat, n = 30), bis 3,1 % der Summe (zwei Häufungen, n = 100).
- **Eine Gewichtsiteration für alle p stimmt nicht.** Ohne Dämpfung konvergiert sie bis p = 3 (bei zwei Häufungen steigt der Zielwert bei p = 3 in 6 % der Läufe), bei p ≥ 4 nie; mit der Dämpfung 1/(p − 1) in höchstens 123 Schritten (größter Wert über alle Läufe mit p ≥ 4).
- **1-Center exakt: 2,38.** Der Plan nannte 2,42 (eine Nelder-Mead-Näherung); mit dem exakten kleinsten umschließenden Kreis ist die Summe im 1-Center 2,375-mal die Summe im geometrischen Median (137,5 % mehr).
- **Zufallsstarts, p < 1:** Der Plan nannte 64 / 13 / 1 % bei p = 0,7 / 0,5 / 0,2 aus einer Ziehung; mit vier Ziehungen sind es 61 / 7 / 3 %. Das Kriterium „innerhalb 0,2 des besten Minimums“ vermischt „falsche Häufung“ und „anderer Kunde derselben Häufung“; „in der richtigen Häufung“ enden 68 % (p = 0,5) und 64 % (p = 0,2) der Starts.
- **Große Sprünge für p < 1 nur bei verschiedener Dichte:** Der Pfad ist eine Treppe von Kunde zu Kunde, der größte Schritt bleibt bei feinerer Schrittweite (2,93 % der Spannweite bei Schrittweite 0,004 statt 3,06 % bei 0,1); groß werden die Sprünge zwischen Häufungen nur bei verschiedener Dichte (H8).
- **Neu:** Kostenmatrix mit exaktem 1-Center, Kompromiss-p, Schwelle des Außenposten-Abstands, leere Zone in der Ebene, Annäherung an das 1-Center, p < 1 in der Ebene (Optimum liegt für p ≥ 0,7 oft im Innern), Dämpfung der Gewichtsiteration.
- **Umsetzung weicht ab:** (1) Der Regler n geht bis 300 (nicht 5.000): Die Karte zeichnet n Kunden, und p < 1 braucht die Abstandsmatrix der Kunden. (2) Der Regler „Fortsetzung des Starts“ des Plans entfällt: Eine Fortsetzung von p = 1 abwärts war in der Vormessung nicht belastbar, und eine Wirkung wird nicht behauptet. (3) Die Achsenansicht hat eigene Regler (Werte, Anzahl, Zufalls-Seed) im Abschnitt E1 statt eines Reglers in der Seitenleiste. (4) Das Kompromiss-p kommt aus einem Raster von neun Werten und n = 10, 30, 100 (Plan: ein Raster, n = 30).

## Modell

- **Instanz.** Kunden an Orten x_i in der Ebene (Einheit 1 km), gleiche Gewichte, eigener SplitMix64-Generator (`lpm_rng.py` ist die wörtliche Kopie aus `reduktionskette-demo/red_scenario.py`, geprüft von `tests/test_copies.py`), damit alle Instanzen auf jeder Plattform gleich sind. Layouts: Normal (N(0, I)), Quadrat, Außenposten (n − 1 normalverteilte Kunden und einer bei (a; a)), zwei Häufungen (Anteil `share` um (0; 0) mit wählbarer Streuung, der Rest mit Streuung 1 um (Abstand; 0)). Auf der Achse zusätzlich lognormal und ein Layout mit zwei Streuungen (Dichte).
- **Aufgabe.** Minimiere Σ ‖x_i − c‖^p. Für p > 1 ist die Zielfunktion streng konvex (genau eine Mitte); für p < 1 ist jeder Kunde ein lokales Minimum.
- **Drei Kriterien.** Summe der Wege (q = 1), Quadratsumme (q = 2), längster Weg (q = ∞); die besten Mitten dafür sind der geometrische Median, der Schwerpunkt und der Mittelpunkt des kleinsten umschließenden Kreises. **Regret** = Kosten_q(c(p)) / Kosten_q(c(q)) − 1.
- **Gewichtsiteration und Dämpfung.** Die Iteration c ← Σ w x / Σ w mit w = d^(p−2) ist für p = 1 das Weiszfeld-Verfahren. Die Dämpfung 1/(p − 1) und die Begründung, warum der Zielwert für p ≤ 2 nicht steigt (Majorisierung), sind **eigene Herleitungen**, keine Quellenaussagen; sie stehen im Expander **📐 Mathematische Formulierung** der App. Für die Lp-Mitte und die Gewichtsiteration wurde **keine Literatur gelesen**; die Konvergenz ist nur gemessen, nicht bewiesen. Der kleinste umschließende Kreis ist nach der Beschreibung des Verfahrens (Welzl) umgesetzt, die Originalarbeit wurde nicht eingesehen.

## Methodik

| Baustein | Datei |
|---|---|
| Kosten, Newton (p > 1), Gewichtsiteration (ungedämpft, gedämpft, viele Starts), Achse exakt, 1-Center (Welzl), p < 1 in der Ebene | `lpm_center.py` |
| Löser für p = 1 (Vardi–Zhang) | `lpm_geomed.py` (wörtliche Kopie einzelner Bausteine aus `weiszfeld-demo`, geprüft von `tests/test_copies.py`) |
| Regret-Matrix, Kompromiss-p, leere Zone | `lpm_regret.py` |
| Generator (SplitMix64-Kopie), Layouts, Achsen-Layouts | `lpm_rng.py`, `lpm_scenario.py` |
| Messreihen E1 bis E7 (Funktionen für die Live-Knöpfe und die Vorrechnung) | `lpm_experiments.py`, `tools/make_results.py` → `data/lpm_results.json` |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `data/lpm_results.json` und sind in `tests/test_claims.py` nachgerechnet.

**E1 Der Pfad.** Zahlenbeispiel auf der Achse (die Werte 1 bis 9 und 40, exakt): p = 1 ist das **Intervall 5 bis 6**, p = 2 gibt 8,500, p = 8 gibt 19,079, p = 32 gibt 20,434, p = ∞ gibt 20,500 (die Mitte der Spannweite): stetig, aber langsam.

Anteil der Pfade, die von p = 1 bis p = ∞ nie gegen ihre Gesamtrichtung laufen („strikt“, in %, Achse, ungerades n, 125 Instanzen je Seed und Zelle):

| Layout | n = 9 | n = 51 | n = 201 |
|---|---|---|---|
| Normal | 43 | 30 | 28 |
| Lognormal | 78 | 98 | 100 |
| Außenposten | 83 | 62 | 58 |
| Zwei Häufungen | 55 | 47 | 44 |

Bei zwei Häufungen (n = 51) sind es bis p = 8 noch 92 %. Der Rückschritt ist klein: bei Normal und n = 51 im Mittel 0,36 % der Spannweite, in der ungünstigsten Instanz 4,7 %.

In der Ebene (14 Stützstellen von p = 1 bis ∞, 50 Instanzen je Seed): Die Summe der Wege wächst nur in 52 % (Normal, n = 30), 34 % (Quadrat), 91 % (Außenposten) und 54 % (zwei Häufungen) der Instanzen strikt mit p; der längste Weg fällt nur in 50 % / 44 % / 88 % / 57 %. Größter Rückschritt aller Zellen: 6,1 % des Wertes.

Wie groß muss p sein (n = 30, 100 Instanzen je Seed)? Abstand von c(p) zum 1-Center in Radien des Kreises:

| p | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| Normal (%) | 12,8 | 7,6 | 4,2 | 2,3 | 1,3 | 0,6 |
| Quadrat (%) | 8,2 | 5,7 | 3,8 | 2,5 | 1,5 | 0,8 |
| Außenposten (%) | 31,1 | 11,7 | 5,0 | 2,5 | 1,3 | 0,7 |
| Zwei Häufungen (%) | 11,8 | 7,6 | 4,5 | 2,6 | 1,4 | 0,8 |

Mit dem Außenposten überschreitet c(p) den längsten Weg des 1-Centers um 29,8 % (p = 4), 9,5 % (p = 8), 2,6 % (p = 16) und 0,7 % (p = 32); die Summe liegt dort gegenüber dem geometrischen Median 47,8 % / 107,1 % / 129,4 % / 136,1 % höher (Normal: 2,0 % / 4,3 % / 6,0 % / 6,8 %).

**E2 Die Kostenmatrix** (Regret in %, Zeile gewählt nach p, Spalte bewertet nach q; n = 30, Mittel über vier Seeds mit je 200 Instanzen, Streuung zwischen den Seeds höchstens 2,1 Prozentpunkte):

| Layout | 1→2 | 1→∞ | 2→1 | 2→∞ | ∞→1 | ∞→2 |
|---|---|---|---|---|---|---|
| Normal | 1,0 | 15,8 | 0,5 | 13,4 | 8,3 | 13,1 |
| Quadrat | 1,2 | 12,3 | 0,5 | 8,3 | 2,7 | 3,6 |
| Außenposten (a = 6) | 1,7 | 59,4 | 1,4 | 55,1 | 137,7 | 214,2 |
| Zwei Häufungen (70/30, Abstand 6) | 13,1 | 44,1 | 7,2 | 21,3 | 22,1 | 14,8 |

Mit n: Normal 1→∞ 22,6 / 15,8 / 10,8 / 8,5 % (n = 10 / 30 / 100 / 300); Quadrat 1→∞ 21,8 / 12,3 / 6,7 / 3,7 %; **Außenposten ∞→2 wächst** (90,5 / 214,2 / 325,8 / 354,8 %), weil ein fester Außenposten gegen immer mehr Kernpunkte steht; bei zwei Häufungen ändert sich 2→1 kaum (8,6 / 7,2 / 6,8 / 6,7 %).

**E3 Der Außenposten** (n = 31, 150 Instanzen je Seed): Regret von p = 1 bewertet nach dem längsten Weg und von p = ∞ bewertet nach der Quadratsumme (in %):

| a | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| 1→∞ | 15,0 | 15,5 | 14,1 | 31,3 | 45,2 | 59,6 | 68,5 | 77,8 | 86,4 |
| ∞→2 | 12,0 | 12,1 | 11,3 | 47,9 | 102,9 | 217,6 | 318,2 | 442,8 | 558,6 |

Bei a = 6 (n = 31, 30 Normalpunkte und ein Außenposten): Summe im Schwerpunkt 1,4 % über dem geometrischen Median, längster Weg im Schwerpunkt 55,5 % und im geometrischen Median 59,6 % über dem Radius des 1-Centers, Summe im 1-Center 137,5 % (das 2,375-fache) über dem geometrischen Median.

**E4 Das Kompromiss-p** (größter Regret über die drei Kriterien, gemittelt über Instanzen, Raster {1; 1,5; 2; 3; 4; 6; 8; 16; ∞}; n = 30, 150 Instanzen je Seed):

| Layout | bestes festes p | größter Regret dort (%) | je Instanz frei (%) | bei p = 1 (%) | bei p = 2 (%) | bei p = ∞ (%) |
|---|---|---|---|---|---|---|
| Normal | 6 | 6,8 | 5,0 | 15,3 | 12,9 | 12,6 |
| Quadrat | 16 | 3,2 | 2,4 | 12,0 | 8,4 | 3,9 |
| Außenposten | 3 | 42,2 | 40,7 | 59,1 | 54,8 | 211,7 |
| Zwei Häufungen | 3 | 14,8 | 13,7 | 44,1 | 21,1 | 21,9 |

Das beste feste p hängt auch von n ab: bei n = 10 / 30 / 100 Normal 6 / 6 / 8, Quadrat 8 / 16 / ∞, Außenposten 3 / 3 / 4, zwei Häufungen 3 / 3 / 3. Das Mittel (p = 2) ist über alle drei Kriterien kein Kompromiss.

**E5 Die leere Zone** (Anteil der Instanzen, in denen der nächste Kunde weiter als 1,5 km von der Mitte entfernt ist; zwei Häufungen, 100 Kunden, 150 Instanzen je Seed):

| Anteil der großen Häufung | 50 % | 60 % | 70 % | 80 % |
|---|---|---|---|---|
| Abstand 10 km, p = 1 (%) | 97 | 0 | 0 | 0 |
| Abstand 10 km, p = 2 (%) | 99 | 80 | 6 | 0 |
| Abstand 10 km, p = ∞ (%) | 98 | 98 | 97 | 97 |

Bei Abstand 6 km liegt keine der drei Mitten in mehr als 2 % der Instanzen in der leeren Zone. Auf der Achse (200 Werte, N(0, 1) und N(8, 1), 100 Instanzen je Seed): Bei 55 % in der großen Häufung liegt c(p) für p = 1,5 / 2 / 3 / 4 in 27 / 60 / 72 / 74 % der Instanzen weiter als 1,0 vom nächsten Wert, bei 60 % für p = 2 / 4 in 18 / 60 %, bei 80 % für p = 4 in 1 %; für p = 0,5 nie (die Mitte ist ein Wert).

**E6 p kleiner als 1.** Dichte gegen Masse auf der Achse (200 Werte, 60 % streuen mit s um 0, 40 % dicht (Streuung 0,5) um 8; Anteil der Instanzen, in denen die Mitte in der dichten kleinen Häufung liegt, 100 Instanzen je Seed):

| p | 2 | 1 | 0,7 | 0,5 | 0,4 | 0,3 | 0,2 | 0,1 |
|---|---|---|---|---|---|---|---|---|
| s = 2 (%) | 0 | 0 | 0 | 0 | 1 | 20 | 58 | 87 |
| s = 3 (%) | 0 | 0 | 2 | 84 | 100 | 100 | 100 | 100 |
| s = 4 (%) | 0 | 37 | 86 | 100 | 100 | 100 | 100 | 100 |

Bei s = 0,5 und s = 1 liegt die Mitte für kein p in der kleinen Häufung. In der Ebene (100 Kunden, 60 % mit Streuung s um (0; 0), 40 % mit Streuung 1 um (8; 0), 30 Instanzen je Seed): Bei s = 3 liegt die Mitte für p = 1 / 0,7 / 0,5 / 0,4 / 0,3 in 31 / 62 / 91 / 98 / 100 % der Instanzen näher an der kleinen Häufung, bei s = 1 nie.

Der Pfad ist eine **Treppe** (größter Schritt zwischen benachbarten p in % der Spannweite, 51 Werte, 30 Instanzen je Seed):

| Layout, Bereich von p | Schrittweite 0,1 | 0,02 | 0,004 |
|---|---|---|---|
| Normal, 0,1 bis 0,9 | 3,06 | 2,96 | 2,93 |
| Normal, 1,1 bis 3 | 0,42 | 0,11 | 0,02 |
| Zwei Häufungen, 0,1 bis 0,9 | 2,82 | 2,59 | 2,57 |
| Zwei Häufungen, 1,1 bis 3 | 1,80 | 0,49 | 0,10 |

Zufallsstarts auf der Achse (200 Werte, 70 % N(0, 1) und 30 % N(8, 1), 200 Starts je Instanz, 6 Instanzen je Seed; B: Anteil der Enden innerhalb 0,2 des besten Minimums, A: in dessen Häufung):

| p | 2 | 0,95 | 0,9 | 0,7 | 0,5 | 0,2 |
|---|---|---|---|---|---|---|
| B (%) | 100 | 100 | 100 | 61 | 7 | 3 |
| A (%) | 100 | 100 | 100 | 100 | 68 | 64 |
| Enden auf einem Kunden (%) | 0 | 100 | 100 | 100 | 100 | 100 |
| verschiedene Enden | 1 | 2 | 3 | 15 | 58 | 84 |

In der Ebene liegt das beste gefundene Minimum (alle Kunden und 20 Zufallsstarts, 50 Instanzen je Seed) bei p = 0,9 / 0,7 / 0,5 / 0,3 / 0,1 auf einem Kunden in 61 / 99 / 100 / 100 / 100 % (Normal, n = 5), 40 / 89 / 100 / 100 / 100 % (n = 10) und 14 / 62 / 98 / 100 / 100 % (n = 30); dazwischen liegt das Optimum oft im Innern, wie bei p = 1. Zufallsstarts in der Ebene (100 Kunden, 70/30, Abstand 8, 100 Starts, 3 Instanzen je Seed): innerhalb 0,2 des besten Minimums enden 100 % (p = 0,7), 99 % (0,5) und 37 % (0,2).

**E7 Die Gewichtsiteration** (30 Kunden, Start im Schwerpunkt, Schritt < 10⁻¹⁰, Grenze 5.000 Schritte, 160 Instanzen je Layout und p):

| p | 1,5 | 2,5 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| ohne Dämpfung: konvergiert (%) | 100 | 100 | 100 | 0 | 0 | 0 |
| ohne Dämpfung: Zielwert steigt (%, zwei Häufungen) | 0 | 0 | 6 | 100 | 100 | 100 |
| ohne Dämpfung: Schritte (Median, Normal) | 17 | 17 | 41 | – | – | – |
| mit Dämpfung: Schritte (Median, Normal) | 17 | 13 | 17 | 24 | 34 | 42 |
| mit Dämpfung: Schritte (Median, zwei Häufungen) | 23 | 17 | 25 | 40 | 66 | 89 |
| Newton: Schritte (Median, Normal) | 3 | 3 | 3 | 3 | 4 | 4 |

Mit Dämpfung konvergiert jeder Lauf, und der Zielwert steigt nie. Bei p = 3 braucht die ungedämpfte Iteration mit zwei Häufungen im Median 198 Schritte.

## Ehrliche Grenzen

- **Luftlinie in der Ebene, gleiche Gewichte, ein einziges Lager.** Keine Straßen, keine Gewichte, keine Kapazität, keine Fixkosten; Summe, Quadratsumme und längster Weg sind Stellvertreter.
- **Simulierte Layouts.** Alle Häufigkeiten und Raten gelten für die vier Layouts dieser Demo; echte Kundenlagen können abweichen.
- **Kompromiss-p nur für ein Raster.** Das beste p stammt aus neun Werten, vier Layouts und n = 10, 30, 100; dazwischen und für andere Layouts ist es nicht untersucht. Bei Normal liegen p = 6 und p = 8 nahe beieinander (6,8 % und 6,9 %).
- **p < 1 mit 20 Zufallsstarts.** In der Ebene werden alle Kunden und 20 Zufallsstarts verglichen: das beste von endlich vielen Kandidaten, kein Beweis des globalen Minimums.
- **Konvergenz nur gemessen.** „Die Gewichtsiteration konvergiert ab p = 4 nicht“ ist eine Messung an 160 Instanzen je Layout und p mit der Grenze von 5.000 Schritten, kein Beweis; ebenso die Monotonie für p ≤ 2 (die Begründung über die Majorisierung ist eine eigene Herleitung, gemessen nur für p = 1,5 und 2,5).
- **Vier Seeds je Zelle.** Die angegebene Streuung ist die zwischen diesen Seeds, kein Konfidenzintervall. Die Zufallsstarts auf der Achse und in der Ebene beruhen auf wenigen Instanzen (6 und 3 je Seed) und streuen stark (B bei p = 0,7: Streuung 21 Prozentpunkte).
- **Fortsetzung als Abhilfe gegen schlechte Starts** (p von 1 abwärts verfolgen) wurde nicht untersucht; die App behauptet keinen Nutzen.
- **Mit echtem Mausklick nicht geprüft:** Die Karte ist nicht anklickbar; alle Eingaben laufen über Regler und Knöpfe.

## Verifikation

- **Kern gegen Handrechnung** (`tests/conftest.py`, `tests/test_center.py`, `tests/test_regret.py`): Achse 0, 0, 0, 12 mit der geschlossenen Formel c(p) = 12 / (1 + 3^(1/(p−1))); Achse 1, 2, 3, 10 (p = 2: 4, p = 3: −2 + √47, p = ∞: 5,5, p = 1: Intervall 2 bis 3); Achse 0, 1, 10, 11 mit p = 0,5 (Gleichstand, Regel: der kleinere Wert); rechtwinkliges Dreieck (1-Center (2; 1,5), Radius 2,5, Schwerpunkt, Summe 7,5 gegen 6,9184, Quadratsumme 18,75 gegen 16,667); zwei Punkte; die Dämpfung 1/(p − 1) als exakter Newton-Schritt auf der Achse.
- **Orakel** (`tests/test_oracle_lp_mitte.py`): Newton gegen `scipy.optimize` (BFGS mit analytischem Gradienten) und die Optimalitätsbedingung (Kräftegleichgewicht), Achse gegen die Nullstelle der Ableitung (Brent) und ein Gitter mit 20.001 Punkten, 1-Center gegen die Aufzählung aller Paare und Tripel und gegen `scipy`, Gewichtsiteration gegen Newton (auf 10⁻⁶), p < 1 gegen ein Gitter, Regret-Matrix gegen `scipy`-Minimierer. Größerer Lauf im Scratchpad (2026-10-25, nicht im Repo): Newton gegen scipy je 500 Instanzen für p = 1,2 / 1,5 / 2 / 3 / 5 / 10 / 30, größte relative Abweichung 7·10⁻¹⁵ (nie ein Fall über 10⁻⁹, Newton in allen ok); 3.000 Instanzen kleinster Kreis gegen Aufzählung: Radius 1,8·10⁻¹⁵, Mittelpunkt 3,3·10⁻¹⁵; gegen scipy (n = 30, 300 Instanzen) fand scipy nie einen kleineren Radius; Achse gegen ein Gitter in 5.500 Fällen nie schlechter; p < 1 in der Ebene gegen ein 801×801-Gitter in 2.100 Fällen nie schlechter (Optimum im Innern bei n = 3 bis 5: 38,7 % bei p = 0,95, 26,3 % bei 0,9, 8,7 % bei 0,8, 0,7 % bei 0,7, darunter 0).
- **Zweig-Tests** (`tests/test_zweige.py`, `tests/test_presets.py`): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab: gedämpfte gegen ungedämpfte Iteration, jeder Löser, jeder Start, Außenposten-Abstand, Anteil, Abstand und Streuung der Häufungen, jedes Layout gegenüber Normal, jedes Bewertungskriterium.
- **Schleifenabbruch und Rechenbudget** (`tests/test_termination.py`, `tests/test_worstcase.py`): Newton, Gewichtsiteration, Bisektion und Welzl enden im eigenen Prozess unter einer Zeitgrenze; die größte Instanz (n = 300) für jedes p-Regime läuft durch.
- **Messreihen** (`tests/test_experiments.py`, `tests/test_results.py`): Hilfsfunktionen an Handfällen, vorgerechnete Zellen werden von den Live-Funktionen mit denselben Seeds reproduziert, die Datei ist aktuell gegen die Quelldateien (Prüfsumme).
- **Mutationsprüfung** (`oracle_mutation.py` aus dem Website-Repo, inkrementell, Zustand in `tests/mutation_state.json`): gezogen wurden alle 151 Stellen von `lpm_center.py` und `lpm_regret.py`; Stand beim Abschluss: **133 gefangen, 18 als äquivalent festgehalten, 0 offen**. Im ersten Lauf überlebten 32 Mutanten die ganze Suite (Rate 79 %); geschlossen sind der Rückfall für kollineare Punkte im Umkreis, der Rahmen der Zufallsstarts für p < 1, die genaue Schrittzahl von Newton und Gewichtsiteration und der Abstand der leeren Zone. Als äquivalent festgehalten sind Stellen ohne Wirkung auf das Ergebnis: Mischung der Reihenfolge und Toleranz im Welzl-Verfahren (der Kreis ist eindeutig), Skalierung und Startwert der Armijo-Suche, Vorzeichen in der Armijo-Bedingung und in der Regularisierung, Abbruch nach der Schrittlänge statt nach der Kraft (eine Iteration mehr), unerreichbare Zweige (Newton-Richtung ohne Abstieg, p = 1 schon vorher behandelt, Gleichstand der Ableitung), 99 statt 100 Bisektionsschritte und ein Nullwächter, der nur Laufzeit spart. Nicht gezogen wurden `lpm_experiments.py`, `lpm_evaluation.py`, `lpm_scenario.py` (das Werkzeug wählt sie nicht als Kernmodul; die Verdikte prüft `tests/test_verdicts.py`), Darstellung und die kopierten Module `lpm_rng.py` und `lpm_geomed.py`. „Mutation sauber“ gilt hier nur für die beiden gezogenen Module.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten. **Aussagen** (`tests/test_claims.py`): jede Zahl dieser README.
- **Oberfläche:** AppTest (jeder Knopf, jeder Regler am Minimum und Maximum, jede Option der Auswahlregler, Eckkombinationen, gesperrte Regler), Layout (aktives Preset, Hervorhebung), Footer, PDF, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App |
| `lpm_constants.py` | Regler-Grenzen, Raster, Farben, Presets mit Hilfetexten |
| `lpm_center.py` | Kern: Kosten, Newton, Gewichtsiteration, Achse exakt, 1-Center, p < 1 |
| `lpm_geomed.py` | Löser für p = 1 (Kopie einzelner Bausteine aus `weiszfeld-demo`) |
| `lpm_regret.py` | Regret-Matrix, Kompromiss-p, leere Zone |
| `lpm_scenario.py`, `lpm_rng.py` | Instanzen; SplitMix64 (Kopie) |
| `lpm_experiments.py` | Messreihen-Funktionen, Prüfsumme der Quellen |
| `lpm_results.py` | Laden der vorgerechneten Messreihen |
| `lpm_evaluation.py` | Kennzahlen der Karte, Verdikte |
| `lpm_visualization.py`, `lpm_pdf_export.py`, `lpm_presets.py`, `lpm_layout.py` | Figuren, PDF, Permalink und Presets, fester Kopf |
| `tools/make_results.py` | erzeugt `data/lpm_results.json` |
| `data/lpm_results.json` | die vorgerechneten Messreihen mit Prüfsumme |
| `tests/` | Tests (siehe oben) |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Laufzeit, Test und Gegenproben (scipy nur dort), CI |

## Bewusst nicht umgesetzt

- **Weiszfeld im Detail, Datenpunkt-Schwelle, Vardi–Zhang:** Stück 5 der Reihe (`weiszfeld-demo`); hier ist p = 1 eine Stelle auf dem Pfad.
- **Median einer Zahlenreihe:** `mittelwert-median-demo`. **Potenzmittel M_r:** Stück 7 der Reihe (ein anderes „p“).
- **Gewichte, Kapazität, Straßennetz, Fixkosten:** Standortplanung mit diskreten Kandidaten (`standortplanung-demo`); **mehrere Mitten:** `kmeans-demo` und `p-center-demo`.
- **Konvergenzbeweis der Gewichtsiteration:** die Konvergenz steht als Messung da, Quellen wurden nicht gelesen.
- **Fortsetzung der Mitte von p = 1 abwärts** als Abhilfe gegen schlechte Starts: nicht untersucht.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Der Code verwendet nur Sprach- und Bibliotheksmerkmale, die unter Python 3.12 (CI) laufen; die Python-Version der Streamlit Cloud ist eine App-Einstellung.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Messreihen werden mit `python tools/make_results.py --part <m1a|m1b|m2|m4|m5|m6a|m6b|m7>` (jeder Teil einzeln, parallel möglich) und `python tools/make_results.py --merge` neu erzeugt; `tests/test_results.py` meldet, wenn die Datei hinter den Quelldateien zurückliegt.

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
