# Cox-Regression: Merkmale ohne Form – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-cox-regression-demo.streamlit.app/)** -->

Teil der Linie **Überlebenszeit** (Zeit bis zum Ereignis bei Beobachtungen, die irgendwann abbrechen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Der Aufhänger ist die Abholung von Containern: Ein **Merkmal** eines Containers verändert die Abholrate; die **Cox-Regression** schätzt das Verhältnis, mit dem es sie vervielfacht,
ohne die Grundform der Rate zu kennen, und Container, die bei Fensterende noch nicht abgeholt sind (**zensiert**), gehen über die Risikomengen ein. Die Demo führt das Verfahren an vier
Containern von Hand vor, hält es in einer simulierten Welt gegen die Regression auf die beobachtete Zeit und schätzt die Grundrate nachträglich.

Gleicher Aufhänger Containerverweilzeit wie [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo), dort ohne Zensierung; hier ist die Verweilzeit weibullverteilt und wird durch ein Beobachtungsfenster abgeschnitten.

**Annahme in dieser Demo, sichtbar in der App und hier: Die Zensierung ist unabhängig vom Ereignis.** Die simulierte Welt erfüllt sie; die Demo prüft sie nicht.

## Warum dieses Problem

Wer wissen will, wie stark ein Merkmal die Abholung beschleunigt, hat meist viele Container, die noch nicht abgeholt sind. Die naheliegende Regression von log(beobachtete Zeit) auf das Merkmal
behandelt jede Zensierzeit wie eine Abholzeit. In der Welt dieser Demo (Weibull-Form 1,5, ein standardnormalverteiltes Merkmal, β = 0,7, 300 Container, Beobachtungsfenster 12 Tage, 54,4 % zensiert,
200 Läufe) liegt ihre Steigung bei −0,21 gegen die Wahrheit −β/k = −0,47, also bei 44 % der Wahrheit; Cox schätzt β im Mittel mit 0,70 gegen 0,70. Die Kernfrage ist: Wie stark verändert das
Merkmal die Rate, und wie weit führt die einfache Regression in die Irre? Ohne Zensierung (Fenster 1000 Tage, 0,8 % zensiert) trifft die Regression dagegen 99 % der Wahrheit: Der Fehler kommt
aus der Zensierung, nicht aus der Regression an sich.

## Modell

Rate eines Containers mit Merkmal $x$: $h(t \mid x) = h_0(t)\,e^{\beta x}$ mit unbekannter Grundrate $h_0$. Die simulierte Welt hat eine Weibull-Grundrate (Form $k$, Skala 8 Tage), ein
standardnormalverteiltes Merkmal und eine Zensierzeit, die zwischen 0 und dem Beobachtungsfenster gleichverteilt und von Merkmal und Abholzeit unabhängig ist. Die partielle Likelihood, Score,
Information, der Newton-Schritt, die Breslow-Grundrate und die Wahrheit der Regression (−β/k bei Daten ohne Zensierung) stehen im Expander „📐 Mathematische Formulierung“ der App, mit Lesehilfe und
Symboltabelle. Gleichstände der Zeiten behandelt der Kern nach Breslow. Der Kern (`cxr_core.py`) ist eine Kopie aus der Linien-Vorlage (`ueberlebenszeit-planung/kern/uz_core.py`, Stand 2026-11-10):
SplitMix64-Zufall, Eingabeprüfung, Kaplan–Meier und Cox mit Newton-Schrittweitenhalbierung und Konvergenzflag.

## Methodik

Die App hat drei Experimente.

1. **Die partielle Likelihood von Hand.** Vier Container mit Merkmal 1, 0, 1, 0 werden zu den Zeiten 1 bis 4 abgeholt. Die App zeigt die Tabelle der Risikomengen bei β = 0 in Brüchen (Mittel, Beitrag
   zum Score, Beitrag zur Information), die Kurve der partiellen Log-Likelihood über β, die Newton-Schritte und, an einem frei wählbaren β, dieselbe Tabelle in Gleitkomma. Vier Muster des Merkmals zeigen
   den Normalfall, das umgekehrte Muster (β̂ wechselt das Vorzeichen), die **perfekte Trennung** (kein endlicher Schätzer) und das **konstante Merkmal** (Information 0).
2. **Cox gegen Regression auf die Zeit** (Kernfrage). 200 Läufe mit festen Zufalls-Seeds; Regler: Container n je Lauf (10 bis 600), Beobachtungsfenster (4 bis 1000 Tage), in der Seitenleiste wahres β,
   Form k und der Zufalls-Seed der Daten. Gezeigt werden der mittlere Zensierungsgrad, Cox β̂, die beiden Regressionssteigungen (alle Container, nur Container mit Ereignis) gegen ihre Wahrheit, die Intervalle
   β̂ ± 1,96·SE der ersten 40 Läufe und die Überdeckung als Zählung. Läufe ohne endlichen Schätzer (Trennung, Information 0, ohne Ereignis) erscheinen als Hinweis mit Zahl und Grund; sie stehen weder im
   Mittel noch im Diagramm.
3. **Grundrate unbekannt.** Ein Datensatz (Lauf 1 der Welt): Die Container werden nach dem Merkmal in drei etwa gleich große Gruppen geteilt. Die Kaplan–Meier-Kurve jeder Gruppe (ohne Modell) steht
   gegen die Cox-Kurve der Gruppe (Breslow-Grundrate, gemittelt über die Mitglieder). Das Verdikt nennt den größten Abstand bis zu der Zeit, zu der in jeder Gruppe noch mindestens 10 Container im Risiko sind.

Jede Nicht-Zahl-Meldung der App kommt aus einem Status des Kerns: Der Wrapper `fit_cox` in `cxr_model.py` gibt entweder einen konvergierten Schätzer oder `ok = False` mit Grund (`ohne_ereignis`,
`singulaer`, `trennung`, `keine_konvergenz`) und nie eine stille Zahl.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus dem Code der Demo (SplitMix64, `cxr_model.py`) mit festen Zufalls-Seeds und stehen eingefroren in `tools/measure_cox_result.json`
(erzeugt mit `tools/measure_cox.py`). Welt, wenn nichts anderes steht: Weibull-Form 1,5, Skala 8 Tage, β = 0,7, standardnormales Merkmal, n = 300 Container, Beobachtungsfenster
12 Tage, 200 Läufe; der Zufalls-Seed der Läufe ist 35, Lauf r hat den Seed 35 · 1 000 003 + r. Die Tabellen unten werden von `tests/test_claims.py` Wort für Wort gegen `tools/measure_cox.py` geprüft
(siehe „Tests“).

**Cox gegen Regression auf die beobachtete Zeit (Messung 1, Standardwelt, 54,4 % zensiert).**

| Schätzer | Mittel über 200 Läufe (± Standardfehler des Mittels) | Wahrheit | Anteil der Wahrheit |
|---|---|---|---|
| Cox β̂ | 0,703 ± 0,007 | 0,70 | 100 % |
| Regression von log(Zeit) auf x, alle Container | −0,206 ± 0,004 | −0,47 | 44 % |
| Regression, nur Container mit Ereignis | −0,213 ± 0,006 | −0,47 | 46 % |

Stabilität über andere Zufalls-Seeds (je 200 Läufe, gleiche Welt):

| Zufalls-Seed | Cox β̂ | Regression, alle | Intervall enthält β |
|---|---|---|---|
| 35 | 0,70 | −0,21 | 193 von 200 |
| 1 | 0,71 | −0,21 | 186 von 200 |
| 2 | 0,69 | −0,21 | 187 von 200 |
| 3 | 0,70 | −0,21 | 184 von 200 |
| 4 | 0,71 | −0,21 | 190 von 200 |
| 5 | 0,72 | −0,21 | 189 von 200 |

Über alle sechs Zufalls-Seeds sind es 1129 von 1200 Läufen (94,1 %), in denen β̂ ± 1,96·SE das wahre β enthält; je Zufalls-Seed schwankt der Anteil zwischen 92,0 % und 96,5 % (Binomial-Standardfehler
bei 200 Läufen: rund 1,6 Prozentpunkte). Eine einzelne Zahl wie 193 von 200 ist daher kein Beleg für eine Überdeckung über 95 %.

**Wirkung des Zensierungsgrades (Messung 2, n = 300, Beobachtungsfenster als Regler).** Je kürzer das Fenster, desto stärker ist die Regression verzerrt; Cox bleibt bei 0,70 bis 0,71.

| Fenster (Tage) | zensiert | Cox β̂ | Regression, alle | Regression, nur Ereignisse | Regression (alle) in % der Wahrheit |
|---|---|---|---|---|---|
| 4 | 85,0 % | 0,71 | −0,06 | −0,07 | 14 % |
| 6 | 76,0 % | 0,70 | −0,11 | −0,10 | 23 % |
| 8 | 67,8 % | 0,70 | −0,14 | −0,14 | 31 % |
| 12 | 54,4 % | 0,70 | −0,21 | −0,21 | 44 % |
| 20 | 37,6 % | 0,71 | −0,29 | −0,30 | 61 % |
| 40 | 20,1 % | 0,71 | −0,37 | −0,39 | 79 % |
| 100 | 8,2 % | 0,71 | −0,43 | −0,44 | 92 % |
| 1000 | 0,8 % | 0,71 | −0,46 | −0,47 | 99 % |

Deutung (nicht gemessen): Die Regression setzt jede Zensierzeit als Abholzeit ein; die wahren Zeiten der Zensierten sind länger, und der Zusammenhang zwischen Merkmal und beobachteter Zeit schwächt sich ab.

**Stichprobengröße, Standardfehler und Überdeckung (Messung 3, Fenster 12 Tage).**

| n | Läufe ohne Schätzer | Cox β̂ (Läufe mit Schätzer) | mittlerer Standardfehler | Streuung der β̂ | Intervall enthält β |
|---|---|---|---|---|---|
| 10 | 9 von 200 | 1,03 | 1,023 | 1,247 | 183 von 191 |
| 20 | 0 von 200 | 0,85 | 0,501 | 0,583 | 193 von 200 |
| 50 | 0 von 200 | 0,72 | 0,255 | 0,268 | 191 von 200 |
| 100 | 0 von 200 | 0,72 | 0,172 | 0,167 | 189 von 200 |
| 300 | 0 von 200 | 0,70 | 0,096 | 0,096 | 193 von 200 |
| 600 | 0 von 200 | 0,70 | 0,067 | 0,065 | 189 von 200 |

Bei n = 10 haben alle 9 Läufe ohne Schätzer eine perfekte Trennung (Status `trennung`). Das Mittel der β̂ liegt bei kleinem n (20 und 10) deutlich über dem wahren 0,70; bei n = 10 fehlen zudem die
Läufe ohne Schätzer im Mittel. Ab n = 50 weicht der mittlere Standardfehler um höchstens 5 % von der tatsächlichen Streuung der β̂ ab (0,255 gegen 0,268 bei n = 50, 0,172 gegen 0,167 bei n = 100, 0,096 gegen 0,096 bei n = 300, 0,067 gegen 0,065 bei n = 600). Das stimmt im Rahmen der Messunsicherheit: Die Streuung ist selbst aus 200 Läufen geschätzt (rund 5 % relativer Standardfehler). Bei n = 10 und 20 liegt der mittlere Standardfehler 18 % und 14 % unter der Streuung.

**Andere Grundformen (Messung 4, Fenster 12 Tage, n = 300).** Die Verzerrung der Regression bleibt, ihre Wahrheit −β/k hängt an k:

| Form k | zensiert | Wahrheit −β/k | Cox β̂ | Regression, alle | Regression (alle) in % der Wahrheit |
|---|---|---|---|---|---|
| 1,0 | 50,9 % | −0,70 | 0,71 | −0,34 | 48 % |
| 1,5 | 54,4 % | −0,47 | 0,70 | −0,21 | 44 % |
| 2,5 | 58,3 % | −0,28 | 0,70 | −0,11 | 40 % |

**Ohne Wirkung (Messung 5, β = 0).** Cox β̂ im Mittel 0,006 ± 0,006, Regression 0,005 (alle) und 0,007 (nur Ereignisse); das Intervall β̂ ± 1,96·SE enthält 0 in 186 von 200 Läufen.

**Grundrate nachträglich (Messung 6, Experiment 3 über 200 Datensätze der Standardwelt).** Größter Abstand der Kurven in Prozentpunkten, je Datensatz das Maximum über die drei Gruppen und über
alle Ereigniszeiten bis τ (τ: letzte Ereigniszeit, zu der in jeder Gruppe noch mindestens 10 Container im Risiko sind), dann Median bzw. 90-%-Quantil über die Datensätze; „Wahrheit“ ist die Kurve der
Gruppe, gemittelt über ihre Mitglieder:

| n | Datensätze (ohne Vergleichsbereich) | Median: größter Abstand Cox–Kaplan–Meier | 90-%-Quantil | Median: Kaplan–Meier gegen Wahrheit | Median: Cox gegen Wahrheit | Median Vergleichsbereich τ |
|---|---|---|---|---|---|---|
| 300 | 200 (0) | 8,2 | 11,7 | 11,0 | 7,2 | 6,4 Tage |
| 100 | 200 (1) | 10,6 | 15,1 | 14,5 | 9,1 | 3,9 Tage |

In dieser Welt, in der das Cox-Modell richtig ist, liegt die Cox-Kurve im Median näher an der Wahrheit als die Kaplan–Meier-Kurve der Gruppe (7,2 gegen 11,0 Prozentpunkte bei n = 300); das gilt für
die richtige Modellform und sagt nichts über Daten, in denen die Raten nicht proportional sind.

**Handbeispiel (Experiment 1, vier Container, Merkmal 1, 0, 1, 0, Zeiten 1 bis 4, alle beobachtet).** Bei β = 0 sind die Risikomengen {1, 0, 1, 0}, {0, 1, 0}, {1, 0}, {0}, die Mittel 1/2, 1/3, 1/2, 0,
der Score 1/2 − 1/3 + 1/2 + 0 = 2/3 und die Information 1/4 + 2/9 + 1/4 + 0 = 13/18. Der erste Newton-Schritt ist (2/3)/(13/18) = 12/13 = 0,923; das Maximum liegt bei β̂ = 0,9406 (Hazard-Verhältnis
2,562, Standardfehler 1,24); die Log-Likelihood bei β = 0 ist −ln 24 = −3,178. Mit vier Containern sagt das nichts über die Welt, es dient nur zum Nachrechnen (`tests/test_model.py`, exakt).
Ein zweites Handbeispiel mit Gleichstand (Zeiten 1, 1, 2, 3, Merkmal 1, 0, 1, 0) hat nach Breslow die geschlossene Lösung e^β̂ = 2, also β̂ = ln 2 (`tests/test_model.py`).

**Zahl → Test.**

| Zahlen | Wo geprüft |
|---|---|
| Handwerte (2/3, 13/18, 12/13, 0,9406, ln 2, −ln 24), Brüche der Tabelle, Breslow-Grundrate von Hand | `tests/test_model.py`, exakt bzw. auf 1e-12 bis 1e-14 |
| Cox gegen Nullstelle des Scores, direkte Minimierung, numerische Differenzen, Trennung als fehlender Vorzeichenwechsel | `tests/test_oracle_cox.py` (andere Rechenwege, siehe „Tests“) |
| Jede Zahl der Verdikt-Texte der App (Schwellen, Brüche, Zählungen) | `tests/test_verdicts.py` an handgesetzten Werten |
| Messung 1 bis 6 (alle Tabellen und Prosa-Zahlen oben) | eingefroren in `tools/measure_cox_result.json`; `tests/test_claims.py` vergleicht jede Tabelle wörtlich und rechnet die Zellen Standardwelt, Seed 1, n = 10, n = 50, Fenster 4, Fenster 1000, Form 2,5, β = 0, Experiment 3 (n = 300) und das Handbeispiel neu (Band 1e-6) |
| Stabilität über Zufalls-Seeds 2 bis 5, n = 20, 100, 600, Fenster 6, 8, 20, 40, 100, Form 1,0, Experiment 3 bei n = 100 | nur eingefroren, nicht im Test neu gerechnet |

**Vormessung zum Vergleich** (Wegwerf-Skript mit numpy-Zufall, `ueberlebenszeit-planung/vormessung/out_uz_v1_bis_v8.txt` V6 und `out_uz_v20_bis_v23.txt` V20b, 200 Läufe): Cox β̂ 0,71 ± 0,01, Regression
−0,21 (alle) und −0,21 (nur Ereignisse) gegen −0,467, 55 % zensiert; Überdeckung 98,5 % (n = 50), 94,5 % (n = 100), 93,5 % (n = 300). Die Werte dieser Demo weichen zufallsbedingt ab (anderer Zufallsgenerator).

## Ehrliche Grenzen

**Annahmen**
- Die Zensierung ist unabhängig vom Ereignis: Ein Container, der zensiert wird, hat dieselbe Abholrate wie die, die noch im Risiko bleiben (hier erzeugt durch ein Beobachtungsfenster mit gleichverteilter Zensierzeit, unabhängig von Merkmal und Abholzeit).
- Proportionale Raten: Das Merkmal vervielfacht die Rate mit einem Faktor exp(β), der über die Zeit gleich bleibt. Die Welt dieser Demo erfüllt das; ob echte Daten es tun, prüft die Demo nicht.
- Ein Merkmal, logarithmisch linear in der Rate; die Grundform der Rate bleibt unbestimmt (sie wird nicht geschätzt, sondern nur über die Risikomengen herausgekürzt).
- Gleichstände der Zeiten werden nach Breslow behandelt (Risikomenge bleibt für alle Ereignisse derselben Zeit gleich).

**Grenzen**
- Alle Zahlen stammen aus einer simulierten Welt (Weibull-Grundrate, ein normalverteiltes Merkmal, Zensierung durch ein Beobachtungsfenster); andere Welten verändern die Größen, nicht unbedingt die Richtung.
- Die Überdeckung des Intervalls β̂ ± 1,96·SE ist nur für das richtige Modell gemessen, nicht bei verletzter Proportionalität.
- Es gibt keine Auswahl unter mehreren Merkmalen; ein Merkmal live.
- Gleichstände: Breslow gegen Efron ist nicht gemessen, die Demo macht dazu keine Zahlenaussage.

**Bekannte Probleme und Unsicherheiten**
- Die Läufe in Experiment 2 und 3 sind Stichproben mit festen Zufalls-Seeds: Anteile wie „in 94 von 100 Läufen“ schwanken um einige Prozentpunkte, wenn man den Zufalls-Seed ändert.
- Bei perfekter Trennung, konstantem Merkmal oder ohne Ereignis gibt es keinen endlichen Schätzer; die App meldet das, statt eine Zahl zu zeigen.
- Die Seite ist im Browser nur am Rechner geprüft (Seitenaufruf, Regler über AppTest), nicht auf Mobilgeräten.

**Weitere Einschränkungen (Negativbefunde bleiben stehen)**
- Die Regression auf log(beobachtete Zeit) ist hier kein Gegner, der „falsch“ ist: Ohne Zensierung trifft sie ihre Wahrheit (99 % bei Fenster 1000). Ihre Wahrheit −β/k setzt außerdem die Weibull-Form und
  die Kenntnis von k voraus, um aus der Steigung ein β zu machen; Cox braucht beides nicht.
- „Nur Container mit Ereignis“ ist keine Abhilfe: Die Regression erreicht damit bei Fenster 12 46 % der Wahrheit, rund 1,5 Prozentpunkte mehr als mit allen Containern (44 %).
- Bei kleinem n (10, 20) überschätzt Cox das β im Mittel (1,03 und 0,85 gegen 0,70), und bei n = 10 gibt es Läufe ohne Schätzer. Das Mittel der Läufe mit Schätzer ist dort kein erwartungstreuer Wert.
- Die Überdeckung von β̂ ± 1,96·SE schwankt zwischen den Zufalls-Seeds von 92,0 % bis 96,5 % (n = 300, 200 Läufe je Zufalls-Seed); die Messungen dieser Demo zeigen weder ein Unter- noch ein Überschreiten von 95 % sicher.
- Der Vergleich der Cox-Kurve mit der Wahrheit (Messung 6) gilt für die richtige Modellform; die Gruppen sind Terzile eines normalverteilten Merkmals, nicht beliebige Gruppen.
- Das Konvergenzflag des Kerns allein reicht nicht: Bei einem einzigen Ereignis (Merkmal des Abgeholten ist das größte seiner Risikomenge) meldete eine frühere Fassung des kopierten Kerns Konvergenz mit β̂ = 16,4 und
  Standardfehler rund 9,5·10^7. Die aktuelle Kopie `cxr_core.py` fängt das mit einer Plausibilitätsgrenze (Standardfehler über 50 oder |β̂| über 20 gilt als verkappte Trennung) ab und meldet `ok = False`. Der Wrapper `fit_cox` erkennt die perfekte Trennung unabhängig davon vorab (`tests/test_oracle_cox.py`).

## Befunde und Korrekturen gegenüber dem Plan

Plan: `ueberlebenszeit-planung/PLAN-04-cox-regression.md`.

- **Zahlen:** Plan 0,71 ± 0,01 und 55 % zensiert (numpy-Zufall), hier 0,703 ± 0,007 und 54,4 % (SplitMix64); die Regression bleibt bei −0,21 gegen −0,47. Es gilt die Zahl dieser Demo.
- **Überdeckung (Plan: V20b):** Der Plan nennt 98,5 / 94,5 / 93,5 % bei n = 50 / 100 / 300 und schließt „bei n = 50 etwas zu weit“. Hier: 191 / 189 / 193 von 200 (95,5 / 94,5 / 96,5 %); über sechs
  Zufalls-Seeds bei n = 300 liegt der Anteil zwischen 92,0 und 96,5 %. Die Aussage „bei n = 50 zu weit“ bestätigt sich nicht; die Schwankung der Anteile ist so groß wie die Unterschiede.
- **Kleines n:** Neu gegenüber dem Plan: Überschätzung bei n = 10 und 20 und Läufe ohne Schätzer bei n = 10 (Tabelle oben). Der Plan nannte nur die leichte Überschätzung 0,71 bis 0,73 bei n = 50 bis 300.
- **Konvergenzflag:** Der Plan verlangt Flag und Meldung statt des Wegwerf-Codes (z. B. β̂ = 31,9 bei Trennung). Umgesetzt für Trennung, Information 0 und fehlendes Ereignis; dazu der oben beschriebene Fund,
  dass eine frühere Fassung des kopierten Kerns bei einem einzigen Ereignis trotzdem „konvergiert“ meldete; die aktuelle Kopie `cxr_core.py` enthält dafür eine Plausibilitätsgrenze. Ob die Vorlage `uz_core.py` sie ebenfalls enthält, ist hier nicht geprüft.
- **Experiment 4 (Gleichstände):** wie im Plan vorgesehen nicht umgesetzt, da Breslow gegen Efron nicht gemessen ist; die App und diese README nennen dazu keine Zahl.
- **Z2 (zwei vorberechnete Merkmale):** nicht umgesetzt, nur ein Merkmal live (siehe „Bewusst nicht umgesetzt“).
- **Experiment 3:** Der Plan nennt „KM-Kurven der Gruppen“. Umgesetzt mit Terzil-Gruppen des Merkmals und der Cox-Kurve als Mittel über die Mitglieder (nicht die Kurve beim mittleren Merkmal), dazu das
  Verdikt mit Zahlen gegen Kaplan–Meier und Wahrheit.
- **Regression „nur Ereignisse“** steht zusätzlich zu „alle“ in der Demo (wie im Plan, V6); beide liegen bei Fenster 12 bei 44 % und 46 % der Wahrheit.
- **Kein PDF-Export** (der Auftrag der Linie verlangt ihn nicht; die App zeigt alle Zahlen auf der Seite).
- **Footer:** Die Linie Überlebenszeit hat noch keine Konzepte-Spezifikation; der Footer verweist deshalb auf `demos.html` und ist vor der Website-Einbindung auf die Linie umzustellen.

## Tests

150 Tests (`pytest --collect-only -q tests`), alle grün. Den größten Teil der Laufzeit machen die AppTest-Läufe (`test_app.py`, `test_layout.py`) und die Neuberechnung der Messzellen (`test_claims.py`) aus.

| Datei | Tests | Inhalt |
|---|---|---|
| `test_core.py` | 10 | kopierte Kernfunktionen: Kaplan–Meier-Handbeispiel, Eingabeprüfung, Cox-Handbeispiel, Eintritt, Trennung und Singularität als Status, Vorzeichen und Skalierung, SplitMix64-Referenzwert; dazu ein Vergleich der Funktionstexte mit der Quelle `uz_core.py`, falls vorhanden |
| `test_model.py` | 32 | Einheiten am Vier-Container-Handbeispiel mit Brüchen (Log-Likelihood, Score 2/3, Information 13/18, Tabelle, Newton-Pfad 12/13), Gleichstands-Beispiel mit Lösung ln 2, Trennungserkennung und Status der vier Muster, Regression von Hand, Breslow-Grundrate von Hand, Gruppen und Vergleichsbereich, Zusammenfassung an gesetzten Läufen, Szenario in geschlossener Form |
| `test_oracle_cox.py` | 10 | **Orakel** mit anderem Rechenweg (reine Schleifen je Container mit Ereignis; Nullstelle des Scores mit `scipy.optimize.brentq`; direkte Minimierung mit `minimize_scalar`; Score und Information als numerische Differenzen; Trennung als fehlender Vorzeichenwechsel; Breslow-Grundrate je Container): 60 bis 200 kleine Instanzen mit Gleichständen und Zensierung je Test, dazu die Simulation mit bekannter Wahrheit als Band und der Fund „ein einziges Ereignis“ |
| `test_eigenschaften_cox.py` | 10 | Eigenschaften: Skalierung des Merkmals (β̂/c), Verschiebung, Zeitverschiebung und monotone Zeittransformation, Umsortieren, vertauschte Gruppen (Vorzeichen), Zeiten permutiert (β̂ um 0), mehr Container, stärkere Wirkung, gleicher Zufalls-Seed, früh zensierte Container ohne Wirkung |
| `test_edge_cases.py` | 9 | n = 1, n = 2, Gleichstand, alles zensiert, konstantes Merkmal, ungültige Zeiten (`SurvivalError`), kleinste Stichprobe, Studie ohne Schätzer, β = 0, Experiment 3 ohne Vergleichsbereich |
| `test_verdicts.py` | 13 | jedes Verdikt und jeder Hinweistext an handgesetzten Werten: Schwellen, Brüche, Zählungen |
| `test_claims.py` | 19 | Tabellen der README wörtlich gegen `tools/measure_cox.py`, Prosa-Zahlen, Neuberechnung ausgewählter Zellen, Annahmen-Text der App in der README |
| `test_presets.py`, `test_layout.py` | 7, 18 | Schnellstart-Szenarien vollständig und in den Grenzen, Permalink-Begrenzung; Schnellstart in der Seitenleiste, fester Kopf, Hervorhebung des aktiven Szenarios, jede Regleränderung hebt sie auf |
| `test_app.py`, `test_footer.py`, `test_runtime_imports.py` | 19, 2, 1 | AppTest: jeder Button, jeder Regler an Minimum und Maximum, jedes Beobachtungsfenster, jedes Muster des Handbeispiels mit seiner Meldung, Eckfall-Matrix, kleinste Stichprobe; Footer; Laufzeit-Importe gegen `requirements.txt` |

Der Orakel-Test braucht SciPy (`requirements-dev.txt`, nicht in der Laufzeit); fehlt es, überspringt er sich (`pytest.importorskip`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: drei Experimente, Kernfrage, Annahmen, Formeln |
| `cxr_core.py` | Kopie aus `ueberlebenszeit-planung/kern/uz_core.py` (SplitMix64, Eingabeprüfung, Kaplan–Meier, Cox mit Konvergenzflag); wird nicht verändert |
| `cxr_model.py` | partielle Log-Likelihood, Score, Information, Newton-Pfad, Trennungserkennung, `fit_cox` mit Status, Breslow-Grundrate, Gruppenkurven, Simulationsstudie |
| `cxr_scenario.py` | Welt (Weibull-Grundrate, Merkmal, Zensierung), Simulation je Lauf, Muster des Handbeispiels |
| `cxr_evaluation.py` | Zahlenformat, Auswertung der drei Experimente, Verdikte als reine Funktionen |
| `cxr_constants.py` | Regler-Grenzen, feste Annahmen, Schnellstart-Szenarien, Text „Annahmen, Grenzen und bekannte Probleme“ |
| `cxr_presets.py` | Permalink (Begrenzen), Schnellstart in der Seitenleiste mit hervorgehobenem aktiven Szenario, Zufalls-Seed-Button |
| `cxr_layout.py` | fester Kopf (Stylesheet) |
| `cxr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/measure_cox.py`, `tools/measure_cox_result.json` | Messskript und eingefrorene Zahlen dieser README |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly); dev zusätzlich pytest und scipy (nur Orakel-Test) |
| `tests/` | Kern, Modell, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, Schnellstart, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Efron-Gleichstandsbehandlung** und jeder Vergleich Breslow gegen Efron (nicht gemessen).
- **Zweites Merkmal, Merkmalsauswahl, nichtlineare Wirkung:** nur ein Merkmal live; die Plan-Entscheidung Z2 (zwei weitere Merkmale vorberechnet) ist nicht gebaut.
- **Prüfung der Proportionalität, abhängige Zensierung, verzögerter Eintritt:** eigene Fragen der Linie Überlebenszeit, hier nicht gemessen. Der kopierte Kern kann Eintritt (`entry`), die App nutzt es nicht.
- **Vergleich mit Lernverfahren** (zum Beispiel Überlebenswäldern): nicht Teil dieser Demo.
- **PDF-Export.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; die in Streamlit Cloud eingestellte Python-Version steht noch nicht fest und ist beim Deploy hier einzutragen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy; Orakel-Tests mit SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
