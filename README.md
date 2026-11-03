MEDE - U16 - CP1 si CP2
======================

Intrebarea de cercetare
-----------------------
Cum influenteaza regimul de operare masuratorile senzorilor si putem
observa mai clar degradarea motorului dupa normalizarea pe regim?

Mod de rulare
-------------
Este necesar Python 3.10 sau mai nou. Pentru CP2 folosim NumPy, pandas
si Matplotlib. Bibliotecile se instaleaza direct, fara requirements.txt.

```powershell
git clone https://github.com/gabixyz/MEDE-U16-NASA.git
Set-Location MEDE-U16-NASA
python -m pip install numpy pandas matplotlib
```

Pentru CP1, incarcarea si verificarea datelor:

```powershell
python src\load_fd002.py --download
```

Pentru CP2, alinierea, normalizarea si calculul trasaturilor:

```powershell
python src\prepare_cp2.py --download
```

Daca datele sunt deja in `data\raw`, omiteti `--download`.
Daca descarcarea nu functioneaza, copiati fisierele `train_FD002.txt`,
`test_FD002.txt` si `RUL_FD002.txt` din arhiva NASA in `data\raw`.
Puteti folosi si datele din alt folder:

```powershell
python src\prepare_cp2.py --raw-dir "C:\cale\catre\MEDE_U16_date"
```

Pentru rularea testelor CP1 si CP2:

```powershell
python -m unittest discover -s tests -v
```

Date folosite
-------------
Folosim C-MAPSS FD002, un dataset NASA cu date simulate despre
functionarea si degradarea motoarelor. S1 contine 21 de senzori, iar
S2 contine 3 setari de operare. Sunt doua perspective ale aceluiasi
dataset, nu doua dataseturi independente.

[Descarcare dataset](https://data.nasa.gov/docs/legacy/CMAPSSData.zip)

[Pagina oficiala NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Ce face CP2
-----------
Reunim senzorii si setarile pe cheia `split, engine_id, cycle`.
Cele 87 750 de cicluri se potrivesc exact, fara interpolare sau imputare.

Din cele 260 de motoare train, 208 sunt pentru fit si 52 pentru validare
(seed 42). Cele 259 de motoare test NASA raman separate. Mediile pe regim,
scalele si eliminarea trasaturilor constante se invata numai din fit.

Folosim ferestre de 10 cicluri, cu pas 1, fara a trece intre motoare.
Calculam media, deviatia standard si panta senzorilor, media si deviatia
standard a setarilor, plus ponderea celor 6 regimuri in fiecare fereastra.
Din 75 de trasaturi candidate eliminam 12 constante si raman 63:
51 din S1 si 12 din S2. La final le standardizam pe baza motoarelor fit.
RUL si identificatorii nu intra in trasaturi.

Iesiri
------
CP1 salveaza senzorii, setarile, valorile RUL si raportul de verificare
in `outputs`. CP2 salveaza rezultatele in `outputs\cp2`:

| Fisier | Continut |
|---|---|
| `aligned_*.csv.gz` | Datele brute reunite si regimul de operare |
| `features_*.csv.gz` | Cele 75 de trasaturi candidate pe ferestre |
| `X_*.csv.gz` | Cele 63 de trasaturi finale |
| `feature_dictionary.csv` | Descrierea trasaturilor si sursa fiecareia |
| `train_engine_partition.csv` | Impartirea motoarelor fit/validare |
| `window_comparison_fit.csv` | Comparatia ferestrelor de 10, 30 si 50 de cicluri |
| `descriptive_*.csv`, `correlation_*.csv` | Statistici si corelatii |
| `cp2_report.json`, `figures` | Parametrii, dimensiunile si graficele |

Matricele au 41 211 de ferestre fit, 10 208 de validare si 31 660 de test,
fiecare cu 63 de trasaturi. Fisierele X mai contin 4 coloane de identificare:
`split`, `engine_id`, `window_start`, `window_end`. Acestea nu sunt
trasaturi; coloanele de intrare sunt cele cu prefixul `S1_` sau `S2_`.
Primele 9 cicluri ale fiecarui motor sunt necesare formarii primei ferestre.

Verificarea datelor
-------------------
Verificam structura fisierelor, valorile lipsa, duplicatele si ordinea ciclurilor.
Daca apar probleme, programul se opreste fara sa modifice datele.
Seriile din test se termina inainte de defectare, conform datasetului.

Limite si pasi urmatori
-----------------------
Datele sunt simulate, iar un ciclu nu are o durata precizata in secunde.
Fereastra de 30 de cicluri din CP0 era o propunere; la CP2 folosim initial
10 si comparam 30/50 numai pe fit. Nu pretindem ca 10 este un optim.
Ferestrele se suprapun, de aceea impartirea se face pe motoare, nu pe randuri.

Pe validare, corelatia S02 - setarea 1 scade de la -0.8693 la -0.0233.
Aceasta arata o reducere a legaturii liniare cu regimul, nu dovedeste
identificarea degradarii. Normalizarea poate elimina si semnal util.
Corelatiile pentru coloane constante sunt nedefinite, nu zero.
Unitatile setarilor de operare nu sunt precizate in readme-ul NASA.

La CP3 urmeaza PCA. La CP2 nu am implementat PCA sau predictia RUL.
Datele brute, rezultatele generate si materialele de prezentare nu sunt
incluse in repository.
