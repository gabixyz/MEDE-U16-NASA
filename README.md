MEDE - U16 - CP1, CP2 si CP3
==========================

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

La CP2 nu am implementat PCA sau predictia RUL. CP3 este descris mai jos.
Datele brute, rezultatele generate si materialele de prezentare nu sunt
incluse in repository.

CP3: PCA versus autoencoder
--------------------------
Instalare suplimentara si rulare completa de la datele brute:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn
python -m unittest discover -s tests -v
python src\reduce_cp3.py --download
```

Sau, pentru datele locale originale, verificate prin SHA-256:

```powershell
python src\reduce_cp3.py --raw-dir "C:\cale\catre\MEDE_U16_date"
```

CP3 reutilizeaza functiile CP2, dar poate rula fara iesiri CP1/CP2 existente.
Pastreaza splitul seed 42: 208 motoare fit / 52 validare. Cele 259 de motoare
test NASA si etichetele RUL nu sunt incarcate in analiza; sunt verificate doar
hash-urile celor trei fisiere brute. Nu optimizam hiperparametri pe validare.

Normalizarea pe regim, eliminarea constantelor si scalarea trasaturilor se
invata numai pe fit, la pasul CP2 de 1 ciclu. Modelele si evaluarea CP3 folosesc
ferestre cauzale de 10 cicluri, nesuprapuse, cu capete 10, 20, 30 etc.:
4 213 ferestre fit si 1 041 validare. Scopul rarefierii este reducerea
dependentei locale dintre vecini; ferestrele raman corelate in acelasi motor.
Matricea CP2 completa ramane 41 211 / 10 208 ferestre, nu a fost inlocuita.

Cele cinci intrari comparate:

| Intrare | d | Rol |
|---|---:|---|
| `sensors_raw` | 63 | S1 brut: baseline cu adevarat pe o singura sursa |
| `operating` | 12 | S2 separat |
| `combined_raw` | 75 | S1 brut + S2 |
| `sensors_normalized` | 51 | S1 normalizat cu ajutorul S2, nu un baseline S1 independent |
| `combined_normalized` | 63 | S1 normalizat + S2, matricea CP2 |

PCA are 2 componente, solver SVD full, si este determinist. Alternativa este
un autoencoder scikit-learn MLPRegressor d-32-2-32-d, activari tanh in straturile
ascunse (inclusiv bottleneck) si iesire liniara, Adam, batch 256, rata 0.001,
alpha 0.001, maximum 300 epoci, fara early stopping pe validare.
Seed-uri: 17, 42, 73; 5 modele PCA si 15 autoencodere. Bugetul este fix;
toate cele 15 autoencodere ating 300 epoci, deci nu afirmam convergenta.
Avertismentele si pierderile fit sunt pastrate in raport.

Metricile pe validare:
- MSE in spatiul standardizat de intrare: mai mic este mai bine; comparati
  PCA cu AE in acelasi rand, nu spatii brute/normalizate diferite.
- MSE_S1 foloseste numai coloanele senzorilor: tinta comuna cand comparam
  senzori cu senzori + regim in aceeasi familie brut/normalizat.
- Trustworthiness k=15, pe aceleasi 800 de ferestre de validare alese cu
  seed 2026: mai mare este mai bine. Metrica se raporteaza la propria intrare;
  nu dovedeste identificarea degradarii.
- Varianta explicata PCA se masoara pe fit, nu pe validare.
- Stabilitate: proportia medie a celor 15 vecini comuni intre fiecare
  pereche de seed-uri, in acelasi esantion; scor invariant la rotatie.
- Subperioade: ferestre de validare cu ultimul ciclu <=100 / >100, fara
  reantrenare. MSE foloseste toate ferestrele din subperioada; trustworthiness
  foloseste pana la 400. Nu sunt etichete de stadiu de degradare.

Intervalele sunt min/max empirice intre cele trei seed-uri, nu intervale de
incredere si nu estimari ale variatiei intre motoare. Nu pretindem
semnificatie statistica. Sensibilitatea la ferestre 10/30/50 se calculeaza
numai pe fit, cu PCA, si nu stabileste un optim predictiv.

Rezultate (aceeasi precizie ca in slide-uri):

| Intrare | EV2 PCA fit | T15 PCA | T15 AE, medie [min; max] | MSE PCA / AE mediu |
|---|---:|---:|---|---:|
| S1 brut | 62.4% | 0.872 | 0.929 [0.925; 0.932] | 0.358 / 0.258 |
| S2 | 62.0% | 0.903 | 0.909 [0.903; 0.916] | 0.376 / 0.342 |
| S1 brut + S2 | 58.8% | 0.849 | 0.916 [0.910; 0.920] | 0.403 / 0.295 |
| S1 normalizat | 33.2% | 0.779 | 0.788 [0.787; 0.789] | 0.639 / 0.611 |
| S1 normalizat + S2 | 27.3% | 0.784 | 0.801 [0.799; 0.804] | 0.681 / 0.644 |

Pe S1 brut, AE reduce MSE fata de PCA cu aproximativ 28%.
Pe tinta comuna S1, adaugarea S2 nu aduce castig mediu in acest protocol:
MSE AE 0.258 -> 0.261 (brut), respectiv 0.611 -> 0.641 (normalizat).
Pentru combinatia bruta, vecinii AE comuni intre seed-uri sunt numai
29.3%-40.1%: scoruri globale apropiate nu inseamna aceleasi grupuri locale.
Nu declaram regimurile sau grupurile ca stadii de degradare; nu facem
predictie RUL si nu demonstram transfer pe motoare reale.

Iesiri in `outputs\cp3`: `metrics_summary.csv`, `cp3_report.json`,
`seeds_*.json`, `embedding_*.csv`, `loadings_*.csv`,
`pca_scree_loadings.png` si `projections_validation.png`.
Raportul include splitul exact, parametrii, toate seed-urile, subperioadele,
sensibilitatea ferestrei, versiunile bibliotecilor si hash-urile codului.

Versiuni folosite: Python 3.12.10, NumPy 2.5.1, pandas 3.0.5,
SciPy 1.18.1, scikit-learn 1.9.1. Pentru aceleasi versiuni:

```powershell
python -m pip install numpy==2.5.1 pandas==3.0.5 scipy==1.18.1 scikit-learn==1.9.1 matplotlib
```

Referinte metodologice: Saxena et al. (2008), DOI 10.1109/PHM.2008.4711414;
Malhotra et al. (2016), arXiv:1608.06154 (motivatie pentru reconstructie,
nu aceeasi arhitectura LSTM); Venna si Kaski (2001),
DOI 10.1007/3-540-44668-0_8 (trustworthiness).
