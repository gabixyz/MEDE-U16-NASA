MEDE - U16 - CP1
================

Intrebarea de cercetare
-----------------------
Cum influenteaza regimul de operare masuratorile senzorilor si putem
observa mai clar degradarea motorului dupa normalizarea pe regim?

Mod de rulare
-------------
Este necesar Python 3.10 sau mai nou, fara biblioteci suplimentare.

```powershell
git clone https://github.com/gabixyz/MEDE-U16-NASA.git
Set-Location MEDE-U16-NASA
python -m unittest discover -s tests -v
python src\load_fd002.py --download
```

Daca descarcarea nu functioneaza, copiati fisierele `train_FD002.txt`,
`test_FD002.txt` si `RUL_FD002.txt` din arhiva NASA in `data\raw`, apoi rulati:

```powershell
python src\load_fd002.py
```

Puteti folosi si datele din alt folder:

```powershell
python src\load_fd002.py --raw-dir "C:\cale\catre\MEDE_U16_date"
```

Date folosite
-------------
Folosim C-MAPSS FD002, un dataset NASA cu date simulate despre
functionarea si degradarea motoarelor.

[Descarcare dataset](https://data.nasa.gov/docs/legacy/CMAPSSData.zip)

[Pagina oficiala NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Iesiri
------
Codul salveaza separat senzorii si setarile de operare, pentru train si test.
Mai genereaza un fisier cu valorile RUL pentru evaluare si un raport de verificare.

Verificarea datelor
-------------------
Verificam structura fisierelor, valorile lipsa, duplicatele si ordinea ciclurilor.
Daca apar probleme, programul se opreste fara sa modifice datele.
Seriile din test se termina inainte de defectare, conform datasetului.

## Limite si pasi urmatori
Datele sunt simulate, iar un ciclu nu are o durata precizata in secunde.
Unitatile senzorilor sunt descrise in data card; cele ale setarilor nu sunt
precizate in readme-ul NASA.

La CP1 am pregatit incarcarea datelor. La CP2 vom analiza distributiile,
corelatiile si evolutia senzorilor pe ferestre de cicluri.
Normalizarea se va calcula doar pe train, iar validarea va folosi motoare
separate. Valorile RUL vor fi folosite doar pentru evaluare.

Documente
---------
- CP1_U16.pdf: prezentarea in trei slide-uri.
- datacard_U16.pdf: descrierea surselor de date.
- SOA_U16.pdf: sinteza celor 10 articole.

Rezultatele din SOA apartin autorilor, nu sunt obtinute in acest proiect.
Studiile folosesc date si metode diferite, deci scorurile nu se compara direct.

Utilizarea datelor
------------------
Datele provin de la NASA PCoE, iar sursa trebuie citata.
Nu am identificat o licenta CC specifica; se respecta conditiile NASA.
Datele brute nu sunt incluse in repository.
