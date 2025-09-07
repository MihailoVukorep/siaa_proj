# Detaljna analiza Cart-Pole Q-Learning implementacije

## 1. StateDiscretizer (discretizer.py)

Ova klasa je odgovorna za konvertovanje kontinuiranih stanja u diskretne vrednosti koje Q-Learning algoritam može da koristi.

### Ključne karakteristike:
- **Adaptivna diskretizacija**: Koristi nelinearno binovanje za ugao theta, sa više binova oko centra (kritična zona)
- **Fokus na stabilnost**: Theta komponenta ima gustiju diskretizaciju između -0.1 i 0.1 radijana
- **Numerička stabilnost**: Koristi `np.clip` za ograničavanje vrednosti u dozvoljene granice
- **Uklanjanje duplikata**: `np.unique` osigurava da nema preklapajućih binova

### Proces diskretizacije:
1. Za theta (indeks 2): Pravi tri regiona - negativni spoljašnji, centralni gusti, pozitivni spoljašnji
2. Za ostale dimenzije: Koristi uniformnu linearnu diskretizaciju
3. Vraća tuple diskretnih indeksa koji predstavljaju "adresu" u Q-tabeli

## 2. CartPoleEnvironment (environment.py)

Simulira fiziku Cart-Pole sistema sa poboljšanjima za stabilnost i realizam.

### Fizički parametri:
- `m = 0.1 kg` - masa štapa
- `M = 1.0 kg` - masa kolica  
- `l = 0.5 m` - dužina štapa
- `g = 9.81 m/s²` - gravitacija
- `T = 0.01 s` - kratko vreme odabiranja za stabilnost

### Ključne metode:

#### `_get_derivatives()`:
Implementira fizičke jednačine kretanja:
- Koristi Lagranžovu mehaniku za izračunavanje ubrzanja
- Dodaje epsilon (1e-6) u imenilac za numeričku stabilnost
- Ograničava brzine da spreči eksploziju

#### `step()`:
- **Podela koraka**: Deli vremenski korak na pola za bolju numeričku integraciju
- **Euler metoda**: Koristi poboljšanu Euler diskretizaciju
- **Ograničavanje**: Klipuje brzine u realnim granicama

#### `_calculate_reward()`:
Kompleksna reward funkcija sa više komponenti:
- **Angle reward**: 1.0 - |theta|/threshold (nagrađuje uspravnost)
- **Position reward**: 1.0 - |x|/threshold (drži blizu centra)
- **Velocity penalty**: Kažnjava velike brzine (-0.1 faktor)
- **Stability bonus**: +0.5 za theta < 0.05 radijana
- **Terminal penalty**: -100 za pad/izlazak

## 3. OptimizedQLearning (q_learning.py)

Napredna implementacija Q-Learning algoritma sa Double Q-Learning tehnikom.

### Ključne funkcionalnosti:

#### Double Q-Learning:
- Koristi **dve Q-tabele** za smanjenje overestimation bias-a
- Jedna tabela bira akciju, druga procenjuje vrednost
- Slučajno bira koju tabelu da ažurira (50-50 šanse)

#### Adaptivni learning rate:
```python
adaptive_lr = learning_rate / (1 + 0.0001 * visit_counts[state])
```
Smanjuje learning rate za često posećena stanja.

#### Epsilon-greedy sa decay:
- Počinje sa epsilon=1.0 (100% eksploracija)
- Eksponencijalno opada: `epsilon *= epsilon_decay`
- Minimum epsilon=0.02 (2% nasumičnih akcija)

#### `choose_action()`:
- Za Double Q: Kombinuje obe tabele (prosek) za bolju procenu
- Koristi epsilon-greedy strategiju
- Vraća indeks najbolje akcije

#### `update()`:
Implementira Double Q-Learning update pravilo:
1. Računa adaptivni learning rate
2. Slučajno bira tabelu za ažuriranje
3. Koristi drugu tabelu za procenu vrednosti sledeće akcije
4. Primenjuje Q-Learning formulu: `Q(s,a) += α[r + γQ'(s',a') - Q(s,a)]`

## 4. Training Pipeline (training.py)

Organizuje ceo proces treniranja i testiranja.

### Hiperparametri:
- **Episodes**: 3000 (optimalno za konvergenciju)
- **Learning rate**: 0.3 (umereno agresivno učenje)
- **Discount factor**: 0.99 (dugoročno planiranje)
- **Force magnitude**: 8.0 N (realna sila)
- **Bins**: 12 po dimenziji (12^4 = 20,736 stanja)

### Proces treniranja:

#### Glavna petlja:
1. **Reset** okruženja - nasumično početno stanje
2. **Epizoda loop**: 
   - Agent bira akciju
   - Okruženje izvršava korak
   - Agent ažurira Q-vrednosti
   - Ponavlja dok epizoda ne završi
3. **Statistike**: Čuva nagrade, dužine, stope uspeha
4. **Epsilon decay**: Smanjuje eksploraciju
5. **Model saving**: Čuva najbolje modele

#### Kriterijumi uspeha:
- **Uspešna epizoda**: ≥200 koraka (smanjen prag)
- **Najbolji model**: Čuva kada dostigne >500 koraka
- **Success rate**: Procenat epizoda ≥200 koraka

### Testiranje:
- Isključuje eksploraciju (epsilon=0)
- Pokreće 10 test epizoda
- Računa statistike performansi
- Vraća detaljne rezultate

## 5. Integracija komponenti

### Tok podataka:
1. **Environment** → kontinuirno stanje (4D vektor)
2. **Discretizer** → diskretno stanje (4D tuple indeksa)
3. **Q-Learning** → akcija (0 ili 1)
4. **Environment** → nova stanja, nagrade, terminal flag
5. **Q-Learning** → ažuriranje Q-tabela

### Optimizacije:
- **Numerička stabilnost**: Clipping, epsilon dodaci
- **Adaptivno učenje**: Smanjuje LR za poznata stanja  
- **Double Q-Learning**: Redukuje bias
- **Adaptivna diskretizacija**: Više preciznosti u kritičnim zonama
- **Rana terminacija**: Štedi vreme na neuspešnim epizodama

---

# Opšti opis koda

Ovaj kod implementira **reinforcement learning** rešenje za klasični **Cart-Pole** problem kontrole. Cilj je da se nauči strategija koja održava štap u uspravnom položaju na pokretnim kolicima što duže moguće.

## Arhitektura:
Kod je podeljen u **4 modularna komponenta**:

1. **Environment** - Fizička simulacija Cart-Pole sistema
2. **Discretizer** - Konvertor kontinuiranih stanja u diskretne
3. **Q-Learning** - Algoritam mašinskog učenja (Double Q-Learning)
4. **Training** - Orkestrator treniranja i testiranja

## Tehnike:
- **Double Q-Learning**: Napredna varijanta koja smanjuje overestimation
- **Adaptivna diskretizacija**: Gušća mreža u kritičnim regionima
- **Epsilon-greedy decay**: Balansira eksploraciju i eksploataciju
- **Multi-component rewards**: Složena reward funkcija sa više kriterijuma
- **Numerical stability**: Clipping i epsilon dodaci za robusnost

## Rezultat:
Trenirani agent uči da kontroliše kolica tako da štap ostane uspravno, koristeći samo informacije o poziciji, brzini, uglu i ugaonoj brzini. Agent postepeno prelazi od nasumičnih akcija ka naučenoj strategiji kroz trial-and-error proces.