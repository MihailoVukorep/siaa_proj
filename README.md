# Opšti opis koda

Ovaj kod implementira **reinforcement learning** rešenje za klasični **Cart-Pole** problem kontrole. Cilj je da se nauči strategija koja održava štap u uspravnom položaju na pokretnim kolicima što duže moguće.

## Arhitektura:
Kod je podeljen u **4 modularna komponenta**:

1. **Environment** ([environment.py](#2-cartpoleenvironment-environmentpy)) - Fizička simulacija Cart-Pole sistema
2. **Discretizer** ([discretizer.py](#1-statediscretizer-discretizerpy)) - Konvertor kontinuiranih stanja u diskretne
3. **Q-Learning** ([q_learning.py](#3-optimizedqlearning-q_learningpy)) - Algoritam mašinskog učenja (Double Q-Learning)
4. **Training** ([training.py](#4-training-pipeline-trainingpy)) - Orkestrator treniranja i testiranja

## Tehnike:
- **Double Q-Learning**: Napredna varijanta koja smanjuje overestimation
- **Adaptivna diskretizacija**: Gušća mreža u kritičnim regionima
- **Epsilon-greedy decay**: Balansira eksploraciju i eksploataciju
- **Multi-component rewards**: Složena reward funkcija sa više kriterijuma
- **Numerical stability**: Clipping i epsilon dodaci za robusnost

## Rezultat:
Trenirani agent uči da kontroliše kolica tako da štap ostane uspravno, koristeći samo informacije o poziciji, brzini, uglu i ugaonoj brzini. Agent postepeno prelazi od nasumičnih akcija ka naučenoj strategiji kroz trial-and-error proces.

# Detaljna analiza Cart-Pole Q-Learning implementacije

## 1. StateDiscretizer (discretizer.py)

Ova klasa je odgovorna za konvertovanje kontinuiranih stanja u diskretne vrednosti koje Q-Learning algoritam može da koristi.

### Ključne karakteristike:
- **Adaptivna diskretizacija**: Koristi nelinearno [binovanje](#binovanje) za ugao theta, sa više binova oko centra (kritična zona)
- **Fokus na stabilnost**: Theta komponenta ima gustiju diskretizaciju između -0.1 i 0.1 radijana
- **Numerička stabilnost**: Koristi `np.clip` za ograničavanje vrednosti u dozvoljene granice
- **Uklanjanje duplikata**: `np.unique` osigurava da nema preklapajućih binova

### Proces diskretizacije:
1. Za theta (indeks 2): Pravi tri regiona - negativni spoljašnji, centralni gusti, pozitivni spoljašnji
2. Za ostale dimenzije: Koristi uniformnu linearnu diskretizaciju
3. Vraća tuple diskretnih indeksa koji predstavljaju "adresu" u Q-tabeli

---

### Binovanje

### Šta su binovi i zašto ih pravimo?

**Binovi** su kao "kutije" u koje stavljamo kontinuirane vrednosti. Predstavljaju diskretne intervale.

#### Problem:
- Q-Learning radi samo sa **diskretnim** stanjima (konačan broj)
- Cart-Pole ima **kontinuirana** stanja (beskonačno mogućih vrednosti)
- Pozicija kolica može biti 1.23456, 1.23457, 1.23458... (beskonačno varijanti)

#### Rešenje - Binovanje:
```
Kontinuirane vrednosti: [-2.4 ... -1.2 ... 0.0 ... 1.2 ... 2.4]
                          ↓
Binovi:                [  Bin0   |  Bin1  |  Bin2  |  Bin3  ]
                        
Primer:
- Pozicija -1.7 → Bin 0
- Pozicija -0.3 → Bin 1  
- Pozicija 0.8  → Bin 2
- Pozicija 2.1  → Bin 3
```

**Q-tabela** tada koristi kombinacije bin indeksa kao "adrese":
```
Q[(bin_x, bin_v, bin_theta, bin_omega)] = [Q_left, Q_right]
```

### Zašto indeks 2 (theta) ima specijalne regione?

**Indeks 2 = ugao štapa (theta)** - najkritičnija varijabla!

#### Problem sa uniformnim binovima za theta:
```
Uniformni binovi: [-0.21 | -0.1 | 0.0 | 0.1 | 0.21]
                     Bin0   Bin1  Bin2  Bin3  Bin4

Problem: Bin2 pokriva [-0.05, 0.05] - KRITIČNU zonu!
- Male razlike u ovoj zoni su VRLO važne
- Theta = 0.01 vs 0.04 - velika razlika u stabilnosti
- Ali su u istom binu → agent ne vidi razliku!
```

#### Rešenje - Adaptivni binovi:
```python
# Više binova oko centra (theta ≈ 0)
center_bins = np.linspace(-0.1, 0.1, n_bins_per_dim // 2)  # Gusti binovi
outer_bins_neg = np.linspace(low, -0.1, n_bins_per_dim // 4)  # Retki binovi  
outer_bins_pos = np.linspace(0.1, high, n_bins_per_dim // 4)   # Retki binovi
```

**Rezultat:**
```
Adaptivni:  [-0.21|-0.15|-0.1|-0.06|-0.02|0.02|0.06|0.1|0.15|0.21]
            Spoljašnji  |    GUSTI CENTRALNI    |   Spoljašnji

Preciznost:
- Oko theta=0: ±0.02 radijana po binu (vrlo precizno)
- Na krajevima: ±0.05 radijana po binu (dovoljno)
```

**Zašto ostale dimenzije uniformno?**
- **Pozicija (x)**: Važna, ali ne toliko kritična kao theta
- **Brzine**: Manje kritične za trenutnu odluku
- **Theta**: NAJVAŽNIJI - razlika između uspehe i pada!

---


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
- Koristi [Lagranžovu mehaniku](#lagranžovu-mehaniku) za izračunavanje ubrzanja
- Dodaje epsilon (1e-6) u imenilac za numeričku stabilnost
- Ograničava brzine da spreči eksploziju

#### `step()`:
- **Podela koraka**: Deli vremenski korak na pola za bolju numeričku integraciju
- **Euler metoda**: Koristi poboljšanu [Euler diskretizaciju](#euler-metoda)
- **Ograničavanje**: Klipuje brzine u realnim granicama

#### `_calculate_reward()`:
Kompleksna reward funkcija sa više komponenti:
- **Angle reward**: 1.0 - |theta|/threshold (nagrađuje uspravnost)
- **Position reward**: 1.0 - |x|/threshold (drži blizu centra)
- **Velocity penalty**: Kažnjava velike brzine (-0.1 faktor)
- **Stability bonus**: +0.5 za theta < 0.05 radijana
- **Terminal penalty**: -100 za pad/izlazak

---

### Lagranžovu mehaniku

### Šta je Lagranžova mehanika?

**Lagranžova mehanika** je napredni pristup za izračunavanje kretanja složenih sistema.

#### Klasična vs Lagranžova:
```
Klasična mehanika (Newton):
F = ma → rešavaj sile direktno
Problem: Kompleksno za sistema sa ograničenjima

Lagranžova mehanika:
L = T - V (kinetička - potencijalna energija)
Automatski daje jednačine kretanja
```

### Cart-Pole sistem:

#### Komponente energije:
1. **Kinetička energija kolica**: `½M(dx/dt)²`
2. **Kinetička energija štapa**: 
   - Translacija: `½m[(dx/dt)² + (dy/dt)²]`
   - Rotacija: `½I(dθ/dt)²`
3. **Potencijalna energija štapa**: `mgh = mg(l cos θ)`

#### Lagranžijan:
```
L = T_kola + T_štap_trans + T_štap_rot - V_štap
L = ½M(dx/dt)² + ½m[(dx/dt + l cos θ(dθ/dt))² + (l sin θ(dθ/dt))²] + ½I(dθ/dt)² - mgl cos θ
```

#### Euler-Lagrange jednačine:
```
d/dt(∂L/∂q̇) - ∂L/∂q = F_ext

Za x: d/dt(∂L/∂ẋ) - ∂L/∂x = F
Za θ: d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
```

**Rezultat** - sistem diferencijalnih jednačina za `d²x/dt²` i `d²θ/dt²`

### Zašto Lagranžova a ne Newton?
- **Automatski** uključuje ograničenja (štap prikačen za kolica)
- **Elegantan** pristup za složene sisteme
- **Manje greške** - ne treba ručno računati sile reakcije
- **Standardan** u robotici i kontroli

---

## Euler metoda

### Šta je Euler metoda?

**Euler metoda** je najjednostavniji način rešavanja diferencijalnih jednačina numerički.

#### Problem:
```
Imamo: dy/dt = f(t,y) sa početnim uslovima y(0) = y₀
Želimo: y(t) za t > 0
Analitičko rešenje često nemoguće → numerička aproksimacija
```

#### Euler pristup:
```
Osnovna idea: dy/dt ≈ Δy/Δt

Algoritam:
1. Podeli vreme na male korake: Δt
2. Za svaki korak: y_{n+1} = y_n + Δt * f(t_n, y_n)
3. Ponavljaj...

Geometrijski: Prati tangens krivulje korak po korak
```

### Primer - Cart-Pole:
```python
# Imamo sistem:
# dx/dt = vx
# dvx/dt = f_x (ubrzanje kolica)  
# dθ/dt = vθ  
# dvθ/dt = f_θ (ubrzanje štapa)

# Euler korak:
new_x = x + Δt * vx                    # pozicija
new_vx = vx + Δt * f_x                 # brzina kolica  
new_θ = θ + Δt * vθ                    # ugao
new_vθ = vθ + Δt * f_θ                 # brzina štapa
```

### Zašto Euler metodu?

#### Prednosti:
- **Jednostavna** za implementaciju
- **Brza** za računanje
- **Stabilna** za male korake
- **Intuitivna** - linearno produžavanje

#### Nedostaci:
- **Greške akumuliraju** tokom vremena
- **Nestabilna** za velike korake
- **Aproksimacija** prvog reda

### Poboljšanje u kodu:
```python
# Umesto jednog velikog koraka:
step_size = 0.01

# Koristi se dva mala koraka:
for _ in range(2):
    half_T = self.T / 2  # 0.005
    # Euler korak sa pola veličine
```

**Razlog**: Manji koraci = veća preciznost = stabilnija simulacija

---


## 3. OptimizedQLearning (q_learning.py)

Napredna implementacija Q-Learning algoritma sa Double Q-Learning tehnikom.

### Ključne funkcionalnosti:

#### Double Q-Learning:
- Koristi **dve Q-tabele** za smanjenje overestimation bias-a
- Jedna tabela bira akciju, druga procenjuje vrednost
- Slučajno bira koju tabelu da ažurira (50-50 šanse)

### Šta je overestimation bias?

**Overestimation bias** = tendencija Q-Learning-a da **precenjuje** Q-vrednosti.

#### Uzrok - max operator:
```python
# Standardni Q-Learning update:
target = reward + γ * max(Q(next_state, a)) 

Problem: max() uvek bira NAJVEĆU vrednost
- Ako su ocene netačne → bira najoptimističniju grešku  
- Greške se propagiraju i pojačavaju
- Konačni rezultat: precenjene Q-vrednosti
```

#### Primer problema:
```
Stvarne Q-vrednosti: [1.0, 1.1, 0.9]  
Ocene sa šumom:      [1.2, 0.9, 1.3] ← šum dodao greške

max(ocene) = 1.3 (odabrao pogrešnu akciju!)  
max(stvarne) = 1.1 (trebao je ovu)

Rezultat: Agent misli da je akcija 2 bolja nego što jeste
```

### Double Q-Learning rešenje

#### Glavna ideja:
```
Umesto jedne tabele → koristi DVE nezavisne tabele
QA i QB treniraju se nezavisno na istim podacima
```

#### Algoritam:
```python
# Za svaki update:
if random() < 0.5:
    # Ažuriraj tabelu A, koristi tabelu B za ocenu
    best_action = argmax(QA[next_state])      # A bira akciju
    target = reward + γ * QB[next_state][best_action]  # B ocenjuje
    QA[state][action] += α * (target - QA[state][action])
else:
    # Ažuriraj tabelu B, koristi tabelu A za ocenu  
    best_action = argmax(QB[next_state])      # B bira akciju
    target = reward + γ * QA[next_state][best_action]  # A ocenjuje
    QB[state][action] += α * (target - QB[state][action])
```

#### Zašto funkcioniše?

**Nezavisnost grešaka:**
```
Tabela A: ocene = [1.2, 0.9, 1.3] (greška u poziciji 0,2)
Tabela B: ocene = [0.8, 1.4, 0.7] (greška u poziciji 1)

A bira akciju 2 (najveća u A)
B ocenjuje akciju 2 = 0.7 (niska vrednost u B)
→ Realističnija procena!

B bira akciju 1 (najveća u B)  
A ocenjuje akciju 1 = 0.9 (niska vrednost u A)
→ Ponovo realističnija procena!
```

**Statistički argument:**
- Verovatnoća da OBE tabele imaju pozitivnu grešku za istu akciju = mala
- Očekivana vrednost kombinacije = bliža stvarnoj vrednosti
- Smanjuje se varijansa procena

### Praktične prednosti:

1. **Stabilnije učenje** - manje oscilacija Q-vrednosti
2. **Bolje konvergencija** - približava se optimalnim vrednostima  
3. **Robusnija strategija** - manje sklon "preoptimizmu"
4. **Empirijski dokazano** - bolje performanse u benchmarks

### U kodu:
```python
# Kombinuje tabele za akciju:
combined_q = (self.q_table[state] + self.q_table_2[state]) / 2.0
return np.argmax(combined_q)  # Koristi prosek za stabilniju odluku
```

Ovo daje agentu **uravnoteženu** sliku o vrednostima akcija umesto da se oslanja na potencijalno pristrasnu jednu tabelu.

---

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

# Pokretanje

## Treniraj od nule
```bash
python main.py --train                                 # treniranje od nule
python main.py --load result/cart_pole_model.pkl --train --episodes 1000
```

## Treniraj sa podešenim brojom epizoda
```bash
python main.py --train --episodes 5000                 # podesene epizode
```

## Učitaj model i nastavi da treniraš
```bash
python main.py --load result/cart_pole_model.pkl --train --episodes 1000
```

## Testiraj model
```bash
python main.py --load result/cart_pole_model.pkl --test
```

## Treniraj i odma testiraj
```bash
python main.py --train --test --episodes 2000
```
```

```
usage: main.py [-h] [--train] [--test] [--load LOAD] [--episodes EPISODES] [--save-path SAVE_PATH] [--test-episodes TEST_EPISODES] [--render]

Cart-Pole Q-Learning Agent

options:
  -h, --help            show this help message and exit
  --train               Train the agent
  --test                Test the agent
  --load LOAD           Load model from specified path (e.g., --load result/cart_pole_model.pkl)
  --episodes EPISODES   Number of training episodes (default: 3000)
  --save-path SAVE_PATH
                        Path to save the trained model (default: result/cart_pole_model.pkl)
  --test-episodes TEST_EPISODES
                        Number of test episodes (default: 10)
  --render              Render the environment during testing
```
