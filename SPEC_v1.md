# SPEC v1 — Modelo Predictivo Mundial 2026: backtesting + fuerza dinámica

> Documento para ejecutar en Claude Code. Objetivo: pasar del MVP (fuerza
> estática) a un modelo (a) **medido** out-of-sample y (b) con **fuerza
> dinámica** que capture forma reciente y declive de planteles — el mecanismo
> detrás de que el campeón vigente raramente repita.

## 0. Estado actual (v0, ya construido)
- `knockout_sim.py`: Poisson `goals ~ C(team)+C(opp)+home` con decaimiento
  temporal (vida media 2 años) sobre `results.csv` (martj42, 49.446 partidos).
- Capa Ridge features→fuerza (R² CV 0.75). Montecarlo 15k torneos.
- **Problema raíz**: la fuerza es ESTÁTICA. No deriva en el tiempo, así que un
  campeón con plantel envejecido o que sobrerrindió queda sobrevalorado. No hay
  validación out-of-sample → no sabemos si el modelo calibra.

## Principio rector
**No se mejora lo que no se mide.** Fase 1 (medición) es prerequisito de todo.
Cada cambio posterior se acepta SOLO si mejora una métrica out-of-sample.

---

## FASE 1 — Harness de backtesting y calibración  *(máxima prioridad)*

### 1.1 Walk-forward estricto
Para cada torneo objetivo T ∈ {WC2010, WC2014, WC2018, WC2022} (y opcional
Euro 2016/2020/2024, Copa América):
1. Entrenar usando SOLO partidos con `date < fecha_inicio(T)`. Prohibido leak.
2. Predecir cada partido de T (y simular el torneo completo).
3. Guardar probabilidades predichas vs resultado real.

### 1.2 Métricas (a nivel partido, sobre 1-X-2)
- **RPS (Ranked Probability Score)** — métrica estándar de oro para fútbol
  (resultado ordenado local/empate/visita). Reportar RPS medio. Menor = mejor.
- **Log-loss** y **Brier score** (multiclase) como complemento.
- **Curva de calibración / reliability diagram**: binnear prob predichas en
  deciles, graficar prob predicha vs frecuencia observada. Reportar
  **ECE (Expected Calibration Error)**.

### 1.3 Métricas a nivel torneo
- Probabilidad que el modelo asignó al campeón real (vs prior uniforme 1/48).
- ¿El finalista/semifinalistas reales estaban en el top-k de prob?
- Log-loss sobre "quién fue campeón".

### 1.4 Benchmark contra el mercado  *(el juez final)*
- Conseguir cuotas de cierre (closing odds) de partidos históricos.
  - Fuente factible: football-data.co.uk (mucho club; internacional es más
    escaso — evaluar Kaggle "International football results" + datasets de odds,
    o oddsportal). Si no hay odds de selecciones, usar eloratings.net como
    benchmark externo de fuerza.
- Quitar el margen (de-vig): normalizar las probabilidades implícitas (método
  básico) o **Shin** (mejor, corrige favorite-longshot bias).
- Comparar RPS del modelo vs RPS del mercado sobre el mismo set de partidos.

### 1.5 Criterio de aceptación Fase 1
- Harness corre los 4 mundiales sin leak (test unitario que verifica
  `max(train.date) < min(test.date)`).
- Produce tabla de RPS/log-loss/Brier/ECE por torneo + curva de calibración.
- Establece el **baseline**: RPS del modelo v0 estático. Todo lo que sigue se
  compara contra este número.

### 1.6 Firmas sugeridas
```python
def train_model(matches: pd.DataFrame, cutoff: pd.Timestamp) -> Model: ...
def predict_match(model, home, away, neutral, host=False) -> tuple[float,float,float]:  # (pL, pE, pV)
    ...
def rps(probs: np.ndarray, outcome_idx: int) -> float: ...
def backtest(tournaments: list[str]) -> pd.DataFrame: ...      # una fila por partido
def calibration_report(bt: pd.DataFrame) -> dict: ...          # ECE + bins
def vs_market(bt: pd.DataFrame, odds: pd.DataFrame) -> dict: ...
```

---

## FASE 2 — Fuerza dinámica (ataque/defensa variables en el tiempo)

Resuelve directo la observación del "campeón no repite": la fuerza deja de ser
un número fijo y **deriva** con los resultados y la edad del plantel.

### 2.1 Nivel 1 — Elo con goles  *(empezar acá: simple y robusto)*
- Rating Elo por selección, actualizado partido a partido. FIFA ya usa un Elo.
- Ajuste por diferencia de goles (factor multiplicador tipo World Football Elo).
- K-factor mayor en torneos oficiales / mundiales; menor en amistosos.
- Mapear diferencia de Elo → prob 1-X-2 (curva logística calibrada en Fase 1).
- Ventaja: la fuerza del campeón cae sola si rinde por debajo de lo esperado.

### 2.2 Nivel 2 — State-space / Poisson bivariado dinámico
- Estados latentes por equipo `θ_t = (ataque_t, defensa_t)` que siguen un
  random walk: `θ_t = θ_{t-1} + η_t`, `η_t ~ N(0, Σ)`.
- Observación: `goles ~ Poisson(exp(μ + ataque_home,t − defensa_away,t + γ·home))`.
- Estimar con **filtro de Kalman** (tras linealizar/usar score-driven) o
  partículas/MCMC. Ref: Koopman & Lit (2015), *A dynamic bivariate Poisson model*.
- Alternativa pragmática: modelo **score-driven (GAS)** — actualización tipo Elo
  pero derivada de la verosimilitud, más principiada que Elo ad-hoc.
- Σ (varianza del random walk) controla cuán rápido "olvida" — calibrar por
  máxima verosimilitud out-of-sample, NO a ojo.

### 2.3 Ajuste por declive de plantel  *(la palanca explícita del campeón)*
- Covariable que mete drift NEGATIVO en el random walk según edad media
  ponderada por minutos del plantel. Plantel viejo → fuerza tiende a bajar.
- Implementar como término en la media del estado: `E[θ_t] = θ_{t-1} − f(edad)`.

### 2.4 Criterio de aceptación Fase 2
- RPS out-of-sample del modelo dinámico **< RPS del v0 estático** (Fase 1).
- Verificar el caso concreto: ¿baja la prob de título de campeones vigentes que
  históricamente decayeron (España 2014, Alemania 2018) en backtest? Debería.

---

## FASE 3 — Features a nivel jugador  *(después de Fase 2)*
- Perfil de edad del plantel, minutos en ligas top-5, disponibilidad/lesiones,
  valor de los 11 titulares (no del plantel completo).
- Fuentes: FBref vía `soccerdata` (pip), Transfermarkt (scrape o dataset).
- Entran como covariables al drift del estado (Fase 2.3) o como prior estructural.

---

## FASE 4 — Refinamientos del motor de goles
- **Dixon-Coles**: corregir dependencia en marcadores bajos (0-0, 1-1), que el
  Poisson puro subestima. Parámetro ρ.
- **Poisson bivariado** real (correlación entre goles de ambos equipos).
- Altitud (Ciudad de México 2.240m), viajes, descanso entre partidos.
- Ventaja de local de anfitriones (MEX/USA/CAN) también en eliminación.

---

## FASE 5 — Incertidumbre
- **Bootstrap** sobre los parámetros del modelo → bandas de confianza para cada
  probabilidad (P(campeón) = 18% ± ?). Hoy el Montecarlo solo captura la
  aleatoriedad de los partidos, no la del modelo.
- Reportar intervalos, no puntos.

---

## Estructura de proyecto sugerida
```
worldcup26/
├── data/
│   ├── results.csv              # martj42 (ya está)
│   ├── odds_hist.csv            # cuotas históricas (Fase 1.4)
│   └── squads/                  # datos de jugador (Fase 3)
├── src/
│   ├── data_loader.py
│   ├── strength_static.py       # v0 (baseline a batir)
│   ├── strength_elo.py          # Fase 2.1
│   ├── strength_statespace.py   # Fase 2.2
│   ├── features.py
│   ├── simulate.py              # Montecarlo torneo (refactor de knockout_sim.py)
│   └── backtest.py              # Fase 1
├── tests/
│   └── test_no_leak.py          # train.date < test.date SIEMPRE
├── notebooks/
│   └── calibration.ipynb
└── SPEC_v1.md
```

## Orden de ejecución (ROI decreciente)
1. **Fase 1** completa → tener el baseline medido. Sin esto, todo lo demás es fe.
2. **Fase 2.1** (Elo) → ataca el problema del campeón, barato.
3. **Fase 1** otra vez → ¿mejoró el RPS? Si no, no avanzar.
4. **Fase 2.2** (state-space) si Elo no alcanza.
5. **Fase 3** (jugador), **Fase 4** (Dixon-Coles), **Fase 5** (bootstrap).

## Notas de datos
- `results.csv`: dominio público, github.com/martj42/international_results.
- Validación externa de Elo: eloratings.net.
- Cuidado con leakage de odds: usar SOLO closing odds previas al partido.
- Para selecciones, la muestra de odds históricas es escasa — si no se consigue,
  usar el Elo externo como benchmark de "mercado" sustituto.
