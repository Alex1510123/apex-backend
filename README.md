# APEX Markets — Backend API

Marktanalyse-Backend mit Yahoo Finance Daten, Sektor-Scoring und Portfolio-Analyse.

## Setup (5 Minuten)

### 1. Python-Umgebung erstellen
```bash
cd apex-backend
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Server starten
```bash
python main.py
```

Server läuft auf `http://localhost:8000`

### 4. Interaktive API-Docs öffnen
Browser öffnen: **http://localhost:8000/docs**

Hier kann man jeden Endpoint direkt testen.

---

## Endpoints

| Endpoint | Methode | Beschreibung |
|---|---|---|
| `/health` | GET | Status-Check |
| `/quote/{ticker}` | GET | Aktueller Kurs (z.B. `/quote/AAPL`) |
| `/history/{ticker}?period=1y` | GET | Kurshistorie (1mo/3mo/6mo/1y/2y/5y) |
| `/sectors` | GET | Alle 11 Sektor-ETFs mit Composite Score |
| `/screener/top?limit=10` | GET | Top Outperformer aus Universum |
| `/macro` | GET | Makroindikatoren (Zinsen, USD, VIX, Gold, Öl, BTC) |
| `/portfolio/analyze` | POST | Portfolio-Analyse mit Risiko-Metriken |

### Beispiel: Sektoren abrufen
```bash
curl http://localhost:8000/sectors
```

### Beispiel: Portfolio analysieren
```bash
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [
      {"ticker": "AAPL", "shares": 50, "avg_cost": 170},
      {"ticker": "MSFT", "shares": 30, "avg_cost": 380}
    ]
  }'
```

---

## Architektur

### Composite Score
Jeder Ticker bekommt einen Score 0–100 basierend auf:
- **Momentum-Score (50%)**: Gewichtete Returns über 1M / 3M / 6M / 12M (40/30/20/10)
- **Relative-Strength-Score (50%)**: Outperformance vs. S&P 500 über 3M

### Caching
- Quotes: 60s TTL
- Historische Daten: 5min TTL
- Screener: 10min TTL

Spart Yahoo-Finance-Calls und beschleunigt Antworten dramatisch.

### Risiko-Metriken (Portfolio)
- Beta (vs. S&P 500)
- Sharpe Ratio (annualisiert, RFR = 4.4%)
- Annualisierte Volatilität
- Max Drawdown (1J)

---

## Deployment auf Railway

1. GitHub-Repo erstellen, Code pushen
2. railway.app → New Project → Deploy from GitHub
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Kosten: ca. 5$/mo (Hobby Tier)

---

## Frontend-Integration

Im React-Dashboard die Datenarrays durch echte API-Calls ersetzen:

```javascript
const API = "http://localhost:8000";  // oder Railway-URL

useEffect(() => {
  fetch(`${API}/sectors`).then(r => r.json()).then(d => setSectors(d.sectors));
  fetch(`${API}/macro`).then(r => r.json()).then(d => setMacro(d.indicators));
  fetch(`${API}/screener/top?limit=10`).then(r => r.json()).then(d => setTopStocks(d.results));
}, []);
```
