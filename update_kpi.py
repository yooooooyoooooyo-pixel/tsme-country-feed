import pandas as pd
import requests
import json
from datetime import datetime

# Dataset pequeño y permanente: CO₂ por país (OWID)
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df = pd.read_csv(url, usecols=['country', 'co2_per_capita', 'population']).dropna()

# Mapa de países a ISO-3
iso_map = requests.get("https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.json").json()
iso_dict = {c['name']: c['alpha-3'] for c in iso_map}

metrics = {}
for _, row in df.iterrows():
    country = row['country']
    iso = iso_dict.get(country)
    if not iso: continue
    vec = [
        row['co2_per_capita'] or 0,
        row['population'] or 0,
    ]
    # Padding a 127D con semilla determinística
    import numpy as np
    np.random.seed(hash(iso) % 1000)
    padded = np.pad(vec, (0, 127 - len(vec)), 'constant')
    theta = np.abs(np.corrcoef(padded)[0, 1]) if len(padded) > 1 else 0.5
    omega = np.std(padded) + np.mean(np.abs(np.diff(padded)))
    M = float(np.clip(theta**2 / (omega + 1e-12), 0, 1))
    metrics[iso.lower()] = {"IE128D": M, "dominantDomain": "Ecológico" if M < 0.5 else "Económico"}

with open("country_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[{datetime.utcnow().isoformat()}] Actualizado: {len(metrics)} países")
