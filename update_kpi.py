import pandas as pd
import requests
import json
from datetime import datetime

url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/World%20Development%20Indicators/World%20Development%20Indicators.csv"
df = pd.read_csv(url, usecols=['Entity', 'GDP per capita (constant 2015 US$)', 'Life expectancy (years)', 'CO₂ emissions (metric tons per capita)']).dropna()

iso_map = requests.get("https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.json").json()
iso_dict = {c['name']: c['alpha-3'] for c in iso_map}

metrics = {}
for _, row in df.iterrows():
    country = row['Entity']
    iso = iso_dict.get(country)
    if not iso: continue
    vec = [
        row['GDP per capita (constant 2015 US$)'] or 0,
        row['Life expectancy (years)'] or 0,
        row['CO₂ emissions (metric tons per capita)'] or 0,
    ]
    import numpy as np
    np.random.seed(hash(iso) % 1000)
    padded = np.pad(vec, (0, 127 - len(vec)), 'constant')
    theta = np.abs(np.corrcoef(padded)[0, 1]) if len(padded) > 1 else 0.5
    omega = np.std(padded) + np.mean(np.abs(np.diff(padded)))
    M = float(np.clip(theta**2 / (omega + 1e-12), 0, 1))
    metrics[iso.lower()] = {"IE128D": M, "dominantDomain": "Económico" if M < 0.5 else "Ecológico"}

with open("country_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[{datetime.utcnow().isoformat()}] Actualizado: {len(metrics)} países")
