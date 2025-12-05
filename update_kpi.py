import pandas as pd
import requests
import json
from datetime import datetime

# Dataset público y actualizado de Our World in Data
url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/world-development-indicators/world_development_indicators.csv"
df = pd.read_csv(url, usecols=['Entity', 'GDP per capita (constant 2015 US$)', 'Life expectancy (years)', 'CO₂ emissions (metric tons per capita)']).dropna()

# Mapa de países a ISO-3
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
    # Padding a 127D con semilla determinística
    import numpy as np
    np.random.seed(hash(iso) % 1000)
    padded = np.pad(vec, (0, 127 - len(vec)), 'constant')
    theta = np.abs(np.corrcoef(padded)[0, 1]) if len(padded) > 1 else 0.5
