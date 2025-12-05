import pandas as pd
import requests
import json
from datetime import datetime
import numpy as np

# Dataset público y permanente: CO₂ por país (OWID)
# Eliminé los espacios en blanco al final de la URL
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df = pd.read_csv(url, usecols=['country', 'co2_per_capita', 'population']).dropna()

# Mapa de países a ISO-3
# Eliminé los espacios en blanco al final de la URL
iso_map = requests.get("https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.json").json()
iso_dict = {c['name']: c['alpha-3'] for c in iso_map}

metrics = {}
for _, row in df.iterrows():
    country = row['country']
    iso = iso_dict.get(country)
    if not iso: 
        continue
    
    vec = [
        row['co2_per_capita'] or 0,
        row['population'] or 0,
    ]
    
    # Padding a 127D con semilla determinística
    np.random.seed(hash(iso) % 1000)
    padded = np.pad(vec, (0, max(0, 127 - len(vec))), 'constant')
    
    # CORRECCIÓN: theta solo se calcula si hay al menos 2 elementos y son diferentes
    if len(padded) < 2:
        theta = 0.0  # o algún valor por defecto
    elif len(padded) == 2 and padded[0] == padded[1]:
        # Si ambos valores son iguales, la correlación no está definida, usar 0
        theta = 0.0
    else:
        # Calcular correlación entre pares de valores si hay suficientes datos
        # Para vectores largos, podrías querer calcular la auto-correlación o usar otro enfoque
        if len(padded) >= 2:
            # Tomamos solo los primeros 2 valores para calcular correlación
            subset = padded[:2]
            if subset[0] != subset[1]:  # Asegurarse que no sean idénticos
                corr_matrix = np.corrcoef([subset], [subset[::-1]])  # Ejemplo simple
                theta = np.abs(corr_matrix[0, 1])
            else:
                theta = 0.0
        else:
            theta = 0.0
    
    omega = np.std(padded) + np.mean(np.abs(np.diff(padded))) if len(padded) > 1 else np.std(padded)
    M = float(np.clip(theta**2 / (omega + 1e-12), 0, 1))
    metrics[iso.lower()] = {"IE128D": M, "dominantDomain": "Ecológico" if M < 0.5 else "Económico"}

with open("country_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[{datetime.utcnow().isoformat()}] Actualizado: {len(metrics)} países")
