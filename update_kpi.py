import pandas as pd
import requests
import json
from datetime import datetime
import numpy as np

# Dataset público y permanente: CO₂ por país (OWID)
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df = pd.read_csv(url, usecols=['country', 'co2_per_capita', 'population']).dropna()

# Mapa de países a ISO-3
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
    
    # CORRECCIÓN: theta solo se puede calcular si hay al menos 2 valores distintos
    if len(padded) < 2:
        theta = 0.0
    elif np.all(padded == padded[0]):  # Todos los valores son iguales
        theta = 0.0
    else:
        # Usar una medida de coherencia más adecuada para vectores multidimensionales
        # Por ejemplo, desviación estándar normalizada o entropía
        # Para simplificar, usar la correlación entre primeros dos valores si son diferentes
        if len(padded) >= 2 and padded[0] != padded[1]:
            try:
                # Correlación entre el vector y una versión desfasada
                theta = min(1.0, abs(np.corrcoef(padded[:-1], padded[1:])[0,1]) if len(padded) > 1 else 0.0)
            except:
                theta = 0.0
        else:
            theta = 0.0
    
    # Calcular omega (medida de complejidad)
    if len(padded) > 1:
        omega = np.std(padded) + np.mean(np.abs(np.diff(padded)))
    else:
        omega = np.std(padded)
    
    M = float(np.clip(theta**2 / (omega + 1e-12), 0, 1))
    metrics[iso.lower()] = {"IE128D": M, "dominantDomain": "Ecológico" if M < 0.5 else "Económico"}

with open("country_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[{datetime.utcnow().isoformat()}] Actualizado: {len(metrics)} países")
