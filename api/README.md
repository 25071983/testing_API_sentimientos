# Fast API

## Python env

```bash

# crear
$ python -m venv venv

# activar
$ ./venv/Scripts/activate
$ cd api
# cambiar a dir FasAPI
$ cd .\ml-service\
# instalamos requisitos
$ pip install -r requirements.txt
```

## API

```bash
# run Fast API
$ python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Test en PowerShell

### Curl - NO funciona

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"excelente atencion\"}"
```

### Invoke-RestMethod

```bash
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"La experiencia fue agradable"}'
```

**Ejemplo Respuesta**

```text
prevision probabilidad model_version
--------- ------------ -------------
Positivo        0,7847 v3.0.1
```

## irm

```bash
irm http://localhost:8000/predict -Method POST -ContentType application/json -Body '{"text":"La experiencia fue agradable"}'
```

**Ejemplo Respuesta**

```text
prevision probabilidad model_version
--------- ------------ -------------
Positivo        0,7847 v3.0.1
```

## Python para invocación

```python
import requests

url = "http://localhost:8000/predict"


payload = {
    "text": "La experiencia fue agradable"
}

response = requests.post(url, json=payload)

# Raise error if request failed (4xx / 5xx)
response.raise_for_status()

data = response.json()
print(data)
```
