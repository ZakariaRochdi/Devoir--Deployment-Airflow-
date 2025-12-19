# Dockerfile

FROM python:3.9-slim-buster
WORKDIR /app

# Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code
COPY . .

# Configuration et Exécution
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 3000
CMD ["python", "app.py"]