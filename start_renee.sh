#!/bin/bash
# Script de démarrage de l'écosystème Renée

# Activation de l'environnement virtuel
source "$(pwd)/venv/bin/activate"

# Démarrage du serveur FastAPI
cd "$(pwd)/src"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
