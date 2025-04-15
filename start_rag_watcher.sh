#!/bin/bash
# Script pour démarrer le système de surveillance de fichiers RAG

# Activation de l'environnement virtuel
cd "$(dirname "$0")"
source venv/bin/activate

# Création des répertoires nécessaires
mkdir -p logs
mkdir -p ~/Desktop/ReneeRAGDocs/processed

# Démarrage du système de surveillance
echo "Démarrage du système de surveillance RAG..."
echo "Les fichiers placés dans ~/Desktop/ReneeRAGDocs/ seront automatiquement ajoutés au RAG"
echo "Les fichiers traités seront déplacés dans ~/Desktop/ReneeRAGDocs/processed/"
echo ""
echo "Formats supportés: .txt, .md, .pdf, .docx, .html, .json, .csv"
echo ""
echo "Pour arrêter le système, utilisez Ctrl+C"
echo ""

# Exécution du script de surveillance
python src/rag/file_watcher.py
