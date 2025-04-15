#!/bin/bash
# Script pour réinitialiser la base de données RAG

# Activation de l'environnement virtuel
cd "$(dirname "$0")"
source venv/bin/activate

# Exécution du script de réinitialisation
echo "Réinitialisation de la base de données RAG..."
python reset_rag_database.py

echo ""
echo "Pour redémarrer le système de surveillance après réinitialisation, exécutez :"
echo "./start_rag_watcher.sh"
echo ""
