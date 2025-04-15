#!/bin/bash

# Configuration
NUM_EXAMPLES=30  # Nombre d'exemples à traiter à chaque itération
DEEPSEEK_MODEL="deepseek-r1:7b"  # Modèle DeepSeek à utiliser
LOG_FILE="training_loop.log"     # Fichier de log pour conserver l'historique des exécutions
OPENAI_API_KEY="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"

# Exporter la clé API OpenAI
export OPENAI_API_KEY

# Exécution en boucle continue
echo "Démarrage de la boucle d'entraînement RLHF pour DeepSeek ($(date))" | tee -a "$LOG_FILE"

while true; do
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "Nouvelle itération démarrée à $(date)" | tee -a "$LOG_FILE"
    
    # Exécuter le script d'entraînement
    python rlhf_wordpress_memory_trainer.py --num_examples "$NUM_EXAMPLES" --deepseek_model "$DEEPSEEK_MODEL" 2>&1 | tee -a "$LOG_FILE"
    
    # Vérifier si l'exécution s'est terminée normalement
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Erreur détectée (code $EXIT_CODE) à $(date)" | tee -a "$LOG_FILE"
        echo "Tentative de redémarrage dans 10 secondes..." | tee -a "$LOG_FILE"
        sleep 10
    else
        echo "Itération terminée avec succès à $(date)" | tee -a "$LOG_FILE"
        echo "Démarrage de la prochaine itération dans 5 secondes..." | tee -a "$LOG_FILE"
        sleep 5
    fi
done
