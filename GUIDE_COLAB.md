# Guide d'utilisation du générateur de dataset pour Google Colab

Ce guide explique comment utiliser le script `generate_enhanced_dataset.py` dans Google Colab pour générer un dataset de fine-tuning DeepSeek optimisé pour la recherche en mémoire, RAG, APIs WordPress et Crawl4ai.

## Configuration dans Google Colab

1. **Créez un nouveau notebook** dans Google Colab.

2. **Importez le script** en l'uploadant ou en le chargeant depuis GitHub :

```python
# Option 1: Télécharger directement le script
!wget https://raw.githubusercontent.com/username/repo/main/generate_enhanced_dataset.py

# Option 2: Copier-coller le contenu du script
%%writefile generate_enhanced_dataset.py
# Collez ici le contenu du script
```

3. **Installez les dépendances** :

```python
!pip install openai nest-asyncio
```

4. **Configurez votre clé API OpenAI** :

```python
import os
os.environ["OPENAI_API_KEY"] = "votre_clé_api_openai"
```

## Utilisation du script

Pour générer 500 exemples (la valeur par défaut) :

```python
!python generate_enhanced_dataset.py
```

Pour personnaliser la génération :

```python
!python generate_enhanced_dataset.py --num_examples 200 --batch_size 10 --concurrent 4
```

## Options disponibles

- `--num_examples` : Nombre d'exemples à générer (défaut: 500)
- `--batch_size` : Taille des lots pour chaque appel API (défaut: 10)
- `--concurrent` : Nombre de tâches parallèles (défaut: 5)
- `--output` : Chemin du fichier de sortie (défaut: ./data/enhanced_dataset.jsonl)

## Suivi de la progression

Le script affiche la progression en temps réel et enregistre également un fichier de progression `generation_progress.json` qui contient des statistiques sur la génération en cours.

## Préparation pour le fine-tuning DeepSeek

Une fois le dataset généré, vous pouvez l'utiliser pour fine-tuner DeepSeek directement dans Colab :

```python
# Créer un répertoire pour télécharger le modèle fine-tuné
!mkdir -p output-model

# Installer Unsloth
!pip install unsloth

# Lancer le fine-tuning
!python -m unsloth.train \
  --model="deepseek-ai/deepseek-r1-llm-7b" \
  --tokenizer="deepseek-ai/deepseek-r1-llm-7b" \
  --dataset_path="./data/enhanced_dataset.jsonl" \
  --output_dir="./output-model" \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --max_seq_length=2048 \
  --bf16 \
  --batch_size=4 \
  --gradient_accumulation_steps=4
```

## Téléchargement du modèle fine-tuné

Une fois le fine-tuning terminé, vous pouvez télécharger le modèle fine-tuné depuis Colab :

```python
# Compresser le modèle
!tar -czvf output-model.tar.gz output-model/

# Télécharger (un lien de téléchargement apparaîtra)
from google.colab import files
files.download('output-model.tar.gz')
```

## Conversion pour Ollama (localement après téléchargement)

```bash
# Installation de llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Conversion GGUF
python convert.py ./output-model --outtype f16 --outfile renee-enhanced.gguf

# Création du modèle Ollama
ollama create renee-enhanced -f ./Modelfile
```

Contenu du Modelfile :
```
FROM renee-enhanced.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<think>"
PARAMETER stop "</think>"

SYSTEM "Tu es Renée, une assistante IA qui répond uniquement en utilisant les informations fournies dans les contextes. Ta réponse doit être directe, concise, et précise, sans étape de réflexion visible. Si l'information n'est pas trouvée dans les contextes fournis, réponds simplement 'Information non disponible'."
```

## Conseils pour optimiser la génération

- Utilisez un runtime GPU dans Colab pour accélérer la génération (T4 gratuit ou A100 avec Colab Pro)
- Ajustez `batch_size` et `concurrent` en fonction des performances observées
- Si vous rencontrez des erreurs de quota OpenAI, réduisez `concurrent` ou introduisez des délais entre les appels
- Pour générer des exemples plus diversifiés, modifiez les listes `TOPICS` et `DATA_SOURCES` dans le script
