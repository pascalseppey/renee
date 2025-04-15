# Guide de Fine-tuning pour DeepSeek avec Unsloth

Ce guide explique comment générer un jeu de données de fine-tuning et comment l'utiliser pour optimiser votre modèle DeepSeek afin qu'il réponde de manière concise et factuelle en se basant uniquement sur les contextes fournis.

## 1. Génération du dataset de fine-tuning

Le script `generate_finetune_dataset.py` permet de générer 500 exemples pour le fine-tuning:

```bash
# D'abord, assurez-vous que la clé API OpenAI est définie
export OPENAI_API_KEY="votre_cle_api_openai"

# Exécutez le script de génération
python generate_finetune_dataset.py
```

Ce script va :
- Générer 500 exemples variés couvrant 15 domaines différents
- Utiliser GPT-4o-mini pour créer des exemples de haute qualité
- Sauvegarder les exemples bruts dans `./data/generated_examples.json`
- Convertir les exemples au format d'entraînement dans `./data/processed_dataset.jsonl`

## 2. Fine-tuning avec Unsloth

Une fois les exemples générés, vous pouvez fine-tuner votre modèle DeepSeek avec Unsloth :

```bash
# Installez Unsloth si ce n'est pas déjà fait
pip install unsloth

# Lancez le fine-tuning
python -m unsloth.train \
  --model="deepseek-ai/deepseek-r1-llm-7b" \
  --tokenizer="deepseek-ai/deepseek-r1-llm-7b" \
  --dataset_path="./data/processed_dataset.jsonl" \
  --output_dir="./output-model" \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --max_seq_length=2048 \
  --load_in_8bit \
  --bf16 \
  --batch_size=4 \
  --gradient_accumulation_steps=4 \
  --save_steps=50 \
  --save_total_limit=3
```

### Paramètres recommandés

- **learning_rate**: 2e-5 est un bon compromis entre rapidité d'apprentissage et stabilité
- **num_train_epochs**: 3 époques suffisent généralement pour un bon fine-tuning
- **batch_size**: Ajustez en fonction de la mémoire GPU disponible (4 est un bon départ)
- **gradient_accumulation_steps**: Permet de simuler des batches plus grands sans augmenter la consommation mémoire

## 3. Conversion pour Ollama

Pour utiliser votre modèle fine-tuné avec Ollama, vous devez le convertir au format GGUF :

```bash
# Installation de llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Conversion GGUF
python convert.py ./output-model --outtype f16 --outfile renee-direct.gguf

# Création du modèle Ollama
ollama create renee-direct -f ./Modelfile
```

Contenu du Modelfile :
```
FROM renee-direct.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<think>"
PARAMETER stop "</think>"

SYSTEM "Tu es Renée, une assistante IA qui répond uniquement en utilisant les informations fournies dans les contextes. Ta réponse doit être directe, concise, et précise, sans étape de réflexion visible. Si l'information n'est pas trouvée dans les contextes fournis, réponds simplement 'Information non disponible'."
```

## 4. Intégration dans votre système de mémoire

Modifiez le fichier `test_memory_interactive.py` pour utiliser votre nouveau modèle :

```python
# Remplacez cette ligne
self.model_service = DeepSeekService(model_name="deepseek-coder:1.3b")

# Par celle-ci
self.model_service = DeepSeekService(model_name="renee-direct")
```

## Notes supplémentaires

- Le fine-tuning prend généralement entre 1 et 4 heures en fonction de votre matériel
- Si vous n'avez pas de GPU puissant, vous pouvez utiliser un service cloud comme Google Colab
- Assurez-vous que les chemins des fichiers sont corrects dans tous les scripts
- Testez le modèle fine-tuné avec des questions variées pour vérifier qu'il répond de manière concise et factuelle

## Dépannage

Si vous rencontrez des problèmes avec Unsloth, essayez ces solutions alternatives :
- Utiliser la bibliothèque transformers standard (plus lente mais plus compatible)
- Réduire la taille du batch ou la longueur de séquence
- Utiliser un modèle DeepSeek plus petit (1.3B au lieu de 7B)
- Vérifier les erreurs dans les logs pour identifier les problèmes spécifiques
