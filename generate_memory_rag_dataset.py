#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour générer un dataset de fine-tuning DeepSeek optimisé pour 
la recherche naturelle dans la mémoire et le RAG.
"""

import os
import json
import time
import random
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import argparse

# Configuration par défaut
DEFAULT_OUTPUT_FILE = "./data/memory_rag_dataset.jsonl"
NUM_EXAMPLES = 500
BATCH_SIZE = 10
MAX_CONCURRENT = 5

# Thèmes variés pour générer des exemples diversifiés
TOPICS = [
    "Informatique et programmation",
    "Science et technologie",
    "Histoire et civilisations",
    "Économie et finance",
    "Santé et médecine",
    "Art et littérature",
    "Psychologie et comportement",
    "Philosophie et éthique",
    "Environnement et développement durable",
    "Business et entrepreneuriat",
    "Politique et société"
]

# Types de questions pour varier les exemples
QUESTION_TYPES = [
    "Factuelle simple",
    "Comparative",
    "Explicative",
    "Analytique",
    "Lien entre concepts",
    "Application pratique",
    "Résumé de connaissances",
    "Contraste d'idées",
    "Chronologique",
    "Définition de concept"
]

class MemoryRAGDatasetGenerator:
    def __init__(self, api_key, output_file, num_examples, batch_size, max_concurrent):
        """Initialise le générateur de dataset pour mémoire et RAG."""
        self.api_key = api_key
        self.output_file = output_file
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.client = OpenAI(api_key=api_key)
        self.examples = []
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Fichier pour sauvegarder l'avancement
        self.progress_file = os.path.join(os.path.dirname(output_file), "generation_progress.json")
    
    def generate_batch(self, batch_id):
        """Génère un lot d'exemples pour l'entraînement."""
        current_batch_size = min(self.batch_size, self.num_examples - batch_id * self.batch_size)
        if current_batch_size <= 0:
            return []
        
        topic = random.choice(TOPICS)
        question_type = random.choice(QUESTION_TYPES)
        
        print(f"🔄 Génération du lot {batch_id+1}: {current_batch_size} exemples - Thème: {topic} - Type: {question_type}")
        
        prompt = f"""
Tu vas créer {current_batch_size} exemples d'entraînement pour fine-tuner un LLM (DeepSeek) à chercher intelligemment dans sa mémoire et son système RAG.

OBJECTIF: Créer des exemples qui entraîneront le LLM à:
1. Identifier automatiquement quand utiliser sa mémoire ou le RAG pour répondre
2. Extraire précisément l'information pertinente des contextes fournis
3. Produire des réponses CONCISES (20-30 mots) basées UNIQUEMENT sur l'information disponible
4. Répondre "Information non disponible" quand le contexte ne suffit pas

THÈME: {topic}
TYPE DE QUESTIONS: {question_type}

FORMAT POUR CHAQUE EXEMPLE:
```
{{
  "conversation_history": [
    {{
      "role": "user", 
      "content": "(question précédente optionnelle)"
    }},
    {{
      "role": "assistant", 
      "content": "(réponse précédente optionnelle)"
    }}
  ],
  "memory_context": "Information stockée dans la mémoire hiérarchique du LLM, représentant des connaissances accumulées sur l'utilisateur et des conversations passées",
  "rag_context": "Information extraite d'une base de connaissances externe, comme des articles, documents, ou wikis sur le sujet",
  "current_question": "Question actuelle de l'utilisateur qui nécessite de chercher dans les contextes",
  "ideal_concise_answer": "Réponse concise et factuelle (20-30 mots) qui utilise UNIQUEMENT l'information des contextes"
}}
```

INSTRUCTIONS:
- Pour chaque exemple, crée un scénario cohérent incluant les contextes et la question
- La 'memory_context' doit contenir des informations personnalisées comme des préférences ou historiques d'interaction
- Le 'rag_context' doit contenir des informations factuelles ou techniques sur le sujet
- Varie la complexité des questions (simples et complexes)
- Inclus des cas où l'information n'est PAS disponible dans les contextes
- L'ideal_concise_answer doit être ultra concise (20-30 mots) et factuelle
- IMPORTANT: La réponse doit se baser UNIQUEMENT sur l'information des contextes fournis

Renvoie uniquement un tableau JSON contenant {current_batch_size} exemples dans le format spécifié.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # On utilise un modèle plus puissant pour générer des exemples de qualité
                messages=[
                    {"role": "system", "content": "Tu es un expert en création de datasets pour fine-tuning de modèles de langage. Tu génères des exemples de haute qualité pour l'entraînement à la recherche en mémoire et RAG."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            batch_data = json.loads(content)
            
            # Normaliser le format en cas de différences dans la structure renvoyée
            examples = []
            if isinstance(batch_data, list):
                examples = batch_data
            elif "examples" in batch_data:
                examples = batch_data["examples"]
            else:
                for key, value in batch_data.items():
                    if isinstance(value, dict):
                        examples.append(value)
                    elif isinstance(value, list):
                        examples.extend(value)
            
            # Validation des exemples
            valid_examples = []
            for example in examples:
                # Vérifier que tous les champs requis sont présents
                required_fields = ["conversation_history", "memory_context", "rag_context", 
                                  "current_question", "ideal_concise_answer"]
                
                if all(field in example for field in required_fields):
                    # Formater l'exemple pour DeepSeek
                    formatted_example = self._format_for_deepseek(example)
                    valid_examples.append(formatted_example)
                else:
                    missing = [field for field in required_fields if field not in example]
                    print(f"⚠️ Exemple incomplet, champs manquants: {missing}")
            
            print(f"✅ Lot {batch_id+1}: {len(valid_examples)}/{current_batch_size} exemples valides")
            
            # Enregistrer les exemples valides
            with self.lock:
                self.examples.extend(valid_examples)
                self._save_progress()
            
            return valid_examples
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération du lot {batch_id+1}: {str(e)}")
            return []
    
    def _format_for_deepseek(self, example):
        """Convertit un exemple au format requis pour le fine-tuning de DeepSeek."""
        # Construire le prompt système avec les contextes
        system_prompt = f"""Tu es Renée, une assistante IA francophone qui suit ces règles strictes:
1. Réponds UNIQUEMENT en utilisant les informations des CONTEXTES ci-dessous
2. Si l'information n'est pas dans les CONTEXTES, réponds simplement "Information non disponible"
3. Sois TRÈS CONCISE (20-30 mots maximum)
4. Ne génère PAS de section <think> - réponds directement
5. Utilise un langage simple et direct

CONTEXTE DE LA MÉMOIRE HIÉRARCHIQUE:
{example.get('memory_context', '')}

CONTEXTE DU SYSTÈME DE RAG:
{example.get('rag_context', '')}

RAPPEL: Réponds UNIQUEMENT avec les informations présentes dans les CONTEXTES ci-dessus.
"""
        
        # Formater au format d'entraînement DeepSeek
        formatted_example = {
            "messages": [
                {"role": "system", "content": system_prompt}
            ]
        }
        
        # Ajouter l'historique de conversation si présent
        conversation_history = example.get('conversation_history', [])
        if conversation_history and isinstance(conversation_history, list):
            formatted_example["messages"].extend(conversation_history)
        
        # Ajouter la question actuelle et la réponse idéale
        formatted_example["messages"].append(
            {"role": "user", "content": example.get('current_question', '')}
        )
        formatted_example["messages"].append(
            {"role": "assistant", "content": example.get('ideal_concise_answer', '')}
        )
        
        return formatted_example
    
    def _save_progress(self):
        """Sauvegarde les exemples actuels au format JSONL et l'avancement."""
        # Sauvegarder les exemples au format JSONL
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Sauvegarder le progrès
        progress = {
            "total_examples": self.num_examples,
            "generated_examples": len(self.examples),
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "percentage_complete": round(len(self.examples) / self.num_examples * 100, 2)
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def generate_dataset(self):
        """Génère l'ensemble du dataset en parallèle."""
        start_time = time.time()
        print(f"🚀 Démarrage de la génération de {self.num_examples} exemples...")
        
        # Calculer le nombre de lots nécessaires
        num_batches = (self.num_examples + self.batch_size - 1) // self.batch_size
        
        # Soumettre les tâches de génération
        futures = []
        for batch_id in range(num_batches):
            future = self.executor.submit(self.generate_batch, batch_id)
            futures.append(future)
        
        # Attendre et traiter les résultats
        total_valid_examples = 0
        for future in futures:
            examples = future.result()
            total_valid_examples += len(examples)
        
        # Si nous n'avons pas assez d'exemples, générer des lots supplémentaires
        additional_batch_id = num_batches
        while total_valid_examples < self.num_examples:
            remaining = self.num_examples - total_valid_examples
            print(f"⚠️ Il manque {remaining} exemples, génération de lots supplémentaires...")
            
            future = self.executor.submit(self.generate_batch, additional_batch_id)
            examples = future.result()
            total_valid_examples += len(examples)
            additional_batch_id += 1
        
        # Limiter aux premiers NUM_EXAMPLES si nous en avons trop généré
        if len(self.examples) > self.num_examples:
            self.examples = self.examples[:self.num_examples]
            self._save_progress()
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"✨ Génération terminée! {len(self.examples)}/{self.num_examples} exemples générés en {duration:.2f} secondes")
        print(f"📊 Vitesse moyenne: {len(self.examples) / duration:.2f} exemples/seconde")
        print(f"💾 Dataset sauvegardé dans: {self.output_file}")
        
        return self.examples

def main():
    parser = argparse.ArgumentParser(description='Générateur de dataset pour fine-tuning DeepSeek sur mémoire et RAG')
    parser.add_argument('--api_key', type=str, help='Clé API OpenAI')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, help='Fichier de sortie au format JSONL')
    parser.add_argument('--num_examples', type=int, default=NUM_EXAMPLES, help='Nombre d\'exemples à générer')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Taille des lots pour la génération')
    parser.add_argument('--concurrent', type=int, default=MAX_CONCURRENT, help='Nombre de tâches parallèles')
    
    args = parser.parse_args()
    
    # Récupérer la clé API soit des arguments, soit des variables d'environnement
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Erreur: Clé API OpenAI requise. Utilisez --api_key ou définissez la variable d'environnement OPENAI_API_KEY")
        return
    
    # Initialiser et lancer le générateur
    generator = MemoryRAGDatasetGenerator(
        api_key=api_key,
        output_file=args.output,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent
    )
    
    generator.generate_dataset()

if __name__ == "__main__":
    main()
