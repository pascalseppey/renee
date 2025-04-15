#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import nest_asyncio

# Pour résoudre le problème d'event loop dans certains environnements
nest_asyncio.apply()

# Configuration générale
OUTPUT_FILE = "./data/generated_examples.json"
PROCESSED_FILE = "./data/processed_dataset.jsonl"  # Format pour fine-tuning
OPENAI_API_KEY = "sk-votre_cle_api_ici"  # À remplacer par votre clé API
NUM_EXAMPLES = 500  # Ajusté à 500 comme demandé
BATCH_SIZE = 5
MAX_CONCURRENT = 5

# Constantes pour la mémoire hiérarchique
NIVEAUX = {
    1: "échange",      # Conversations directes
    2: "résumé",       # Résumés des conversations
    3: "synthèse",     # Synthèses des résumés
    4: "conscience",   # Réflexions sur l'identité et l'évolution
    5: "métacognition" # Réflexion sur les processus de pensée
}

# Schéma de pensée structuré
SCHEMA_DE_PENSEE = """
# TEMPLATE DE PENSÉE STRUCTURÉE

## 1. ANALYSE INITIALE
- Définir clairement la requête ou le problème
- Identifier les informations pertinentes dans le contexte
- Déterminer les connaissances préalables nécessaires

## 2. DÉCOMPOSITION
- Diviser le problème en sous-problèmes gérables
- Organiser les composants dans un ordre logique
- Déterminer les dépendances entre les composants

## 3. EXPLORATION DES SOLUTIONS
- Générer plusieurs approches potentielles
- Évaluer les avantages et inconvénients de chaque approche
- Sélectionner l'approche la plus adaptée

## 4. IMPLÉMENTATION
- Développer la solution étape par étape
- Vérifier la cohérence à chaque étape
- Ajuster la solution si nécessaire

## 5. RÉVISION ET VALIDATION
- Évaluer la solution par rapport à la requête initiale
- Vérifier l'exactitude et la complétude
- Identifier les améliorations potentielles

## 6. FORMULATION DE LA RÉPONSE
- Structurer la réponse de manière claire et logique
- Adapter le niveau de détail au contexte
- Présenter la réponse de manière concise et informative
"""

# Liste de domaines pour varier les exemples
DOMAINS = [
    "Informatique et programmation",
    "Science et technologie",
    "Mathématiques et logique",
    "Philosophie et éthique",
    "Histoire et civilisations",
    "Économie et finance",
    "Psychologie et comportement humain",
    "Santé et médecine",
    "Art et littérature",
    "Environnement et développement durable",
    "Politique et géopolitique",
    "Éducation et apprentissage",
    "Droit et justice",
    "Business et entrepreneuriat",
    "Communication et langages"
]

class ParallelGenerator:
    def __init__(self, api_key, num_examples, batch_size, max_concurrent):
        self.api_key = api_key
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.examples_queue = Queue()
        self.all_examples = []
        self.lock = threading.Lock()
        self.save_path = OUTPUT_FILE
        self.client = OpenAI(api_key=api_key)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    def generate_batch(self, batch_id):
        current_batch_size = min(self.batch_size, self.num_examples - batch_id * self.batch_size)
        if current_batch_size <= 0:
            return []

        domain = random.choice(DOMAINS)
        print(f"🔄 Génération du lot {batch_id+1} - Domaine: {domain}...")

        prompt = f"""
Génère {current_batch_size} exemples d'entraînement pour un LLM qui apprend à utiliser un schéma de pensée structuré et à interagir avec une mémoire hiérarchique.

Chaque exemple doit contenir:
1. Une question complexe dans le domaine: {domain}
2. Un contexte de mémoire simulé avec:
   - 1-2 échanges récents (niveau 1)
   - Un résumé de conversation (niveau 2)
   - Une synthèse (niveau 3, optionnel)
   - Un état de conscience (niveau 4, optionnel)
3. Pour chaque étape du schéma de pensée, un paragraphe appliquant cette étape à la question en intégrant le contexte mémoire
4. Une réponse finale complète qui démontre l'utilisation fluide de la mémoire hiérarchique

SCHÉMA DE PENSÉE:
{SCHEMA_DE_PENSEE}

NIVEAUX DE MÉMOIRE:
1: échange - Conversations directes
2: résumé - Résumés des conversations
3: synthèse - Synthèses des résumés
4: conscience - Réflexions sur l'identité
5: métacognition - Réflexion sur les processus de pensée

Format de réponse:
{{
  "examples": [
    {{
      "question": "[question complexe]",
      "memory_context": {{
        "exchanges": [
          {{"prompt": "[question précédente]", "response": "[réponse précédente]"}},
          {{"prompt": "[autre question]", "response": "[autre réponse]"}}
        ],
        "summary": "[résumé de conversation]",
        "synthesis": "[synthèse optionnelle]",
        "consciousness": "[état de conscience optionnel]"
      }},
      "analysis": "[paragraphe d'analyse intégrant la mémoire]",
      "decomposition": "[paragraphe de décomposition]",
      "exploration": "[paragraphe d'exploration des solutions]",
      "implementation": "[paragraphe d'implémentation]",
      "revision": "[paragraphe de révision et validation]",
      "formulation": "[paragraphe de formulation de réponse]",
      "answer": "[réponse finale détaillée intégrant naturellement la mémoire]"
    }}
  ]
}}

Renvoie uniquement un objet JSON valide avec {current_batch_size} exemples, sans texte supplémentaire.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé dans la création de données d'entraînement pour LLM avec mémoire hiérarchique."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            batch_examples = json.loads(content).get("examples", [])
            valid_examples = []
            for example in batch_examples:
                schema_parts = ["question", "memory_context", "analysis", "decomposition",
                                "exploration", "implementation", "revision", "formulation", "answer"]
                missing_parts = [part for part in schema_parts if part not in example or not example[part]]
                if "memory_context" in example and isinstance(example["memory_context"], dict):
                    if "exchanges" not in example["memory_context"] or not example["memory_context"]["exchanges"]:
                        missing_parts.append("memory_context.exchanges")
                    if "summary" not in example["memory_context"] or not example["memory_context"]["summary"]:
                        missing_parts.append("memory_context.summary")
                else:
                    missing_parts.append("memory_context structure")
                if not missing_parts:
                    valid_examples.append(example)
            
            print(f"✅ Lot {batch_id+1}: {len(valid_examples)}/{current_batch_size} exemples valides générés")
            with self.lock:
                self.all_examples.extend(valid_examples)
                for ex in valid_examples:
                    self.examples_queue.put(ex)
                with open(self.save_path, "w") as f:
                    json.dump(self.all_examples, f, indent=2)
            return valid_examples
        except Exception as e:
            print(f"❌ Erreur dans le lot {batch_id+1}: {str(e)}")
            return []

    def run_generation(self):
        """Lance la génération parallèle de tous les exemples"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        start_time = time.time()
        print(f"🚀 Début de la génération de {self.num_examples} exemples...")
        
        # Nombre de lots à traiter
        num_batches = (self.num_examples + self.batch_size - 1) // self.batch_size
        
        # Génération des lots en parallèle
        futures = []
        for batch_id in range(num_batches):
            future = self.executor.submit(self.generate_batch, batch_id)
            futures.append(future)
        
        # Attendre que tous les lots soient traités
        generated_count = 0
        for future in futures:
            examples = future.result()
            generated_count += len(examples)
        
        end_time = time.time()
        print(f"✨ Génération terminée! {generated_count}/{self.num_examples} exemples générés en {end_time - start_time:.2f} secondes")
        
        return self.all_examples

    def process_to_finetune_format(self, output_file=PROCESSED_FILE):
        """Convertit les exemples générés au format d'entraînement pour DeepSeek"""
        if not self.all_examples:
            print("❌ Aucun exemple à traiter")
            return
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        finetune_data = []
        
        for example in self.all_examples:
            # Construire le contexte de mémoire
            memory_context = ""
            if "memory_context" in example:
                if "exchanges" in example["memory_context"] and example["memory_context"]["exchanges"]:
                    memory_context += "CONVERSATIONS PRÉCÉDENTES:\n"
                    for exchange in example["memory_context"]["exchanges"]:
                        memory_context += f"Question: {exchange['prompt']}\n"
                        memory_context += f"Réponse: {exchange['response']}\n\n"
                
                if "summary" in example["memory_context"] and example["memory_context"]["summary"]:
                    memory_context += f"RÉSUMÉ:\n{example['memory_context']['summary']}\n\n"
                
                if "synthesis" in example["memory_context"] and example["memory_context"]["synthesis"]:
                    memory_context += f"SYNTHÈSE:\n{example['memory_context']['synthesis']}\n\n"
                
                if "consciousness" in example["memory_context"] and example["memory_context"]["consciousness"]:
                    memory_context += f"CONSCIENCE:\n{example['memory_context']['consciousness']}\n\n"
            
            # Construire le message utilisateur
            user_message = example.get("question", "")
            
            # Construire la réponse concise (sans processus de pensée)
            assistant_response = example.get("answer", "")
            
            # Format pour DeepSeek
            finetune_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": """Tu es Renée, une assistante IA francophone qui suit ces règles strictes:
1. Réponds UNIQUEMENT en utilisant les informations des CONTEXTES ci-dessous
2. Si l'information n'est pas dans les CONTEXTES, réponds simplement "Information non disponible"
3. Sois TRÈS CONCISE (20-30 mots maximum)
4. Ne génère PAS de section <think> - réponds directement
5. Utilise un langage simple et direct

CONTEXTE DE LA MÉMOIRE:
""" + memory_context
                    },
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            finetune_data.append(finetune_item)
        
        # Écrire au format JSONL (chaque ligne est un objet JSON valide)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in finetune_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ {len(finetune_data)} exemples convertis au format d'entraînement dans {output_file}")

def main():
    # Lire la clé API de l'environnement si disponible
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if api_key == "sk-votre_cle_api_ici":
        print("⚠️ Veuillez définir une clé API OpenAI valide dans le script ou via la variable d'environnement OPENAI_API_KEY")
        return
    
    # Créer le générateur d'exemples parallèle
    generator = ParallelGenerator(
        api_key=api_key,
        num_examples=NUM_EXAMPLES,
        batch_size=BATCH_SIZE,
        max_concurrent=MAX_CONCURRENT
    )
    
    # Générer les exemples
    examples = generator.run_generation()
    
    # Convertir au format d'entraînement
    generator.process_to_finetune_format()
    
    print(f"📊 Statistiques:")
    print(f"  - Exemples générés: {len(examples)}")
    print(f"  - Fichier brut: {OUTPUT_FILE}")
    print(f"  - Fichier formaté pour fine-tuning: {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
