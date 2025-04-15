#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour générer un dataset de fine-tuning DeepSeek optimisé pour:
1. La recherche naturelle dans la mémoire hiérarchique
2. L'utilisation efficace du RAG
3. L'interaction avec les APIs HTTPS (ex: WordPress)
4. L'utilisation de Crawl4ai pour la recherche web en direct
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
import sys

# Configuration par défaut
DEFAULT_OUTPUT_FILE = "./data/enhanced_dataset.jsonl"
NUM_EXAMPLES = 500
BATCH_SIZE = 10
MAX_CONCURRENT = 5
DEFAULT_API_KEY = "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"

# Thèmes variés pour générer des exemples diversifiés
TOPICS = [
    "Informatique et programmation",
    "WordPress et gestion de contenu",
    "SEO et marketing digital",
    "E-commerce et plateformes web",
    "Développement web et APIs",
    "Automatisation et scripts",
    "Science des données et ML",
    "Actualités et tendances tech",
    "Histoire et civilisations",
    "Économie et finance",
    "Santé et médecine",
    "Art et littérature",
    "Psychologie et comportement",
    "Philosophie et éthique",
    "Environnement et développement durable"
]

# Types de sources de données
DATA_SOURCES = [
    "memory_only",         # Seulement mémoire hiérarchique
    "rag_only",            # Seulement RAG
    "memory_and_rag",      # Combinaison mémoire + RAG
    "wordpress_api",       # API WordPress
    "crawl4ai",            # Crawl4ai pour recherche web
    "mixed_sources",       # Combinaison de plusieurs sources
]

# Types de requêtes API (pour WordPress et autres APIs)
API_QUERY_TYPES = [
    {
        "type": "wordpress_posts",
        "endpoint": "/wp-json/wp/v2/posts",
        "description": "Récupération d'articles WordPress"
    },
    {
        "type": "wordpress_pages",
        "endpoint": "/wp-json/wp/v2/pages",
        "description": "Récupération de pages WordPress"
    },
    {
        "type": "wordpress_users",
        "endpoint": "/wp-json/wp/v2/users",
        "description": "Récupération d'utilisateurs WordPress"
    },
    {
        "type": "wordpress_categories",
        "endpoint": "/wp-json/wp/v2/categories",
        "description": "Récupération de catégories WordPress"
    },
    {
        "type": "wordpress_tags",
        "endpoint": "/wp-json/wp/v2/tags",
        "description": "Récupération de tags WordPress"
    },
    {
        "type": "wordpress_media",
        "endpoint": "/wp-json/wp/v2/media",
        "description": "Récupération de médias WordPress"
    },
    {
        "type": "wordpress_comments",
        "endpoint": "/wp-json/wp/v2/comments",
        "description": "Récupération de commentaires WordPress"
    },
    {
        "type": "crawl4ai_search",
        "endpoint": "/api/search",
        "description": "Recherche web via Crawl4ai"
    },
    {
        "type": "crawl4ai_extract",
        "endpoint": "/api/extract",
        "description": "Extraction de contenu via Crawl4ai"
    }
]

class EnhancedDatasetGenerator:
    def __init__(self, api_key, output_file, num_examples, batch_size, max_concurrent):
        """Initialise le générateur de dataset amélioré avec support APIs et Crawl4ai."""
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
        data_source = random.choice(DATA_SOURCES)
        
        print(f"🔄 Génération du lot {batch_id+1}: {current_batch_size} exemples - Thème: {topic} - Source: {data_source}")
        
        # Ajuster le prompt selon la source de données
        if data_source == "wordpress_api":
            api_query = random.choice(API_QUERY_TYPES[:7])  # Les 7 premiers sont des types WordPress
            return self._generate_wordpress_api_examples(batch_id, current_batch_size, topic, api_query)
        elif data_source == "crawl4ai":
            api_query = random.choice(API_QUERY_TYPES[7:])  # Les 2 derniers sont des types Crawl4ai
            return self._generate_crawl4ai_examples(batch_id, current_batch_size, topic, api_query)
        else:
            return self._generate_memory_rag_examples(batch_id, current_batch_size, topic, data_source)
    
    def _generate_memory_rag_examples(self, batch_id, current_batch_size, topic, data_source):
        """Génère des exemples basés sur la mémoire et le RAG."""
        prompt = f"""
Tu vas créer {current_batch_size} exemples d'entraînement pour fine-tuner un LLM (DeepSeek) à chercher intelligemment dans sa mémoire et son système RAG.

OBJECTIF: Créer des exemples qui entraîneront le LLM à:
1. Identifier automatiquement quand utiliser sa mémoire ou le RAG pour répondre
2. Extraire précisément l'information pertinente des contextes fournis
3. Produire des réponses CONCISES (20-30 mots) basées UNIQUEMENT sur l'information disponible
4. Répondre "Information non disponible" quand le contexte ne suffit pas

THÈME: {topic}
SOURCE DE DONNÉES: {data_source}

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
        return self._call_openai_api(prompt, batch_id, current_batch_size)
    
    def _generate_wordpress_api_examples(self, batch_id, current_batch_size, topic, api_query):
        """Génère des exemples spécifiques à l'API WordPress."""
        prompt = f"""
Tu vas créer {current_batch_size} exemples d'entraînement pour fine-tuner un LLM (DeepSeek) à utiliser efficacement l'API WordPress.

OBJECTIF: Créer des exemples qui entraîneront le LLM à:
1. Comprendre comment interroger une API WordPress REST pour différentes ressources
2. Extraire et traiter correctement les données de l'API WordPress
3. Produire des réponses CONCISES (20-30 mots) basées UNIQUEMENT sur l'information disponible
4. Répondre "Information non disponible" quand le contexte ne suffit pas

THÈME: {topic}
TYPE D'API WordPress: {api_query['type']} ({api_query['description']})
ENDPOINT: {api_query['endpoint']}

FORMAT POUR CHAQUE EXEMPLE:
```
{{
  "conversation_history": [
    {{
      "role": "user", 
      "content": "(question précédente optionnelle sur WordPress)"
    }},
    {{
      "role": "assistant", 
      "content": "(réponse précédente optionnelle)"
    }}
  ],
  "memory_context": "Information stockée dans la mémoire hiérarchique du LLM, relevant des interactions passées avec l'utilisateur concernant WordPress",
  "api_context": {{
    "endpoint": "{api_query['endpoint']}",
    "method": "GET",
    "response": {{
      // Exemple de réponse JSON de l'API WordPress simulée
    }}
  }},
  "current_question": "Question actuelle de l'utilisateur qui nécessite d'interroger l'API WordPress",
  "ideal_concise_answer": "Réponse concise et factuelle (20-30 mots) qui utilise UNIQUEMENT l'information des contextes"
}}
```

INSTRUCTIONS:
- Pour chaque exemple, crée un scénario cohérent où l'utilisateur veut des informations de son site WordPress
- Simule une réponse d'API WordPress réaliste pour l'endpoint {api_query['endpoint']}
- Inclus des cas typiques comme lister des articles, consulter des pages, gérer des médias, etc.
- L'ideal_concise_answer doit être ultra concise (20-30 mots) et factuelle
- IMPORTANT: La réponse doit se baser UNIQUEMENT sur l'information des contextes fournis

Renvoie uniquement un tableau JSON contenant {current_batch_size} exemples dans le format spécifié.
"""
        return self._call_openai_api(prompt, batch_id, current_batch_size)
    
    def _generate_crawl4ai_examples(self, batch_id, current_batch_size, topic, api_query):
        """Génère des exemples spécifiques à Crawl4ai pour la recherche web."""
        prompt = f"""
Tu vas créer {current_batch_size} exemples d'entraînement pour fine-tuner un LLM (DeepSeek) à utiliser efficacement Crawl4ai pour la recherche web en direct.

OBJECTIF: Créer des exemples qui entraîneront le LLM à:
1. Comprendre comment utiliser Crawl4ai pour rechercher des informations sur le web
2. Extraire et synthétiser correctement les informations trouvées sur le web
3. Produire des réponses CONCISES (20-30 mots) basées UNIQUEMENT sur l'information disponible
4. Répondre "Information non disponible" quand la recherche ne donne pas de résultats pertinents

THÈME: {topic}
TYPE DE REQUÊTE Crawl4ai: {api_query['type']} ({api_query['description']})
ENDPOINT: {api_query['endpoint']}

FORMAT POUR CHAQUE EXEMPLE:
```
{{
  "conversation_history": [
    {{
      "role": "user", 
      "content": "(question précédente optionnelle sur un sujet web)"
    }},
    {{
      "role": "assistant", 
      "content": "(réponse précédente optionnelle)"
    }}
  ],
  "memory_context": "Information stockée dans la mémoire hiérarchique du LLM, relevant des interactions passées avec l'utilisateur",
  "crawl4ai_context": {{
    "query": "Terme de recherche pour Crawl4ai",
    "results": [
      {{
        "url": "https://exemple.com/page1",
        "title": "Titre de la page",
        "snippet": "Extrait pertinent du contenu de la page"
      }},
      // Autres résultats simulés...
    ]
  }},
  "current_question": "Question actuelle de l'utilisateur qui nécessite une recherche web en direct",
  "ideal_concise_answer": "Réponse concise et factuelle (20-30 mots) qui utilise UNIQUEMENT l'information des résultats de recherche"
}}
```

INSTRUCTIONS:
- Pour chaque exemple, crée un scénario cohérent où l'utilisateur a besoin d'informations récentes du web
- Simule des résultats de recherche Crawl4ai réalistes et pertinents pour la question
- Inclus des cas variés comme rechercher des actualités, des informations techniques, des avis, etc.
- L'ideal_concise_answer doit être ultra concise (20-30 mots) et factuelle
- IMPORTANT: La réponse doit se baser UNIQUEMENT sur l'information des contextes fournis

Renvoie uniquement un tableau JSON contenant {current_batch_size} exemples dans le format spécifié.
"""
        return self._call_openai_api(prompt, batch_id, current_batch_size)
    
    def _call_openai_api(self, prompt, batch_id, current_batch_size):
        """Appelle l'API OpenAI pour générer les exemples."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # On utilise un modèle puissant pour générer des exemples de qualité
                messages=[
                    {"role": "system", "content": "Tu es un expert en création de datasets pour fine-tuning de modèles de langage. Tu génères des exemples de haute qualité pour l'entraînement."},
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
            
            # Validation et formatage des exemples
            valid_examples = []
            for example in examples:
                # Vérifier la validité et formater pour DeepSeek
                formatted_example = self._format_for_deepseek(example)
                if formatted_example:
                    valid_examples.append(formatted_example)
            
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
        # Vérifier que l'exemple contient au moins les champs obligatoires
        required_fields = ["current_question", "ideal_concise_answer"]
        if not all(field in example for field in required_fields):
            return None
            
        # Construire le prompt système avec les contextes
        system_prompt = """Tu es Renée, une assistante IA francophone qui suit ces règles strictes:
1. Réponds UNIQUEMENT en utilisant les informations des CONTEXTES ci-dessous
2. Si l'information n'est pas dans les CONTEXTES, réponds simplement "Information non disponible"
3. Sois TRÈS CONCISE (20-30 mots maximum)
4. Ne génère PAS de section <think> - réponds directement
5. Utilise un langage simple et direct

"""
        
        # Ajouter les contextes selon leur disponibilité
        context_parts = []
        
        # Mémoire hiérarchique
        if "memory_context" in example and example["memory_context"]:
            context_parts.append(f"CONTEXTE DE LA MÉMOIRE HIÉRARCHIQUE:\n{example['memory_context']}")
        
        # RAG
        if "rag_context" in example and example["rag_context"]:
            context_parts.append(f"CONTEXTE DU SYSTÈME DE RAG:\n{example['rag_context']}")
            
        # API WordPress
        if "api_context" in example and example["api_context"]:
            api_ctx = example["api_context"]
            api_json = json.dumps(api_ctx.get("response", {}), ensure_ascii=False, indent=2)
            context_parts.append(f"CONTEXTE DE L'API WORDPRESS ({api_ctx.get('endpoint', '')}):\n{api_json}")
            
        # Crawl4ai
        if "crawl4ai_context" in example and example["crawl4ai_context"]:
            crawl_ctx = example["crawl4ai_context"]
            results_text = ""
            for i, result in enumerate(crawl_ctx.get("results", [])):
                results_text += f"Résultat {i+1}:\n"
                results_text += f"URL: {result.get('url', '')}\n"
                results_text += f"Titre: {result.get('title', '')}\n"
                results_text += f"Extrait: {result.get('snippet', '')}\n\n"
            context_parts.append(f"RÉSULTATS DE RECHERCHE WEB POUR: '{crawl_ctx.get('query', '')}'\n{results_text}")
        
        # Ajouter les contextes au prompt système
        if context_parts:
            system_prompt += "\n" + "\n\n".join(context_parts) + "\n"
        else:
            system_prompt += "\nAucun contexte disponible.\n"
            
        system_prompt += "\nRAPPEL: Réponds UNIQUEMENT avec les informations présentes dans les CONTEXTES ci-dessus."
        
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
        
        # Si nous n'avons pas assez d'exemples, générer des lots supplémentaires (avec un maximum de tentatives)
        additional_batch_id = num_batches
        max_additional_attempts = 20  # Limite de sécurité pour éviter la boucle infinie
        attempts = 0
        
        while total_valid_examples < self.num_examples and attempts < max_additional_attempts:
            remaining = self.num_examples - total_valid_examples
            print(f"⚠️ Il manque {remaining} exemples, génération de lots supplémentaires... (tentative {attempts+1}/{max_additional_attempts})")
            
            # Utiliser une taille de lot adaptée pour les lots restants
            current_batch_size = min(self.batch_size, remaining)
            
            # Ajouter un délai pour éviter de surcharger l'API
            time.sleep(1)
            
            future = self.executor.submit(self.generate_batch, additional_batch_id)
            examples = future.result()
            total_valid_examples += len(examples)
            additional_batch_id += 1
            attempts += 1
            
            # Si on n'obtient aucun exemple valide après plusieurs tentatives, ajuster les paramètres
            if len(examples) == 0 and attempts >= 3:
                print("⚠️ Plusieurs tentatives sans exemples valides, ajustement des paramètres...")
                # Réduire la complexité pour augmenter les chances de succès
                current_batch_size = 1
                
            # Enregistrer les progrès après chaque tentative
            self._save_progress()
        
        # Message d'alerte si nous n'avons pas atteint le nombre d'exemples demandés
        if total_valid_examples < self.num_examples:
            print(f"⚠️ ATTENTION: Impossible de générer le nombre demandé d'exemples. {total_valid_examples}/{self.num_examples} ont été générés.")
        
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
    # Version compatibles avec Colab et Jupyter qui passent des arguments supplémentaires
    import sys
    
    # Filtrer les arguments qui commencent par '-f' (pour Jupyter/Colab)
    filtered_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-f"):
            skip_next = True
            continue
        filtered_args.append(arg)
    
    # Remplacer les arguments filtrés
    sys.argv = [sys.argv[0]] + filtered_args
    
    # Créer le parser avec les arguments supportés
    parser = argparse.ArgumentParser(description='Générateur de dataset de fine-tuning DeepSeek avec support mémoire, RAG, WordPress et Crawl4ai')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, help='Clé API OpenAI')
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
    generator = EnhancedDatasetGenerator(
        api_key=api_key,
        output_file=args.output,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        max_concurrent=args.concurrent
    )
    
    generator.generate_dataset()

if __name__ == "__main__":
    main()
