# Système de Mémoire Hiérarchique pour Renée

## Introduction

Ce document décrit l'implémentation du système de mémoire hiérarchique à 5 niveaux pour Renée, une intelligence artificielle conversationnelle. Ce système permet d'organiser, de condenser et de gérer efficacement les connaissances à différentes échelles temporelles.

## Architecture du Système

Le système de mémoire hiérarchique est composé de 5 niveaux interconnectés :

### Niveau 1 : Mémoire à Court Terme (ShortTermMemory)

- **Fonction** : Stockage des interactions récentes sous leur forme brute
- **Durée de rétention** : Minutes à heures
- **Structure** : Conversations complètes (entrées utilisateur et réponses système)
- **Capacité** : Limitée en nombre d'éléments
- **Caractéristiques** :
  - Utilisation d'embeddings pour la recherche sémantique
  - Index FAISS optimisé pour la recherche rapide
  - Gestion automatique de la capacité (suppression des éléments les plus anciens)

### Niveau 2 : Condensateur 5 Minutes (Level2Condenser)

- **Fonction** : Génération de résumés des conversations récentes
- **Cycle** : Toutes les 5 minutes
- **Technique** : Clustering de conversations similaires et génération de résumés
- **Caractéristiques** :
  - Utilisation de BERTopic pour le clustering
  - Génération de résumés par un LLM
  - Index sémantique pour rechercher des résumés pertinents

### Niveau 3 : Concepts Horaires (Level3HourlyConcepts)

- **Fonction** : Extraction de méta-concepts à partir des résumés
- **Cycle** : Toutes les heures
- **Technique** : Analyse des résumés pour identifier des tendances et des patterns
- **Caractéristiques** :
  - Clustering de résumés similaires
  - Création d'un graphe conceptuel (relations entre concepts)
  - Génération de méta-concepts via un LLM

### Niveau 4 : Connaissances Journalières (Level4DailyKnowledge)

- **Fonction** : Consolidation des méta-concepts en connaissances à long terme
- **Cycle** : Une fois par jour
- **Technique** : Dérivation de règles et génération de réflexions synthétiques
- **Caractéristiques** :
  - Extraction des concepts les plus importants
  - Dérivation de règles générales (méthode TRAN)
  - Mécanisme d'oubli intelligent pour éliminer les connaissances obsolètes

### Niveau 5 : Orchestrateur (Level5Orchestrator)

- **Fonction** : Assemblage intelligent des informations de tous les niveaux
- **Objectif** : Produire un contexte optimal pour le LLM
- **Technique** : Allocation de tokens entre les différents niveaux et composition contextuelle
- **Caractéristiques** :
  - Pondération adaptative entre les niveaux
  - Formatage structuré du contexte
  - Extraction contextuelle basée sur la requête

## Fonctionnement Détaillé

### Flux de Données

1. Les conversations avec l'utilisateur sont stockées dans la mémoire à court terme (Niveau 1)
2. Toutes les 5 minutes, le niveau 2 regroupe les conversations similaires et génère des résumés
3. Toutes les heures, le niveau 3 analyse les résumés pour extraire des méta-concepts
4. Chaque jour, le niveau 4 consolide les méta-concepts en connaissances à long terme
5. À chaque requête utilisateur, le niveau 5 compose un contexte optimal à partir de tous les niveaux

### Technologies Clés

- **FAISS** : Bibliothèque pour la recherche efficace de similarité
- **SentenceTransformers** : Génération d'embeddings pour la recherche sémantique
- **BERTopic** (optionnel) : Clustering et modélisation de sujets
- **Threading** : Exécution des tâches de condensation en arrière-plan

### Optimisations

- **Détection automatique du device** : Utilisation de GPU/MPS si disponible
- **Précision mixte** : Économie de mémoire sur les modèles d'embeddings
- **Traitement par lots** : Optimisation pour les grandes quantités de données
- **Gestion de la mémoire** : Mécanismes d'oubli intelligent à tous les niveaux

## Méthodes Principales par Niveau

### Niveau 1 : ShortTermMemory
- `add_conversation` : Ajoute une nouvelle conversation
- `get_similar_conversations` : Récupère les conversations similaires à une requête
- `get_recent_conversations` : Récupère les conversations les plus récentes

### Niveau 2 : Level2Condenser
- `condense_recent_memory` : Génère des résumés à partir des conversations récentes
- `get_relevant_summaries` : Récupère les résumés pertinents pour une requête

### Niveau 3 : Level3HourlyConcepts
- `condense_level2_to_level3` : Génère des méta-concepts à partir des résumés
- `cluster_summaries` : Regroupe les résumés par similarité sémantique
- `get_relevant_concepts` : Récupère les concepts pertinents pour une requête

### Niveau 4 : Level4DailyKnowledge
- `daily_consolidation` : Effectue la consolidation journalière
- `extract_key_concepts` : Identifie les concepts les plus importants
- `derive_rules_from_patterns` : Dérive des règles générales
- `apply_intelligent_forgetting` : Supprime les connaissances obsolètes

### Niveau 5 : Level5Orchestrator
- `compose_context` : Assemble un contexte optimal pour le LLM
- `get_relevant_knowledge` : Récupère les connaissances pertinentes

## Conclusion

Ce système de mémoire hiérarchique permet à Renée de condenser et de gérer efficacement ses connaissances au fil du temps. En utilisant une approche à plusieurs niveaux temporels, Renée peut maintenir un équilibre entre l'information récente et détaillée (niveaux 1-2) et les connaissances consolidées à plus long terme (niveaux 3-4), tout en optimisant la composition du contexte pour chaque requête (niveau 5).

Cette architecture s'inspire de la façon dont le cerveau humain gère les informations à différentes échelles temporelles, permettant une meilleure adaptabilité et une utilisation plus efficace des capacités de traitement.
