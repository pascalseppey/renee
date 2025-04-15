#!/usr/bin/env python3
# memory_fix.py
# Correctifs pour le système de mémoire hiérarchique

import os
import sys
import uuid
import numpy as np
import faiss
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import pickle
import math

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryFix")

# Répertoire de données pour les tests
TEST_DIR = Path("./memory_fixed")
TEST_DIR.mkdir(exist_ok=True)
MEMORY_DIR = TEST_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True, parents=True)

# Classe pour simuler le service LLM
class SimpleLLMService:
    """Service LLM simplifié pour les tests"""
    
    def generate(self, prompt):
        """Génère une réponse simulée"""
        if "résumé" in prompt.lower():
            return "Voici un résumé des discussions récentes sur l'intelligence artificielle et ses applications."
        elif "meta-concept" in prompt.lower() or "concept" in prompt.lower():
            return "Intelligence Artificielle: Domaine de l'informatique visant à créer des systèmes capables d'apprendre et de s'adapter."
        elif "règle" in prompt.lower():
            return "1. Fournir des explications adaptées au niveau technique de l'utilisateur.\n2. Présenter des exemples concrets pour illustrer les concepts abstraits."
        elif "réflexion" in prompt.lower():
            return "Les utilisateurs s'intéressent principalement aux aspects pratiques de l'IA plutôt qu'aux détails théoriques."
        else:
            return "Réponse générique du LLM simulé."

# Classes de base pour la mémoire (versions simplifiées)
@dataclass
class UserPrompt:
    id: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemResponse:
    id: str
    prompt_id: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    id: str
    prompt: UserPrompt
    response: SystemResponse
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Summary:
    id: str
    content: str
    created_at: datetime
    conversation_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetaConcept:
    id: str
    name: str
    description: str
    created_at: datetime
    summary_ids: List[str] = field(default_factory=list)
    source_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeItem:
    id: str
    name: str
    content: str
    created_at: datetime
    concept_ids: List[str] = field(default_factory=list)
    importance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# Classe ShortTermMemory corrigée et compatible
class FixedShortTermMemory:
    """
    Niveau 1: Mémoire à court terme qui conserve les interactions récentes
    Version corrigée et compatible
    """
    def __init__(self, data_dir: str, max_items: int = 15):
        """
        Initialise la mémoire à court terme.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            max_items: Nombre maximum d'éléments à conserver
        """
        self.data_dir = data_dir
        self.max_items = max_items
        self.conversations = {}  # id -> Conversation
        self.embeddings = {}  # id -> embedding
        self.conversation_ids = []  # Pour maintenir l'ordre
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialiser l'index FAISS
        self.init_faiss_index()
        
        # Charger les données existantes
        self.load_data()
        
        logger.info(f"Mémoire à court terme initialisée: {len(self.conversations)} conversations")
    
    def init_faiss_index(self):
        """Initialise l'index FAISS pour la recherche par similarité."""
        # Dimension des embeddings (1536 pour les embeddings OpenAI)
        embedding_dim = 1536
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"Index FAISS initialisé (dim={embedding_dim})")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Génère un embedding pour un texte en utilisant OpenAI.
        Compatible avec openai>=1.0.0
        """
        try:
            import openai
            from openai import OpenAI
            # Récupère la clé API de l'environnement
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                logger.warning("Clé API OpenAI non disponible. Utilisation de vecteurs aléatoires.")
                return np.random.random(1536).astype(np.float32)
            
            # Création du client OpenAI (nouvelle interface 1.0+)
            client = OpenAI(api_key=api_key)
            
            # Génération de l'embedding avec la nouvelle interface OpenAI
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"  # Modèle d'embedding d'OpenAI
            )
            
            # Extraction et conversion du vecteur en numpy array (nouvelle structure)
            embedding = np.array(response.data[0].embedding).astype(np.float32)
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'embedding avec OpenAI: {e}")
            # Fallback à la génération aléatoire en cas d'erreur
            return np.random.random(1536).astype(np.float32)
    
    def add_conversation(self, user_input: str, system_response: str,
                         user_metadata: Dict[str, Any] = None,
                         response_metadata: Dict[str, Any] = None):
        """
        Ajoute une nouvelle conversation à la mémoire.
        
        Args:
            user_input: Texte de l'entrée utilisateur
            system_response: Texte de la réponse système
            user_metadata: Métadonnées utilisateur (optionnel)
            response_metadata: Métadonnées de réponse (optionnel)
        """
        # Création de l'ID pour la conversation
        conversation_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Création des objets pour la conversation
        prompt = UserPrompt(
            id=str(uuid.uuid4()),
            content=user_input,
            created_at=current_time,
            metadata=user_metadata or {}
        )
        
        response = SystemResponse(
            id=str(uuid.uuid4()),
            prompt_id=prompt.id,
            content=system_response,
            created_at=current_time,
            metadata=response_metadata or {}
        )
        
        conversation = Conversation(
            id=conversation_id,
            prompt=prompt,
            response=response,
            created_at=current_time
        )
        
        # Ajout à la mémoire
        self.conversations[conversation_id] = conversation
        self.conversation_ids.append(conversation_id)
        
        # Génération et stockage de l'embedding
        text_to_embed = f"{user_input} {system_response}"
        embedding = self.generate_embedding(text_to_embed)
        self.embeddings[conversation_id] = embedding
        
        # Ajout à l'index FAISS
        self.faiss_index.add(np.array([embedding]))
        
        # Nettoyage si on dépasse la limite
        self._clean_old_conversations()
        
        # Sauvegarde des données
        self.save_data()
        
        logger.info(f"Conversation ajoutée: {conversation_id}")
        return conversation_id

    def _clean_old_conversations(self):
        """Nettoie les conversations anciennes si la limite est dépassée."""
        if len(self.conversations) <= self.max_items:
            return
        
        # Tri par date (plus ancien en premier)
        sorted_ids = sorted(
            self.conversation_ids,
            key=lambda x: self.conversations[x].created_at if x in self.conversations else datetime.max
        )
        
        # Nombre d'éléments à supprimer
        to_remove = len(sorted_ids) - self.max_items
        
        if to_remove <= 0:
            return
        
        # IDs à supprimer
        remove_ids = sorted_ids[:to_remove]
        
        # Suppression des conversations et embeddings
        for conv_id in remove_ids:
            if conv_id in self.conversations:
                del self.conversations[conv_id]
            if conv_id in self.embeddings:
                del self.embeddings[conv_id]
            if conv_id in self.conversation_ids:
                self.conversation_ids.remove(conv_id)
        
        # Reconstruction de l'index FAISS
        embeddings_to_keep = [self.embeddings[conv_id] for conv_id in self.conversation_ids if conv_id in self.embeddings]
        self._rebuild_faiss_index(embeddings_to_keep)
        
        logger.info(f"Nettoyage effectué: {to_remove} conversations supprimées")
    
    def _rebuild_faiss_index(self, embeddings_to_keep):
        """Reconstruit l'index FAISS avec les embeddings à conserver."""
        if not embeddings_to_keep:
            self.init_faiss_index()
            return
        
        # Réinitialisation de l'index
        embedding_dim = embeddings_to_keep[0].shape[0]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        # Ajout des embeddings
        if embeddings_to_keep:
            self.faiss_index.add(np.array(embeddings_to_keep))
    
    def get_relevant_conversations(self, query: str, k: int = 5):
        """
        Récupère les conversations les plus pertinentes pour une requête.
        
        Args:
            query: Requête utilisateur
            k: Nombre de conversations à récupérer
            
        Returns:
            Liste des conversations pertinentes
        """
        if not self.conversations:
            return []
        
        # Génération de l'embedding pour la requête
        query_embedding = self.generate_embedding(query)
        
        # Limiter k au nombre de conversations disponibles
        k = min(k, len(self.conversations))
        
        if k == 0:
            return []
        
        # Recherche dans l'index FAISS
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Récupération des conversations
        relevant_conversations = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.conversation_ids):
                continue
            
            # Récupération de la conversation par son ID
            conv_id = self.conversation_ids[idx]
            if conv_id in self.conversations:
                relevant_conversations.append(self.conversations[conv_id])
        
        return relevant_conversations
    
    def get_recent_conversations(self, limit: int = 10):
        """
        Récupère les conversations les plus récentes.
        
        Args:
            limit: Nombre maximum de conversations à récupérer
            
        Returns:
            Liste des conversations récentes
        """
        # Tri par date (plus récent en premier)
        sorted_ids = sorted(
            self.conversation_ids,
            key=lambda x: self.conversations[x].created_at if x in self.conversations else datetime.min,
            reverse=True
        )
        
        # Limitation au nombre demandé
        limited_ids = sorted_ids[:limit]
        
        # Récupération des conversations
        recent = [self.conversations[conv_id] for conv_id in limited_ids if conv_id in self.conversations]
        
        return recent
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "short_term_memory.pkl")
        
        if not os.path.exists(pickle_path):
            logger.info("Aucune donnée de mémoire à court terme à charger")
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Chargement des conversations
            loaded_conversations = {}
            for k, v in data.get("conversations", {}).items():
                conversation = Conversation(
                    id=v["id"],
                    prompt=UserPrompt(
                        id=v["prompt"]["id"],
                        content=v["prompt"]["content"],
                        created_at=v["prompt"]["created_at"],
                        metadata=v["prompt"].get("metadata", {})
                    ),
                    response=SystemResponse(
                        id=v["response"]["id"],
                        prompt_id=v["response"]["prompt_id"],
                        content=v["response"]["content"],
                        created_at=v["response"]["created_at"],
                        metadata=v["response"].get("metadata", {})
                    ),
                    created_at=v["created_at"],
                    metadata=v.get("metadata", {})
                )
                loaded_conversations[k] = conversation
            
            # Mise à jour des données
            self.conversations = loaded_conversations
            self.conversation_ids = data.get("conversation_ids", [])
            self.embeddings = {k: np.array(v) for k, v in data.get("embeddings", {}).items()}
            
            # Reconstruction de l'index FAISS
            embeddings_list = list(self.embeddings.values())
            if embeddings_list:
                self.init_faiss_index()
                self.faiss_index.add(np.array(embeddings_list))
            
            logger.info(f"Données de mémoire à court terme chargées: {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de mémoire à court terme: {e}")
            # Initialisation des structures vides
            self.conversations = {}
            self.conversation_ids = []
            self.embeddings = {}
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        try:
            # Préparation des données
            conversations_data = {}
            for k, conv in self.conversations.items():
                conv_data = {
                    "id": conv.id,
                    "prompt": {
                        "id": conv.prompt.id,
                        "content": conv.prompt.content,
                        "created_at": conv.prompt.created_at,
                        "metadata": conv.prompt.metadata
                    },
                    "response": {
                        "id": conv.response.id,
                        "prompt_id": conv.response.prompt_id,
                        "content": conv.response.content,
                        "created_at": conv.response.created_at,
                        "metadata": conv.response.metadata
                    },
                    "created_at": conv.created_at,
                    "metadata": conv.metadata
                }
                conversations_data[k] = conv_data
            
            data = {
                "conversations": conversations_data,
                "conversation_ids": self.conversation_ids,
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items() if v is not None}
            }
            
            # Sauvegarde dans un fichier pickle
            with open(os.path.join(self.data_dir, "short_term_memory.pkl"), "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Données de mémoire à court terme sauvegardées: {len(self.conversations)} conversations")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données de mémoire à court terme: {e}")

# Classe de niveau 2 corrigée
class FixedLevel2Condenser:
    """
    Niveau 2: Condensateur qui condense les conversations récentes en résumés
    Version compatible avec le système corrigé
    """
    def __init__(self, data_dir: str, short_term_memory, 
                 time_window_minutes: int = 5,
                 min_conversations_for_summary: int = 3,
                 min_cluster_size: int = 2,
                 llm_service=None):
        """
        Initialise le condensateur de niveau 2.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            short_term_memory: Instance de la mémoire à court terme
            time_window_minutes: Fenêtre de temps pour la condensation
            min_conversations_for_summary: Nombre minimum de conversations pour générer un résumé
            min_cluster_size: Taille minimum d'un cluster pour générer un résumé
            llm_service: Service LLM à utiliser pour la génération de résumés
        """
        self.data_dir = data_dir
        self.short_term_memory = short_term_memory
        self.time_window_minutes = time_window_minutes
        self.min_conversations_for_summary = min_conversations_for_summary
        self.min_cluster_size = min_cluster_size
        self.llm_service = llm_service
        
        # Structures de données
        self.summaries = {}  # id -> Summary
        self.summary_embeddings = {}  # id -> embedding
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialiser l'index FAISS
        self.init_faiss_index()
        
        # Charger les données existantes
        self.load_data()
        
        # Démarrer le processus automatique de condensation
        self.start_auto_condense()
        
        logger.info(f"Condensateur de niveau 2 initialisé: {len(self.summaries)} résumés")
    
    def init_faiss_index(self):
        """Initialise l'index FAISS pour la recherche par similarité."""
        # Dimension des embeddings (utilise la même que pour la mémoire à court terme)
        embedding_dim = 1536
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        logger.info(f"Index FAISS initialisé (dim={embedding_dim})")
    
    def start_auto_condense(self):
        """
        Démarre une tâche en arrière-plan pour condenser la mémoire toutes les 5 minutes.
        Utilise threading au lieu d'asyncio pour éviter les problèmes d'event loop.
        """
        def auto_condense_task():
            while True:
                try:
                    # Attente de 5 minutes
                    time.sleep(300)  # 5 minutes
                    
                    # Condensation de la mémoire récente
                    summaries = self.condense_recent_memory()
                    
                    logger.info(f"Condensation automatique effectuée: {len(summaries)} résumés générés")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la condensation automatique: {e}")
        
        # Démarrage du thread
        condense_thread = threading.Thread(target=auto_condense_task, daemon=True)
        condense_thread.start()
        
        logger.info("Tâche de condensation automatique démarrée")
    
    def condense_recent_memory(self):
        """
        Condense la mémoire à court terme en résumés.
        
        Returns:
            Liste des nouveaux résumés générés
        """
        # Récupération des conversations récentes
        recent_conversations = self.get_recent_conversations()
        
        if len(recent_conversations) < self.min_conversations_for_summary:
            logger.info(f"Pas assez de conversations récentes pour générer un résumé ({len(recent_conversations)} < {self.min_conversations_for_summary})")
            return []
        
        # Clustering des conversations
        conversation_clusters = self.cluster_conversations(recent_conversations)
        
        # Génération des résumés
        new_summaries = []
        
        for cluster_idx, cluster_convos in conversation_clusters.items():
            if len(cluster_convos) < self.min_cluster_size:
                continue
            
            # Génération du résumé pour ce cluster
            summary_content = self.generate_summary_for_cluster(cluster_convos)
            
            # Création d'un objet Summary
            summary_id = str(uuid.uuid4())
            summary = Summary(
                id=summary_id,
                content=summary_content,
                created_at=datetime.now(),
                conversation_ids=[conv.id for conv in cluster_convos]
            )
            
            # Stockage du résumé
            self.summaries[summary_id] = summary
            
            # Génération et stockage de l'embedding
            summary_embedding = self.short_term_memory.generate_embedding(summary_content)
            self.summary_embeddings[summary_id] = summary_embedding
            
            # Mise à jour de l'index FAISS
            if summary_embedding is not None:
                self.faiss_index.add(np.array([summary_embedding]))
            
            # Ajout à la liste des nouveaux résumés
            new_summaries.append(summary)
        
        # Sauvegarde des données
        self.save_data()
        
        return new_summaries

    def get_recent_conversations(self):
        """
        Récupère les conversations récentes pour la condensation.
        
        Returns:
            Liste des conversations récentes dans la fenêtre de temps spécifiée
        """
        # Récupération de toutes les conversations récentes
        all_recent_conversations = self.short_term_memory.get_recent_conversations(limit=100)
        
        # Filtrage par fenêtre de temps
        time_threshold = datetime.now() - timedelta(minutes=self.time_window_minutes)
        recent_conversations = [
            conv for conv in all_recent_conversations
            if conv.created_at >= time_threshold
        ]
        
        return recent_conversations
    
    def cluster_conversations(self, conversations):
        """
        Regroupe les conversations par similarité.
        Version simplifiée: utilisation d'un clustering basique.
        
        Args:
            conversations: Liste des conversations à regrouper
            
        Returns:
            Dictionnaire de clusters {cluster_id -> liste de conversations}
        """
        if not conversations:
            return {}
        
        # Version simplifiée: clustering naïf basé sur les embeddings
        # Dans une implémentation réelle, on utiliserait un algorithme de clustering plus sophistiqué
        
        # Collection des embeddings
        embeddings = []
        for conv in conversations:
            if conv.id in self.short_term_memory.embeddings:
                embeddings.append(self.short_term_memory.embeddings[conv.id])
        
        if not embeddings:
            return {}
        
        # Clustering simple: division en 2 groupes (pourrait être amélioré)
        n_clusters = min(2, len(conversations))
        clusters = {}
        
        if n_clusters == 1:
            # Un seul cluster
            clusters[0] = conversations
        else:
            # Division naïve: moitié-moitié
            mid = len(conversations) // 2
            clusters[0] = conversations[:mid]
            clusters[1] = conversations[mid:]
        
        return clusters
    
    def generate_summary_for_cluster(self, conversations):
        """
        Génère un résumé pour un cluster de conversations.
        
        Args:
            conversations: Liste des conversations du cluster
            
        Returns:
            Texte du résumé
        """
        # Préparation des conversations pour le prompt
        conversation_texts = []
        for conv in conversations:
            conversation_texts.append(f"User: {conv.prompt.content}\nRenée: {conv.response.content}")
        
        conversations_joined = "\n\n".join(conversation_texts)
        
        # Utilisation du service LLM si disponible
        if self.llm_service:
            prompt = f"""
            Voici un ensemble de conversations similaires:
            
            {conversations_joined}
            
            Générez un résumé concis qui capture les points essentiels de ces conversations.
            """
            
            summary = self.llm_service.generate(prompt)
        else:
            # Génération d'un résumé basique sans LLM
            topics = set()
            for conv in conversations:
                # Extraction naïve de sujets (à améliorer dans une implémentation réelle)
                words = conv.prompt.content.lower().split()
                potential_topics = [word for word in words if len(word) > 5]
                topics.update(potential_topics[:2])  # Ajout de quelques mots longs comme sujets potentiels
            
            topics_str = ", ".join(list(topics)[:3])
            summary = f"Résumé des conversations portant sur {topics_str if topics else 'divers sujets'}."
        
        return summary
    
    def get_relevant_summaries(self, query: str, k: int = 3):
        """
        Récupère les résumés les plus pertinents pour une requête.
        
        Args:
            query: Requête utilisateur
            k: Nombre de résumés à récupérer
            
        Returns:
            Liste des résumés pertinents
        """
        if not self.summaries:
            return []
        
        # Génération de l'embedding pour la requête
        query_embedding = self.short_term_memory.generate_embedding(query)
        
        # Limiter k au nombre de résumés disponibles
        k = min(k, len(self.summaries))
        
        if k == 0:
            return []
        
        # Recherche dans l'index FAISS
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Récupération des résumés
        relevant_summaries = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.summaries):
                continue
            
            # Récupération du résumé par son ID
            summary_ids = list(self.summaries.keys())
            if idx < len(summary_ids):
                summary_id = summary_ids[idx]
                if summary_id in self.summaries:
                    relevant_summaries.append(self.summaries[summary_id])
        
        return relevant_summaries
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        try:
            # Préparation des données
            data = {
                "summaries": {k: asdict(v) for k, v in self.summaries.items()},
                "summary_embeddings": {k: v.tolist() for k, v in self.summary_embeddings.items() if v is not None}
            }
            
            # Sauvegarde dans un fichier pickle
            with open(os.path.join(self.data_dir, "level2_summaries.pkl"), "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Données du niveau 2 sauvegardées: {len(self.summaries)} résumés")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données du niveau 2: {e}")
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "level2_summaries.pkl")
        
        if not os.path.exists(pickle_path):
            logger.info("Aucune donnée de niveau 2 à charger")
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Chargement des résumés
            loaded_summaries = {}
            for k, v in data.get("summaries", {}).items():
                summary = Summary(
                    id=v["id"],
                    content=v["content"],
                    created_at=v["created_at"],
                    conversation_ids=v["conversation_ids"],
                    metadata=v.get("metadata", {})
                )
                loaded_summaries[k] = summary
            
            # Mise à jour des données
            self.summaries = loaded_summaries
            self.summary_embeddings = {k: np.array(v) for k, v in data.get("summary_embeddings", {}).items()}
            
            # Reconstruction de l'index FAISS
            embeddings_list = list(self.summary_embeddings.values())
            if embeddings_list:
                self.init_faiss_index()
                self.faiss_index.add(np.array(embeddings_list))
            
            logger.info(f"Données du niveau 2 chargées: {len(self.summaries)} résumés")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données du niveau 2: {e}")
            # Initialisation des structures vides
            self.summaries = {}
            self.summary_embeddings = {}

# Classe de niveau 3 corrigée
class FixedLevel3HourlyConcepts:
    """
    Niveau 3: Concepts horaires qui condensent les résumés en méta-concepts
    Version compatible avec le système corrigé
    """
    def __init__(self, data_dir: str, level2_condenser,
                llm_service=None):
        """
        Initialise le générateur de méta-concepts de niveau 3.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            level2_condenser: Instance du condensateur de niveau 2
            llm_service: Service LLM à utiliser pour la génération de méta-concepts
        """
        self.data_dir = data_dir
        self.level2_condenser = level2_condenser
        self.llm_service = llm_service
        
        # Structures de données
        self.meta_concepts = {}  # id -> MetaConcept
        self.concept_embeddings = {}  # id -> embedding
        self.concept_graph = {}  # id -> [id1, id2, ...] (relations entre concepts)
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialiser l'index FAISS
        self.init_faiss_index()
        
        # Charger les données existantes
        self.load_data()
        
        # Démarrer le processus automatique de condensation
        self.start_auto_process()
        
        logger.info(f"Générateur de méta-concepts de niveau 3 initialisé: {len(self.meta_concepts)} concepts")
    
    def init_faiss_index(self):
        """Initialise l'index FAISS pour la recherche par similarité."""
        # Dimension des embeddings (utilise la même que pour la mémoire à court terme)
        embedding_dim = 1536
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        logger.info(f"Index FAISS initialisé (dim={embedding_dim})")
    
    def start_auto_process(self):
        """
        Démarre une tâche en arrière-plan pour condenser les résumés en méta-concepts toutes les heures.
        Utilise threading au lieu d'asyncio pour éviter les problèmes d'event loop.
        """
        def auto_condense_task():
            while True:
                try:
                    # Attente d'une heure
                    time.sleep(3600)  # 1 heure
                    
                    # Condensation des résumés en méta-concepts
                    concepts = self.condense_level2_to_level3()
                    
                    logger.info(f"Condensation horaire automatique effectuée: {len(concepts)} méta-concepts générés")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la condensation horaire automatique: {e}")
        
        # Démarrage du thread
        condense_thread = threading.Thread(target=auto_condense_task, daemon=True)
        condense_thread.start()
        
        logger.info("Tâche de condensation horaire automatique démarrée")
    
    def condense_level2_to_level3(self):
        """
        Condense les résumés de niveau 2 en méta-concepts de niveau 3.
        
        Returns:
            Liste des nouveaux méta-concepts générés
        """
        # Récupération des résumés récents
        recent_summaries = self.get_recent_summaries()
        
        if not recent_summaries:
            logger.info("Aucun résumé récent à condenser")
            return []
        
        # Clustering des résumés
        summary_clusters = self.cluster_summaries(recent_summaries)
        
        # Génération des méta-concepts
        new_concepts = []
        
        for cluster_idx, cluster_summaries in summary_clusters.items():
            if len(cluster_summaries) < 2:
                continue
            
            # Génération du méta-concept pour ce cluster
            concept_name, concept_description = self.generate_meta_concept(cluster_summaries)
            
            # Création d'un objet MetaConcept
            concept_id = str(uuid.uuid4())
            meta_concept = MetaConcept(
                id=concept_id,
                name=concept_name,
                description=concept_description,
                created_at=datetime.now(),
                summary_ids=[summary.id for summary in cluster_summaries]
            )
            
            # Stockage du méta-concept
            self.meta_concepts[concept_id] = meta_concept
            
            # Génération et stockage de l'embedding
            concept_text = f"{concept_name}: {concept_description}"
            concept_embedding = self.level2_condenser.short_term_memory.generate_embedding(concept_text)
            self.concept_embeddings[concept_id] = concept_embedding
            
            # Mise à jour de l'index FAISS
            if concept_embedding is not None:
                self.faiss_index.add(np.array([concept_embedding]))
            
            # Mise à jour du graphe conceptuel
            self.update_concept_graph(meta_concept)
            
            # Ajout à la liste des nouveaux méta-concepts
            new_concepts.append(meta_concept)
        
        # Sauvegarde des données
        self.save_data()
        
        return new_concepts

    def get_recent_summaries(self):
        """
        Récupère les résumés récents pour la condensation.
        
        Returns:
            Liste des résumés récents (dernière heure)
        """
        # Récupération de tous les résumés
        all_summaries = list(self.level2_condenser.summaries.values())
        
        # Filtrage par fenêtre de temps (1 heure)
        time_threshold = datetime.now() - timedelta(hours=1)
        recent_summaries = [
            summary for summary in all_summaries
            if summary.created_at >= time_threshold
        ]
        
        return recent_summaries
    
    def cluster_summaries(self, summaries):
        """
        Regroupe les résumés par similarité.
        
        Args:
            summaries: Liste des résumés à regrouper
            
        Returns:
            Dictionnaire de clusters {cluster_id -> liste de résumés}
        """
        if not summaries:
            return {}
        
        # Version simplifiée: clustering naïf basé sur les embeddings
        # Dans une implémentation réelle, on utiliserait un algorithme de clustering plus sophistiqué
        
        # Collection des embeddings
        embeddings = []
        summaries_with_embeddings = []
        
        for summary in summaries:
            if summary.id in self.level2_condenser.summary_embeddings:
                embeddings.append(self.level2_condenser.summary_embeddings[summary.id])
                summaries_with_embeddings.append(summary)
        
        if not embeddings:
            return {}
        
        # Clustering simple: division en 2 groupes (pourrait être amélioré)
        n_clusters = min(2, len(summaries_with_embeddings))
        clusters = {}
        
        if n_clusters == 1:
            # Un seul cluster
            clusters[0] = summaries_with_embeddings
        else:
            # Division naïve: moitié-moitié
            mid = len(summaries_with_embeddings) // 2
            clusters[0] = summaries_with_embeddings[:mid]
            clusters[1] = summaries_with_embeddings[mid:]
        
        return clusters
    
    def generate_meta_concept(self, summaries):
        """
        Génère un méta-concept à partir d'un cluster de résumés.
        
        Args:
            summaries: Liste des résumés du cluster
            
        Returns:
            Tuple (nom du concept, description du concept)
        """
        # Préparation des résumés pour le prompt
        summary_texts = [summary.content for summary in summaries]
        summaries_joined = "\n\n".join(summary_texts)
        
        # Utilisation du service LLM si disponible
        if self.llm_service:
            prompt = f"""
            Voici un ensemble de résumés qui traitent de sujets similaires:
            
            {summaries_joined}
            
            Générez un méta-concept qui capture l'essence de ces résumés.
            Format de réponse: "[Nom du concept]: [Description détaillée du concept]"
            
            Le nom du concept devrait être court (2-5 mots) et la description devrait détailler
            ce que le concept englobe.
            """
            
            response = self.llm_service.generate(prompt)
            
            # Extraction du nom et de la description
            try:
                parts = response.split(":", 1)
                if len(parts) == 2:
                    concept_name = parts[0].strip()
                    concept_description = parts[1].strip()
                else:
                    concept_name = "Concept Généré"
                    concept_description = response.strip()
            except:
                concept_name = "Concept Généré"
                concept_description = response.strip()
        else:
            # Génération d'un méta-concept basique sans LLM
            # Extraction de mots-clés communs
            common_words = set()
            for summary in summaries:
                words = summary.content.lower().split()
                # Filtrer les mots courts ou communs
                keywords = [word for word in words if len(word) > 5]
                if not common_words:
                    common_words = set(keywords)
                else:
                    common_words = common_words.intersection(set(keywords))
            
            if common_words:
                concept_name = " ".join(list(common_words)[:2]).title()
            else:
                # Prendre quelques mots du premier résumé
                words = summaries[0].content.split()
                concept_name = " ".join(words[:2]).title()
            
            concept_description = f"Ce concept englobe les discussions sur {concept_name.lower()} qui apparaissent dans plusieurs résumés de conversations."
        
        return concept_name, concept_description
    
    def update_concept_graph(self, new_concept):
        """
        Met à jour le graphe conceptuel avec un nouveau méta-concept.
        
        Args:
            new_concept: Nouveau méta-concept à ajouter au graphe
        """
        # Initialisation du nœud dans le graphe
        self.concept_graph[new_concept.id] = []
        
        # Recherche de concepts similaires pour établir des connexions
        if len(self.meta_concepts) > 1:
            # Génération de l'embedding pour le nouveau concept
            concept_text = f"{new_concept.name}: {new_concept.description}"
            query_embedding = self.level2_condenser.short_term_memory.generate_embedding(concept_text)
            
            # Recherche des concepts similaires (exclure le nouveau concept)
            k = min(3, len(self.meta_concepts) - 1)
            if k > 0 and self.faiss_index.ntotal > 1:
                distances, indices = self.faiss_index.search(np.array([query_embedding]), k + 1)
                
                # Création des connexions
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.meta_concepts) and idx != self.faiss_index.ntotal - 1:
                        # Récupération du concept par son ID
                        concept_ids = list(self.meta_concepts.keys())
                        if idx < len(concept_ids):
                            related_id = concept_ids[idx]
                            if related_id != new_concept.id:
                                # Ajout de la connexion bidirectionnelle
                                self.concept_graph[new_concept.id].append(related_id)
                                if related_id in self.concept_graph:
                                    self.concept_graph[related_id].append(new_concept.id)
                                else:
                                    self.concept_graph[related_id] = [new_concept.id]
    
    def get_relevant_concepts(self, query: str, k: int = 3):
        """
        Récupère les méta-concepts les plus pertinents pour une requête.
        
        Args:
            query: Requête utilisateur
            k: Nombre de méta-concepts à récupérer
            
        Returns:
            Liste des méta-concepts pertinents
        """
        if not self.meta_concepts:
            return []
        
        # Génération de l'embedding pour la requête
        query_embedding = self.level2_condenser.short_term_memory.generate_embedding(query)
        
        # Limiter k au nombre de méta-concepts disponibles
        k = min(k, len(self.meta_concepts))
        
        if k == 0:
            return []
        
        # Recherche dans l'index FAISS
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Récupération des méta-concepts
        relevant_concepts = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.meta_concepts):
                continue
            
            # Récupération du méta-concept par son ID
            concept_ids = list(self.meta_concepts.keys())
            if idx < len(concept_ids):
                concept_id = concept_ids[idx]
                if concept_id in self.meta_concepts:
                    relevant_concepts.append(self.meta_concepts[concept_id])
        
        return relevant_concepts
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        try:
            # Préparation des données
            data = {
                "meta_concepts": {k: asdict(v) for k, v in self.meta_concepts.items()},
                "concept_embeddings": {k: v.tolist() for k, v in self.concept_embeddings.items() if v is not None},
                "concept_graph": self.concept_graph
            }
            
            # Sauvegarde dans un fichier pickle
            with open(os.path.join(self.data_dir, "level3_meta_concepts.pkl"), "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Données du niveau 3 sauvegardées: {len(self.meta_concepts)} concepts")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données du niveau 3: {e}")
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "level3_meta_concepts.pkl")
        
        if not os.path.exists(pickle_path):
            logger.info("Aucune donnée de niveau 3 à charger")
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Chargement des méta-concepts
            loaded_meta_concepts = {}
            for k, v in data.get("meta_concepts", {}).items():
                meta_concept = MetaConcept(
                    id=v["id"],
                    name=v["name"],
                    description=v["description"],
                    created_at=v["created_at"],
                    summary_ids=v["summary_ids"],
                    source_concepts=v["source_concepts"],
                    metadata=v.get("metadata", {})
                )
                loaded_meta_concepts[k] = meta_concept
            
            # Mise à jour des données
            self.meta_concepts = loaded_meta_concepts
            self.concept_embeddings = {k: np.array(v) for k, v in data.get("concept_embeddings", {}).items()}
            self.concept_graph = data.get("concept_graph", {})
            
            # Reconstruction de l'index FAISS
            embeddings_list = list(self.concept_embeddings.values())
            if embeddings_list:
                self.init_faiss_index()
                self.faiss_index.add(np.array(embeddings_list))
            
            logger.info(f"Données du niveau 3 chargées: {len(self.meta_concepts)} concepts")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données du niveau 3: {e}")
            # Initialisation des structures vides
            self.meta_concepts = {}
            self.concept_embeddings = {}
            self.concept_graph = {}

# Classe de niveau 4 corrigée
class FixedLevel4DailyKnowledge:
    """
    Niveau 4: Connaissances quotidiennes qui consolident les méta-concepts en connaissances à long terme
    Version compatible avec le système corrigé
    """
    def __init__(self, data_dir: str, level3_concepts,
                llm_service=None):
        """
        Initialise le gestionnaire de connaissances de niveau 4.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            level3_concepts: Instance du générateur de méta-concepts de niveau 3
            llm_service: Service LLM à utiliser pour la consolidation des connaissances
        """
        self.data_dir = data_dir
        self.level3_concepts = level3_concepts
        self.llm_service = llm_service
        
        # Structures de données
        self.knowledge_items = {}  # id -> KnowledgeItem
        self.knowledge_embeddings = {}  # id -> embedding
        self.knowledge_graph = {}  # id -> [id1, id2, ...] (relations entre connaissances)
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialiser l'index FAISS
        self.init_faiss_index()
        
        # Charger les données existantes
        self.load_data()
        
        # Démarrer le processus automatique de consolidation
        self.start_auto_process()
        
        logger.info(f"Gestionnaire de connaissances de niveau 4 initialisé: {len(self.knowledge_items)} items")
    
    def init_faiss_index(self):
        """Initialise l'index FAISS pour la recherche par similarité."""
        # Dimension des embeddings (utilise la même que pour les niveaux précédents)
        embedding_dim = 1536
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        logger.info(f"Index FAISS initialisé (dim={embedding_dim})")
    
    def start_auto_process(self):
        """
        Démarre une tâche en arrière-plan pour consolider les méta-concepts en connaissances quotidiennes.
        Utilise threading au lieu d'asyncio pour éviter les problèmes d'event loop.
        """
        def auto_consolidate_task():
            while True:
                try:
                    # Attente d'un jour
                    time.sleep(86400)  # 24 heures
                    
                    # Consolidation des méta-concepts en connaissances
                    knowledge_items = self.consolidate_level3_to_level4()
                    
                    logger.info(f"Consolidation quotidienne automatique effectuée: {len(knowledge_items)} connaissances générées")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la consolidation quotidienne automatique: {e}")
        
        # Démarrage du thread
        consolidate_thread = threading.Thread(target=auto_consolidate_task, daemon=True)
        consolidate_thread.start()
        
        logger.info("Tâche de consolidation quotidienne automatique démarrée")
    
    def consolidate_level3_to_level4(self):
        """
        Consolide les méta-concepts de niveau 3 en connaissances de niveau 4.
        
        Returns:
            Liste des nouvelles connaissances générées
        """
        # Récupération des méta-concepts récents
        recent_concepts = self.get_recent_concepts()
        
        if not recent_concepts:
            logger.info("Aucun méta-concept récent à consolider")
            return []
        
        # Clustering des méta-concepts
        concept_clusters = self.cluster_concepts(recent_concepts)
        
        # Génération des connaissances
        new_knowledge = []
        
        for cluster_idx, cluster_concepts in concept_clusters.items():
            if len(cluster_concepts) < 2:
                continue
            
            # Génération de la connaissance pour ce cluster
            knowledge_name, knowledge_content = self.generate_knowledge_item(cluster_concepts)
            
            # Création d'un objet KnowledgeItem
            knowledge_id = str(uuid.uuid4())
            knowledge_item = KnowledgeItem(
                id=knowledge_id,
                name=knowledge_name,
                content=knowledge_content,
                created_at=datetime.now(),
                concept_ids=[concept.id for concept in cluster_concepts],
                importance=self.calculate_importance(cluster_concepts)
            )
            
            # Stockage de la connaissance
            self.knowledge_items[knowledge_id] = knowledge_item
            
            # Génération et stockage de l'embedding
            knowledge_text = f"{knowledge_name}: {knowledge_content}"
            knowledge_embedding = self.level3_concepts.level2_condenser.short_term_memory.generate_embedding(knowledge_text)
            self.knowledge_embeddings[knowledge_id] = knowledge_embedding
            
            # Mise à jour de l'index FAISS
            if knowledge_embedding is not None:
                self.faiss_index.add(np.array([knowledge_embedding]))
            
            # Mise à jour du graphe de connaissances
            self.update_knowledge_graph(knowledge_item, cluster_concepts)
            
            # Ajout à la liste des nouvelles connaissances
            new_knowledge.append(knowledge_item)
        
        # Sauvegarde des données
        self.save_data()
        
        return new_knowledge
    
    def get_recent_concepts(self):
        """
        Récupère les méta-concepts récents pour la consolidation.
        
        Returns:
            Liste des méta-concepts récents (dernière journée)
        """
        # Récupération de tous les méta-concepts
        all_concepts = list(self.level3_concepts.meta_concepts.values())
        
        # Filtrage par fenêtre de temps (1 journée)
        time_threshold = datetime.now() - timedelta(days=1)
        recent_concepts = [
            concept for concept in all_concepts
            if concept.created_at >= time_threshold
        ]
        
        return recent_concepts
    
    def cluster_concepts(self, concepts):
        """
        Regroupe les méta-concepts par similarité.
        
        Args:
            concepts: Liste des méta-concepts à regrouper
            
        Returns:
            Dictionnaire de clusters {cluster_id -> liste de méta-concepts}
        """
        if not concepts:
            return {}
        
        # Version simplifiée: clustering naïf basé sur les embeddings
        # Dans une implémentation réelle, on utiliserait un algorithme de clustering plus sophistiqué
        
        # Collection des embeddings
        embeddings = []
        concepts_with_embeddings = []
        
        for concept in concepts:
            if concept.id in self.level3_concepts.concept_embeddings:
                embeddings.append(self.level3_concepts.concept_embeddings[concept.id])
                concepts_with_embeddings.append(concept)
        
        if not embeddings:
            return {}
        
        # Clustering simple: division en 2 groupes (pourrait être amélioré)
        n_clusters = min(2, len(concepts_with_embeddings))
        clusters = {}
        
        if n_clusters == 1:
            # Un seul cluster
            clusters[0] = concepts_with_embeddings
        else:
            # Division naïve: moitié-moitié
            mid = len(concepts_with_embeddings) // 2
            clusters[0] = concepts_with_embeddings[:mid]
            clusters[1] = concepts_with_embeddings[mid:]
        
        return clusters
    
    def generate_knowledge_item(self, concepts):
        """
        Génère une connaissance à partir d'un cluster de méta-concepts.
        
        Args:
            concepts: Liste des méta-concepts du cluster
            
        Returns:
            Tuple (nom de la connaissance, contenu de la connaissance)
        """
        # Préparation des méta-concepts pour le prompt
        concept_texts = [f"{concept.name}: {concept.description}" for concept in concepts]
        concepts_joined = "\n\n".join(concept_texts)
        
        # Utilisation du service LLM si disponible
        if self.llm_service:
            prompt = f"""
            Voici un ensemble de méta-concepts qui traitent de sujets similaires:
            
            {concepts_joined}
            
            Générez une connaissance qui capture l'essence de ces méta-concepts.
            Format de réponse: "[Nom de la connaissance]: [Contenu détaillé de la connaissance]"
            
            Le nom de la connaissance devrait être court (2-5 mots) et le contenu devrait détailler
            ce que la connaissance englobe.
            """
            
            response = self.llm_service.generate(prompt)
            
            # Extraction du nom et du contenu
            try:
                parts = response.split(":", 1)
                if len(parts) == 2:
                    knowledge_name = parts[0].strip()
                    knowledge_content = parts[1].strip()
                else:
                    knowledge_name = "Connaissance Générée"
                    knowledge_content = response.strip()
            except:
                knowledge_name = "Connaissance Générée"
                knowledge_content = response.strip()
        else:
            # Génération d'une connaissance basique sans LLM
            # Extraction de mots-clés communs
            common_words = set()
            for concept in concepts:
                words = concept.name.lower().split()
                # Filtrer les mots courts ou communs
                keywords = [word for word in words if len(word) > 5]
                if not common_words:
                    common_words = set(keywords)
                else:
                    common_words = common_words.intersection(set(keywords))
            
            if common_words:
                knowledge_name = " ".join(list(common_words)[:2]).title()
            else:
                # Prendre quelques mots du premier méta-concept
                words = concepts[0].name.split()
                knowledge_name = " ".join(words[:2]).title()
            
            knowledge_content = f"Ce concept englobe les discussions sur {knowledge_name.lower()} qui apparaissent dans plusieurs méta-concepts."
        
        return knowledge_name, knowledge_content
    
    def update_knowledge_graph(self, new_knowledge, concepts):
        """
        Met à jour le graphe de connaissances avec une nouvelle connaissance.
        
        Args:
            new_knowledge: Nouvelle connaissance à ajouter au graphe
            concepts: Liste des méta-concepts associés à la connaissance
        """
        # Initialisation du nœud dans le graphe
        self.knowledge_graph[new_knowledge.id] = []
        
        # Recherche de connaissances similaires pour établir des connexions
        if len(self.knowledge_items) > 1:
            # Génération de l'embedding pour la nouvelle connaissance
            knowledge_text = f"{new_knowledge.name}: {new_knowledge.content}"
            query_embedding = self.level3_concepts.level2_condenser.short_term_memory.generate_embedding(knowledge_text)
            
            # Recherche des connaissances similaires (exclure la nouvelle connaissance)
            k = min(3, len(self.knowledge_items) - 1)
            if k > 0 and self.faiss_index.ntotal > 1:
                distances, indices = self.faiss_index.search(np.array([query_embedding]), k + 1)
                
                # Création des connexions
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.knowledge_items) and idx != self.faiss_index.ntotal - 1:
                        # Récupération de la connaissance par son ID
                        knowledge_ids = list(self.knowledge_items.keys())
                        if idx < len(knowledge_ids):
                            related_id = knowledge_ids[idx]
                            if related_id != new_knowledge.id:
                                # Ajout de la connexion bidirectionnelle
                                self.knowledge_graph[new_knowledge.id].append(related_id)
                                if related_id in self.knowledge_graph:
                                    self.knowledge_graph[related_id].append(new_knowledge.id)
                                else:
                                    self.knowledge_graph[related_id] = [new_knowledge.id]
    
    def get_relevant_knowledge(self, query: str, k: int = 3):
        """
        Récupère les connaissances les plus pertinentes pour une requête.
        
        Args:
            query: Requête utilisateur
            k: Nombre de connaissances à récupérer
            
        Returns:
            Liste des connaissances pertinentes
        """
        if not self.knowledge_items:
            return []
        
        # Génération de l'embedding pour la requête
        query_embedding = self.level3_concepts.level2_condenser.short_term_memory.generate_embedding(query)
        
        # Limiter k au nombre de connaissances disponibles
        k = min(k, len(self.knowledge_items))
        
        if k == 0:
            return []
        
        # Recherche dans l'index FAISS
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Récupération des connaissances
        relevant_knowledge = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.knowledge_items):
                continue
            
            # Récupération de la connaissance par son ID
            knowledge_ids = list(self.knowledge_items.keys())
            if idx < len(knowledge_ids):
                knowledge_id = knowledge_ids[idx]
                if knowledge_id in self.knowledge_items:
                    relevant_knowledge.append(self.knowledge_items[knowledge_id])
        
        return relevant_knowledge
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        try:
            # Préparation des données
            data = {
                "knowledge_items": {k: asdict(v) for k, v in self.knowledge_items.items()},
                "knowledge_embeddings": {k: v.tolist() for k, v in self.knowledge_embeddings.items() if v is not None},
                "knowledge_graph": self.knowledge_graph
            }
            
            # Sauvegarde dans un fichier pickle
            with open(os.path.join(self.data_dir, "level4_knowledge.pkl"), "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Données du niveau 4 sauvegardées: {len(self.knowledge_items)} connaissances")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données du niveau 4: {e}")
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "level4_knowledge.pkl")
        
        if not os.path.exists(pickle_path):
            logger.info("Aucune donnée de niveau 4 à charger")
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Chargement des connaissances
            loaded_knowledge = {}
            for k, v in data.get("knowledge_items", {}).items():
                knowledge_item = KnowledgeItem(
                    id=v["id"],
                    name=v["name"],
                    content=v["content"],
                    created_at=v["created_at"],
                    concept_ids=v["concept_ids"],
                    importance=v["importance"],
                    metadata=v.get("metadata", {})
                )
                loaded_knowledge[k] = knowledge_item
            
            # Mise à jour des données
            self.knowledge_items = loaded_knowledge
            self.knowledge_embeddings = {k: np.array(v) for k, v in data.get("knowledge_embeddings", {}).items()}
            self.knowledge_graph = data.get("knowledge_graph", {})
            
            # Reconstruction de l'index FAISS
            embeddings_list = list(self.knowledge_embeddings.values())
            if embeddings_list:
                self.init_faiss_index()
                self.faiss_index.add(np.array(embeddings_list))
            
            logger.info(f"Données du niveau 4 chargées: {len(self.knowledge_items)} connaissances")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données du niveau 4: {e}")
            # Initialisation des structures vides
            self.knowledge_items = {}
            self.knowledge_embeddings = {}
            self.knowledge_graph = {}
    
    def calculate_importance(self, concepts):
        """
        Calcule l'importance d'une connaissance en fonction des méta-concepts associés.
        
        Args:
            concepts: Liste des méta-concepts associés
            
        Returns:
            Score d'importance (0.0 à 1.0)
        """
        # Calcul basique: plus il y a de concepts associés, plus c'est important
        base_score = min(len(concepts) / 10.0, 0.7)  # Max 0.7 pour ce critère
        
        # Bonus pour les concepts récents
        now = datetime.now()
        recency_scores = []
        for concept in concepts:
            # Calcul de la fraîcheur relative (1.0 = très récent, 0.0 = ancien)
            days_old = (now - concept.created_at).total_seconds() / 86400.0
            if days_old < 1.0:
                recency = 1.0 - days_old
            else:
                recency = 0.3 * math.exp(-days_old / 7.0)  # Décroissance exponentielle
            recency_scores.append(recency)
        
        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.0
        recency_score = 0.3 * avg_recency  # Max 0.3 pour ce critère
        
        # Score final
        importance = base_score + recency_score
        
        return min(importance, 1.0)  # Limiter à 1.0

# Classe de niveau 5 corrigée (Orchestrateur)
class FixedHierarchicalMemoryOrchestrator:
    """
    Niveau 5: Orchestrateur qui coordonne tous les niveaux de mémoire
    Version compatible avec le système corrigé
    """
    def __init__(self, data_dir: str, 
                 short_term_memory=None, 
                 level2_condenser=None, 
                 level3_concepts=None, 
                 level4_knowledge=None,
                 llm_service=None):
        """
        Initialise l'orchestrateur de mémoire hiérarchique.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            short_term_memory: Instance de la mémoire à court terme (optionnel, sera créé si absent)
            level2_condenser: Instance du condensateur de niveau 2 (optionnel, sera créé si absent)
            level3_concepts: Instance du générateur de méta-concepts (optionnel, sera créé si absent)
            level4_knowledge: Instance du gestionnaire de connaissances (optionnel, sera créé si absent)
            llm_service: Service LLM à utiliser pour les différents niveaux
        """
        self.data_dir = data_dir
        self.llm_service = llm_service
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Configuration des niveaux de mémoire
        if short_term_memory is None:
            short_term_memory = FixedShortTermMemory(
                data_dir=os.path.join(data_dir, "level1"),
                max_items=15  # Paramètre supporté, valeur par défaut
            )
        self.short_term_memory = short_term_memory
        
        if level2_condenser is None:
            level2_condenser = FixedLevel2Condenser(
                data_dir=os.path.join(data_dir, "level2"),
                short_term_memory=self.short_term_memory,
                llm_service=llm_service,
                time_window_minutes=5
            )
        self.level2_condenser = level2_condenser
        
        if level3_concepts is None:
            level3_concepts = FixedLevel3HourlyConcepts(
                data_dir=os.path.join(data_dir, "level3"),
                level2_condenser=self.level2_condenser,
                llm_service=llm_service
            )
        self.level3_concepts = level3_concepts
        
        if level4_knowledge is None:
            level4_knowledge = FixedLevel4DailyKnowledge(
                data_dir=os.path.join(data_dir, "level4"),
                level3_concepts=self.level3_concepts,
                llm_service=llm_service
            )
        self.level4_knowledge = level4_knowledge
        
        logger.info(f"Orchestrateur de mémoire hiérarchique initialisé")
    
    def add_conversation(self, user_input: str, system_response: str, 
                         user_metadata: Dict[str, Any] = None, 
                         response_metadata: Dict[str, Any] = None):
        """
        Ajoute une conversation à la mémoire à court terme.
        
        Args:
            user_input: Texte de l'entrée utilisateur
            system_response: Texte de la réponse du système
            user_metadata: Métadonnées associées à l'entrée utilisateur
            response_metadata: Métadonnées associées à la réponse du système
            
        Returns:
            ID de la conversation ajoutée
        """
        # Ajout à la mémoire à court terme
        conversation_id = self.short_term_memory.add_conversation(
            user_input, system_response, user_metadata, response_metadata
        )
        
        logger.info(f"Conversation ajoutée à la mémoire: {conversation_id}")
        
        return conversation_id
    
    def get_context_for_query(self, query: str, max_items_per_level: int = 3):
        """
        Récupère le contexte pertinent de tous les niveaux pour une requête.
        
        Args:
            query: Requête utilisateur
            max_items_per_level: Nombre maximum d'éléments à récupérer par niveau
            
        Returns:
            Dictionnaire contenant le contexte de chaque niveau
        """
        context = {
            "short_term": [],
            "summaries": [],
            "concepts": [],
            "knowledge": []
        }
        
        # Récupération du contexte de la mémoire à court terme
        recent_conversations = self.short_term_memory.get_relevant_conversations(
            query, max_items_per_level
        )
        context["short_term"] = recent_conversations
        
        # Récupération du contexte des résumés
        relevant_summaries = self.level2_condenser.get_relevant_summaries(
            query, max_items_per_level
        )
        context["summaries"] = relevant_summaries
        
        # Récupération du contexte des méta-concepts
        relevant_concepts = self.level3_concepts.get_relevant_concepts(
            query, max_items_per_level
        )
        context["concepts"] = relevant_concepts
        
        # Récupération du contexte des connaissances
        relevant_knowledge = self.level4_knowledge.get_relevant_knowledge(
            query, max_items_per_level
        )
        context["knowledge"] = relevant_knowledge
        
        total_items = (
            len(context["short_term"]) +
            len(context["summaries"]) +
            len(context["concepts"]) +
            len(context["knowledge"])
        )
        
        logger.info(f"Contexte récupéré pour la requête: {total_items} éléments au total")
        
        return context
    
    def compose_context(self, query: str, max_tokens: int = 1500, prioritize_knowledge: bool = True):
        """
        Compose un contexte compact pour une requête, en respectant une limite de tokens.
        
        Args:
            query: Requête utilisateur
            max_tokens: Nombre maximum de tokens dans le contexte
            prioritize_knowledge: Si True, privilégie les connaissances de niveau supérieur
            
        Returns:
            Texte du contexte composé
        """
        # Récupération du contexte complet
        context = self.get_context_for_query(query)
        
        # Estimation approximative des tokens
        def estimate_tokens(text):
            # Estimation simple: environ 4 caractères par token
            return len(text) // 4
        
        # Composition du contexte avec priorité
        composed_text = "Contexte pertinent pour la conversation :\n\n"
        total_tokens = estimate_tokens(composed_text)
        
        # Déterminer l'ordre des niveaux en fonction de la priorité
        if prioritize_knowledge:
            level_order = ["knowledge", "concepts", "summaries", "short_term"]
        else:
            level_order = ["short_term", "summaries", "concepts", "knowledge"]
        
        # Dictionnaire des formatages par niveau
        formatters = {
            "knowledge": lambda item: f"CONNAISSANCE IMPORTANTE: {item.name} - {item.content}",
            "concepts": lambda item: f"CONCEPT: {item.name} - {item.description}",
            "summaries": lambda item: f"RÉSUMÉ DE CONVERSATIONS PASSÉES: {item.content}",
            "short_term": lambda item: f"CONVERSATION RÉCENTE:\nUser: {item.prompt.content}\nRenée: {item.response.content}"
        }
        
        # Ajout des éléments en fonction de la priorité et de la limite de tokens
        for level in level_order:
            items = context[level]
            
            # Trier les connaissances par importance si c'est le niveau 4
            if level == "knowledge":
                items = sorted(items, key=lambda x: x.importance, reverse=True)
            
            for item in items:
                item_text = "\n" + formatters[level](item) + "\n"
                item_tokens = estimate_tokens(item_text)
                
                if total_tokens + item_tokens <= max_tokens:
                    composed_text += item_text
                    total_tokens += item_tokens
                else:
                    # Si on dépasse la limite mais qu'on n'a encore rien ajouté, inclure au moins un élément
                    if not any(len(context[l]) > 0 for l in level_order[:level_order.index(level)]):
                        composed_text += item_text
                    break
        
        logger.info(f"Contexte composé: environ {total_tokens} tokens")
        
        return composed_text
    
    def trigger_manual_condensation(self):
        """
        Déclenche manuellement les processus de condensation pour tous les niveaux.
        Utile pour les tests ou pour forcer des mises à jour.
        
        Returns:
            Dictionnaire avec les résultats des différentes condensations
        """
        results = {
            "level2": None,
            "level3": None,
            "level4": None
        }
        
        # Niveau 2: Condensation des conversations en résumés
        try:
            level2_results = self.level2_condenser.condense_recent_memory()
            results["level2"] = {
                "success": True,
                "count": len(level2_results),
                "items": level2_results
            }
        except Exception as e:
            results["level2"] = {
                "success": False,
                "error": str(e)
            }
        
        # Niveau 3: Condensation des résumés en méta-concepts
        try:
            level3_results = self.level3_concepts.condense_level2_to_level3()
            results["level3"] = {
                "success": True,
                "count": len(level3_results),
                "items": level3_results
            }
        except Exception as e:
            results["level3"] = {
                "success": False,
                "error": str(e)
            }
        
        # Niveau 4: Consolidation des méta-concepts en connaissances
        try:
            level4_results = self.level4_knowledge.consolidate_level3_to_level4()
            results["level4"] = {
                "success": True,
                "count": len(level4_results),
                "items": level4_results
            }
        except Exception as e:
            results["level4"] = {
                "success": False,
                "error": str(e)
            }
        
        logger.info(f"Condensation manuelle déclenchée: {results}")
        
        return results
