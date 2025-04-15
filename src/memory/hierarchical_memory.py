# hierarchical_memory.py
# Implémentation du système de mémoire hiérarchique à 5 niveaux pour Renée

import os
import json
import uuid
import torch
import numpy as np
import faiss
import logging
import asyncio
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from contextlib import nullcontext

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Détermination du device pour les calculs selon la configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config/config.json")
with open(config_path) as f:
    config = json.load(f)

# Détermination du device pour les calculs
device_config = config.get("acceleration", {})
use_mps = device_config.get("use_mps", False) and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
use_gpu = device_config.get("use_gpu", False) and torch.cuda.is_available()
mixed_precision = device_config.get("mixed_precision", False)
batch_size = device_config.get("batch_size", 16)

if use_mps:
    DEVICE = "mps"
    logger.info("Utilisation de l'accélération MPS (Metal)")
elif use_gpu:
    DEVICE = "cuda"
    logger.info("Utilisation de l'accélération CUDA (GPU)")
else:
    DEVICE = "cpu"
    logger.info("Utilisation du CPU pour les calculs")

# Structures de données pour la mémoire
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
class ErrorMemory:
    id: str
    description: str
    created_at: datetime
    error_type: str
    context: str
    solution: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersonaAspect:
    id: str
    aspect_type: str
    description: str
    created_at: datetime
    origin: str = "system"
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

class ShortTermMemory:
    """
    Niveau 1: Mémoire à court terme qui conserve les interactions récentes
    """
    def __init__(self, data_dir: str, max_items: int = 15, 
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 use_device: str = DEVICE, mixed_precision: bool = mixed_precision,
                 batch_size: int = batch_size):
        """
        Initialise la mémoire à court terme avec optimisations pour MPS/GPU.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            max_items: Nombre maximum d'éléments à conserver
            embedding_model: Modèle d'embedding à utiliser
            use_device: Device à utiliser (mps, cuda, cpu)
            mixed_precision: Utiliser la précision mixte pour économiser la mémoire
            batch_size: Taille des lots pour le traitement des embeddings
        """
        self.data_dir = data_dir
        self.max_items = max_items
        self.device = use_device
        self.mixed_precision = mixed_precision and (use_device in ["cuda", "mps"])
        self.batch_size = batch_size
        
        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialisation du modèle d'embedding
        self.model = SentenceTransformer(embedding_model, device=self.device)
        
        # Utiliser la précision mixte si activée
        if self.mixed_precision:
            self.model = self.model.half()  # Convertit en FP16
        
        # Initialisation de l'index FAISS
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        self.init_faiss_index()
        
        # Structures de données en mémoire
        self.conversations: Dict[str, Conversation] = {}
        self.prompts: Dict[str, UserPrompt] = {}
        self.responses: Dict[str, SystemResponse] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Chargement des données persistantes
        self.load_data()
        
        logger.info(f"ShortTermMemory initialisé avec {len(self.conversations)} conversations")
    
    def init_faiss_index(self):
        """Initialise l'index FAISS optimisé pour le device configuré."""
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info(f"Index FAISS GPU initialisé (dim={self.vector_dimension})")
        else:
            logger.info(f"Index FAISS CPU initialisé (dim={self.vector_dimension})")
    
    def generate_embedding(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Génère un embedding optimisé pour un texte ou une liste de textes.
        Utilise le traitement par lots et la précision mixte si configuré.
        """
        # Mode d'inférence pour optimiser la mémoire
        with torch.inference_mode():
            # Utiliser la précision mixte si activée
            if self.mixed_precision and self.device in ["cuda", "mps"]:
                with torch.autocast(device_type=self.device):
                    embedding = self.model.encode(
                        text,
                        show_progress_bar=False,
                        normalize_embeddings=normalize,
                        convert_to_tensor=True
                    )
            else:
                embedding = self.model.encode(
                    text,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    convert_to_tensor=True
                )
            
            # Retour au format numpy pour compatibilité
            return embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
    
    def batch_generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Génère des embeddings pour plusieurs textes en utilisant le traitement par lots."""
        all_embeddings = []
        
        # Traitement par lots pour optimiser l'utilisation mémoire
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self.generate_embedding(batch)
            
            if len(batch) == 1:
                all_embeddings.append(batch_embeddings)
            else:
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def add_conversation(self, user_input: str, system_response: str, 
                         user_metadata: Dict[str, Any] = None, 
                         response_metadata: Dict[str, Any] = None) -> Conversation:
        """
        Ajoute une nouvelle conversation à la mémoire à court terme.
        Génère et stocke les embeddings appropriés.
        """
        now = datetime.now()
        
        # Création des identifiants uniques
        prompt_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        
        # Création des objets
        prompt = UserPrompt(
            id=prompt_id,
            content=user_input,
            created_at=now,
            metadata=user_metadata or {}
        )
        
        response = SystemResponse(
            id=response_id,
            prompt_id=prompt_id,
            content=system_response,
            created_at=now,
            metadata=response_metadata or {}
        )
        
        conversation = Conversation(
            id=conversation_id,
            prompt=prompt,
            response=response,
            created_at=now
        )
        
        # Génération et stockage des embeddings
        combined_text = f"User: {user_input} Renée: {system_response}"
        embedding = self.generate_embedding(combined_text)
        
        # Ajout à l'index FAISS
        if hasattr(self, 'gpu_index'):
            self.gpu_index.add(np.array([embedding]))
        else:
            self.index.add(np.array([embedding]))
        
        # Stockage en mémoire
        self.prompts[prompt_id] = prompt
        self.responses[response_id] = response
        self.conversations[conversation_id] = conversation
        self.embeddings[conversation_id] = embedding
        
        # Nettoyage si dépassement de la limite
        self._clean_old_conversations()
        
        # Sauvegarde des données
        self.save_data()
        
        return conversation
    
    def _clean_old_conversations(self):
        """Nettoie les conversations les plus anciennes si la limite est dépassée."""
        if len(self.conversations) <= self.max_items:
            return
        
        # Trier par date et ne garder que les max_items plus récentes
        sorted_convs = sorted(
            self.conversations.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        # Garder uniquement les max_items plus récentes
        to_keep = sorted_convs[:self.max_items]
        to_remove = sorted_convs[self.max_items:]
        
        # Reconstruire l'index FAISS
        self._rebuild_faiss_index([self.embeddings[conv_id] for conv_id, _ in to_keep])
        
        # Supprimer les conversations, prompts, réponses et embeddings
        for conv_id, conv in to_remove:
            prompt_id = conv.prompt.id
            response_id = conv.response.id
            
            del self.conversations[conv_id]
            del self.prompts[prompt_id]
            del self.responses[response_id]
            del self.embeddings[conv_id]
    
    def _rebuild_faiss_index(self, embeddings_to_keep: List[np.ndarray]):
        """Reconstruit l'index FAISS avec les embeddings à conserver."""
        embeddings_array = np.array(embeddings_to_keep)
        
        # Réinitialiser l'index
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        
        if hasattr(self, 'gpu_index'):
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.gpu_index.add(embeddings_array)
        else:
            self.index.add(embeddings_array)
    
    def get_similar_conversations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les k conversations les plus similaires à une requête.
        Utilise l'index FAISS pour une recherche efficace.
        """
        if not self.conversations:
            return []
        
        # Générer l'embedding de la requête
        query_embedding = self.generate_embedding(query)
        
        # Recherche dans l'index FAISS
        if hasattr(self, 'gpu_index'):
            D, I = self.gpu_index.search(np.array([query_embedding]), k)
        else:
            D, I = self.index.search(np.array([query_embedding]), k)
        
        # Reconstruire les résultats
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0 and idx < len(self.conversations):  # Vérifier la validité de l'index
                conv_id = list(self.conversations.keys())[idx]
                conv = self.conversations[conv_id]
                
                results.append({
                    "id": conv.id,
                    "similarity": float(D[0][i]),
                    "user_input": conv.prompt.content,
                    "system_response": conv.response.content,
                    "created_at": conv.created_at.isoformat(),
                    "metadata": conv.metadata
                })
        
        return results
    
    def get_recent_conversations(self, limit: int = 10) -> List[Conversation]:
        """Récupère les conversations les plus récentes."""
        sorted_convs = sorted(
            self.conversations.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return sorted_convs[:limit]
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        data = {
            "conversations": {k: asdict(v) for k, v in self.conversations.items()},
            "prompts": {k: asdict(v) for k, v in self.prompts.items()},
            "responses": {k: asdict(v) for k, v in self.responses.items()},
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()}
        }
        
        # Sauvegarde dans un fichier pickle
        with open(os.path.join(self.data_dir, "short_term_memory.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "short_term_memory.pkl")
        
        if not os.path.exists(pickle_path):
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Conversion dictionnaire -> objets
            for conv_id, conv_dict in data["conversations"].items():
                self.conversations[conv_id] = Conversation(**conv_dict)
            
            for prompt_id, prompt_dict in data["prompts"].items():
                self.prompts[prompt_id] = UserPrompt(**prompt_dict)
            
            for response_id, response_dict in data["responses"].items():
                self.responses[response_id] = SystemResponse(**response_dict)
            
            for embedding_id, embedding_list in data["embeddings"].items():
                self.embeddings[embedding_id] = np.array(embedding_list)
            
            # Reconstruire l'index FAISS
            embeddings_list = list(self.embeddings.values())
            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                
                if hasattr(self, 'gpu_index'):
                    self.gpu_index.add(embeddings_array)
                else:
                    self.index.add(embeddings_array)
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")

class Level2Condenser:
    """
    Niveau 2: Condensation des interactions récentes (toutes les 5 minutes)
    Utilise BERTopic et des techniques de clustering pour générer des résumés.
    """
    def __init__(self, data_dir: str, short_term_memory: ShortTermMemory,
                 use_device: str = DEVICE, mixed_precision: bool = mixed_precision,
                 batch_size: int = batch_size):
        """
        Initialise le condenseur de niveau 2.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            short_term_memory: Instance de ShortTermMemory
            use_device: Device à utiliser (mps, cuda, cpu)
            mixed_precision: Utiliser la précision mixte pour économiser la mémoire
            batch_size: Taille des lots pour le traitement des embeddings
        """
        self.data_dir = data_dir
        self.short_term_memory = short_term_memory
        self.device = use_device
        self.mixed_precision = mixed_precision and (use_device in ["cuda", "mps"])
        self.batch_size = batch_size
        
        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialisation de l'index FAISS
        self.vector_dimension = self.short_term_memory.vector_dimension
        self.init_faiss_index()
        
        # Structures de données en mémoire
        self.summaries: Dict[str, Summary] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Chargement des données persistantes
        self.load_data()
        
        # Initialisation du thread de condensation automatique
        self.condense_thread = None
        self.start_auto_condense()
    
    def init_faiss_index(self):
        """Initialise l'index FAISS optimisé pour le device configuré."""
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info(f"Index FAISS GPU initialisé (dim={self.vector_dimension})")
        else:
            logger.info(f"Index FAISS CPU initialisé (dim={self.vector_dimension})")
    
    def start_auto_condense(self):
        """Démarre le thread de condensation automatique."""
        import threading
        
        def auto_condense_task():
            while True:
                try:
                    logger.info("Démarrage de la condensation de niveau 2")
                    summaries = self.condense_recent_memory()
                    logger.info(f"Condensation de niveau 2 terminée: {len(summaries)} résumés générés")
                except Exception as e:
                    logger.error(f"Erreur lors de la condensation automatique: {e}")
                
                # Attendre 5 minutes
                import time
                time.sleep(300)
        
        # Démarrer la tâche
        self.condense_thread = threading.Thread(target=auto_condense_task, daemon=True)
        self.condense_thread.start()
    
    def condense_recent_memory(self) -> List[Summary]:
        """
        Condense les conversations récentes en résumés en utilisant BERTopic.
        Optimisé pour l'exécution avec MPS/GPU.
        """
        try:
            # Récupération des conversations récentes
            recent_conversations = self.short_term_memory.get_recent_conversations(limit=50)
            
            if not recent_conversations:
                return []
            
            # Préparation des données pour le clustering
            conversation_texts = [
                f"User: {conv.prompt.content}\nRenée: {conv.response.content}"
                for conv in recent_conversations
            ]
            
            # Génération des embeddings par lots (optimisé pour le device)
            embeddings = self.short_term_memory.batch_generate_embeddings(conversation_texts)
            
            # Configuration de UMAP pour la réduction de dimension
            try:
                from umap import UMAP
                import hdbscan
                from sklearn.feature_extraction.text import CountVectorizer
                from bertopic import BERTopic
                
                # Configuration de UMAP optimisée pour le device choisi
                if self.device == "cuda" and torch.cuda.is_available():
                    import cuml
                    umap_model = cuml.manifold.UMAP(
                        n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        n_jobs=-1
                    )
                else:
                    # Configuration standard pour CPU/MPS
                    umap_model = UMAP(
                        n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        low_memory=True,
                        random_state=42
                    )
                
                # Configuration du clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=2,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Configuration du vectoriseur
                vectorizer = CountVectorizer(stop_words="english")
                
                # Initialisation de BERTopic
                topic_model = BERTopic(
                    embedding_model=None,  # On utilise nos propres embeddings
                    umap_model=umap_model,
                    hdbscan_model=clusterer,
                    vectorizer_model=vectorizer,
                    calculate_probabilities=True,
                    verbose=True
                )
                
                # Extraction des thèmes
                topics, probs = topic_model.fit_transform(
                    conversation_texts, 
                    embeddings=embeddings
                )
                
            except ImportError as e:
                logger.error(f"Erreur lors du chargement des modules de clustering: {e}")
                # Fallback sur une méthode de clustering plus simple
                topics = self._fallback_clustering(embeddings)
                probs = np.ones((len(topics), 1))  # Probabilités fictives
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction des thèmes: {e}")
                # Fallback sur une méthode de clustering plus simple
                topics = self._fallback_clustering(embeddings)
                probs = np.ones((len(topics), 1))  # Probabilités fictives
            
            # Générer des résumés pour chaque cluster
            summaries = []
            unique_clusters = set(topics)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # -1 est le cluster de bruit dans HDBSCAN
                    continue
                
                # Sélectionner les conversations du cluster
                cluster_indices = [i for i, topic in enumerate(topics) if topic == cluster_id]
                cluster_conversations = [recent_conversations[i] for i in cluster_indices]
                
                if not cluster_conversations:
                    continue
                
                # Créer un prompt pour le résumé
                cluster_text = "\n\n".join([
                    f"User: {conv.prompt.content}\nRenée: {conv.response.content}"
                    for conv in cluster_conversations
                ])
                
                # Générer le résumé (simulé ici - à remplacer par un appel au LLM)
                summary_text = self._generate_summary_text(cluster_text, topic_model, cluster_id)
                
                # Créer l'objet Summary
                summary = Summary(
                    id=str(uuid.uuid4()),
                    content=summary_text,
                    created_at=datetime.now(),
                    conversation_ids=[conv.id for conv in cluster_conversations]
                )
                
                # Générer et stocker l'embedding du résumé
                summary_embedding = self.short_term_memory.generate_embedding(summary_text)
                
                # Ajouter à l'index FAISS
                if hasattr(self, 'gpu_index'):
                    self.gpu_index.add(np.array([summary_embedding]))
                else:
                    self.index.add(np.array([summary_embedding]))
                
                self.summaries[summary.id] = summary
                self.embeddings[summary.id] = summary_embedding
                
                summaries.append(summary)
            
            # Sauvegarder les données
            self.save_data()
            
            return summaries
            
        except Exception as e:
            logger.error(f"Erreur lors de la condensation des mémoires: {e}")
            return []
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> List[int]:
        """Méthode de clustering de secours en cas d'échec de BERTopic."""
        try:
            from sklearn.cluster import KMeans
            
            # Déterminer le nombre de clusters (maximum 5, minimum 1)
            n_clusters = min(5, max(1, len(embeddings) // 2))
            
            # Clustering K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"Erreur lors du clustering de secours: {e}")
            # Fallback ultime : tout dans un seul cluster
            return [0] * len(embeddings)
    
    def _generate_summary_text(self, cluster_text: str, topic_model=None, cluster_id=None) -> str:
        """
        Génère un résumé textuel pour un cluster.
        Cette fonction simule la génération de résumé qui serait faite par un LLM.
        """
        try:
            if topic_model is not None and cluster_id is not None and cluster_id != -1:
                # Extraire les mots-clés du cluster à l'aide de BERTopic
                try:
                    topic_info = topic_model.get_topic_info()
                    topic_words = topic_model.get_topic(cluster_id)
                    
                    # Créer un résumé basé sur les mots-clés
                    keywords = [word for word, _ in topic_words[:5]]
                    
                    # Résumé simple basé sur les mots-clés
                    summary = f"Ce résumé regroupe des conversations autour des thèmes: {', '.join(keywords)}. "
                    summary += "Ces conversations portent principalement sur ces sujets, "
                    summary += "avec plusieurs échanges qui explorent ces concepts en détail."
                    
                    return summary
                except:
                    pass
            
            # Fallback simple pour la génération de résumé
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Initialiser le vectoriseur TF-IDF
            vectorizer = CountVectorizer(
                max_features=50,
                stop_words="french",
                ngram_range=(1, 2)
            )
            
            # Transformer le texte
            X = vectorizer.fit_transform([cluster_text])
            
            # Obtenir les termes les plus importants
            feature_names = vectorizer.get_feature_names_out()
            importance = X.toarray()[0]
            
            # Trier les termes par importance
            top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
            
            # Extraire les N premiers termes les plus importants
            top_terms = [term for term, _ in top_features[:5]]
            
            # Créer un résumé basé sur les termes
            summary = f"Ce résumé regroupe des conversations autour des thèmes: {', '.join(top_terms)}. "
            summary += "Ces échanges abordent ces sujets à travers plusieurs discussions connexes."
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé: {e}")
            # Résumé par défaut si l'extraction de mots-clés échoue
            return "Ce résumé regroupe plusieurs conversations sur des thèmes connexes."
    
    def get_relevant_summaries(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Récupère les k résumés les plus pertinents pour une requête.
        Utilise l'index FAISS pour une recherche efficace.
        """
        if not self.summaries:
            return []
        
        # Générer l'embedding de la requête
        query_embedding = self.short_term_memory.generate_embedding(query)
        
        # Recherche dans l'index FAISS
        if hasattr(self, 'gpu_index'):
            D, I = self.gpu_index.search(np.array([query_embedding]), k)
        else:
            D, I = self.index.search(np.array([query_embedding]), k)
        
        # Reconstruire les résultats
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0 and idx < len(self.summaries):  # Vérifier la validité de l'index
                summary_id = list(self.summaries.keys())[idx]
                summary = self.summaries[summary_id]
                
                results.append({
                    "id": summary.id,
                    "similarity": float(D[0][i]),
                    "content": summary.content,
                    "created_at": summary.created_at.isoformat(),
                    "conversation_ids": summary.conversation_ids
                })
        
        return results
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        data = {
            "summaries": {k: asdict(v) for k, v in self.summaries.items()},
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()}
        }
        
        # Sauvegarde dans un fichier pickle
        with open(os.path.join(self.data_dir, "level2_summaries.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "level2_summaries.pkl")
        
        if not os.path.exists(pickle_path):
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Conversion dictionnaire -> objets
            for summary_id, summary_dict in data["summaries"].items():
                self.summaries[summary_id] = Summary(**summary_dict)
            
            for embedding_id, embedding_list in data["embeddings"].items():
                self.embeddings[embedding_id] = np.array(embedding_list)
            
            # Reconstruire l'index FAISS
            embeddings_list = list(self.embeddings.values())
            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                
                if hasattr(self, 'gpu_index'):
                    self.gpu_index.add(embeddings_array)
                else:
                    self.index.add(embeddings_array)
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")

class Level3HourlyConcepts:
    """
    Niveau 3: Condensation Horaire des méta-concepts
    
    Ce niveau condense les résumés du niveau 2 en méta-concepts qui identifient 
    les tendances et relations plus profondes. Il opère sur une base horaire.
    """
    
    def __init__(self, data_dir: str, level2_condenser, 
                 use_device: str = DEVICE, mixed_precision: bool = mixed_precision,
                 batch_size: int = batch_size, llm_service=None):
        """
        Initialise le système de méta-concepts horaires.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            level2_condenser: Instance de Level2Condenser
            use_device: Device à utiliser (mps, cuda, cpu)
            mixed_precision: Utiliser la précision mixte pour économiser la mémoire
            batch_size: Taille des lots pour le traitement
            llm_service: Service LLM pour la génération de méta-concepts
        """
        self.data_dir = data_dir
        self.level2_condenser = level2_condenser
        self.device = use_device
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        
        # Structures pour les données
        self.meta_concepts = {}
        self.embeddings = {}
        self.concept_graph = defaultdict(list)
        
        # Service LLM pour la génération
        self.llm_service = llm_service
        
        # Dimension du vecteur (correspond à celle du modèle d'embedding du niveau 2)
        self.vector_dimension = 1024
        
        # Initialisation de l'index
        self.init_faiss_index()
        
        # Chargement des données existantes
        self.load_data()
        
        # Tâche de condensation automatique
        self.start_auto_condense()
    
    def init_faiss_index(self):
        """Initialise l'index FAISS optimisé pour le device configuré."""
        # Index FAISS de base
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        
        # Optimisation GPU si disponible
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info(f"Index FAISS GPU initialisé (dim={self.vector_dimension})")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'index GPU: {e}")
                logger.info(f"Index FAISS CPU initialisé (dim={self.vector_dimension})")
        else:
            logger.info(f"Index FAISS CPU initialisé (dim={self.vector_dimension})")
    
    def start_auto_condense(self):
        """Démarre la tâche de condensation automatique horaire."""
        import threading
        
        def auto_condense_task():
            while True:
                try:
                    # Attendre une heure
                    import time
                    time.sleep(3600)  # 1 heure
                    logger.info("Démarrage de la condensation horaire (niveau 3)")
                    self.condense_level2_to_level3()
                    logger.info("Condensation horaire terminée")
                except Exception as e:
                    logger.error(f"Erreur lors de la condensation automatique niveau 3: {e}")
        
        # Démarrer dans un thread séparé
        condense_thread = threading.Thread(target=auto_condense_task, daemon=True)
        condense_thread.start()
        logger.info("Tâche de condensation horaire démarrée")
    
    def condense_level2_to_level3(self):
        """
        Condense les résumés du niveau 2 en méta-concepts.
        Utilise un graphe sémantique et le LLM pour la synthèse.
        """
        # Récupération des résumés récents du niveau 2
        recent_summaries = self.get_recent_level2_summaries()
        
        if not recent_summaries:
            logger.info("Aucun résumé récent à condenser")
            return
        
        # Groupement des résumés par thèmes sémantiques
        concept_clusters = self.cluster_by_semantic_similarity(recent_summaries)
        
        # Pour chaque cluster, générer un méta-concept
        for cluster_id, summaries in concept_clusters.items():
            try:
                # Création d'un méta-concept
                meta_concept = self.generate_meta_concept(summaries)
                
                # Génération de l'embedding pour le méta-concept
                concept_embedding = self.level2_condenser.short_term_memory.generate_embedding(
                    meta_concept.name + ": " + meta_concept.description
                )
                
                # Stockage du méta-concept et de son embedding
                self.meta_concepts[meta_concept.id] = meta_concept
                self.embeddings[meta_concept.id] = concept_embedding
                
                # Mise à jour du graphe conceptuel
                self.update_concept_graph(meta_concept, summaries)
                
                # Ajout à l'index FAISS
                if hasattr(self, 'gpu_index'):
                    self.gpu_index.add(np.array([concept_embedding]))
                else:
                    self.index.add(np.array([concept_embedding]))
                
                logger.info(f"Méta-concept créé: {meta_concept.name}")
            
            except Exception as e:
                logger.error(f"Erreur lors de la création d'un méta-concept: {e}")
        
        # Sauvegarde des données
        self.save_data()
    
    def get_recent_level2_summaries(self):
        """Récupère les résumés récents du niveau 2."""
        # Temps limite: dernière heure
        time_limit = datetime.now() - timedelta(hours=1)
        
        recent_summaries = []
        for summary_id, summary in self.level2_condenser.summaries.items():
            if summary.created_at >= time_limit:
                recent_summaries.append(summary)
        
        return recent_summaries
    
    def cluster_by_semantic_similarity(self, summaries):
        """
        Regroupe les résumés par similarité sémantique.
        Utilise un clustering basé sur les embeddings.
        """
        if not summaries:
            return {}
        
        # Si nous n'avons qu'un seul résumé, pas besoin de clustering
        if len(summaries) == 1:
            return {0: summaries}
        
        try:
            # Récupération des embeddings des résumés
            summary_embeddings = np.array([
                self.level2_condenser.embeddings.get(s.id) 
                for s in summaries 
                if s.id in self.level2_condenser.embeddings
            ])
            
            if len(summary_embeddings) < 2:
                # Si nous avons moins de 2 embeddings valides, pas de clustering
                return {0: summaries}
            
            # Clustering simple basé sur la similarité cosinus
            from sklearn.cluster import AgglomerativeClustering
            
            # Normalisation pour utiliser la similarité cosinus
            normalized_embeddings = summary_embeddings / np.linalg.norm(summary_embeddings, axis=1, keepdims=True)
            
            # Calcul des similarités
            similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # Conversion en distances (1 - similarité)
            distances = 1 - similarities
            
            # Détermination automatique du nombre de clusters (entre 1 et 5)
            n_clusters = min(5, max(1, len(summaries) // 3))
            
            # Clustering hiérarchique
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            ).fit(distances)
            
            # Organisation des résumés par cluster
            clusters = defaultdict(list)
            for i, label in enumerate(clustering.labels_):
                if i < len(summaries):
                    clusters[label].append(summaries[i])
            
            return dict(clusters)
        
        except Exception as e:
            logger.error(f"Erreur lors du clustering: {e}")
            # En cas d'erreur, retourner tous les résumés dans un seul cluster
            return {0: summaries}
    
    def generate_meta_concept(self, summaries):
        """
        Génère un méta-concept à partir d'un ensemble de résumés.
        Utilise le LLM pour la synthèse avancée.
        """
        # Génération d'un identifiant unique
        concept_id = str(uuid.uuid4())
        
        # Extraction des textes des résumés
        summary_texts = [s.content for s in summaries]
        summary_ids = [s.id for s in summaries]
        
        # Si nous avons un service LLM, l'utiliser pour générer le méta-concept
        if self.llm_service:
            prompt = f"""
            Voici plusieurs résumés liés thématiquement:
            
            {chr(10).join([f"- {text}" for text in summary_texts])}
            
            Génère un méta-concept qui synthétise ces résumés. 
            Format:
            1. Nom du concept (court, 2-5 mots)
            2. Description détaillée (2-3 phrases)
            
            Répondre uniquement au format:
            NOM: [nom du concept]
            DESCRIPTION: [description]
            """
            
            response = self.llm_service.generate(prompt)
            
            # Extraction du nom et de la description
            try:
                name_part = response.split("NOM:")[1].split("DESCRIPTION:")[0].strip()
                description_part = response.split("DESCRIPTION:")[1].strip()
            except:
                # Fallback en cas d'erreur de format
                name_part = f"Concept-{concept_id[:8]}"
                description_part = " ".join(summary_texts[:2]) + "..."
        else:
            # Méthode simple si pas de LLM disponible
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Extraction des mots clés avec TF-IDF
            vectorizer = TfidfVectorizer(max_features=10)
            try:
                X = vectorizer.fit_transform(summary_texts)
                feature_names = vectorizer.get_feature_names_out()
                importance = np.asarray(X.sum(axis=0)).flatten()
                top_keywords = [feature_names[i] for i in importance.argsort()[-5:]]
                
                name_part = " ".join(top_keywords[:3]).title()
                description_part = f"Ce méta-concept regroupe des thèmes autour de {', '.join(top_keywords)}. "
                description_part += "Il représente une tendance significative dans les conversations récentes."
            except:
                # Fallback en cas d'erreur
                name_part = f"Concept-{concept_id[:8]}"
                description_part = "Ce méta-concept regroupe plusieurs thèmes connexes identifiés dans les conversations récentes."
        
        # Création du méta-concept
        meta_concept = MetaConcept(
            id=concept_id,
            name=name_part,
            description=description_part,
            created_at=datetime.now(),
            summary_ids=summary_ids,
            source_concepts=[]
        )
        
        return meta_concept
    
    def update_concept_graph(self, new_concept, related_summaries):
        """
        Met à jour le graphe conceptuel avec le nouveau méta-concept.
        Établit des relations entre les concepts existants et le nouveau.
        """
        # Pour chaque résumé lié au nouveau concept
        for summary in related_summaries:
            # Vérifier s'il existe des concepts liés à ce résumé
            for concept_id, concept in self.meta_concepts.items():
                if summary.id in concept.summary_ids and concept_id != new_concept.id:
                    # Créer des liens bidirectionnels entre les concepts
                    self.concept_graph[new_concept.id].append(concept_id)
                    self.concept_graph[concept_id].append(new_concept.id)
        
        # Ajouter le concept à la source de lui-même pour faciliter les requêtes
        if new_concept.id not in self.concept_graph:
            self.concept_graph[new_concept.id] = []
    
    def get_related_concepts(self, concept_id, max_depth=2):
        """
        Récupère les concepts liés à un concept donné jusqu'à une profondeur maximale.
        Effectue une recherche en largeur dans le graphe conceptuel.
        """
        if concept_id not in self.concept_graph:
            return []
        
        visited = set([concept_id])
        related = []
        queue = [(concept_id, 0)]  # (concept_id, depth)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            # Si nous avons atteint la profondeur maximale, ne pas explorer plus loin
            if depth >= max_depth:
                continue
            
            # Explorer les voisins
            for neighbor_id in self.concept_graph[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    related.append(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
        
        return related
    
    def get_relevant_concepts(self, query, k=3):
        """
        Récupère les k méta-concepts les plus pertinents pour une requête.
        Utilise l'index FAISS pour une recherche efficace.
        """
        if not self.meta_concepts:
            return []
        
        # Générer l'embedding de la requête
        query_embedding = self.level2_condenser.short_term_memory.generate_embedding(query)
        
        # Recherche dans l'index FAISS
        if hasattr(self, 'gpu_index'):
            D, I = self.gpu_index.search(np.array([query_embedding]), k)
        else:
            D, I = self.index.search(np.array([query_embedding]), k)
        
        # Reconstruire les résultats
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0 and idx < len(self.meta_concepts):
                concept_id = list(self.meta_concepts.keys())[idx]
                concept = self.meta_concepts[concept_id]
                
                results.append({
                    "id": concept.id,
                    "name": concept.name,
                    "description": concept.description,
                    "similarity": float(D[0][i]),
                    "created_at": concept.created_at.isoformat(),
                    "related_summaries": concept.summary_ids
                })
        
        return results
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        data = {
            "meta_concepts": {k: asdict(v) for k, v in self.meta_concepts.items()},
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "concept_graph": dict(self.concept_graph)
        }
        
        # Sauvegarde dans un fichier pickle
        with open(os.path.join(self.data_dir, "level3_concepts.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "level3_concepts.pkl")
        
        if not os.path.exists(pickle_path):
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Conversion dictionnaire -> objets
            for concept_id, concept_dict in data["meta_concepts"].items():
                self.meta_concepts[concept_id] = MetaConcept(**concept_dict)
            
            for embedding_id, embedding_list in data["embeddings"].items():
                self.embeddings[embedding_id] = np.array(embedding_list)
            
            # Chargement du graphe conceptuel
            self.concept_graph = defaultdict(list, data["concept_graph"])
            
            # Reconstruire l'index FAISS
            embeddings_list = list(self.embeddings.values())
            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                
                if hasattr(self, 'gpu_index'):
                    self.gpu_index.add(embeddings_array)
                else:
                    self.index.add(embeddings_array)
                    
            logger.info(f"Niveau 3: {len(self.meta_concepts)} méta-concepts chargés")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données niveau 3: {e}")

class Level4DailyKnowledge:
    """
    Niveau 4: Condensation Journalière (Mémoire à Long Terme)
    
    Ce niveau représente la mémoire à long terme, avec une consolidation quotidienne 
    des informations les plus importantes et significatives.
    """
    
    def __init__(self, data_dir: str, level3_concepts, 
                 use_device: str = DEVICE, mixed_precision: bool = mixed_precision,
                 batch_size: int = batch_size, llm_service=None):
        """
        Initialise le système de mémoire à long terme.
        
        Args:
            data_dir: Répertoire pour les données persistantes
            level3_concepts: Instance de Level3HourlyConcepts
            use_device: Device à utiliser (mps, cuda, cpu)
            mixed_precision: Utiliser la précision mixte pour économiser la mémoire
            batch_size: Taille des lots pour le traitement
            llm_service: Service LLM pour la génération de connaissances
        """
        self.data_dir = data_dir
        self.level3_concepts = level3_concepts
        self.device = use_device
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        
        # Structures pour les données
        self.long_term_knowledge = {}
        self.derived_rules = {}
        self.embeddings = {}
        
        # Service LLM pour la génération
        self.llm_service = llm_service
        
        # Dimension du vecteur (identique aux autres niveaux)
        self.vector_dimension = 1024
        
        # Initialisation de l'index
        self.init_faiss_index()
        
        # Chargement des données existantes
        self.load_data()
        
        # Tâche de consolidation automatique
        self.start_auto_consolidate()
    
    def init_faiss_index(self):
        """Initialise l'index FAISS optimisé pour le device configuré."""
        # Index FAISS de base
        self.index = faiss.IndexFlatIP(self.vector_dimension)
        
        # Optimisation GPU si disponible
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info(f"Index FAISS GPU initialisé (dim={self.vector_dimension})")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'index GPU: {e}")
                logger.info(f"Index FAISS CPU initialisé (dim={self.vector_dimension})")
        else:
            logger.info(f"Index FAISS CPU initialisé (dim={self.vector_dimension})")
    
    def start_auto_consolidate(self):
        """Démarre la tâche de consolidation automatique journalière."""
        import threading
        
        def auto_consolidate_task():
            while True:
                try:
                    # Attendre 24 heures (86400 secondes)
                    # Pour les tests, on peut réduire ce délai
                    import time
                    time.sleep(86400)  # 24 heures
                    logger.info("Démarrage de la consolidation journalière (niveau 4)")
                    self.daily_consolidation()
                    logger.info("Consolidation journalière terminée")
                except Exception as e:
                    logger.error(f"Erreur lors de la consolidation journalière: {e}")
        
        # Démarrer dans un thread séparé
        consolidate_thread = threading.Thread(target=auto_consolidate_task, daemon=True)
        consolidate_thread.start()
        logger.info("Tâche de consolidation journalière démarrée")
    
    def daily_consolidation(self):
        """
        Effectue la consolidation journalière des méta-concepts en connaissances à long terme.
        Applique des techniques d'oubli intelligent et d'auto-réflexion.
        """
        # Récupération des méta-concepts récents (dernières 24h)
        recent_concepts = self.get_recent_level3_concepts()
        
        if not recent_concepts:
            logger.info("Aucun méta-concept récent à consolider")
            return
        
        # Identification des concepts importants
        key_concepts = self.extract_key_concepts(recent_concepts)
        
        # Dérivation de règles (méthode TRAN)
        derived_rules = self.derive_rules_from_patterns(key_concepts)
        
        # Génération d'une réflexion synthétique
        reflection_content, reflection_id = self.generate_reflection(key_concepts, derived_rules)
        
        # Mécanisme d'oubli intelligent (suppression des connaissances obsolètes ou redondantes)
        self.apply_intelligent_forgetting()
        
        # Sauvegarde des données
        self.save_data()
        
        logger.info(f"Consolidation journalière terminée: {len(key_concepts)} concepts clés, {len(derived_rules)} règles")
    
    def get_recent_level3_concepts(self):
        """Récupère les méta-concepts récents du niveau 3."""
        # Temps limite: dernières 24 heures
        time_limit = datetime.now() - timedelta(days=1)
        
        recent_concepts = []
        for concept_id, concept in self.level3_concepts.meta_concepts.items():
            if concept.created_at >= time_limit:
                recent_concepts.append(concept)
        
        return recent_concepts
    
    def extract_key_concepts(self, concepts):
        """
        Identifie les concepts les plus importants à retenir.
        Utilise différentes métriques pour évaluer l'importance.
        """
        if not concepts:
            return []
        
        # Calcul des scores d'importance pour chaque concept
        concept_scores = []
        
        for concept in concepts:
            # Facteurs d'importance
            # 1. Nombre de résumés associés
            summary_count = len(concept.summary_ids)
            
            # 2. Nombre de connexions dans le graphe conceptuel
            connection_count = len(self.level3_concepts.concept_graph.get(concept.id, []))
            
            # 3. Longueur et richesse de la description
            description_length = len(concept.description.split())
            
            # Calcul du score composite
            importance_score = (
                0.4 * summary_count + 
                0.4 * connection_count + 
                0.2 * min(1.0, description_length / 50)
            )
            
            concept_scores.append((concept, importance_score))
        
        # Tri par score d'importance décroissant
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Sélection des concepts les plus importants (max 10)
        top_concepts = [c for c, _ in concept_scores[:min(10, len(concept_scores))]]
        
        return top_concepts
    
    def derive_rules_from_patterns(self, key_concepts):
        """
        Dérive des règles générales à partir des patterns observés.
        Utilise la méthode TRAN (accumulation de règles).
        """
        # Si nous n'avons pas de concepts, pas de règles à dériver
        if not key_concepts:
            return []
        
        derived_rules = []
        
        # Si nous avons un service LLM, l'utiliser pour la dérivation de règles
        if self.llm_service:
            # Préparation des données pour le LLM
            concepts_text = []
            for concept in key_concepts:
                concepts_text.append(f"- {concept.name}: {concept.description}")
            
            concepts_formatted = "\n".join(concepts_text)
            
            # Prompt pour dériver des règles
            prompt = f"""
            Sur la base des méta-concepts suivants:
            
            {concepts_formatted}
            
            Dérive 3 à 5 règles ou principes généraux qui émergent de ces concepts.
            Ces règles doivent être formulées comme des connaissances générales 
            qui pourraient être appliquées à de futures interactions.
            
            Format:
            1. [Règle 1]
            2. [Règle 2]
            3. [Règle 3]
            ...
            """
            
            response = self.llm_service.generate(prompt)
            
            # Extraction des règles (lignes commençant par un nombre suivi d'un point)
            import re
            rule_pattern = r'\d+\.\s+(.+)'
            matches = re.findall(rule_pattern, response)
            
            for rule_text in matches:
                rule_id = str(uuid.uuid4())
                derived_rules.append({
                    "id": rule_id,
                    "content": rule_text.strip(),
                    "created_at": datetime.now(),
                    "source_concepts": [c.id for c in key_concepts],
                    "confidence": 0.8  # Confiance par défaut
                })
        else:
            # Méthode simple de dérivation de règles sans LLM
            # Création d'une règle par concept important
            for concept in key_concepts[:3]:  # Limiter à 3 règles
                rule_id = str(uuid.uuid4())
                rule_content = f"Dans les conversations sur {concept.name}, considérer que {concept.description}"
                
                derived_rules.append({
                    "id": rule_id,
                    "content": rule_content,
                    "created_at": datetime.now(),
                    "source_concepts": [concept.id],
                    "confidence": 0.7  # Confiance plus faible sans LLM
                })
        
        # Stockage des règles dérivées
        for rule in derived_rules:
            self.derived_rules[rule["id"]] = rule
        
        return derived_rules
    
    def generate_reflection(self, key_concepts, derived_rules):
        """
        Génère une réflexion synthétique basée sur les concepts clés et les règles dérivées.
        Utilise le LLM pour la génération de contenu.
        """
        # Génération d'un identifiant unique
        reflection_id = str(uuid.uuid4())
        
        # Si nous avons un service LLM, l'utiliser pour la génération de la réflexion
        if self.llm_service:
            # Préparation des données pour le LLM
            concepts_text = []
            for concept in key_concepts:
                concepts_text.append(f"- {concept.name}: {concept.description}")
            
            rules_text = []
            for rule in derived_rules:
                rules_text.append(f"- {rule['content']}")
            
            concepts_formatted = "\n".join(concepts_text)
            rules_formatted = "\n".join(rules_text)
            
            # Prompt pour la génération de la réflexion
            prompt = f"""
            Sur la base des méta-concepts suivants:
            
            {concepts_formatted}
            
            Et des règles dérivées:
            
            {rules_formatted}
            
            Génère une réflexion synthétique qui intègre ces concepts et règles.
            Cette réflexion doit être une synthèse approfondie qui explore les relations entre ces éléments.
            
            Format:
            [Réflexion]
            """
            
            response = self.llm_service.generate(prompt)
            
            # Extraction de la réflexion
            reflection_content = response.strip()
        else:
            # Méthode simple de génération de réflexion sans LLM
            reflection_content = "Réflexion synthétique sur les concepts clés et les règles dérivées."
        
        # Stockage de la réflexion
        self.long_term_knowledge[reflection_id] = {
            "id": reflection_id,
            "content": reflection_content,
            "created_at": datetime.now(),
            "source_concepts": [c.id for c in key_concepts],
            "source_rules": [r["id"] for r in derived_rules]
        }
        
        return reflection_content, reflection_id
    
    def apply_intelligent_forgetting(self):
        """
        Applique un mécanisme d'oubli intelligent pour supprimer les connaissances obsolètes ou redondantes.
        Utilise des critères de suppression basés sur la pertinence et la fréquence d'utilisation.
        """
        # Critères de suppression
        # 1. Âge des connaissances (supprimer les plus anciennes)
        # 2. Fréquence d'utilisation (supprimer les moins utilisées)
        # 3. Pertinence (supprimer les moins pertinentes)
        
        # Suppression des connaissances obsolètes ou redondantes
        for knowledge_id, knowledge in list(self.long_term_knowledge.items()):
            # Vérifier les critères de suppression
            if knowledge["created_at"] < datetime.now() - timedelta(days=30):
                # Supprimer les connaissances plus anciennes que 30 jours
                del self.long_term_knowledge[knowledge_id]
            elif knowledge["source_concepts"] == [] and knowledge["source_rules"] == []:
                # Supprimer les connaissances sans concepts ou règles associés
                del self.long_term_knowledge[knowledge_id]
            elif knowledge["content"] == "":
                # Supprimer les connaissances avec un contenu vide
                del self.long_term_knowledge[knowledge_id]
        
        # Sauvegarde des données
        self.save_data()
    
    def save_data(self):
        """Sauvegarde les données sur disque."""
        data = {
            "long_term_knowledge": self.long_term_knowledge,
            "derived_rules": self.derived_rules,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()}
        }
        
        # Sauvegarde dans un fichier pickle
        with open(os.path.join(self.data_dir, "level4_knowledge.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    def load_data(self):
        """Charge les données depuis le disque."""
        pickle_path = os.path.join(self.data_dir, "level4_knowledge.pkl")
        
        if not os.path.exists(pickle_path):
            return
        
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Conversion dictionnaire -> objets
            self.long_term_knowledge = data["long_term_knowledge"]
            self.derived_rules = data["derived_rules"]
            self.embeddings = {k: np.array(v) for k, v in data["embeddings"].items()}
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")

class Level5Orchestrator:
    """
    Niveau 5: Orchestrateur Hiérarchique
    
    Ce niveau agit comme un chef d'orchestre entre les différents niveaux de mémoire.
    Il assemble intelligemment les informations pertinentes à partir de tous les niveaux
    pour produire un contexte optimal pour le LLM.
    """
    
    def __init__(self, 
                 short_term_memory, 
                 level2_condenser,
                 level3_concepts,
                 level4_knowledge,
                 use_device: str = DEVICE):
        """
        Initialise l'orchestrateur de la mémoire hiérarchique.
        
        Args:
            short_term_memory: Instance de ShortTermMemory (niveau 1)
            level2_condenser: Instance de Level2Condenser (niveau 2)
            level3_concepts: Instance de Level3HourlyConcepts (niveau 3)
            level4_knowledge: Instance de Level4DailyKnowledge (niveau 4)
            use_device: Device à utiliser (mps, cuda, cpu)
        """
        self.short_term_memory = short_term_memory
        self.level2_condenser = level2_condenser
        self.level3_concepts = level3_concepts
        self.level4_knowledge = level4_knowledge
        self.device = use_device
        
        logger.info("Orchestrateur de mémoire hiérarchique initialisé")
    
    def compose_context(self, query: str, max_tokens: int = 2048) -> str:
        """
        Compose un contexte optimal en assemblant les informations pertinentes
        des différents niveaux de mémoire, en fonction de la requête.
        
        Args:
            query: Requête utilisateur
            max_tokens: Nombre maximum de tokens pour le contexte (approximatif)
            
        Returns:
            Contexte optimal pour le LLM
        """
        # Allocation des tokens par niveau (approximatif)
        # - Niveau 1 (court terme): 40%
        # - Niveau 2 (résumés): 25%
        # - Niveau 3 (méta-concepts): 20%
        # - Niveau 4 (long terme): 15%
        
        st_tokens = int(max_tokens * 0.4)
        l2_tokens = int(max_tokens * 0.25)
        l3_tokens = int(max_tokens * 0.2)
        l4_tokens = int(max_tokens * 0.15)
        
        # Récupération des informations de chaque niveau
        # Ces récupérations peuvent s'exécuter en parallèle avec asyncio ou Thread
        # pour une meilleure efficacité sur GPU
        
        # 1. Récupération des conversations récentes (niveau 1)
        recent_convos = self.get_relevant_short_term_memory(query, token_limit=st_tokens)
        
        # 2. Récupération des résumés pertinents (niveau 2)
        summaries = self.get_relevant_summaries(query, token_limit=l2_tokens)
        
        # 3. Récupération des méta-concepts pertinents (niveau 3)
        concepts = self.get_relevant_concepts(query, token_limit=l3_tokens)
        
        # 4. Récupération des connaissances à long terme (niveau 4)
        knowledge = self.get_relevant_knowledge(query, token_limit=l4_tokens)
        
        # Composition du contexte final
        context_parts = []
        
        # Section 1: Connaissances à long terme (niveau 4)
        if knowledge:
            context_parts.append("## Connaissances à long terme")
            context_parts.append(knowledge)
            context_parts.append("")
        
        # Section 2: Méta-concepts (niveau 3)
        if concepts:
            context_parts.append("## Concepts pertinents")
            context_parts.append(concepts)
            context_parts.append("")
        
        # Section 3: Résumés (niveau 2)
        if summaries:
            context_parts.append("## Résumés des conversations précédentes")
            context_parts.append(summaries)
            context_parts.append("")
        
        # Section 4: Conversations récentes (niveau 1)
        if recent_convos:
            context_parts.append("## Conversations récentes")
            context_parts.append(recent_convos)
            context_parts.append("")
        
        # Assemblage du contexte final
        composed_context = "\n".join(context_parts)
        
        return composed_context
    
    def get_relevant_short_term_memory(self, query: str, token_limit: int = 800) -> str:
        """
        Récupère les conversations récentes pertinentes.
        
        Args:
            query: Requête utilisateur
            token_limit: Limite approximative de tokens
            
        Returns:
            Texte formaté des conversations récentes
        """
        # Récupération des conversations similaires
        similar_convos = self.short_term_memory.get_similar_conversations(query, k=5)
        
        # Formatage des conversations
        formatted_convos = []
        total_length = 0
        avg_token_len = 4  # Estimation du nombre de caractères par token
        
        for convo in similar_convos:
            user_text = convo.get('user_input', convo.get('prompt_content', ''))
            system_text = convo.get('system_response', convo.get('response_content', ''))
            
            formatted_convo = f"Utilisateur: {user_text}\nRenée: {system_text}"
            convo_length = len(formatted_convo) // avg_token_len
            
            if total_length + convo_length <= token_limit:
                formatted_convos.append(formatted_convo)
                total_length += convo_length
            else:
                break
        
        return "\n\n".join(formatted_convos)
    
    def get_relevant_summaries(self, query: str, token_limit: int = 500) -> str:
        """
        Récupère les résumés pertinents.
        
        Args:
            query: Requête utilisateur
            token_limit: Limite approximative de tokens
            
        Returns:
            Texte formaté des résumés
        """
        # Récupération des résumés pertinents
        summaries = self.level2_condenser.get_relevant_summaries(query, k=3)
        
        # Formatage des résumés
        formatted_summaries = []
        total_length = 0
        avg_token_len = 4  # Estimation du nombre de caractères par token
        
        for summary in summaries:
            summary_text = summary.get('content', '')
            summary_length = len(summary_text) // avg_token_len
            
            if total_length + summary_length <= token_limit:
                formatted_summaries.append(f"- {summary_text}")
                total_length += summary_length
            else:
                break
        
        return "\n".join(formatted_summaries)
    
    def get_relevant_concepts(self, query: str, token_limit: int = 400) -> str:
        """
        Récupère les méta-concepts pertinents.
        
        Args:
            query: Requête utilisateur
            token_limit: Limite approximative de tokens
            
        Returns:
            Texte formaté des méta-concepts
        """
        # Récupération des concepts pertinents
        concepts = self.level3_concepts.get_relevant_concepts(query, k=3)
        
        # Formatage des concepts
        formatted_concepts = []
        total_length = 0
        avg_token_len = 4  # Estimation du nombre de caractères par token
        
        for concept in concepts:
            concept_name = concept.get('name', '')
            concept_desc = concept.get('description', '')
            
            formatted_concept = f"- **{concept_name}**: {concept_desc}"
            concept_length = len(formatted_concept) // avg_token_len
            
            if total_length + concept_length <= token_limit:
                formatted_concepts.append(formatted_concept)
                total_length += concept_length
            else:
                break
        
        return "\n".join(formatted_concepts)
    
    def get_relevant_knowledge(self, query: str, token_limit: int = 300) -> str:
        """
        Récupère les connaissances à long terme pertinentes.
        
        Args:
            query: Requête utilisateur
            token_limit: Limite approximative de tokens
            
        Returns:
            Texte formaté des connaissances
        """
        # Pour l'instant, nous n'avons pas d'index de recherche spécifique pour le niveau 4
        # Nous allons donc utiliser une approche simplifiée basée sur les règles dérivées
        # et nous compléterons plus tard avec une recherche sémantique
        
        # Mise à jour des pondérations entre règles et connaissances générales
        if "question" in query.lower() or "comment" in query.lower():
            # Pour les questions, favoriser les connaissances générales
            rules_weight = 0.3
            knowledge_weight = 0.7
        else:
            # Pour les autres requêtes, équilibrer
            rules_weight = 0.5
            knowledge_weight = 0.5
        
        # Préparation des connaissances et règles
        formatted_items = []
        total_length = 0
        avg_token_len = 4  # Estimation du nombre de caractères par token
        
        # Budget pour chaque section
        rules_budget = int(token_limit * rules_weight)
        knowledge_budget = int(token_limit * knowledge_weight)
        
        # Ajout des règles dérivées (si disponibles)
        if self.level4_knowledge.derived_rules:
            recent_rules = sorted(
                self.level4_knowledge.derived_rules.values(),
                key=lambda x: x["created_at"],
                reverse=True
            )[:3]  # Prendre les 3 règles les plus récentes
            
            for rule in recent_rules:
                rule_text = f"- Règle: {rule['content']}"
                rule_length = len(rule_text) // avg_token_len
                
                if total_length + rule_length <= rules_budget:
                    formatted_items.append(rule_text)
                    total_length += rule_length
                else:
                    break
        
        # Ajout des connaissances à long terme (si disponibles)
        if self.level4_knowledge.long_term_knowledge:
            recent_knowledge = sorted(
                self.level4_knowledge.long_term_knowledge.values(),
                key=lambda x: x["created_at"],
                reverse=True
            )[:2]  # Prendre les 2 connaissances les plus récentes
            
            for knowledge in recent_knowledge:
                knowledge_text = f"- {knowledge['content']}"
                knowledge_length = len(knowledge_text) // avg_token_len
                
                if total_length + knowledge_length <= token_limit:
                    formatted_items.append(knowledge_text)
                    total_length += knowledge_length
                else:
                    break
        
        return "\n".join(formatted_items)

class ShortTermMemory:
    # ...

    def get_similar_conversations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les k conversations les plus similaires à une requête.
        Utilise l'index FAISS pour une recherche efficace.
        """
        if not self.conversations:
            return []
        
        # Générer l'embedding de la requête
        query_embedding = self.generate_embedding(query)
        
        # Recherche dans l'index FAISS
        if hasattr(self, 'gpu_index'):
            D, I = self.gpu_index.search(np.array([query_embedding]), k)
        else:
            D, I = self.index.search(np.array([query_embedding]), k)
        
        # Reconstruire les résultats
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0 and idx < len(self.conversations):  # Vérifier la validité de l'index
                conv_id = list(self.conversations.keys())[idx]
                conv = self.conversations[conv_id]
                
                results.append({
                    "id": conv.id,
                    "similarity": float(D[0][i]),
                    "user_input": conv.prompt.content,
                    "system_response": conv.response.content,
                    "created_at": conv.created_at.isoformat(),
                    "metadata": conv.metadata
                })
        
        return results

class Level2Condenser:
    # ...

    def condense_recent_memory(self) -> List[Summary]:
        """
        Condense les conversations récentes en résumés en utilisant BERTopic.
        Optimisé pour l'exécution avec MPS/GPU.
        """
        try:
            # Récupération des conversations récentes
            recent_conversations = self.short_term_memory.get_recent_conversations(limit=50)
            
            if not recent_conversations:
                return []
            
            # Préparation des données pour le clustering
            conversation_texts = [
                f"User: {conv.prompt.content}\nRenée: {conv.response.content}"
                for conv in recent_conversations
            ]
            
            # Génération des embeddings par lots (optimisé pour le device)
            embeddings = self.short_term_memory.batch_generate_embeddings(conversation_texts)
            
            # Configuration de UMAP pour la réduction de dimension
            try:
                from umap import UMAP
                import hdbscan
                from sklearn.feature_extraction.text import CountVectorizer
                from bertopic import BERTopic
                
                # Configuration de UMAP optimisée pour le device choisi
                if self.device == "cuda" and torch.cuda.is_available():
                    import cuml
                    umap_model = cuml.manifold.UMAP(
                        n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        n_jobs=-1
                    )
                else:
                    # Configuration standard pour CPU/MPS
                    umap_model = UMAP(
                        n_neighbors=15,
                        n_components=5,
                        min_dist=0.0,
                        metric='cosine',
                        low_memory=True,
                        random_state=42
                    )
                
                # Configuration du clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=2,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Configuration du vectoriseur
                vectorizer = CountVectorizer(stop_words="english")
                
                # Initialisation de BERTopic
                topic_model = BERTopic(
                    embedding_model=None,  # On utilise nos propres embeddings
                    umap_model=umap_model,
                    hdbscan_model=clusterer,
                    vectorizer_model=vectorizer,
                    calculate_probabilities=True,
                    verbose=True
                )
                
                # Extraction des thèmes
                topics, probs = topic_model.fit_transform(
                    conversation_texts, 
                    embeddings=embeddings
                )
                
            except ImportError as e:
                logger.error(f"Erreur lors du chargement des modules de clustering: {e}")
                # Fallback sur une méthode de clustering plus simple
                topics = self._fallback_clustering(embeddings)
                probs = np.ones((len(topics), 1))  # Probabilités fictives
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction des thèmes: {e}")
                # Fallback sur une méthode de clustering plus simple
                topics = self._fallback_clustering(embeddings)
                probs = np.ones((len(topics), 1))  # Probabilités fictives
            
            # Générer des résumés pour chaque cluster
            summaries = []
            unique_clusters = set(topics)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # -1 est le cluster de bruit dans HDBSCAN
                    continue
                
                # Sélectionner les conversations du cluster
                cluster_indices = [i for i, topic in enumerate(topics) if topic == cluster_id]
                cluster_conversations = [recent_conversations[i] for i in cluster_indices]
                
                if not cluster_conversations:
                    continue
                
                # Créer un prompt pour le résumé
                cluster_text = "\n\n".join([
                    f"User: {conv.prompt.content}\nRenée: {conv.response.content}"
                    for conv in cluster_conversations
                ])
                
                # Générer le résumé (simulé ici - à remplacer par un appel au LLM)
                summary_text = self._generate_summary_text(cluster_text, topic_model, cluster_id)
                
                # Créer l'objet Summary
                summary = Summary(
                    id=str(uuid.uuid4()),
                    content=summary_text,
                    created_at=datetime.now(),
                    conversation_ids=[conv.id for conv in cluster_conversations]
                )
                
                # Générer et stocker l'embedding du résumé
                summary_embedding = self.short_term_memory.generate_embedding(summary_text)
                
                # Ajouter à l'index FAISS
                if hasattr(self, 'gpu_index'):
                    self.gpu_index.add(np.array([summary_embedding]))
                else:
                    self.index.add(np.array([summary_embedding]))
                
                self.summaries[summary.id] = summary
                self.embeddings[summary.id] = summary_embedding
                
                summaries.append(summary)
            
            # Sauvegarder les données
            self.save_data()
            
            return summaries
            
        except Exception as e:
            logger.error(f"Erreur lors de la condensation des mémoires: {e}")
            return []
