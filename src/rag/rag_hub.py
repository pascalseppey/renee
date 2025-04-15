# rag_hub.py
# Implémentation du hub RAG Auto-Correctif pour Renée

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configuration du logging
logger = logging.getLogger(__name__)

# Chargement de la configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config/config.json")
with open(config_path) as f:
    config = json.load(f)

class RAGHub:
    """
    Hub RAG (Retrieval-Augmented Generation) Auto-Correctif pour Renée.
    Cette classe gère la récupération d'informations pertinentes à partir de sources diverses.
    """
    
    def __init__(self, vector_db: str = "qdrant", 
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 reranking_enabled: bool = True):
        """
        Initialise le hub RAG avec les paramètres spécifiés.
        
        Args:
            vector_db: Base de données vectorielle à utiliser ('qdrant', 'chroma', etc.)
            embedding_model: Modèle d'embedding à utiliser
            reranking_enabled: Activer ou non le re-ranking des résultats
        """
        self.vector_db_type = vector_db
        self.embedding_model_name = embedding_model
        self.reranking_enabled = reranking_enabled
        
        # Chargement des dépendances en fonction de la configuration
        self._load_dependencies()
        
        # Création de la collection si elle n'existe pas
        self._ensure_collection_exists()
        
        logger.info(f"RAGHub initialisé avec {vector_db} et {embedding_model}")
    
    def _load_dependencies(self):
        """
        Charge les dépendances nécessaires en fonction de la configuration.
        Utilise le lazy loading pour éviter de charger des modules inutiles.
        """
        try:
            # Chargement du modèle d'embedding
            from sentence_transformers import SentenceTransformer
            
            # Détermination du device (mps, cuda, cpu)
            device_config = config.get("acceleration", {})
            use_mps = device_config.get("use_mps", False)
            use_gpu = device_config.get("use_gpu", False)
            
            if use_mps and hasattr(import_module('torch').backends, 'mps') and import_module('torch').backends.mps.is_available():
                device = "mps"
            elif use_gpu and import_module('torch').cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
            # Chargement du modèle d'embedding
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            
            # Initialisation de la base de données vectorielle
            if self.vector_db_type.lower() == "qdrant":
                from qdrant_client import QdrantClient
                self.vector_db = QdrantClient(":memory:")  # Mode mémoire pour les tests
                logger.info("Base de données vectorielle Qdrant initialisée")
            elif self.vector_db_type.lower() == "chroma":
                import chromadb
                self.vector_db = chromadb.Client()
                logger.info("Base de données vectorielle ChromaDB initialisée")
            else:
                raise ValueError(f"Base de données vectorielle non supportée: {self.vector_db_type}")
            
            # Chargement du modèle de re-ranking si activé
            if self.reranking_enabled:
                try:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
                    logger.info("Modèle de re-ranking chargé")
                except ImportError:
                    logger.warning("Impossible de charger le modèle de re-ranking, fonctionnalité désactivée")
                    self.reranking_enabled = False
        
        except ImportError as e:
            logger.error(f"Erreur lors du chargement des dépendances: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """
        Assure que la collection nécessaire existe dans la base de données vectorielle.
        Crée la collection si elle n'existe pas déjà.
        """
        try:
            if self.vector_db_type.lower() == "qdrant":
                from qdrant_client.models import VectorParams, Distance
                
                # Vérification si la collection existe
                collections = self.vector_db.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                # Création de la collection si elle n'existe pas
                if "knowledge_base" not in collection_names:
                    # Détermination de la dimension du modèle d'embedding
                    test_embedding = self.embedding_model.encode("Test text")
                    dimension = len(test_embedding)
                    
                    # Création de la collection
                    self.vector_db.create_collection(
                        collection_name="knowledge_base",
                        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                    )
                    logger.info("Collection 'knowledge_base' créée dans Qdrant")
            
            elif self.vector_db_type.lower() == "chroma":
                # Vérification si la collection existe
                try:
                    self.vector_db.get_collection("knowledge_base")
                except Exception:
                    # Création de la collection
                    self.vector_db.create_collection(name="knowledge_base")
                    logger.info("Collection 'knowledge_base' créée dans ChromaDB")
        
        except Exception as e:
            logger.error(f"Erreur lors de la création de la collection: {e}")
            raise
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Effectue une requête dans le système RAG.
        
        Args:
            query_text: Texte de la requête
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste des résultats les plus pertinents
        """
        try:
            # Génération de l'embedding de la requête
            query_embedding = self.embedding_model.encode(query_text)
            
            # Recherche dans la base de données vectorielle
            if self.vector_db_type.lower() == "qdrant":
                # Exemple pour Qdrant
                results = self.vector_db.search(
                    collection_name="knowledge_base",
                    query_vector=query_embedding.tolist(),
                    limit=top_k * 2 if self.reranking_enabled else top_k
                )
            elif self.vector_db_type.lower() == "chroma":
                # Exemple pour ChromaDB
                results = self.vector_db.collection("knowledge_base").query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k * 2 if self.reranking_enabled else top_k
                )
            
            # Application du re-ranking si activé
            if self.reranking_enabled:
                results = self._rerank_results(query_text, results, top_k)
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de la requête RAG: {e}")
            return []
    
    def _rerank_results(self, query_text: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Applique le re-ranking aux résultats initiaux pour améliorer la pertinence.
        
        Args:
            query_text: Texte de la requête
            results: Résultats initiaux de la recherche vectorielle
            top_k: Nombre de résultats à retourner après re-ranking
            
        Returns:
            Liste des résultats re-classés
        """
        # Préparation des paires pour le re-ranking
        pairs = [(query_text, result.get('content', '')) for result in results]
        
        # Calcul des scores de similarité
        scores = self.reranker.predict(pairs)
        
        # Tri des résultats par score
        reranked_results = [(score, result) for score, result in zip(scores, results)]
        reranked_results.sort(reverse=True, key=lambda x: x[0])
        
        # Retourner les top_k meilleurs résultats
        return [result for _, result in reranked_results[:top_k]]
    
    def add_document(self, document_content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ajoute un document à la base de connaissances.
        
        Args:
            document_content: Contenu du document
            metadata: Métadonnées associées au document
            
        Returns:
            Identifiant du document ajouté
        """
        try:
            # Génération de l'embedding du document
            document_embedding = self.embedding_model.encode(document_content)
            
            # Génération d'un identifiant unique
            import uuid
            doc_id = str(uuid.uuid4())
            
            # Ajout à la base de données vectorielle
            if self.vector_db_type.lower() == "qdrant":
                # Exemple pour Qdrant
                from qdrant_client.models import PointStruct
                
                # Création d'un point avec PointStruct au lieu d'un dictionnaire
                point = PointStruct(
                    id=doc_id,
                    vector=document_embedding.tolist(),
                    payload={
                        "content": document_content,
                        "metadata": metadata or {}
                    }
                )
                
                self.vector_db.upsert(
                    collection_name="knowledge_base",
                    points=[point]
                )
            elif self.vector_db_type.lower() == "chroma":
                # Exemple pour ChromaDB
                self.vector_db.collection("knowledge_base").add(
                    ids=[doc_id],
                    embeddings=[document_embedding.tolist()],
                    metadatas=[{"content": document_content, **(metadata or {})}]
                )
            
            logger.info(f"Document ajouté avec succès, ID: {doc_id}")
            return doc_id
        
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du document: {e}")
            return None
    
    def reset_collection(self):
        """
        Réinitialise la collection de la base de données vectorielle en supprimant
        tous les documents et en recréant la collection.
        
        Returns:
            bool: True si la réinitialisation a réussi, False sinon
        """
        try:
            if self.vector_db_type.lower() == "qdrant":
                # Supprimer la collection si elle existe
                collections = self.vector_db.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if "knowledge_base" in collection_names:
                    # Supprimer l'ancienne collection
                    self.vector_db.delete_collection(collection_name="knowledge_base")
                    logger.info("Collection 'knowledge_base' supprimée dans Qdrant")
                
                # Recréer la collection
                from qdrant_client.models import VectorParams, Distance
                
                # Détermination de la dimension du modèle d'embedding
                test_embedding = self.embedding_model.encode("Test text")
                dimension = len(test_embedding)
                
                # Création de la collection
                self.vector_db.create_collection(
                    collection_name="knowledge_base",
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info("Collection 'knowledge_base' recréée dans Qdrant")
            
            elif self.vector_db_type.lower() == "chroma":
                # Supprimer et recréer la collection pour ChromaDB
                try:
                    self.vector_db.delete_collection("knowledge_base")
                    logger.info("Collection 'knowledge_base' supprimée dans ChromaDB")
                except Exception:
                    pass  # La collection n'existait peut-être pas
                
                # Création de la collection
                self.vector_db.create_collection(name="knowledge_base")
                logger.info("Collection 'knowledge_base' recréée dans ChromaDB")
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation de la collection: {e}")
            return False
