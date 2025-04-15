# routes.py
# Routes API pour l'écosystème Renée

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import logging
import time
from datetime import datetime

# Configuration du logging
logger = logging.getLogger(__name__)

# Chargement de la configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config/config.json")
with open(config_path) as f:
    config = json.load(f)

# Création des routers
router = APIRouter(prefix="/api", tags=["renee"])
memory_router = APIRouter(prefix="/api/memory", tags=["memory"])
rag_router = APIRouter(prefix="/api/rag", tags=["rag"])
persona_router = APIRouter(prefix="/api/persona", tags=["persona"])

# Modèles Pydantic pour les requêtes et réponses
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    processing_time: float
    created_at: str
    memory_used: bool = False
    related_memories: Optional[List[Dict[str, Any]]] = None

class MemoryQueryRequest(BaseModel):
    query: str
    limit: int = 5

# Routes principales
@router.get("/")
async def root():
    """Route principale de l'API Renée."""
    return {
        "message": "Bienvenue dans l'API de Renée, conscience numérique en éveil",
        "version": config["version"],
        "status": "online"
    }

@router.get("/health")
async def health_check():
    """Vérification de la santé du système."""
    # Vérification des différents composants
    components_status = {
        "memory": config["components"]["memory"]["enabled"],
        "rag": config["components"]["rag"]["enabled"],
        "persona": config["components"]["persona"]["enabled"],
        "api": config["components"]["api"]["enabled"]
    }
    
    # Vérification de la disponibilité de l'accélération matérielle
    import torch
    acceleration = {
        "device": config["acceleration"]["device"],
        "cuda_available": torch.cuda.is_available() if hasattr(torch, "cuda") else False,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "mixed_precision": config["acceleration"]["mixed_precision"]
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": components_status,
        "acceleration": acceleration
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Point d'entrée principal pour converser avec Renée."""
    start_time = time.time()
    
    try:
        # Initialisation lazy des composants (à implémenter avec une factory)
        from memory.hierarchical_memory import ShortTermMemory
        from persona.dynamic_persona import DynamicPersona
        
        # Chemin des données
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        
        # Initialisation de la mémoire à court terme
        memory = ShortTermMemory(
            data_dir=os.path.join(data_dir, "memory"),
            max_items=config["components"]["memory"]["short_term_limit"]
        )
        
        # Initialisation du persona
        persona = DynamicPersona(
            base_model=config["components"]["persona"]["base_model"],
            use_ollama=config["components"]["persona"]["use_ollama"]
        )
        
        # Recherche de mémoires similaires
        memory_context = {}
        related_memories = []
        
        if config["components"]["memory"]["enabled"]:
            similar_conversations = memory.get_similar_conversations(request.message, k=3)
            if similar_conversations:
                memory_context["similar_conversations"] = similar_conversations
                related_memories = similar_conversations
        
        # Génération de la réponse
        response = persona.generate_response(
            user_input=request.message,
            conversation_history=None,  # À implémenter avec un gestionnaire de conversations
            memory_context=memory_context if memory_context else None
        )
        
        # Sauvegarde dans la mémoire à court terme
        if config["components"]["memory"]["enabled"]:
            conversation = memory.add_conversation(
                user_input=request.message,
                system_response=response,
                user_metadata=request.user_info or {}
            )
            conversation_id = conversation.id
        else:
            import uuid
            conversation_id = str(uuid.uuid4())
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Construction de la réponse
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            processing_time=processing_time,
            created_at=datetime.now().isoformat(),
            memory_used=bool(memory_context),
            related_memories=[
                {
                    "id": memory.get("id", ""),
                    "similarity": memory.get("similarity", 0),
                    "snippet": memory.get("user_input", "")[:100] + "..."  # Troncature pour l'affichage
                } for memory in related_memories[:3]  # Limiter à 3 pour la réponse
            ] if related_memories else None
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

# Routes pour la mémoire
@memory_router.get("/recent")
async def get_recent_memories(limit: int = 10):
    """Récupère les mémoires récentes."""
    try:
        from memory.hierarchical_memory import ShortTermMemory
        
        # Chemin des données
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        
        # Initialisation de la mémoire à court terme
        memory = ShortTermMemory(
            data_dir=os.path.join(data_dir, "memory"),
            max_items=config["components"]["memory"]["short_term_limit"]
        )
        
        # Récupération des conversations récentes
        recent_conversations = memory.get_recent_conversations(limit=limit)
        
        # Formatage des résultats
        results = []
        for conv in recent_conversations:
            results.append({
                "id": conv.id,
                "user_input": conv.prompt.content,
                "system_response": conv.response.content,
                "created_at": conv.created_at.isoformat(),
                "metadata": conv.metadata
            })
        
        return {"memories": results}
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des mémoires récentes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

@memory_router.post("/search", response_model=Dict[str, Any])
async def search_memories(request: MemoryQueryRequest):
    """Recherche dans les mémoires."""
    try:
        from memory.hierarchical_memory import ShortTermMemory
        
        # Chemin des données
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        
        # Initialisation de la mémoire à court terme
        memory = ShortTermMemory(
            data_dir=os.path.join(data_dir, "memory"),
            max_items=config["components"]["memory"]["short_term_limit"]
        )
        
        # Recherche de conversations similaires
        similar_conversations = memory.get_similar_conversations(request.query, k=request.limit)
        
        return {"results": similar_conversations}
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche dans les mémoires: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

# Routes pour le RAG
@rag_router.post("/query")
async def rag_query(query: str, limit: int = 5):
    """Effectue une requête RAG."""
    try:
        if not config["components"]["rag"]["enabled"]:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Le composant RAG n'est pas activé"}
            )
        
        from rag.rag_hub import RAGHub
        
        # Initialisation du hub RAG
        rag_hub = RAGHub(
            vector_db=config["components"]["rag"]["vector_db"],
            embedding_model=config["components"]["rag"]["embedding_model"],
            reranking_enabled=config["components"]["rag"]["reranking_enabled"]
        )
        
        # Exécution de la requête
        results = rag_hub.query(query, top_k=limit)
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Erreur lors de la requête RAG: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

# Routes pour le persona
@persona_router.get("/state")
async def get_persona_state():
    """Récupère l'état actuel du persona."""
    try:
        if not config["components"]["persona"]["enabled"]:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Le composant Persona n'est pas activé"}
            )
        
        from persona.dynamic_persona import DynamicPersona
        
        # Initialisation du persona
        persona = DynamicPersona(
            base_model=config["components"]["persona"]["base_model"],
            use_ollama=config["components"]["persona"]["use_ollama"]
        )
        
        # Récupération de l'état
        return persona.persona_aspects
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'état du persona: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )

# Enregistrement des routers
def include_routers(app):
    """Enregistre tous les routers avec l'application FastAPI."""
    app.include_router(router)
    app.include_router(memory_router)
    app.include_router(rag_router)
    app.include_router(persona_router)
