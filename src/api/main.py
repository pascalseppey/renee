from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import sys
import logging
import json
import asyncio
from typing import Dict, Any, Optional

# Ajout du chemin du projet au sys.path pour les importations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import des routes directement avec des chemins relatifs
from routes import include_routers
from wordpress_routes import router as wordpress_router

# Import du système RAG et des connecteurs
try:
    from rag.rag_hub import RAGHub
    from connectors.wordpress_connector import WordPressConnector
except ImportError:
    # Chemin alternatif si on exécute depuis un autre répertoire
    sys.path.insert(0, os.path.dirname(current_dir))
    from rag.rag_hub import RAGHub
    from connectors.wordpress_connector import WordPressConnector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs/api.log"), mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("ReneeAPI")

# Création du dossier de logs s'il n'existe pas
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

# Chargement de la configuration
config_path = os.path.join(project_root, "config/config.json")
try:
    with open(config_path) as f:
        config = json.load(f)
except FileNotFoundError:
    logger.warning(f"Fichier de configuration introuvable: {config_path}")
    config = {
        "components": {
            "rag": {
                "vector_db": "chroma",
                "embedding_model": "local",
                "reranking_enabled": False
            }
        }
    }

# Initialisation du RAG Hub pour les requêtes
try:
    rag_hub = RAGHub(
        vector_db=config["components"]["rag"]["vector_db"],
        embedding_model=config["components"]["rag"]["embedding_model"],
        reranking_enabled=config["components"]["rag"]["reranking_enabled"]
    )
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation du RAG Hub: {e}")
    # Initialisation par défaut en cas d'erreur
    rag_hub = None

# Modèles de données pour le chat
class ChatRequest(BaseModel):
    message: str
    wordpress_site: Optional[str] = None

# Création de l'application
app = FastAPI(
    title="Renée API",
    description="API pour interagir avec le système Renée",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À ajuster en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Enregistrement des routers
include_routers(app)
app.include_router(wordpress_router)

# Route principale - servir l'interface web
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open(os.path.join(current_dir, "static/index.html"), "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("Fichier index.html introuvable")
        return HTMLResponse(content="<html><body><h1>Erreur: Fichier index.html introuvable</h1></body></html>")

# Route de statut
@app.get("/status")
async def status():
    return {
        "name": "Renée API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }

# Route de chat avec Renée
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Log la requête
        logger.info(f"Requête de chat reçue: {request.message}")
        
        # Contexte de la requête
        context = {
            "wordpress_site": request.wordpress_site
        }
        
        # Récupérer des informations du RAG si disponible
        rag_context = ""
        if rag_hub:
            try:
                # Récupérer des informations du RAG
                rag_results = await asyncio.to_thread(
                    rag_hub.query,
                    query_text=request.message,
                    top_k=3
                )
                
                # Formater les résultats du RAG pour le contexte
                if rag_results:
                    rag_context = "Informations pertinentes de la base de connaissances:\n\n"
                    for i, result in enumerate(rag_results):
                        content = result.get("payload", {}).get("content", "")
                        # Limiter la taille du contenu pour éviter des réponses trop longues
                        if len(content) > 500:
                            content = content[:500] + "..."
                        rag_context += f"{i+1}. {content}\n\n"
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des informations du RAG: {e}")
        
        # Préparer la réponse en fonction de la demande
        if "wordpress" in request.message.lower() or "elementor" in request.message.lower():
            # Vérifier si un site WordPress est spécifié
            if not request.wordpress_site:
                return {
                    "response": "Pour interagir avec WordPress, veuillez d'abord sélectionner un site dans le panneau de droite."
                }
            
            # Si c'est une demande WordPress, préparer une réponse spécifique
            if "crée" in request.message.lower() or "créer" in request.message.lower() or "ajoute" in request.message.lower():
                return {
                    "response": f"Je vais vous aider à créer du contenu sur votre site WordPress '{request.wordpress_site}'.\n\nVoici ce que je comprends de votre demande :\n\n{request.message}\n\nJe vais utiliser les informations de la base de connaissances pour réaliser cette tâche de manière optimale.\n\nJe vais maintenant procéder à la modification. Veuillez patienter..."
                }
            elif "modifie" in request.message.lower() or "change" in request.message.lower() or "mettre à jour" in request.message.lower():
                return {
                    "response": f"Je vais vous aider à modifier votre site WordPress '{request.wordpress_site}'.\n\nVoici ce que je comprends de votre demande :\n\n{request.message}\n\nJe vais utiliser les informations de la base de connaissances sur WordPress et Elementor pour réaliser cette tâche correctement.\n\nJe vais maintenant procéder à la modification. Veuillez patienter..."
                }
            else:
                return {
                    "response": f"Je suis prêt à vous aider avec votre site WordPress '{request.wordpress_site}'.\n\nPour commencer, pourriez-vous me préciser ce que vous souhaitez que je fasse? Par exemple:\n- Créer une nouvelle page\n- Modifier le design d'une page existante\n- Ajouter des fonctionnalités\n\nJ'ai accès à votre site et à une base de connaissances sur WordPress et Elementor pour vous aider efficacement."
                }
        else:
            # Réponse générique pour les autres types de demandes
            return {
                "response": f"Bonjour! Je suis Renée, votre assistant IA spécialisé en WordPress et Elementor.\n\nJe peux vous aider à modifier votre site web, créer des pages, personnaliser votre design et bien plus encore.\n\nPour commencer, veuillez sélectionner votre site WordPress dans le panneau de droite et me préciser ce que vous souhaitez faire."
            }
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête de chat: {e}")
        return {
            "response": "Désolé, une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
