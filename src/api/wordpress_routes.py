from fastapi import APIRouter, HTTPException, Depends, Form, Body, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# Ajout du chemin du projet au sys.path pour les importations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import du connecteur WordPress
try:
    from connectors.wordpress_connector import WordPressConnector, WordPressError
except ImportError:
    # Chemin alternatif si on exécute depuis un autre répertoire
    sys.path.insert(0, os.path.dirname(current_dir))
    from connectors.wordpress_connector import WordPressConnector, WordPressError

# Configuration du logging
logger = logging.getLogger("ReneeAPI")

# Création du router
router = APIRouter(
    prefix="/wordpress",
    tags=["wordpress"],
    responses={404: {"description": "Not found"}},
)

# Chemin vers le fichier de configuration des sites WordPress
CREDENTIALS_PATH = os.path.join(project_root, "config/wordpress_credentials.json")

# Modèles de données
class WordPressCredentials(BaseModel):
    site_url: HttpUrl = Field(..., description="URL du site WordPress")
    username: str = Field(..., description="Nom d'utilisateur WordPress")
    app_password: str = Field(..., description="Mot de passe d'application")

class WordPressSite(BaseModel):
    name: str = Field(..., description="Nom du site WordPress")
    url: HttpUrl = Field(..., description="URL du site WordPress")
    username: Optional[str] = Field(None, description="Nom d'utilisateur WordPress")
    app_password: Optional[str] = Field(None, description="Mot de passe d'application")

class WordPressPost(BaseModel):
    title: str = Field(..., description="Titre de l'article")
    content: str = Field(..., description="Contenu de l'article en HTML")
    status: str = Field("draft", description="Statut de l'article (draft, publish, etc.)")
    excerpt: Optional[str] = Field(None, description="Extrait de l'article")
    categories: Optional[List[int]] = Field(None, description="IDs des catégories")
    tags: Optional[List[int]] = Field(None, description="IDs des tags")

class WordPressPage(BaseModel):
    title: str = Field(..., description="Titre de la page")
    content: str = Field(..., description="Contenu de la page en HTML")
    status: str = Field("draft", description="Statut de la page (draft, publish, etc.)")
    parent: Optional[int] = Field(None, description="ID de la page parente")
    template: Optional[str] = Field(None, description="Template utilisé par la page")


# Fonction pour obtenir un connecteur WordPress
async def get_wordpress_connector(
    site_url: HttpUrl = Query(..., description="URL du site WordPress"),
    username: Optional[str] = Query(None, description="Nom d'utilisateur WordPress"),
    app_password: Optional[str] = Query(None, description="Mot de passe d'application"),
    site_name: Optional[str] = Query(None, description="Nom du site dans la configuration")
):
    """
    Crée un connecteur WordPress avec les identifiants fournis
    ou récupérés depuis la configuration.
    """
    # Si un nom de site est fourni, on essaie de récupérer les identifiants
    if site_name:
        try:
            # Charger la configuration
            with open(CREDENTIALS_PATH, 'r') as f:
                config = json.load(f)
            
            # Rechercher le site dans la configuration
            site_config = next(
                (site for site in config.get("sites", []) if site.get("name") == site_name),
                None
            )
            
            if site_config:
                site_url = site_url or site_config.get("url")
                username = username or site_config.get("username")
                app_password = app_password or site_config.get("app_password")
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des identifiants: {e}")
    
    # Créer le connecteur
    connector = WordPressConnector(
        site_url=site_url,
        username=username,
        app_password=app_password
    )
    
    # Vérifier la connexion
    if not await connector.verify_connection():
        raise HTTPException(
            status_code=401,
            detail="Impossible de se connecter au site WordPress avec les identifiants fournis"
        )
    
    return connector


# Routes API
@router.get("/sites", response_model=List[WordPressSite])
async def list_sites():
    """Liste les sites WordPress configurés."""
    try:
        # Charger la configuration
        with open(CREDENTIALS_PATH, 'r') as f:
            config = json.load(f)
        
        # Masquer les mots de passe dans la réponse
        sites = []
        for site in config.get("sites", []):
            site_copy = {**site}
            if "app_password" in site_copy:
                site_copy["app_password"] = None if site_copy["app_password"] else None
            sites.append(site_copy)
        
        return sites
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sites: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des sites: {str(e)}"
        )


@router.get("/verify")
async def verify_connection(
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Vérifie la connexion au site WordPress.
    
    Retourne les informations du site si la connexion est établie.
    """
    try:
        # La connexion a déjà été vérifiée dans le dépendance
        return {"status": "connected", "site_url": connector.site_url}
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la connexion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vérification de la connexion: {str(e)}"
        )


@router.get("/posts")
async def get_posts(
    page: int = Query(1, description="Numéro de page"),
    per_page: int = Query(10, description="Nombre d'éléments par page"),
    search: Optional[str] = Query(None, description="Terme de recherche"),
    status: Optional[str] = Query(None, description="Statut des articles (publish, draft, etc.)"),
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Récupère les articles du site WordPress.
    
    Permet la pagination, la recherche et le filtrage par statut.
    """
    try:
        # Préparation des paramètres
        params = {
            "page": page,
            "per_page": per_page
        }
        
        if search:
            params["search"] = search
        
        if status:
            params["status"] = status
        
        # Requête API
        posts = await connector.get_posts(params=params)
        return posts
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des articles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des articles: {str(e)}"
        )


@router.post("/posts")
async def create_post(
    post_data: WordPressPost,
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Crée un nouvel article sur le site WordPress.
    """
    try:
        # Conversion du modèle Pydantic en dictionnaire
        post_dict = post_data.dict(exclude_none=True)
        
        # Création de l'article
        new_post = await connector.create_post(post_dict)
        return new_post
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'article: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création de l'article: {str(e)}"
        )


@router.put("/posts/{post_id}")
async def update_post(
    post_id: int = Path(..., description="ID de l'article à mettre à jour"),
    post_data: WordPressPost = Body(...),
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Met à jour un article existant sur le site WordPress.
    """
    try:
        # Conversion du modèle Pydantic en dictionnaire
        post_dict = post_data.dict(exclude_none=True)
        
        # Mise à jour de l'article
        updated_post = await connector.update_post(post_id, post_dict)
        return updated_post
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'article: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la mise à jour de l'article: {str(e)}"
        )


@router.delete("/posts/{post_id}")
async def delete_post(
    post_id: int = Path(..., description="ID de l'article à supprimer"),
    force: bool = Query(False, description="Supprimer définitivement l'article"),
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Supprime un article du site WordPress.
    
    Si force=False, l'article est mis à la corbeille.
    Si force=True, l'article est supprimé définitivement.
    """
    try:
        # Suppression de l'article
        result = await connector.delete_post(post_id, force=force)
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de l'article: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la suppression de l'article: {str(e)}"
        )


@router.get("/pages")
async def get_pages(
    page: int = Query(1, description="Numéro de page"),
    per_page: int = Query(10, description="Nombre d'éléments par page"),
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Récupère les pages du site WordPress.
    """
    try:
        # Préparation des paramètres
        params = {
            "page": page,
            "per_page": per_page
        }
        
        # Requête API
        pages = await connector.get_pages(params=params)
        return pages
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des pages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des pages: {str(e)}"
        )


@router.post("/pages")
async def create_page(
    page_data: WordPressPage,
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Crée une nouvelle page sur le site WordPress.
    """
    try:
        # Conversion du modèle Pydantic en dictionnaire
        page_dict = page_data.dict(exclude_none=True)
        
        # Création de la page
        new_page = await connector.create_page(page_dict)
        return new_page
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de la page: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création de la page: {str(e)}"
        )


@router.put("/pages/{page_id}")
async def update_page(
    page_id: int = Path(..., description="ID de la page à mettre à jour"),
    page_data: WordPressPage = Body(...),
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Met à jour une page existante sur le site WordPress.
    """
    try:
        # Conversion du modèle Pydantic en dictionnaire
        page_dict = page_data.dict(exclude_none=True)
        
        # Mise à jour de la page
        updated_page = await connector.update_page(page_id, page_dict)
        return updated_page
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la page: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la mise à jour de la page: {str(e)}"
        )


@router.post("/custom")
async def execute_custom_query(
    endpoint: str = Body(..., description="Endpoint de l'API WordPress"),
    method: str = Body("GET", description="Méthode HTTP (GET, POST, PUT, DELETE)"),
    params: Optional[Dict[str, Any]] = Body(None, description="Paramètres de la requête"),
    data: Optional[Dict[str, Any]] = Body(None, description="Données de la requête"),
    connector: WordPressConnector = Depends(get_wordpress_connector)
):
    """
    Exécute une requête personnalisée vers l'API WordPress.
    
    Permet d'accéder aux endpoints spécifiques ou aux extensions.
    """
    try:
        # Exécution de la requête personnalisée
        result = await connector.execute_custom_query(
            endpoint=endpoint,
            method=method,
            params=params,
            data=data
        )
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la requête personnalisée: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'exécution de la requête personnalisée: {str(e)}"
        )

@router.get("/sites", response_model=List[WordPressSite])
async def list_wordpress_sites():
    """Liste tous les sites WordPress configurés."""
    try:
        # Charger la configuration
        with open(CREDENTIALS_PATH, 'r') as f:
            config = json.load(f)
        
        sites = []
        
        for site_name, site_config in config.items():
            sites.append(WordPressSite(
                name=site_name,
                url=site_config.get("site_url", ""),
                username=site_config.get("username", "")
            ))
        
        return sites
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sites WordPress: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des sites WordPress: {str(e)}"
        )
