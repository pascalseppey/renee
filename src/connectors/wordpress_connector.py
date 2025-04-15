#!/usr/bin/env python3
# wordpress_connector.py
# Module de connexion pour interagir avec n'importe quel site WordPress via l'API REST

import os
import json
import logging
import base64
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Union

# Configuration du logging
logger = logging.getLogger(__name__)

class WordPressError(Exception):
    """
    Exception spécifique pour les erreurs WordPress.
    """
    def __init__(self, message: str, status_code: int = None, response_data: Any = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(message)

class WordPressConnector:
    """
    Connecteur WordPress pour interagir avec n'importe quel site WordPress via l'API REST.
    Cette classe gère l'authentification et les appels API vers WordPress.
    """
    
    def __init__(self, site_url: str, username: str = None, app_password: str = None, token_file: str = None):
        """
        Initialise le connecteur WordPress.
        
        Args:
            site_url: URL du site WordPress (ex: https://monsite.com)
            username: Nom d'utilisateur WordPress
            app_password: Mot de passe d'application généré dans WordPress
            token_file: Chemin vers un fichier contenant les identifiants
        """
        # S'assurer que l'URL se termine par un slash
        self.site_url = site_url if site_url.endswith('/') else f"{site_url}/"
        self.api_base_url = f"{self.site_url}wp-json/wp/v2"
        self.username = username
        self.app_password = app_password
        
        # Charger les identifiants depuis un fichier si fourni
        if token_file and os.path.exists(token_file):
            with open(token_file, 'r') as f:
                credentials = json.load(f)
                self.username = self.username or credentials.get('username')
                self.app_password = self.app_password or credentials.get('app_password')
        
        # Vérifier si les identifiants sont disponibles
        self.auth_available = bool(self.username and self.app_password)
        
        # En-têtes pour l'authentification
        self.headers = {}
        if self.auth_available:
            auth_string = f"{self.username}:{self.app_password}"
            auth_token = base64.b64encode(auth_string.encode()).decode()
            self.headers["Authorization"] = f"Basic {auth_token}"
        
        # Session HTTP asynchrone
        self.session = None
    
    async def _create_session(self):
        """Crée une session HTTP si elle n'existe pas déjà."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
    
    async def verify_connection(self) -> bool:
        """
        Vérifie que la connexion au site WordPress est fonctionnelle.
        
        Returns:
            bool: True si la connexion est établie, False sinon
        """
        try:
            session = await self._create_session()
            
            # Vérifier que l'API REST est disponible
            async with session.get(f"{self.site_url}wp-json/") as response:
                if response.status != 200:
                    logger.error(f"Impossible de se connecter à l'API REST WordPress: {response.status}")
                    return False
                
                data = await response.json()
                if not data or not isinstance(data, dict):
                    logger.error("Réponse API WordPress invalide")
                    return False
                
                logger.info(f"Connexion établie avec le site WordPress: {data.get('name', self.site_url)}")
                
                # Vérifier l'authentification si des identifiants ont été fournis
                if self.auth_available:
                    async with session.get(f"{self.api_base_url}/users/me") as auth_response:
                        if auth_response.status != 200:
                            logger.error(f"Échec de l'authentification: {auth_response.status}")
                            return False
                        
                        user_data = await auth_response.json()
                        logger.info(f"Authentifié en tant que: {user_data.get('name', self.username)}")
                
                return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la connexion: {e}")
            return False
    
    async def get_posts(self, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Récupère les articles du site WordPress.
        
        Args:
            params: Paramètres de la requête (pagination, filtres, etc.)
            
        Returns:
            Liste des articles
        """
        session = await self._create_session()
        async with session.get(f"{self.api_base_url}/posts", params=params) as response:
            return await self._handle_response(response)
    
    async def create_post(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crée un nouvel article sur le site WordPress.
        
        Args:
            data: Données de l'article (title, content, status, etc.)
            
        Returns:
            Données de l'article créé
        """
        session = await self._create_session()
        async with session.post(f"{self.api_base_url}/posts", json=data) as response:
            return await self._handle_response(response)
    
    async def update_post(self, post_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour un article existant.
        
        Args:
            post_id: ID de l'article à mettre à jour
            data: Nouvelles données de l'article
            
        Returns:
            Données de l'article mis à jour
        """
        session = await self._create_session()
        async with session.post(f"{self.api_base_url}/posts/{post_id}", json=data) as response:
            return await self._handle_response(response)
    
    async def delete_post(self, post_id: int, force: bool = False) -> Dict[str, Any]:
        """
        Supprime un article.
        
        Args:
            post_id: ID de l'article à supprimer
            force: Si True, supprime définitivement (sinon, met à la corbeille)
            
        Returns:
            Résultat de la suppression
        """
        params = {"force": "true"} if force else {}
        session = await self._create_session()
        async with session.delete(f"{self.api_base_url}/posts/{post_id}", params=params) as response:
            return await self._handle_response(response)
    
    async def get_pages(self, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Récupère les pages du site WordPress.
        
        Args:
            params: Paramètres de la requête (pagination, filtres, etc.)
            
        Returns:
            Liste des pages
        """
        session = await self._create_session()
        async with session.get(f"{self.api_base_url}/pages", params=params) as response:
            return await self._handle_response(response)
    
    async def create_page(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crée une nouvelle page sur le site WordPress.
        
        Args:
            data: Données de la page (title, content, status, etc.)
            
        Returns:
            Données de la page créée
        """
        session = await self._create_session()
        async with session.post(f"{self.api_base_url}/pages", json=data) as response:
            return await self._handle_response(response)
    
    async def update_page(self, page_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour une page existante.
        
        Args:
            page_id: ID de la page à mettre à jour
            data: Nouvelles données de la page
            
        Returns:
            Données de la page mise à jour
        """
        session = await self._create_session()
        async with session.post(f"{self.api_base_url}/pages/{page_id}", json=data) as response:
            return await self._handle_response(response)
    
    async def get_media(self, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Récupère les médias du site WordPress.
        
        Args:
            params: Paramètres de la requête (pagination, filtres, etc.)
            
        Returns:
            Liste des médias
        """
        session = await self._create_session()
        async with session.get(f"{self.api_base_url}/media", params=params) as response:
            return await self._handle_response(response)
    
    async def upload_media(self, file_path: str, post_id: int = 0) -> Dict[str, Any]:
        """
        Téléverse un fichier média sur le site WordPress.
        
        Args:
            file_path: Chemin du fichier à téléverser
            post_id: ID de l'article/page auquel associer le média (0 = aucun)
            
        Returns:
            Données du média téléversé
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        # Détecter le type MIME à partir de l'extension du fichier
        import mimetypes
        content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        # Préparer les en-têtes pour l'upload
        headers = dict(self.headers)
        headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(file_path)}"'
        headers["Content-Type"] = content_type
        
        # Lire le fichier
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # URL pour l'upload
        url = f"{self.api_base_url}/media"
        if post_id > 0:
            url += f"?post={post_id}"
        
        # Envoyer le fichier
        session = await self._create_session()
        async with session.post(url, headers=headers, data=data) as response:
            return await self._handle_response(response)
    
    async def execute_custom_query(self, endpoint: str, method: str = "GET", 
                                  params: Dict[str, Any] = None, 
                                  data: Dict[str, Any] = None) -> Any:
        """
        Exécute une requête personnalisée vers un endpoint de l'API REST WordPress.
        
        Args:
            endpoint: Endpoint de l'API REST (ex: "wp/v2/posts", "wc/v3/products", etc.)
            method: Méthode HTTP (GET, POST, PUT, DELETE)
            params: Paramètres de la requête
            data: Données à envoyer (pour POST, PUT)
            
        Returns:
            Résultat de la requête
        """
        # S'assurer que l'endpoint ne commence pas par un slash
        endpoint = endpoint.lstrip('/')
        
        # Déterminer l'URL complète
        if endpoint.startswith('wp/'):
            # C'est un endpoint standard de l'API WordPress
            url = f"{self.site_url}wp-json/{endpoint}"
        else:
            # C'est un endpoint personnalisé
            url = f"{self.site_url}wp-json/{endpoint}"
        
        session = await self._create_session()
        async with session.request(method, url, params=params, json=data) as response:
            return await self._handle_response(response)
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Any:
        """
        Traite la réponse de l'API WordPress.
        
        Args:
            response: Réponse HTTP
            
        Returns:
            Données de la réponse
            
        Raises:
            WordPressError: En cas d'erreur
        """
        if response.status < 200 or response.status >= 300:
            error_text = await response.text()
            try:
                error_data = json.loads(error_text)
                error_message = error_data.get('message', error_text)
            except json.JSONDecodeError:
                error_message = error_text
            
            logger.error(f"Erreur API WordPress ({response.status}): {error_message}")
            raise WordPressError(f"Erreur API WordPress: {error_message}", response.status, error_data)
        
        # Pour les requêtes non-JSON, renvoyer la réponse brute
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('application/json'):
            return await response.text()
        
        try:
            return await response.json()
        except json.JSONDecodeError:
            return await response.text()


# Fonction utilitaire pour exécuter des requêtes WordPress de manière synchrone
def run_wordpress_request(func, *args, **kwargs):
    """
    Exécute une fonction asynchrone de manière synchrone.
    
    Args:
        func: Fonction asynchrone à exécuter
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Résultat de la fonction
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(func(*args, **kwargs))
