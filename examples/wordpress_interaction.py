#!/usr/bin/env python3
# wordpress_interaction.py
# Exemple d'utilisation du connecteur WordPress

import os
import asyncio
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from pprint import pprint

# Ajout du chemin parent au sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import du connecteur WordPress
from src.connectors.wordpress_connector import WordPressConnector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WordPressExample")


async def list_posts(connector: WordPressConnector):
    """Liste les articles récents du site."""
    logger.info("Récupération des articles récents...")
    posts = await connector.get_posts(params={"per_page": 5})
    
    print("\n=== ARTICLES RÉCENTS ===")
    for post in posts:
        print(f"ID: {post['id']} | Titre: {post['title']['rendered']} | Date: {post['date']}")
    
    return posts


async def list_pages(connector: WordPressConnector):
    """Liste les pages du site."""
    logger.info("Récupération des pages...")
    pages = await connector.get_pages(params={"per_page": 5})
    
    print("\n=== PAGES ===")
    for page in pages:
        print(f"ID: {page['id']} | Titre: {page['title']['rendered']}")
    
    return pages


async def create_test_post(connector: WordPressConnector):
    """Crée un article de test."""
    logger.info("Création d'un article de test...")
    
    post_data = {
        "title": f"Article de test Renée - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": """
            <p>Ceci est un article de test créé par Renée via l'API REST WordPress.</p>
            <p>Le système fonctionne correctement !</p>
        """,
        "status": "draft",  # draft, publish, private, etc.
        "categories": [1],  # Catégorie par défaut (non-classé)
    }
    
    try:
        new_post = await connector.create_post(post_data)
        print("\n=== ARTICLE DE TEST CRÉÉ ===")
        print(f"ID: {new_post['id']}")
        print(f"Titre: {new_post['title']['rendered']}")
        print(f"Statut: {new_post['status']}")
        if new_post['status'] == 'draft':
            print(f"Lien de prévisualisation: {new_post['link']}")
        else:
            print(f"Lien public: {new_post['link']}")
        
        return new_post
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'article: {e}")
        return None


async def main_async(site_url, username, app_password):
    """Fonction principale d'exemple asynchrone."""
    logger.info(f"Connexion au site WordPress: {site_url}")
    
    # Initialisation du connecteur
    connector = WordPressConnector(
        site_url=site_url,
        username=username,
        app_password=app_password
    )
    
    try:
        # Vérification de la connexion
        connection_ok = await connector.verify_connection()
        if not connection_ok:
            logger.error("Impossible de se connecter au site WordPress")
            return
        
        # Récupération des articles récents
        await list_posts(connector)
        
        # Récupération des pages
        await list_pages(connector)
        
        # Si des identifiants sont fournis, on peut créer un article de test
        if username and app_password:
            await create_test_post(connector)
        else:
            logger.warning("Aucun identifiant fourni, impossible de créer un article de test")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
    finally:
        # Fermeture propre de la session
        if connector.session:
            await connector.session.close()


def main(site_url, username, app_password):
    """Fonction principale d'exemple (wrapper synchrone)."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async(site_url, username, app_password))


if __name__ == "__main__":
    # Analyse des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Exemple d'interaction avec WordPress via API REST")
    parser.add_argument("--site", required=True, help="URL du site WordPress (ex: https://monsite.com)")
    parser.add_argument("--user", help="Nom d'utilisateur WordPress")
    parser.add_argument("--password", help="Mot de passe d'application WordPress")
    parser.add_argument("--token-file", help="Fichier JSON contenant les identifiants")
    
    args = parser.parse_args()
    
    # Récupération des identifiants
    username = args.user
    app_password = args.password
    
    # Si un fichier de token est fourni, on le charge
    if args.token_file and os.path.exists(args.token_file):
        with open(args.token_file, 'r') as f:
            credentials = json.load(f)
            username = username or credentials.get('username')
            app_password = app_password or credentials.get('app_password')
    
    # Exécution du programme principal
    main(args.site, username, app_password)
