#!/usr/bin/env python3
# file_watcher.py
# Surveillance continue d'un dossier pour ajouter des documents au RAG

import os
import sys
import time
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import threading
import mimetypes
import hashlib

# Watchdog pour la surveillance de fichiers
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/renee-project/logs/rag_watcher.log"))
    ]
)
logger = logging.getLogger("ReneeRAGWatcher")

# Import du module RAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.rag_hub import RAGHub

# Chargement de la configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config/config.json")
with open(config_path) as f:
    config = json.load(f)

class DocumentProcessor:
    """
    Processeur de documents pour la conversion et l'extraction de contenu
    avant l'ajout à la base de connaissances RAG.
    """
    
    def __init__(self, rag_hub: RAGHub):
        """
        Initialise le processeur de documents.
        
        Args:
            rag_hub: Instance du hub RAG pour ajouter les documents
        """
        self.rag_hub = rag_hub
        self.supported_formats = {
            '.txt': self._process_text,
            '.md': self._process_text,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.html': self._process_html,
            '.json': self._process_json,
            '.csv': self._process_csv
        }
        
        # Initialisation des dépendances (lazy loading)
        self._dependencies_loaded = False
        
    def _load_dependencies(self):
        """Charge les dépendances Python nécessaires en fonction des besoins."""
        if self._dependencies_loaded:
            return
        
        # Dictionnaire des modules à importer
        modules = {}
        
        try:
            # PDF
            try:
                import pypdf
                modules['pdf'] = pypdf
                logger.info("Module PyPDF chargé")
            except ImportError:
                logger.warning("Module PyPDF non disponible, les fichiers PDF ne seront pas traités correctement")
            
            # DOCX
            try:
                import docx
                modules['docx'] = docx
                logger.info("Module python-docx chargé")
            except ImportError:
                logger.warning("Module python-docx non disponible, les fichiers DOCX ne seront pas traités correctement")
            
            # HTML
            try:
                import bs4
                modules['bs4'] = bs4
                logger.info("Module BeautifulSoup chargé")
            except ImportError:
                logger.warning("Module BeautifulSoup non disponible, les fichiers HTML ne seront pas traités correctement")
            
            # CSV
            try:
                import pandas as pd
                modules['pandas'] = pd
                logger.info("Module pandas chargé")
            except ImportError:
                logger.warning("Module pandas non disponible, les fichiers CSV ne seront pas traités correctement")
            
            self.modules = modules
            self._dependencies_loaded = True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des dépendances: {e}")
            raise
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Traite un fichier et extrait son contenu pour l'ajout au RAG.
        
        Args:
            file_path: Chemin du fichier à traiter
            
        Returns:
            Dictionnaire avec le contenu extrait et les métadonnées
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Vérification de l'extension
        if extension not in self.supported_formats:
            logger.warning(f"Format non supporté: {extension}")
            return {"error": f"Format non supporté: {extension}"}
        
        # Chargement des dépendances
        self._load_dependencies()
        
        # Traitement du fichier selon son format
        processor = self.supported_formats[extension]
        try:
            content = processor(file_path)
            
            # Calcul du hash du fichier pour garantir l'unicité
            file_hash = self._calculate_file_hash(file_path)
            
            # Métadonnées du fichier
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "extension": extension,
                "date_added": datetime.now().isoformat(),
                "file_hash": file_hash,
                "file_size": os.path.getsize(file_path)
            }
            
            # Création du document pour le RAG
            document = {
                "content": content,
                "metadata": metadata
            }
            
            # Conversion en JSON
            json_path = file_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Document converti et sauvegardé en JSON: {json_path}")
            
            return document
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {file_path}: {e}")
            return {"error": str(e)}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcule le hash SHA-256 d'un fichier."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _process_text(self, file_path: Path) -> str:
        """Traite un fichier texte avec gestion de plusieurs encodages."""
        # Liste des encodages à essayer
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'iso-8859-15']
        
        # Essayer différents encodages
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"Fichier {file_path} ouvert avec l'encodage {encoding}")
                return content
            except UnicodeDecodeError:
                continue
        
        # Si aucun encodage ne fonctionne, essayer de lire en mode binaire et faire une conversion
        try:
            with open(file_path, 'rb') as f:
                binary_content = f.read()
            
            # Tenter de détecter l'encodage avec chardet si disponible
            try:
                import chardet
                detected = chardet.detect(binary_content)
                if detected['encoding']:
                    logger.info(f"Encodage détecté pour {file_path}: {detected['encoding']}")
                    return binary_content.decode(detected['encoding'], errors='replace')
            except ImportError:
                pass
            
            # Fallback: conversion avec remplacement des caractères non reconnus
            return binary_content.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Impossible de lire le fichier {file_path}: {e}")
            return f"[Erreur de lecture du fichier. Contenu illisible: {str(e)}]"
    
    def _process_pdf(self, file_path: Path) -> str:
        """Traite un fichier PDF."""
        if 'pdf' not in self.modules:
            return "PDF processing not available (install pypdf package)"
        
        pypdf = self.modules['pdf']
        text = ""
        
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        
        return text
    
    def _process_docx(self, file_path: Path) -> str:
        """Traite un fichier DOCX."""
        if 'docx' not in self.modules:
            return "DOCX processing not available (install python-docx package)"
        
        docx = self.modules['docx']
        doc = docx.Document(file_path)
        
        # Extraire le texte des paragraphes
        text = "\n".join([para.text for para in doc.paragraphs])
        
        return text
    
    def _process_html(self, file_path: Path) -> str:
        """Traite un fichier HTML."""
        if 'bs4' not in self.modules:
            return "HTML processing not available (install beautifulsoup4 package)"
        
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
            # Supprimer les scripts et les styles
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extraire le texte
            text = soup.get_text(separator='\n')
            
            # Nettoyer les lignes vides
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return text
    
    def _process_json(self, file_path: Path) -> str:
        """Traite un fichier JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Si le JSON contient déjà un champ 'content', l'utiliser
        if isinstance(data, dict) and 'content' in data:
            return data['content']
        
        # Sinon, convertir tout le JSON en texte
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _process_csv(self, file_path: Path) -> str:
        """Traite un fichier CSV."""
        if 'pandas' not in self.modules:
            return "CSV processing not available (install pandas package)"
        
        pd = self.modules['pandas']
        df = pd.read_csv(file_path)
        
        # Convertir en JSON pour une meilleure représentation
        return df.to_json(orient='records', indent=2)
    
    def add_to_rag(self, document: Dict[str, Any]) -> Optional[str]:
        """
        Ajoute un document au système RAG.
        
        Args:
            document: Document à ajouter avec contenu et métadonnées
            
        Returns:
            Identifiant du document ajouté ou None en cas d'erreur
        """
        try:
            if "error" in document:
                logger.error(f"Impossible d'ajouter le document au RAG: {document['error']}")
                return None
            
            # Ajout au RAG
            doc_id = self.rag_hub.add_document(
                document_content=document["content"],
                metadata=document["metadata"]
            )
            
            logger.info(f"Document ajouté au RAG avec ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout au RAG: {e}")
            return None


class RAGFileEventHandler(FileSystemEventHandler):
    """
    Gestionnaire d'événements pour la surveillance de fichiers.
    Détecte les nouveaux fichiers et les traite.
    """
    
    def __init__(self, watched_dir: str, processor: DocumentProcessor, 
                 processed_dir: Optional[str] = None):
        """
        Initialise le gestionnaire d'événements.
        
        Args:
            watched_dir: Répertoire à surveiller
            processor: Processeur de documents
            processed_dir: Répertoire pour déplacer les fichiers traités (optionnel)
        """
        self.watched_dir = os.path.abspath(watched_dir)
        self.processor = processor
        self.processed_dir = processed_dir
        self.being_processed = set()  # Ensemble des fichiers en cours de traitement
        
        # Créer le dossier de traitement s'il n'existe pas
        if processed_dir:
            os.makedirs(processed_dir, exist_ok=True)
    
    def on_created(self, event):
        """Gère les événements de création de fichiers."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Ignorer les fichiers déjà traités
        if "processed" in str(file_path):
            return
            
        # Ignorer les fichiers JSON générés automatiquement par notre système
        if file_path.suffix.lower() == '.json' and self._is_system_generated_json(file_path):
            return
            
        logger.info(f"Traitement du fichier: {file_path}")
        self._process_file(file_path)
    
    def _is_system_generated_json(self, file_path: Path) -> bool:
        """
        Détermine si un fichier JSON a été généré par notre système.
        
        Vérifie si le contenu du fichier JSON contient les champs typiques
        de notre format de traitement interne.
        """
        if not file_path.suffix.lower() == '.json':
            return False
            
        try:
            with open(file_path, 'r') as f:
                content = f.read(100)  # Lire juste le début pour vérifier
                # Si c'est un JSON généré par notre système, il contiendra ces champs
                return '"content":' in content and '"metadata":' in content
        except Exception:
            return False
    
    def _process_file(self, file_path: Path):
        """Traite un fichier en fonction de son type."""
        try:
            # Utiliser le processeur pour traiter le fichier
            document = self.processor.process_file(file_path)
            
            if document:
                # Ajouter au RAG
                try:
                    doc_id = self.processor.add_to_rag(document)
                    logger.info(f"Document ajouté au RAG avec ID: {doc_id}")
                    
                    # Déplacer le fichier original vers le dossier "processed" avec horodatage
                    self._move_to_processed(file_path)
                    
                    # Si un fichier JSON a été créé, on le déplace aussi
                    json_path = file_path.with_suffix('.json')
                    if json_path.exists():
                        self._move_to_processed(json_path)
                except Exception as e:
                    logger.error(f"Impossible d'ajouter le document au RAG: {e}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {file_path}: {e}")
            
    def _move_to_processed(self, file_path: Path):
        """Déplace un fichier vers le dossier 'processed' avec horodatage."""
        # Créer le dossier processed s'il n'existe pas
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(exist_ok=True)
        
        # Créer un nom de fichier avec horodatage pour éviter les collisions
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_filename = f"{timestamp}_{file_path.name}"
        target_path = processed_dir / new_filename
        
        # Déplacer le fichier
        shutil.move(str(file_path), str(target_path))
        logger.info(f"Fichier déplacé vers: {target_path}")


class RAGFileWatcher:
    """
    Surveillance continue d'un dossier pour l'ajout automatique de documents au RAG.
    """
    
    def __init__(self, watched_dir: str, processed_dir: Optional[str] = None):
        """
        Initialise le surveillant de fichiers.
        
        Args:
            watched_dir: Répertoire à surveiller
            processed_dir: Répertoire pour les fichiers traités (optionnel)
        """
        self.watched_dir = os.path.abspath(watched_dir)
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(self.watched_dir, exist_ok=True)
        
        # Définir le répertoire des fichiers traités
        if processed_dir:
            self.processed_dir = os.path.abspath(processed_dir)
        else:
            self.processed_dir = os.path.join(self.watched_dir, "processed")
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialisation du RAG Hub
        self.rag_hub = RAGHub(
            vector_db=config["components"]["rag"]["vector_db"],
            embedding_model=config["components"]["rag"]["embedding_model"],
            reranking_enabled=config["components"]["rag"]["reranking_enabled"]
        )
        
        # Initialisation du processeur de documents
        self.processor = DocumentProcessor(self.rag_hub)
        
        # Initialisation du gestionnaire d'événements
        self.event_handler = RAGFileEventHandler(
            watched_dir=self.watched_dir,
            processor=self.processor,
            processed_dir=self.processed_dir
        )
        
        # Initialisation de l'observateur
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.watched_dir, recursive=True)
        
        logger.info(f"Surveillance configurée pour le répertoire: {self.watched_dir}")
        logger.info(f"Les fichiers traités seront déplacés vers: {self.processed_dir}")
    
    def start(self):
        """Démarre la surveillance."""
        self.observer.start()
        logger.info("Surveillance démarrée")
        
        try:
            # Traiter les fichiers déjà présents dans le répertoire
            self._process_existing_files()
            
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Arrête la surveillance."""
        self.observer.stop()
        self.observer.join()
        logger.info("Surveillance arrêtée")
    
    def _process_existing_files(self):
        """Traite les fichiers déjà présents dans le répertoire surveillé."""
        for root, _, files in os.walk(self.watched_dir):
            for filename in files:
                # Ignorer les fichiers cachés et les fichiers temporaires
                if filename.startswith('.') or filename.endswith('~') or filename.startswith('~$'):
                    continue
                
                # Ignorer les fichiers dans les sous-dossiers processed
                if "processed" in root:
                    continue
                
                file_path = Path(os.path.join(root, filename))
                
                # Vérifier que c'est un fichier régulier et non un dossier
                if os.path.isfile(file_path):
                    # Ignorer les fichiers JSON générés par notre système
                    if file_path.suffix.lower() == '.json' and self.event_handler._is_system_generated_json(file_path):
                        continue
                        
                    logger.info(f"Traitement du fichier existant: {file_path}")
                    self.event_handler._process_file(file_path)


def run_watcher(watched_dir: str, processed_dir: Optional[str] = None):
    """
    Fonction principale pour exécuter le surveillant de fichiers.
    
    Args:
        watched_dir: Répertoire à surveiller
        processed_dir: Répertoire pour les fichiers traités (optionnel)
    """
    watcher = RAGFileWatcher(watched_dir, processed_dir)
    
    try:
        logger.info("Démarrage de la surveillance...")
        watcher.start()
    except KeyboardInterrupt:
        logger.info("Arrêt de la surveillance...")
        watcher.stop()
    except Exception as e:
        logger.error(f"Erreur lors de la surveillance: {e}")
        watcher.stop()


if __name__ == "__main__":
    # Répertoire à surveiller (par défaut: ~/Desktop/ReneeRAGDocs)
    default_watched_dir = os.path.expanduser("~/Desktop/ReneeRAGDocs")
    
    # Utiliser le répertoire fourni en argument ou le répertoire par défaut
    watched_dir = sys.argv[1] if len(sys.argv) > 1 else default_watched_dir
    
    # Répertoire pour les fichiers traités
    processed_dir = os.path.join(watched_dir, "processed")
    
    run_watcher(watched_dir, processed_dir)
