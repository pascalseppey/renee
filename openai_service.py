#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Service d'intégration avec l'API OpenAI pour Renée.
Ce module fournit une interface pour générer des résumés, concepts et réponses
en utilisant les modèles de langage d'OpenAI.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
import openai
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("openai_service")

# Chargement des variables d'environnement depuis .env si présent
load_dotenv()

class OpenAIService:
    """Service d'intégration avec l'API OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialise le service OpenAI.
        
        Args:
            api_key: Clé API OpenAI (facultatif, peut être définie via env var OPENAI_API_KEY)
            model: Modèle à utiliser pour les générations
        """
        # Configuration de la clé API
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("Aucune clé API OpenAI fournie. Le service ne fonctionnera pas.")
        
        # Configuration du client
        openai.api_key = self.api_key
        
        # Paramètres du modèle
        self.model = model
        
        logger.info(f"Service OpenAI initialisé avec le modèle: {model}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Texte d'entrée
            max_tokens: Nombre maximum de tokens dans la réponse
            
        Returns:
            Texte généré
        """
        if not self.api_key:
            logger.error("Aucune clé API configurée. Impossible de générer une réponse.")
            return "Erreur: Service LLM non configuré."
        
        try:
            # Définition du contexte système
            messages = [
                {"role": "system", "content": "Tu es Renée, une assistante IA française qui se souvient des conversations passées."},
                {"role": "user", "content": prompt}
            ]
            
            # Appel à l'API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0,
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.5
            )
            
            # Extraction et retour du texte généré
            return response.choices[0].message['content'].strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération OpenAI: {e}")
            return f"Erreur: {str(e)}"
    
    def generate_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Génère un résumé à partir d'un ensemble de conversations.
        
        Args:
            conversations: Liste de conversations (contenant prompt et response)
            
        Returns:
            Texte du résumé
        """
        # Construction du prompt pour la génération du résumé
        conversations_text = ""
        for i, conv in enumerate(conversations):
            prompt_content = conv.get("prompt", {}).get("content", "")
            response_content = conv.get("response", {}).get("content", "")
            conversations_text += f"Conversation {i+1}:\nUtilisateur: {prompt_content}\nRenée: {response_content}\n\n"
        
        prompt = f"""
        Voici un ensemble de conversations entre un utilisateur et Renée (assistant IA):
        
        {conversations_text}
        
        Génère un résumé concis (3-5 phrases) qui capture les points essentiels de ces conversations.
        Le résumé doit mettre en évidence les sujets principaux et les informations importantes.
        """
        
        return self.generate(prompt, max_tokens=200)
    
    def generate_concept(self, summaries: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Génère un méta-concept à partir d'un ensemble de résumés.
        
        Args:
            summaries: Liste de résumés
            
        Returns:
            Dictionnaire contenant le nom et la description du concept
        """
        # Construction du prompt pour la génération du concept
        summaries_text = ""
        for i, summary in enumerate(summaries):
            content = summary.get("content", "")
            summaries_text += f"Résumé {i+1}: {content}\n\n"
        
        prompt = f"""
        Voici un ensemble de résumés de conversations:
        
        {summaries_text}
        
        Génère un méta-concept qui capture l'essence de ces résumés.
        Réponds dans ce format exact:
        Nom du concept: [nom court du concept]
        Description: [description détaillée du concept]
        """
        
        response = self.generate(prompt, max_tokens=300)
        
        # Extraction du nom et de la description du concept
        concept_name = "Concept Généré"
        concept_description = response
        
        try:
            if "Nom du concept:" in response and "Description:" in response:
                parts = response.split("Description:", 1)
                name_part = parts[0].replace("Nom du concept:", "").strip()
                desc_part = parts[1].strip()
                
                concept_name = name_part
                concept_description = desc_part
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du concept: {e}")
        
        return {
            "name": concept_name,
            "description": concept_description
        }
    
    def generate_knowledge(self, concepts: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Génère une connaissance à partir d'un ensemble de concepts.
        
        Args:
            concepts: Liste de concepts (contenant name et description)
            
        Returns:
            Dictionnaire contenant le nom et le contenu de la connaissance
        """
        # Construction du prompt pour la génération de la connaissance
        concepts_text = ""
        for i, concept in enumerate(concepts):
            name = concept.get("name", "")
            description = concept.get("description", "")
            concepts_text += f"Concept {i+1}: {name} - {description}\n\n"
        
        prompt = f"""
        Voici un ensemble de concepts:
        
        {concepts_text}
        
        Génère une connaissance qui intègre ces concepts en une compréhension plus large.
        Réponds dans ce format exact:
        Nom de la connaissance: [nom court de la connaissance]
        Contenu: [contenu détaillé de la connaissance]
        """
        
        response = self.generate(prompt, max_tokens=400)
        
        # Extraction du nom et du contenu de la connaissance
        knowledge_name = "Connaissance Générée"
        knowledge_content = response
        
        try:
            if "Nom de la connaissance:" in response and "Contenu:" in response:
                parts = response.split("Contenu:", 1)
                name_part = parts[0].replace("Nom de la connaissance:", "").strip()
                content_part = parts[1].strip()
                
                knowledge_name = name_part
                knowledge_content = content_part
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la connaissance: {e}")
        
        return {
            "name": knowledge_name,
            "content": knowledge_content
        }
    
    def generate_coherent_response(self, user_input: str, memory_context: str) -> str:
        """
        Génère une réponse cohérente en tenant compte du contexte mémoire.
        
        Args:
            user_input: Message de l'utilisateur
            memory_context: Contexte récupéré depuis la mémoire hiérarchique
            
        Returns:
            Réponse cohérente
        """
        prompt = f"""
        Voici le contexte des conversations précédentes:
        
        {memory_context}
        
        Message de l'utilisateur: {user_input}
        
        Génère une réponse qui est cohérente avec le contexte fourni et répond précisément
        à la demande de l'utilisateur. Si l'utilisateur fait référence à une information
        présente dans le contexte, assure-toi de l'inclure dans ta réponse.
        """
        
        return self.generate(prompt, max_tokens=500)

# Exemple d'utilisation
if __name__ == "__main__":
    # Test du service avec une clé API définie dans l'environnement
    service = OpenAIService()
    
    if service.api_key:
        response = service.generate("Comment fonctionnent les systèmes de mémoire hiérarchique?")
        print(f"Réponse: {response}")
    else:
        print("Aucune clé API disponible pour le test.")
