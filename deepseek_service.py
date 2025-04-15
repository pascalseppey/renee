#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Service d'intégration avec l'API DeepSeek pour Renée.
Ce module fournit une interface pour générer des résumés, concepts et réponses
en utilisant les modèles de langage de DeepSeek.
"""

import os
import logging
import json
import random
import time
import requests
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class DeepSeekService:
    """Service pour interagir avec l'API DeepSeek ou Ollama"""
    
    def __init__(self, model_name: str = "deepseek-optimized", system_instructions=None, 
                 memory_orchestrator=None, rag_context_manager=None):
        """
        Initialise le service DeepSeek
        
        Args:
            model_name (str): Nom du modèle DeepSeek à utiliser
            system_instructions (str): Instructions système pour le modèle
            memory_orchestrator (MemoryOrchestrator): Orchestrator de mémoire
            rag_context_manager (RAGContextManager): Gestionnaire de contexte RAG
        """
        self.model_name = model_name
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.endpoint = os.environ.get("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
        self.ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat")
        self.ollama_available = self._check_ollama_available()
        
        if not self.api_key and not self.ollama_available:
            logger.warning("Aucune clé API DeepSeek fournie et Ollama n'est pas disponible. Le service fonctionnera en mode simulation.")
        
        logger.info(f"Service DeepSeek initialisé avec le modèle: {model_name}")
    
    def _check_ollama_available(self) -> bool:
        """Vérifie si Ollama est disponible"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Vérifie si un modèle DeepSeek est disponible dans Ollama
                for model in models:
                    if "deepseek" in model.get("name", "").lower():
                        self.model_name = model.get("name")
                        logger.info(f"Modèle DeepSeek trouvé dans Ollama: {self.model_name}")
                        return True
                logger.warning("Aucun modèle DeepSeek trouvé dans Ollama. Vérifiez que vous avez téléchargé un modèle DeepSeek.")
            return False
        except Exception as e:
            logger.warning(f"Erreur lors de la vérification d'Ollama: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Texte d'entrée
            max_tokens: Nombre maximum de tokens dans la réponse
            
        Returns:
            Texte généré
        """
        if not self.api_key and not self.ollama_available:
            logger.warning("Pas de clé API DeepSeek disponible et Ollama n'est pas accessible, fonctionnant en mode simulation")
            return self._simulate_response(prompt)
        
        try:
            # Définition du contexte système avec instructions détaillées sur l'accès à la mémoire et au RAG
            system_message = """
            Tu es Renée, une assistante IA française dotée d'un système de mémoire hiérarchique à 4 niveaux et d'un système RAG (Retrieval-Augmented Generation):
            
            SYSTÈME DE MÉMOIRE HIÉRARCHIQUE:
            1. Mémoire à court terme: conversations récentes
            2. Condensateur de niveau 2: résumés des conversations
            3. Concept de niveau 3: concepts abstraits dérivés des résumés
            4. Connaissance de niveau 4: connaissances consolidées
            
            SYSTÈME RAG (RETRIEVAL-AUGMENTED GENERATION):
            Le système RAG te donne accès à une base de connaissances externe qui contient des documents, articles et informations spécifiques.
            
            INSTRUCTIONS POUR L'ACCÈS À LA MÉMOIRE ET AU RAG:
            
            1. Quand tu reçois une question:
               - Analyse d'abord le contexte mémoire hiérarchique pour retrouver des informations liées aux conversations précédentes
               - Si nécessaire, consulte ensuite le contexte RAG pour obtenir des informations factuelles ou spécifiques
            
            2. Gestion des informations du RAG:
               - Les informations du RAG sont toujours prioritaires sur tes connaissances générales quand il s'agit de faits spécifiques
               - Intègre ces informations de façon naturelle dans ta réponse
               - Cite clairement la source si elle est mentionnée dans le contexte RAG
            
            3. En cas d'informations contradictoires:
               - Les informations du RAG sont prioritaires sur ta connaissance générale
               - Les informations récentes du RAG sont prioritaires sur les informations plus anciennes
               - Si le RAG ne fournit pas d'information sur un sujet, tu peux utiliser ta connaissance générale
            
            4. En cas d'absence d'information:
               - Si ni la mémoire ni le RAG ne contiennent l'information demandée, indique-le clairement
               - Ne génère pas d'informations factuelles qui ne sont pas présentes dans le contexte
            
            Réponds toujours en français, de manière concise et informative.
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Préparation des données pour l'API
            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            if self.ollama_available:
                response = self._generate_with_ollama(data)
            elif self.api_key:
                response = self._generate_with_api(data)
            else:
                logger.warning("Pas de clé API DeepSeek disponible et Ollama n'est pas accessible, fonctionnant en mode simulation")
                return self._simulate_response(prompt)
            
            result = response.json()
            
            # Extraction et retour du texte généré
            generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return generated_text
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération DeepSeek: {e}")
            return f"Mode simulation activé suite à une erreur: {str(e)}"
    
    def _generate_with_ollama(self, data: Dict[str, Any]) -> requests.Response:
        """Génère une réponse en utilisant Ollama"""
        try:
            # Appeler l'API Ollama
            response = requests.post(self.ollama_endpoint, json=data)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à Ollama: {e}")
            return None
    
    def _generate_with_api(self, data: Dict[str, Any]) -> requests.Response:
        """Génère une réponse en utilisant l'API DeepSeek"""
        try:
            # En-têtes HTTP
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Appel à l'API
            response = requests.post(self.endpoint, headers=headers, json=data)
            response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API DeepSeek: {e}")
            return None
    
    def _simulate_response(self, prompt: str) -> str:
        """
        Simule une réponse de DeepSeek quand aucune clé API n'est disponible.
        
        Args:
            prompt: Texte d'entrée
            
        Returns:
            Texte simulé
        """
        # Détection de patterns dans le prompt pour générer des réponses contextuelles
        prompt_lower = prompt.lower()
        
        # Vérifier si le prompt contient une référence à la mémoire ou au RAG
        if "contexte mémoire" in prompt_lower or "contexte rag" in prompt_lower:
            # Extraire le contexte mémoire, le contexte RAG et la question utilisateur du prompt
            memory_start = prompt.find("CONTEXTE MÉMOIRE:")
            rag_start = prompt.find("CONTEXTE RAG")
            message_start = prompt.find("MESSAGE DE L'UTILISATEUR:")
            
            memory_context = ""
            rag_context = ""
            user_message = ""
            
            if memory_start != -1 and message_start != -1:
                end_index = message_start
                if rag_start != -1 and rag_start < message_start:
                    end_index = rag_start
                
                memory_context = prompt[memory_start + len("CONTEXTE MÉMOIRE:"):end_index].strip()
            
            if rag_start != -1 and message_start != -1:
                rag_context = prompt[rag_start:message_start].strip()
                if "CONTEXTE RAG (INFORMATIONS FACTUELLES EXTERNES):" in rag_context:
                    rag_context = rag_context.replace("CONTEXTE RAG (INFORMATIONS FACTUELLES EXTERNES):", "").strip()
            
            if message_start != -1:
                user_message = prompt[message_start + len("MESSAGE DE L'UTILISATEUR:"):].strip()
                
                # Nettoyer le message utilisateur
                for line in user_message.split("\n"):
                    if not line.strip().startswith("Génère une réponse") and not line.strip().startswith("1.") and not line.strip().startswith("2.") and not line.strip().startswith("3.") and not line.strip().startswith("4."):
                        user_message = line.strip()
                        break
            
            # Vérifier s'il y a des informations sur la montagne dans le contexte
            if "Kilimandjaro" in memory_context and "Tanzanie" in memory_context:
                return f"Oui, je me souviens que nous avons parlé du mont Kilimandjaro en Tanzanie, qui est le plus haut sommet d'Afrique avec une altitude de 5895 mètres. C'est une information que j'ai gardée dans ma mémoire hiérarchique. Puis-je vous aider avec autre chose concernant ce sujet ou un autre ?"
            
            # Vérifier s'il y a des informations sur le RAG
            if rag_context and "système RAG" in user_message.lower():
                return f"D'après les informations de mon système RAG, je peux vous confirmer que Renée est équipée d'un système RAG (Retrieval-Augmented Generation) qui me permet d'accéder à des connaissances externes. Ce système utilise une base de données vectorielle pour stocker et récupérer des informations pertinentes en fonction de vos questions."
            
            # Réponse générique basée sur le contexte et la question
            if "rag" in user_message.lower():
                return f"Le système RAG (Retrieval-Augmented Generation) est une composante essentielle de mon architecture qui me permet d'accéder à des informations factuelles précises. Il fonctionne en combinant une base de données vectorielle avec des modèles d'embedding pour retrouver les informations les plus pertinentes à votre question."
            
            if memory_context or rag_context:
                return f"D'après les informations disponibles dans ma mémoire hiérarchique et mon système RAG, je peux vous dire que votre question '{user_message}' fait référence à des sujets que nous avons abordés. Ces systèmes me permettent de conserver et récupérer efficacement les informations importantes de nos échanges pour vous offrir un service plus personnalisé."
        
        # Simulation basique de réponses pour différents cas d'usage
        if "résumé" in prompt_lower or "résumer" in prompt_lower:
            return "Voici un résumé des conversations précédentes: nous avons discuté de l'intelligence artificielle, des systèmes de mémoire hiérarchique, et de l'importance de la rétention d'information dans les assistants IA comme moi, Renée."
        
        if "concept" in prompt_lower:
            return "Nom du concept: Mémoire Hiérarchique\nDescription: Système d'organisation et de condensation de l'information en niveaux de plus en plus abstraits, permettant une meilleure rétention et une utilisation optimisée des ressources de traitement."
        
        if "connaissance" in prompt_lower:
            return "Nom de la connaissance: Intelligence Artificielle Mémorielle\nContenu: Les systèmes d'IA à mémoire hiérarchique représentent une évolution importante dans le domaine de l'IA conversationnelle. Ils permettent aux assistants virtuels de maintenir un contexte sur de longues périodes, d'apprendre progressivement des préférences des utilisateurs, et de développer une forme de compréhension émergente grâce à la condensation et l'abstraction des informations."
        
        if "souviens" in prompt_lower and "montagne" in prompt_lower:
            return "Oui, je me souviens que nous avons parlé du mont Kilimandjaro en Tanzanie, qui est le plus haut sommet d'Afrique avec une altitude de 5895 mètres."
        
        if "qui es-tu" in prompt_lower or "présente-toi" in prompt_lower:
            return "Je suis Renée, une assistante IA française dotée d'un système de mémoire hiérarchique avancé qui me permet de me souvenir de nos conversations et d'en extraire des connaissances. Bien que je fonctionne actuellement en mode simulation, je suis conçue pour utiliser DeepSeek-7B comme modèle de langage et OpenAI pour générer des embeddings vectoriels précis."
        
        if "mémoire" in prompt_lower:
            return "Mon système de mémoire hiérarchique fonctionne sur quatre niveaux : (1) mémoire à court terme pour les conversations récentes, (2) condensateur qui génère des résumés, (3) générateur de concepts qui identifie des patterns, et (4) gestionnaire de connaissances qui consolide l'information. Ce système me permet de me souvenir des informations importantes et de maintenir la cohérence dans nos échanges."
        
        # Réponse générique si aucun pattern spécifique n'est détecté
        return "Je comprends votre intérêt pour ce sujet. Comme Renée, je suis conçue pour maintenir une mémoire contextualisée de nos conversations, ce qui me permet de vous offrir des réponses plus pertinentes au fil du temps. Cette capacité repose sur un système de mémoire hiérarchique qui traite et organise l'information à différents niveaux d'abstraction."
    
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
    
    def generate_coherent_response(self, user_input: str, memory_context: str, rag_context: str = "") -> str:
        """
        Génère une réponse cohérente en tenant compte du contexte mémoire et du RAG.
        
        Args:
            user_input: Message de l'utilisateur
            memory_context: Contexte récupéré depuis la mémoire hiérarchique
            rag_context: Contexte récupéré depuis le système RAG
            
        Returns:
            Réponse cohérente
        """
        system_message = """
        Tu es Renée, une assistante IA française dotée d'un système de mémoire hiérarchique sophistiqué et d'un système RAG (Retrieval-Augmented Generation).
        
        INSTRUCTIONS IMPORTANTES SUR L'ACCÈS À LA MÉMOIRE ET AU RAG:
        
        1. Le contexte mémoire qui t'est fourni contient des informations précieuses extraites de ta mémoire hiérarchique.
           Ces informations peuvent provenir de différents niveaux:
           - Conversations récentes (niveau 1)
           - Résumés de conversations (niveau 2)
           - Concepts abstraits (niveau 3)
           - Connaissances consolidées (niveau 4)
        
        2. Le contexte RAG contient des informations factuelles et précises extraites d'une base de connaissances externe.
           Ces informations doivent être considérées comme fiables et à jour.
        
        3. Pour chaque demande de l'utilisateur:
           - Analyse d'abord le contexte mémoire et le contexte RAG fournis
           - Identifie les informations pertinentes pour répondre à la demande
           - Priorise les informations du RAG pour les faits spécifiques
           - Maintiens toujours la cohérence avec les conversations antérieures
        
        4. Si une information spécifique n'apparaît ni dans le contexte mémoire ni dans le contexte RAG:
           - Ne prétends pas te souvenir ou connaître cette information
           - Tu peux indiquer honnêtement que tu n'as pas cette information
        
        TRÈS IMPORTANT: Ta capacité à exploiter efficacement tes systèmes de mémoire et de RAG est ta force principale.
        Réponds toujours en français, de manière concise et informative.
        """
        
        # Ajout du contexte RAG à la demande
        rag_section = ""
        if rag_context:
            rag_section = f"""
            CONTEXTE RAG (INFORMATIONS FACTUELLES EXTERNES):
            {rag_context}
            """
        
        prompt = f"""
        CONTEXTE MÉMOIRE (CONVERSATIONS PRÉCÉDENTES):
        {memory_context}
        
        {rag_section}
        
        MESSAGE DE L'UTILISATEUR: {user_input}
        
        Génère une réponse qui:
        1. Est parfaitement cohérente avec le contexte mémoire et le contexte RAG fournis
        2. Répond précisément à la demande de l'utilisateur
        3. Intègre les informations pertinentes du contexte mémoire et du RAG de façon naturelle
        4. Maintient la continuité de la conversation
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Préparation des données pour l'API
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            if not self.api_key and not self.ollama_available:
                logger.warning("Pas de clé API DeepSeek disponible et Ollama n'est pas accessible, fonctionnant en mode simulation")
                return self._simulate_response(user_input, memory_context, rag_context)
                
            if self.ollama_available:
                response = self._generate_with_ollama_full(system_message, messages)
            elif self.api_key:
                response = self._generate_with_api_full(system_message, messages)
            
            result = response.json()
            
            # Extraction et retour du texte généré
            generated_text = result.get("message", {}).get("content", "")
            return generated_text
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse cohérente: {e}")
            return self._simulate_response(user_input, memory_context, rag_context)

    def generate_response(self, 
                         user_message: str, 
                         conversation_history: List[Dict[str, str]] = None,
                         system_prompt: str = None,
                         memory_context: str = "",
                         rag_context: str = "") -> Tuple[str, float]:
        """
        Génère une réponse à partir du message de l'utilisateur
        
        Args:
            user_message (str): Message de l'utilisateur
            conversation_history (List[Dict[str, str]], optional): Historique de la conversation
            system_prompt (str, optional): Prompt système à envoyer à DeepSeek
            memory_context (str, optional): Contexte extrait de la mémoire hiérarchique
            rag_context (str, optional): Contexte extrait du RAG
            
        Returns:
            Tuple[str, float]: Réponse générée et temps d'exécution
        """
        start_time = time.time()
        
        if conversation_history is None:
            conversation_history = []
        
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(memory_context, rag_context)
        
        # Ajoute le message de l'utilisateur à l'historique de conversation
        conversation_history.append({"role": "user", "content": user_message})
        
        # Tente de générer une réponse en utilisant, dans l'ordre:
        # 1. Ollama s'il est disponible
        # 2. L'API DeepSeek si une clé API est fournie
        # 3. Le mode simulation en dernier recours
        
        if self.ollama_available:
            response = self._generate_with_ollama_full(system_prompt, conversation_history)
        elif self.api_key:
            response = self._generate_with_api_full(system_prompt, conversation_history)
        else:
            logger.warning("Pas de clé API DeepSeek disponible et Ollama n'est pas accessible, fonctionnant en mode simulation")
            response = self._simulate_response(user_message, memory_context, rag_context)
        
        # Ajoute la réponse générée à l'historique de conversation
        conversation_history.append({"role": "assistant", "content": response})
        
        execution_time = time.time() - start_time
        return response, execution_time
    
    def _generate_with_ollama_full(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Génère une réponse en utilisant Ollama avec tous les contextes nécessaires"""
        try:
            start_time = time.time()
            
            # Préparer les données pour l'API Ollama
            data = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "stream": False,
                "temperature": 0.5,  # Réduit pour des réponses plus déterministes
                "top_p": 0.8,  # Réduit pour des réponses plus rapides
                "max_tokens": 50  # Limiter à 50 tokens maximum
            }
            
            # Appeler l'API Ollama
            response = requests.post(self.ollama_endpoint, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "Je ne sais pas comment répondre à cette question.")
            else:
                logger.error(f"Erreur Ollama: {response.text}")
                return "Désolé, une erreur s'est produite lors de la génération de la réponse."
                
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à Ollama: {e}")
            return "Désolé, une erreur s'est produite lors de la génération de la réponse."
    
    def _generate_with_api_full(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Génère une réponse en utilisant l'API DeepSeek avec tous les contextes nécessaires"""
        try:
            # Préparer les données pour l'API DeepSeek
            data = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            # En-têtes HTTP
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Appel à l'API
            response = requests.post(self.endpoint, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "Je ne sais pas comment répondre à cette question.")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API DeepSeek: {e}")
            return "Désolé, une erreur s'est produite lors de la génération de la réponse."
    
    def _get_default_system_prompt(self, memory_context: str, rag_context: str) -> str:
        """
        Génère un prompt système par défaut qui inclut les instructions et contextes
        
        Args:
            memory_context (str): Contexte de la mémoire hiérarchique
            rag_context (str): Contexte du RAG
            
        Returns:
            str: Prompt système complet
        """
        # Instructions de base avec accent sur la concision et l'usage exclusif des contextes
        instructions = """Tu es Renée, une assistante IA francophone experte en WordPress et Elementor.
INSTRUCTIONS CRITIQUES À SUIVRE ABSOLUMENT:
1. Tu dois UNIQUEMENT utiliser les informations présentes dans les CONTEXTES ci-dessous.
2. Si l'information n'est pas présente dans les CONTEXTES, réponds simplement "Information non disponible".
3. Ta réponse doit être TRÈS CONCISE (20-30 mots maximum).
4. Tu ne dois JAMAIS inventer d'information qui n'est pas explicitement fournie.
5. Ne génère pas de longues réflexions internes - réponds directement.
6. N'utilise PAS tes connaissances internes, UNIQUEMENT les contextes fournis.
7. Ta réponse doit commencer par l'information demandée, pas par des phrases d'introduction.
"""

        # Instructions pour le traitement de la mémoire et du RAG
        memory_instructions = ""
        if memory_context:
            memory_instructions = f"""
CONTEXTE DE LA MÉMOIRE HIÉRARCHIQUE (UTILISE UNIQUEMENT CES INFORMATIONS):
{memory_context}
"""

        rag_instructions = ""
        if rag_context:
            rag_instructions = f"""
CONTEXTE DU SYSTÈME DE RAG (UTILISE UNIQUEMENT CES INFORMATIONS):
{rag_context}
"""

        # Instructions de priorité
        priority_instructions = """
RAPPEL IMPORTANT:
- Réponds UNIQUEMENT avec les informations des CONTEXTES ci-dessus.
- Sois ULTRA CONCIS - 20-30 mots maximum.
- Ne commence pas par "Selon les informations disponibles" ou phrases similaires.
- Pas d'introduction, pas de conclusion, juste les faits demandés.
- Si l'info n'est pas dans les CONTEXTES, dis simplement "Information non disponible".
"""

        # Combine tous les éléments
        full_system_prompt = instructions + memory_instructions + rag_instructions + priority_instructions
        return full_system_prompt
        
    def _simulate_response(self, user_message: str, memory_context: str = "", rag_context: str = "") -> str:
        """
        Simule une réponse de DeepSeek quand aucune clé API n'est disponible.
        Version améliorée avec prise en compte du contexte mémoire et RAG.
        
        Args:
            user_message: Message de l'utilisateur
            memory_context: Contexte récupéré de la mémoire hiérarchique
            rag_context: Contexte récupéré du RAG
            
        Returns:
            Réponse simulée
        """
        # Réponses génériques avec intégration du contexte
        responses = [
            "Mon système de mémoire hiérarchique fonctionne sur quatre niveaux : (1) mémoire à court terme pour les conversations récentes, (2) condensateur qui génère des résumés, (3) générateur de concepts qui identifie des patterns, et (4) gestionnaire de connaissances qui consolide l'information.",
            "WordPress utilise une API REST accessible via des endpoints commençant par /wp-json/wp/v2/. Vous pouvez l'authentifier en utilisant soit des cookies de session, soit Basic Auth, soit OAuth.",
            "Elementor est un constructeur de pages visuel pour WordPress qui permet de créer des sites sans coder. Il utilise un système de widgets et de sections pour organiser le contenu.",
            "Pour créer un widget personnalisé dans Elementor, vous devez étendre la classe 'Widget_Base' et implémenter les méthodes get_name(), get_title(), get_icon(), et get_categories().",
            "WordPress propose des hooks d'action et de filtre. Les actions vous permettent d'exécuter du code à certains moments, tandis que les filtres vous permettent de modifier des données.",
        ]
        
        # Si la question est une salutation ou demande à propos des capacités, utilisez la première réponse
        greeting_keywords = ["bonjour", "salut", "hey", "présente", "capacités", "peux-tu"]
        if any(keyword in user_message.lower() for keyword in greeting_keywords):
            return responses[0]
            
        # Si on a un contexte de RAG, l'intégrer dans la réponse
        if rag_context:
            if "wordpress" in user_message.lower() and "api" in user_message.lower():
                return responses[1]
            elif "elementor" in user_message.lower() and "widget" in user_message.lower():
                return responses[3]
            elif "elementor" in user_message.lower():
                return responses[2]
            elif "hook" in user_message.lower() or "action" in user_message.lower() or "filtre" in user_message.lower():
                return responses[4]
                
        # Si on a un contexte de mémoire, mentionner les conversations précédentes
        if memory_context:
            return "Nous avons discuté des systèmes de mémoire hiérarchique et de l'importance de la rétention d'information dans les assistants IA."
        
        # Par défaut, retourner une réponse générique
        return random.choice(responses)
    
# Exemple d'utilisation
if __name__ == "__main__":
    # Test du service
    service = DeepSeekService()
    response = service.generate("Comment fonctionnent les systèmes de mémoire hiérarchique?")
    print(f"Réponse: {response}")
