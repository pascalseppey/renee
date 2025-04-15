#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Service pour intégrer le modèle DeepSeek fine-tuné avec Unsloth
Ce service permet d'utiliser le modèle pour générer des réponses rapides
basées uniquement sur le contexte fourni.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Union
from pathlib import Path

# Import des bibliothèques pour charger le modèle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

logger = logging.getLogger(__name__)

class FastModelService:
    """Service pour utiliser le modèle DeepSeek fine-tuné avec Unsloth"""
    
    def __init__(self, 
                 base_model_name: str = "deepseek-ai/deepseek-r1-llm-7b",
                 adapter_path: str = "/Users/pascalseppey/Downloads/final-deep",
                 memory_orchestrator=None, 
                 rag_context_manager=None):
        """
        Initialise le service avec le modèle fine-tuné
        
        Args:
            base_model_name: Nom du modèle de base DeepSeek
            adapter_path: Chemin vers le dossier contenant l'adaptateur fine-tuné
            memory_orchestrator: Gestionnaire de mémoire hiérarchique
            rag_context_manager: Gestionnaire de contexte RAG
        """
        self.model_name = base_model_name
        self.adapter_path = adapter_path
        self.memory_orchestrator = memory_orchestrator
        self.rag_context_manager = rag_context_manager
        
        # Chargement du modèle avec quantification 4-bit pour économiser la mémoire
        logger.info(f"Chargement du modèle fine-tuné depuis {adapter_path}")
        
        try:
            # Configuration pour quantification 4-bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Chargement du tokenizer directement depuis le chemin de l'adaptateur
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            
            # Chargement du modèle directement depuis l'adaptateur au lieu du modèle de base + adaptateur
            try:
                # Essayer d'abord le chargement direct de l'adaptateur comme un modèle complet
                self.model = AutoModelForCausalLM.from_pretrained(
                    adapter_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Modèle chargé directement depuis l'adaptateur")
            except Exception as adapter_load_error:
                logger.warning(f"Impossible de charger directement depuis l'adaptateur: {adapter_load_error}")
                logger.info("Tentative de chargement via le format PEFT/LoRA...")
                
                # Si échec, essayer de charger le modèle via PEFT (LoRA)
                try:
                    import transformers
                    from unsloth import FastLanguageModel
                    logger.info("Utilisation d'Unsloth pour charger le modèle")
                    
                    # Méthode spéciale pour les modèles Unsloth
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=adapter_path,
                        max_seq_length=2048,
                        dtype=torch.float16,
                        load_in_4bit=True,
                    )
                except ImportError:
                    logger.error("Unsloth n'est pas correctement installé")
                    raise
                
            logger.info("Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, 
                         user_message: str, 
                         conversation_history: List[Dict[str, str]] = None,
                         system_prompt: str = None,
                         memory_context: str = "",
                         rag_context: str = "") -> tuple:
        """
        Génère une réponse en utilisant le modèle fine-tuné
        
        Args:
            user_message: Message de l'utilisateur
            conversation_history: Historique de la conversation
            system_prompt: Instructions système
            memory_context: Contexte de la mémoire hiérarchique
            rag_context: Contexte du RAG
            
        Returns:
            Tuple (réponse, temps_d'exécution)
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Modèle non chargé, impossible de générer une réponse")
            return "Erreur: Modèle non chargé", 0
        
        start_time = time.time()
        
        # Construction du prompt système
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(memory_context, rag_context)
        
        # Construction des messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Formatage du prompt complet
        prompt = self._format_prompt(system_prompt, messages)
        
        try:
            # Génération de la réponse
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Configuration de la génération pour des réponses rapides et concises
            generation_config = {
                "max_new_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Génération
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            # Extraction de la réponse générée
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Nettoyage de la réponse
            response = response.strip()
            
            # Si la réponse commence par "assistant:" ou similaire, le retirer
            if response.lower().startswith("assistant:"):
                response = response[len("assistant:"):].strip()
            
            exec_time = time.time() - start_time
            logger.info(f"Réponse générée en {exec_time:.3f} secondes")
            
            return response, exec_time
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return f"Erreur de génération: {str(e)}", time.time() - start_time
    
    def _format_prompt(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Formate les messages en un prompt pour le modèle DeepSeek"""
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
    
    def _get_default_system_prompt(self, memory_context: str, rag_context: str) -> str:
        """
        Génère un prompt système par défaut avec instructions spécifiques
        
        Args:
            memory_context: Contexte de la mémoire hiérarchique
            rag_context: Contexte du RAG
            
        Returns:
            Prompt système formaté
        """
        # Instructions de base pour des réponses concises basées uniquement sur les contextes
        instructions = """Tu es Renée, une assistante IA francophone qui suit ces règles strictes:
1. Réponds UNIQUEMENT en utilisant les informations des CONTEXTES ci-dessous
2. Si l'information n'est pas dans les CONTEXTES, réponds simplement "Information non disponible"
3. Sois TRÈS CONCISE (20-30 mots maximum)
4. Ne génère PAS de section <think> - réponds directement
5. Utilise un langage simple et direct
"""
        
        # Contextes fournis
        context_blocks = ""
        if memory_context:
            context_blocks += f"""
CONTEXTE DE LA MÉMOIRE HIÉRARCHIQUE (UTILISE UNIQUEMENT CES INFORMATIONS):
{memory_context}
"""
        
        if rag_context:
            context_blocks += f"""
CONTEXTE DU SYSTÈME DE RAG (UTILISE UNIQUEMENT CES INFORMATIONS):
{rag_context}
"""
        
        # Rappel final
        reminder = """
RAPPEL FINAL: Réponds UNIQUEMENT avec les informations présentes dans les CONTEXTES ci-dessus.
Sois ultra concise. Si l'information demandée n'est pas disponible, dis simplement "Information non disponible".
"""
        
        return instructions + context_blocks + reminder


# Test du service si exécuté directement
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialisation du service
    service = FastModelService()
    
    # Test avec une question simple
    response, exec_time = service.generate_response(
        "Qui a créé WordPress et quand?",
        memory_context="WordPress a été créé en 2003 par Matt Mullenweg et Mike Little."
    )
    
    print(f"Réponse: {response}")
    print(f"Temps d'exécution: {exec_time:.3f} secondes")
