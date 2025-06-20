import json
import logging
import autogen
import os

from typing import List, Dict
from promptflow.tracing import start_trace, trace


from src.utils.translation_processing_utils import parse_translation_data
from src.utils.prompt_utils import FlexiblePromptTemplate


# TODO: add this at another level
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = '/Users/fvelasco/data/research/mt-agents-research/'


class BasicTranslationWorkflow:
    """
    A basic class to manage the translation workflow.
    Attributes:
        source_lang (str): The source language code.
        target_lang (str): The target language code.
        translator (autogen.AssistantAgent): The agent responsible for performing translations.
    Methods:
        translate(text: str) -> dict:
    """
    def __init__(self, source_lang: str, target_lang: str):
        self.workflow_name = "BasicTranslationWorkflow"
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        start_trace(collection=f"autogen-translation-{self.workflow_name}-{self.source_lang}-{self.target_lang}")

        # Basic translator agent
        self.translator = autogen.AssistantAgent(
            name="Translator",
            system_message=f"You are a professional translator converting text from {source_lang} to {target_lang}. "
            "Provide accurate, context-aware translations."
        )
    
    def translate(self, text: str) -> dict:
        """
        Translate text from source language to target language.
        
        :param text: Text to translate
        :return: Dictionary with translation results
        """
        # Initial translation request
        initial_request = {
            "content": f"Translate the following text from {self.source_lang} to {self.target_lang}: {text}",
            "role": "user"
        }
        
        # Generate translation
        translation = self.translator.generate_reply([initial_request])
        
        # Parse the JSON response
        try:
            translation_dict = json.loads(translation)
            return {
                "original_text": text,
                "translation_result": translation_dict,
                "source_language": self.source_lang,
                "target_language": self.target_lang
            }
        except json.JSONDecodeError:
            return {
                "original_text": text,
                "error": "Failed to parse translation response",
                "raw_response": translation,
                "source_language": self.source_lang,
                "target_language": self.target_lang
            }
    
    @trace
    def multi_inference(self, texts: List[str]) -> List[dict]:
        """
        Perform translation on multiple texts.
        
        :param texts: List of texts to translate
        :return: List of dictionaries with translation results
        """
        results = []
        for text in texts:
            result = self.translate(text)
            results.append(result)
        return results

class BaselineTranslationWorkflow(BasicTranslationWorkflow):
    """
    A class to manage the translation workflow using specialized agents for translation and quality checking.
    Attributes:
        source_lang (str): The source language code.
        target_lang (str): The target language code.
        config_list (List[Dict[str, str]]): Optional configuration list for the language model.
        translator (autogen.AssistantAgent): The agent responsible for performing translations.
        quality_checker (autogen.AssistantAgent): The agent responsible for checking translation quality.
        group_chat (autogen.GroupChat): The group chat instance managing the conversation between agents.
        group_chat_manager (autogen.GroupChatManager): The manager for handling group chat interactions.
    Methods:
        translate(text: str) -> dict:
        _trace_message(message: Dict[str, str], description: str):
    """
    def __init__(self, source_lang: str, target_lang: str, config_list: List[Dict[str, str]] = None,
                 workflow_name: str = "BaselineTranslationWorkflow"):
        super().__init__(source_lang, target_lang)
        
        self.workflow_name = workflow_name
        self.config_list = config_list
        self.translator_prompt_path = f"{BASE_PATH}prompts/translation/baseline_translator_prompt.txt"
        self.quality_checker_prompt_path = f"{BASE_PATH}prompts/translation/baseline_quality_checker.txt"
        self.initial_message_template_path = f"{BASE_PATH}prompts/translation/baseline_initial_message.txt"
        
        # Specialized agents
        # Load system prompts from files
        translator_system_message = FlexiblePromptTemplate(self.translator_prompt_path).render(
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        quality_checker_system_message = FlexiblePromptTemplate(self.quality_checker_prompt_path).render(
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Create initial message template
        self.initial_message_template = FlexiblePromptTemplate(self.initial_message_template_path)
        

        self.translator = autogen.AssistantAgent(
            name="Translator",
            system_message=translator_system_message,
            llm_config={"config_list": self.config_list}
        )
        
        self.quality_checker = autogen.AssistantAgent(
            name="QualityChecker",
            system_message=quality_checker_system_message,
            llm_config={"config_list": self.config_list}
        )
        
        # Group chat to manage translation workflow
        self.group_chat = autogen.GroupChat(
            agents=[self.translator, self.quality_checker],
            messages=[],
            max_round=3
        )
        
        self.group_chat_manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config={"config_list": self.config_list}
        )
    
    @trace
    def translate(self, text: str) -> dict:
        """
        Translate text from source language to target language with quality checking.
        
        :param text: Text to translate
        :return: Dictionary with translation results
        """
        # Reset group chat messages
        self.group_chat.messages = []
        
        # Initial translation request
        initial_request = {
            "content": self.initial_message_template.render(
                source_lang=self.source_lang, 
                target_lang=self.target_lang, 
                text=text
            ),
            "role": "user"
        }

        self.group_chat.messages.append(initial_request)
        
        # Trace the initial request
        self._trace_message(initial_request, "Initial Translation Request")
        
        # Generate initial translation
        translation = self.group_chat_manager.generate_reply(
            self.group_chat.messages
        )
        
        # Trace the translator's response
        self._trace_message({"content": translation, "role": "assistant"}, "Translator's First Response")
        
        # Add quality check request to the conversation
        quality_check_request = {
            "content": f"""Initial translation: {translation}. Please review and suggest improvements. Provide the response in this exact JSON format:
            {{
                "translation": "translated text here",
                "explanation": "detailed explanation of translation choices"
            }}""",
            "role": "user"
        }
        self.group_chat.messages.append(quality_check_request)
        
        # Trace the quality check request
        self._trace_message(quality_check_request, "Quality Check Request")
        
        # Perform quality check and potential refinement
        final_translation = self.group_chat_manager.generate_reply(
            self.group_chat.messages
        )
        
        # Trace the final translation
        self._trace_message({"content": final_translation, "role": "assistant"}, "Final Translation")
        
        # Log full conversation
        logger.info("Conversation History:")
        for msg in self.group_chat.messages:
            logger.info(f"{msg.get('role', 'N/A')}: {msg.get('content', 'N/A')}")
        
        # Parse the JSON response
        try:
            translation_dict = parse_translation_data(final_translation)
            return {
                "original_text": text,
                "translation_result": translation_dict,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                "conversation_history": self.group_chat.messages
            }
        except json.JSONDecodeError:
            return {
                "original_text": text,
                "error": "Failed to parse translation response",
                "raw_response": final_translation,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                "conversation_history": self.group_chat.messages
            }
    
    @trace
    def _trace_message(self, message: Dict[str, str], description: str):
        """
        Helper method to trace individual messages with a description.
        
        :param message: The message dictionary to trace
        :param description: A description of the message's purpose
        """
        trace_data = {
            "content": message.get("content", ""),
            "role": message.get("role", ""),
            "description": description
        }
        # Use trace to log the message
        return trace_data
    