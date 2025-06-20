import json
import logging
import faiss
import numpy as np


from typing import List, Dict
from sentence_transformers import SentenceTransformer
from promptflow.tracing import start_trace, trace


from src.utils.translation_processing_utils import parse_translation_data
from src.translation.translation_workflow import BaselineTranslationWorkflow

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = '/Users/fvelasco/data/research/mt-agents-research/'


class RAGTranslationWorkflow():
    """
    Enhanced translation workflow that incorporates RAG (Retrieval Augmented Generation)
    capabilities to improve translation quality using previous translations.
    """
    def __init__(
        self, 
        source_lang: str, 
        target_lang: str, 
        config_list: List[Dict[str, str]] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        # Create a baseline workflow instance without inheriting from it
        self.workflow_name = "RAGTranslationWorkflow"
        self.baseline_workflow = BaselineTranslationWorkflow(source_lang, target_lang, 
                                                             config_list, workflow_name=self.workflow_name)        
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.stored_translations = []
        
        # Get references to the group chat components
        self.translator = self.baseline_workflow.translator
        self.quality_checker = self.baseline_workflow.quality_checker
        self.group_chat = self.baseline_workflow.group_chat
        self.group_chat_manager = self.baseline_workflow.group_chat_manager
        
        # Initialize FAISS index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the FAISS index for similarity search."""
        # 768 is the dimension of the sentence-transformer embeddings
        self.index = faiss.IndexFlatL2(768)  
        
    def load_translation_memory(self, translations: List[Dict[str, str]]):
        """
        Load translation memory from a list of previous translations.
        
        :param translations: List of dictionaries containing previous translations
        """
        if not translations:
            return
            
        self.stored_translations = translations
        
        # Create embeddings for source texts
        try:
            source_texts = [t["source_text"] for t in translations]
        except KeyError:
            source_texts = [t["source"] for t in translations]
        embeddings = self.embedding_model.encode(source_texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
    def _retrieve_similar_translations(self, text: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar previous translations for the input text.
        
        :param text: Input text to find similar translations for
        :param k: Number of similar translations to retrieve
        :return: List of similar translations with their metadata
        """
        if not self.stored_translations:
            return []
            
        # Get embedding for input text
        query_embedding = self.embedding_model.encode([text])
        
        # Search in FAISS index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Get similar translations
        similar_translations = []
        for idx in indices[0]:
            if idx < len(self.stored_translations):
                similar_translations.append(self.stored_translations[idx])
                
        return similar_translations
    
    @trace
    def translate(self, text: str) -> dict:
        """
        Enhanced translation method that incorporates similar previous translations.
        
        :param text: Text to translate
        :return: Dictionary with translation results and reference translations
        """
        trace_data = {
            'workflow_type': 'RAG_TRANSLATION',
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'original_text': text,
            'steps': []
        }
        
        # Retrieve similar translations
        similar_translations = self._retrieve_similar_translations(text)
        trace_data['steps'].append({
            'step': 'retrieve_similar_translations',
            'data': {
                'similar_translations_count': len(similar_translations)
            }
        })
        
        # Reset group chat messages
        self.group_chat.messages = []
        
        # Create enhanced initial request with similar translations
        similar_translations_str = ""
        if similar_translations:
            similar_translations_str = "\nReference translations:\n" + "\n".join([
                f"- Source: {t['source']}\n  Target: {t['target']}"
                for t in similar_translations
            ])
            
        # Initial translation request with RAG context
        initial_request = {
            "content": self.baseline_workflow.initial_message_template.render(
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                text=text,
                similar_translations=similar_translations_str
            ),
            "role": "user"
        }
        
        self.group_chat.messages.append(initial_request)
        trace_data['steps'].append({
            'step': 'initial_request',
            'data': initial_request
        })
        
        # Generate initial translation
        translation = self.group_chat_manager.generate_reply(
            self.group_chat.messages
        )
        
        trace_data['steps'].append({
            'step': 'initial_translation',
            'data': {
                'content': translation,
                'role': 'assistant'
            }
        })
        
        # Add quality check request with reference translations
        quality_check_request = {
            "content": f"""Initial translation: {translation}
            
            Reference translations for context:
            {similar_translations_str}
            
            Please review and suggest improvements, considering the reference translations.
            Provide the response in this exact JSON format:
            {{
                "translation": "translated text here",
                "explanation": "detailed explanation of translation choices",

            }}""",
            "role": "user"
        }

        # "reference_translations_used": "explanation of how reference translations influenced the final result"
        
        self.group_chat.messages.append(quality_check_request)
        trace_data['steps'].append({
            'step': 'quality_check_request',
            'data': quality_check_request
        })
        
        # Perform quality check and potential refinement
        final_translation = self.group_chat_manager.generate_reply(
            self.group_chat.messages
        )
        
        trace_data['steps'].append({
            'step': 'final_translation',
            'data': {
                'content': final_translation,
                'role': 'assistant'
            }
        })
        
        # Log full conversation
        # logger.info("Conversation History:")
        # for msg in self.group_chat.messages:
        #     logger.info(f"{msg.get('role', 'N/A')}: {msg.get('content', 'N/A')}")
        
        # Parse the JSON response
        try:
            translation_dict = parse_translation_data(final_translation)
            result = {
                "original_text": text,
                "translation_result": translation_dict,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                "similar_translations_used": similar_translations,
                "conversation_history": self.group_chat.messages,
                "trace_data": trace_data
            }
            trace_data['steps'].append({
                'step': 'completion',
                'status': 'success'
            })
            return result
        except json.JSONDecodeError:
            error_result = {
                "original_text": text,
                "error": "Failed to parse translation response",
                "raw_response": final_translation,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                "similar_translations_used": similar_translations,
                "conversation_history": self.group_chat.messages,
                "trace_data": trace_data
            }
            trace_data['steps'].append({
                'step': 'completion',
                'status': 'error',
                'error_type': 'json_parse_error'
            })
            return error_result        

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

if __name__ == "__main__":

    config_list = [
        {
            "model": "SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF", # "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf:0",
            "price" : [0.0, 0.0], # [prompt_price_per_1k, completion_token_price_per_1k]}
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            

        },
    ]

    # Initialize the RAG-enhanced workflow
    workflow = RAGTranslationWorkflow(
        source_lang="en",
        target_lang="es",
        config_list=config_list
    )
    DATA_PATH = '/Users/fvelasco/data/research/mt-agents-research/data/trad_doc_store/'
    
    # Load your translation memory
	# Load your translations
    with open(f'{DATA_PATH}translations.json', 'r') as f:
        translations = json.load(f)['translations']
    workflow.load_translation_memory(translations)

    # Translate text
    text = "Hello, my name is Iñigo Montoya. You killed my father. Prepare to die"
    result = workflow.translate(text)
    print(result['translation_result']['translation'])