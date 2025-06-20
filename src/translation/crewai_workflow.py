import json

from promptflow.tracing import trace
from crewai import LLM

from src.translation.translation_workflow import BasicTranslationWorkflow
from src.utils.translation_processing_utils import parse_translation_data

from src.translation.crewai_trad_project.crew import TradProject

class BaselineCrewAITranslationWorkflow(BasicTranslationWorkflow):
    """
    Translation workflow that uses the Crew AI translation model.
    """
    def __init__(self, source_lang: str, target_lang: str, llm: LLM):
        super().__init__(source_lang, target_lang)
        self.workflow_name = "CrewAITranslationWorkflow"
        self.translation_crew = TradProject(llm).crew()
    
    @trace        
    def translate(self, text):
        final_translation =  self.translation_crew.kickoff(inputs={'source_lang': self.source_lang, 
                                                     'target_lang': self.target_lang, 
                                                     'segment': text})
        # return final_translation
    
            # Parse the JSON response
        try:
            translation_dict = parse_translation_data(final_translation.raw)
            return {
                "original_text": text,
                "translation_result": translation_dict,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                # "conversation_history": self.group_chat.messages
            }
        except json.JSONDecodeError:
            return {
                "original_text": text,
                "error": "Failed to parse translation response",
                "raw_response": final_translation,
                "source_language": self.source_lang,
                "target_language": self.target_lang,
                # "conversation_history": self.group_chat.messages
            }
    



if __name__ == "__main__":
    llm = LLM(
				model="openai/llama3.1",
				# "SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF", # "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf:0",
				base_url="http://localhost:1234/v1",
				api_key="asdf"
				)
    workflow = BaselineCrewAITranslationWorkflow(source_lang="en", target_lang="es", llm=llm)
    translation = workflow.translate("Hello, my name is Iñigo Montoya. You killed my father. Prepare to die")
    print(translation)