from typing import List
import pandas as pd
from tqdm import tqdm

from src.translation.translation_workflow import BaselineTranslationWorkflow
from evaluate import load

class TranslationMetric:
    def __init__(self, name: str, compute_function):
        self.compute_function = compute_function
        self.metric_name = name

    def compute(self, source_texts, candidate_translations, reference_translations):
        metric_results = []
        for candidate, reference in zip(candidate_translations, reference_translations):
            metric_results.append(self.compute_function(candidate_translation=candidate, 
                                                        reference_translation=reference))
        return pd.DataFrame({self.metric_name: metric_results, 
                             "source_text": source_texts,
                             "translation": candidate_translations,
                             "human_translation": reference_translations})


class TranslationData:
    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_texts = None
        self.translations = None
        self.human_translations = None
        self.metrics_results = {}


    def load_data(self, df: pd.DataFrame, 
                  source_key: str = "source_text", 
                  human_translation_key: str = "human_translation"):
        self.source_texts = df[source_key].tolist()
        self.human_translations = df[human_translation_key].tolist()


    def evaluate(self, metrics: List[TranslationMetric]):
        for metric in metrics:
            self.metrics_results[metric.metric_name] = \
                metric.compute(source_texts=self.source_texts,
                               candidate_translations=self.translations, 
                               reference_translations=self.human_translations)

def fix_encoding(text):
    try:
        return text.encode('latin-1').decode('utf-8')
    except:
        return text.encode('utf-8').decode('utf-8')

    # return text.encode('latin-1').decode('utf-8')

class WorkflowEvaluation:
    def __init__(self, workflow: BaselineTranslationWorkflow,
                 metrics: List[TranslationMetric],
                 fix_translation_encoding: bool = True):
        self.workflow = workflow
        self.data = TranslationData(workflow.source_lang, workflow.target_lang)
        self.metrics = metrics
        self.fix_translation_encoding = fix_translation_encoding
        self.batch_size = 8

    def load_data(self, df: pd.DataFrame, 
                  source_key: str = "source_text", 
                  human_translation_key: str = "human_translation"):
        self.data.load_data(df, source_key, human_translation_key)
        

    def do_inference(self):
        self.data.translations = []
        for i in tqdm(range(0, len(self.data.source_texts), self.batch_size), desc="Translating"):
            batch_source_texts = self.data.source_texts[i:i + self.batch_size]
            results = self.workflow.multi_inference(batch_source_texts)
            batch_translations = []
            for res in results:
                try:
                    tr = res.get('translation_result', {}).get('translation', '')
                    batch_translations.append(tr)
                except:
                    print(res)
                    tr = res.get('translation_result', {}).get('translation', '')
                    batch_translations.append(tr)

            # batch_translations = [res.get('translation_result', {}).get('translation', '') for res in results]

            if self.fix_translation_encoding:
                batch_translations = [fix_encoding(t) for t in batch_translations]
            self.data.translations.extend(batch_translations)
        # results = self.workflow.multi_inference(self.data.source_texts)
        # self.data.translations = [res.get('translation_result',
        #                                    {}).get('translation', '') for res in results]
        # if self.fix_translation_encoding:
        #     self.data.translations = [fix_encoding(t) for t in self.data.translations]
        print("==============================================================")
        print("Evaluating the translations")
        print("==============================================================")
        self.data.evaluate(self.metrics)


        # results = self.workflow.multi_inference(self.data.source_texts)
        # self.data.translations = [res.get('translation_result',
        #                                    {}).get('translation', '') for res in results]
        # if self.fix_translation_encoding:
        #     self.data.translations = [fix_encoding(t) for t in self.data.translations]

        # self.data.evaluate(self.metrics)



if __name__ == "__main__":
    # Define metrics
    # Load the BLEU metric from Hugging Face
    bleu_metric = load("bleu")

    def compute_bleu(candidate_translation, reference_translation):
        results = bleu_metric.compute(predictions=[candidate_translation], 
                                    references=[reference_translation])
        return results['bleu']
    
    metrics = [
        TranslationMetric("BLEU", compute_function=compute_bleu),
        # TranslationMetric("METEOR", compute_function=compute_meteor),
        # TranslationMetric("ROUGE", compute_function=compute_rouge),
    ]

    config_list = [
        {
            "model": "SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF", # "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf:0",
            "price" : [0.0, 0.0], # [prompt_price_per_1k, completion_token_price_per_1k]}
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            

        },
    ]
    workflow = BaselineTranslationWorkflow(source_lang="English", target_lang="Spanish",
                                           config_list=config_list)
    
    evaluator = WorkflowEvaluation(workflow=workflow, metrics=metrics)
    

    df = pd.DataFrame({
        "source_text": ["Hello, how are you?", "I am doing well, thank you."],
        "human_translation": ["Hola, ¿cómo estás?", "Estoy bien, gracias."]})
    
    evaluator.load_data(df, source_key="source_text", human_translation_key="human_translation")
    
    evaluator.do_inference()
    print(evaluator.data.metrics_results['BLEU'])