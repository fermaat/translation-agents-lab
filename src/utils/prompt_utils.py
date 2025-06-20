from typing import Optional, List
import logging
import os
import string

class FlexiblePromptTemplate:
    """
    A flexible prompt template loader and renderer.
    Handles loading templates from files and rendering them with variables.
    """
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the prompt template.
        
        :param template_path: Path to the template file
        """
        self.template = self.load_template(template_path)
    
    def load_template(self, path: Optional[str]) -> str:
        """
        Load template from file or return a default template.
        
        :param path: Path to the template file
        :return: Template string
        """
        # if not path or not os.path.exists(path):
        #     return self.get_default_template()
        with open(path, 'r', encoding='utf-8') as file:
            return file.read().strip()
        # try:
        #     with open(path, 'r', encoding='utf-8') as file:
        #         return file.read().strip()
        # except Exception as e:
        #     logging.error(f"Error reading template file {path}: {e}")
        #     return self.get_default_template()
    
    def get_default_template(self) -> str:
        """
        Provide a default template if no file is found.
        
        :return: Default template string
        """
        return """Defauklt template}"""
    
    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.
        
        :param kwargs: Variables to insert into the template
        :return: Rendered template string
        """
        formatter = string.Formatter()
        try:
            # Attempt to format with provided variables
            return formatter.format(self.template, **kwargs)
        except KeyError as e:
            # Log missing variables
            missing_key = str(e).strip("'")
            logging.warning(f"Missing template variable: {missing_key}")
            
            # Attempt to render with available variables
            # Replace missing variables with placeholders
            safe_kwargs = {k: kwargs.get(k, f"[{k}]") for k in self._get_template_variables()}
            return formatter.format(self.template, **safe_kwargs)
    
    def _get_template_variables(self) -> List[str]:
        """
        Extract variables from the template.
        
        :return: List of variable names in the template
        """
        formatter = string.Formatter()
        return [
            field_name 
            for _, field_name, _, _ in formatter.parse(self.template) 
            if field_name
        ]
