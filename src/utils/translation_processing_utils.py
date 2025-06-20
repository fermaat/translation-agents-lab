import re
import json

def replace_outer_quotes(text):
    """
    Replace all double quotes with single quotes outside the JSON structure.
    Args:
        text (str): Input text containing JSON structures and other content.

    Returns:
        str: Text with double quotes replaced by single quotes outside JSON structures.
    """
    json_pattern = r'(\{(?:[^{}]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^{}])*?\})'
    parts = re.split(json_pattern, text)
    for i in range(len(parts)):
        if not re.match(json_pattern, parts[i]):
            parts[i] = parts[i].replace('"', "'")
    return ''.join(parts)
def clean_inner_quotes(json_string):
    """
    Remove inner quotes from JSON string values while preserving the JSON structure.
    Args:
        json_string (str): Input JSON string

    Returns:
        str: Cleaned JSON string
    """
    # Pattern to match content between JSON string quotes
    pattern = r'": "(.*?)"(?=(,|}))'

    def replace_quotes(match):
        content = match.group(1)
        # Replace inner double quotes with single quotes
        content = content.replace('"', "'")
        return f'": "{content}"'

    # Apply the replacement
    cleaned_json = re.sub(pattern, replace_quotes, json_string)

    return cleaned_json


def preprocess_text_and_load_json(text, json_key, debug=False):

    """
    Preprocess the given text by replacing specific Unicode escape sequences with their corresponding characters,
    extract the JSON part from the cleaned text, and load it as a Python dictionary.
    Args:
        text (str): The input text containing Unicode escape sequences and a JSON part.
        json_key (str): The key that must be present in the JSON object.
        debug (bool): If True, print debug information during processing.
    Returns:
        dict: The parsed JSON object as a Python dictionary.
    Raises:
        AttributeError: If the JSON part cannot be found in the text or if no valid JSON with the required key is found.
    """
    def debug_print(msg):
        if debug:
            print(f"DEBUG: {msg}")

    # Pre-process to handle potential XML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\'"', '"', text)
    text = re.sub(r'"\'', '"', text)
    
    text = clean_inner_quotes(text)
    text = text.strip()

    # Find JSON-like structures using regex first
    json_pattern = r'(\{(?:[^{}]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^{}])*?\})'

    # Modified: Decode Unicode escapes before other processing
    text = text.encode().decode('unicode_escape')
    
    text = re.sub(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})', lambda m: m.group(0).encode().decode('unicode_escape'), text)
    text = text.replace('\\"', '"')
    placeholder_pattern = r'\{\d+\}'
    text = re.sub(placeholder_pattern, lambda x: f"'{x.group(0)}'", text)

    potential_jsons = re.finditer(json_pattern, text, re.DOTALL)
    parsed_jsons = []

    for match in potential_jsons:
        json_str = match.group(0)
        debug_print(f"Found potential JSON: {json_str}...")

        # Normalize whitespace and line breaks
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = re.sub('\'"', '\'', json_str)
        json_str = re.sub('"\'', '\'', json_str)
        json_str = json_str.strip()
        debug_print(f"JSON pre loads: {json_str}...")

        try:
            # Try parsing with direct approach
            parsed_json = json.loads(json_str)
            if json_key in parsed_json:
                debug_print("Successfully parsed JSON with direct approach")
                parsed_jsons.append(parsed_json)
                continue
        except json.JSONDecodeError as e:
            debug_print(f"Direct parsing failed: {e}")

        try:
            # Try cleaning up the string
            cleaned_json = (json_str
                .replace('\\"', '"')  # Fix double escaped quotes
                .replace('\\\\', '\\')  # Fix double escaped backslashes
                .replace('\\\n', '\\n')  # Fix escaped newlines
            )
            cleaned_json = re.sub(r'\{(\d+)\}', r'{\1}', cleaned_json)
            parsed_json = json.loads(cleaned_json)
            if json_key in parsed_json:
                debug_print("Successfully parsed JSON after cleaning")
                parsed_jsons.append(parsed_json)
                continue
        except json.JSONDecodeError as e:
            debug_print(f"Cleaned parsing failed: {e}")

    if not parsed_jsons:
        raise AttributeError(f"No valid JSON objects found containing the key: {json_key}")

    return parsed_jsons[-1]


# def is_mostly_capitalized(s, threshold=0.5):
#     # Calculate the number of uppercase letters
#     num_uppercase = sum(1 for c in s if c.isupper())
#     # Calculate the number of alphabetic characters
#     num_alpha = sum(1 for c in s if c.isalpha())
#     # Check if more than half of the alphabetic characters are uppercase
#     return num_uppercase / num_alpha > threshold
# def convert_if_mostly_capitalized(s):
#     if is_mostly_capitalized(s):
#         return s.lower()
#     return s

def find_json_boundaries(text):
    """Find the start and end indices of the first complete JSON object"""
    start_idx = text.find('{')
    if start_idx == -1:
        return None, None

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == '\\':
            escape = True
            continue

        if char == '"' and not escape:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return start_idx, i + 1

    return None, None

def find_last_json_boundaries(text):
    """Find the start and end indices of the last complete JSON object"""
    end_idx = text.rfind('}')
    if end_idx == -1:
        return None, None

    brace_count = 0
    in_string = False
    escape = False

    for i in range(end_idx, -1, -1):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == '\\':
            escape = True
            continue

        if char == '"' and not escape:
            in_string = not in_string
            continue

        if not in_string:
            if char == '}':
                brace_count += 1
            elif char == '{':
                brace_count -= 1
                if brace_count == 0:
                    return i, end_idx + 1

    return None, None

def remove_after_last_quote(s):
    last_quote_index = s.rfind('"')
    if last_quote_index != -1:
        return s[:last_quote_index + 1]
    return s


def parse_translation_data(text, verbose=False):
    """
    Parse a JSON-like structure containing translation and explanation from text.

    Args:
        text (str): Input text containing the JSON-like structure

    Returns:
        dict: Dictionary with 'translation' and 'explanation' keys, or None if parsing fails
    """
    try:
        # Find JSON boundaries
        start_idx, end_idx = find_json_boundaries(text)
        if start_idx is None or end_idx is None:
            if verbose:
                print("======================================")
                print("Failed to find JSON boundaries")
                print(text)
                print("======================================")
                print("Adding curly braces to the end")
            if not text.strip().endswith("}"):
                text = remove_after_last_quote(text)
                text += "}"
            start_idx, end_idx = find_json_boundaries(text)

        # Extract the JSON object
        json_str = text[start_idx:end_idx]

        # Replace unescaped quotes in explanation with escaped ones
        # First, find the explanation part
        match = re.search(r'"explanation":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', json_str)
        if not match or start_idx is None or end_idx is None:
            start_idx, end_idx = find_last_json_boundaries(text)
            # Extract the JSON object
            json_str = text[start_idx:end_idx]

            # Replace unescaped quotes in explanation with escaped ones
            # First, find the explanation part
            match = re.search(r'"explanation":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', json_str)

        if match:
            explanation = match.group(1)
            explanation = explanation.replace(r"\u", "barrau")
            # explanation = convert_if_mostly_capitalized(explanation)
            # TODO: This is a temporary fix to handle multiple lines in the explanation
            explanation = explanation.replace('\n', ' ')
            # Replace unescaped quotes with escaped ones
            new_explanation = re.sub(r'(?<!\\)(?<!\\\\)"', '\\"', explanation)
            # new_explanation = re.sub(r'(?<!\\)(?<!\\\\)\'', "\\'", new_explanation)
            new_explanation = new_explanation.replace("\\'", "'")
            # new_explanation = new_explanation.replace('\\"', '"')
            # Replace back in the JSON string
            json_str = json_str[:match.start(1)] + new_explanation + json_str[match.end(1):]



        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, try additional cleanup
            json_str = re.sub(r'\\+"', '\\"', json_str)  # Fix multiple backslashes
            json_str = json_str.replace('\\"', "'")
            # this is the good one
            json_str = json_str.replace("'\"", "'")
            json_str = json_str.replace("\\\'", "'")
            # return json_str
            data = json.loads(json_str)

        # # Verify required keys
        # if not all(key in data for key in ('translation', 'explanation')):
        #     return None

        return {
            'translation': data['translation'].strip(),
            'explanation': data['explanation'].strip()
        }

    except (ValueError, json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Parsing error: {e}")
        print(f"Failed to parse: {json_str}")
        print(f"Original text: {text}")
        # return json_str