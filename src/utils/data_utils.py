import pandas as pd
import json

LENGTH_THRESHOLD = 5

def load_tm_dataframe_from_excel(file_path, tm_ds=False):
    """
    Load a DataFrame from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_excel(file_path)
        source_lang, target_lang = df.columns[1], df.columns[2]
        df = df[df[source_lang].str.split().str.len() > LENGTH_THRESHOLD]
        df = df[df[target_lang].str.split().str.len() > LENGTH_THRESHOLD]
        
        
        if tm_ds:
            df.rename(columns={df.columns[1]: 'source',
                           df.columns[2]: 'target'}, inplace=True)
            df['source_lang'] = source_lang.lower()
            df['target_lang'] = target_lang.lower()
        else:
            df.rename(columns={df.columns[1]: 'source_text',
                               df.columns[2]: 'human_translation'}, inplace=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"An error occurred while loading the Excel file: {e}")
        return None
    


def tm_dataframe_to_json(df, json_file_path):
    """
    Convert a DataFrame to a JSON file with a specific structure.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert.
    json_file_path (str): The path to the JSON file to save.
    """
    try:
        translations = df.to_dict(orient='records')
        data = {"translations": translations}
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Data successfully written to {json_file_path}")
    except Exception as e:
        print(f"An error occurred while converting the DataFrame to JSON: {e}")


def load_and_convert_excel_to_json(excel_file_path, json_file_path):
    """
    Load a dataset from an Excel file, process it, and save it as a JSON file.

    Parameters:
    excel_file_path (str): The path to the Excel file.
    json_file_path (str): The path to the JSON file to save.
    """
    df = load_tm_dataframe_from_excel(excel_file_path)
    if df is not None:
        df.rename(columns={'source_text': 'source', 
                           'human_translation': 'target'}, inplace=True)
        tm_dataframe_to_json(df, json_file_path)
    else:
        print("Failed to load the Excel file.")

        # Example usage
        # load_and_convert_excel_to_json('/path/to/excel/file.xlsx', '/path/to/output/file.json')