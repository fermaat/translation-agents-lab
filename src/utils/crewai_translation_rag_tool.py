import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from crewai_tools import tool

class FAISSRAGTool:
    """Tool for querying translations using FAISS."""

    def __init__(self, json_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index, self.knowledge_docs = self._build_index(json_path)

        # Create a callable tool function that captures the instance
        self.query_translation = self._create_tool_function()

    def _build_index(self, json_path):
        # Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        knowledge_docs = []
        embeddings = []

        for entry in data['translations']:
            # Combine fields into a single text block
            text = (
                f"Source: {entry['source']}\n"
                f"Target: {entry['target']}\n"
                f"Context: {entry['context']}\n"
                f"Source Language: {entry['source_lang']}\n"
                f"Target Language: {entry['target_lang']}"
            )
            knowledge_docs.append(text)
            embeddings.append(self.model.encode(text, convert_to_tensor=False))

        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        return index, knowledge_docs

    def _query_translation(self, query: str, top_k: int = 5) -> str:
        """Internal method to query FAISS index."""
        query_embedding = self.model.encode(query, convert_to_tensor=False).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve top-k matching documents
        results = [self.knowledge_docs[idx] for idx in indices[0]]
        return "\n\n".join(results)

    def _create_tool_function(self):
        """Create a callable tool function that captures the instance."""
        @tool
        def query_translation(query: str, top_k: int = 5) -> str:
            """
            Translates the given query using the specified translation model.
            Args:
                query (str): The query string to be translated.
                top_k (int, optional): The number of top translations to consider. Defaults to 5.
            Returns:
                str: The translated query.
            """
            return self._query_translation(query=query, top_k=top_k)

        return query_translation
