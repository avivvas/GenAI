from PyPDF2 import PdfReader
import chromadb
from openai import OpenAI
class InfoAgent:
    def __init__(self, model_name : str):

        company_info_file_name = 'Python Developer Job Description.pdf'
        # Extract text from PDF
        reader = PdfReader(company_info_file_name)
        doc_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        docs = [doc_text]

        self.openai_client = OpenAI()
        self.embedding_model = "text-embedding-3-large"
        
        chunks = self._chunk_text(doc_text, chunk_size=700, overlap=120)

        embeddings_response = self.openai_client.embeddings.create(
            input=chunks,
            model=self.embedding_model,
        )

        embeddings = [list(d.embedding) for d in embeddings_response.data]
        
        self.chroma_client = chromadb.Client() 
        self.collection = self.chroma_client.create_collection(name="company_and_role_info")
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

        self.agent_model = model_name

    def _chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 120):
        text = " ".join(text.split())
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap

        return chunks

    def invoke(self, user_input : str):
        
        user_input_embedding = self.openai_client.embeddings.create(
            input = [user_input],
            model = self.embedding_model
        ).data[0].embedding

        retrived_results = self.collection.query(
            query_embeddings=[user_input_embedding],
            n_results=1
        )

        retrived_docs = retrived_results["documents"][0]

        context = "\n".join(retrived_docs)

        prompt = f"""
                You are answering a candidate's question about the company or the Python Developer role.

                Answer using ONLY the context below.
                Be concise, specific, and recruiter-like.
                Do not imply on any external source or document that you rely on, just answer naturally based on your knowledge according to these documents.
                If the context does not explicitly contain the answer, say so clearly and do not guess.
                Context:
                {context}
                Question:
                {user_input}
                """

        response = self.openai_client.responses.create(
            model=self.agent_model,
            input=prompt
        )

        return response.output_text.strip()





