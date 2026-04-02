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
        self.model = "text-embedding-3-large"
        response = self.openai_client.embeddings.create(
            input=docs, 
            model=self.model)
        embeddings = [list(d.embedding) for d in response.data]
        
        self.chroma_client = chromadb.Client() 
        self.collection = self.chroma_client.create_collection(name="my_collection")
        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=["python_dev_doc"]
        )

        self.agent_model = model_name

    def invoke(self, user_input : str):
        
        user_input_embedding = self.openai_client.embeddings.create(
            input = [user_input],
            model = self.model
        ).data[0].embedding

        retrived_results = self.collection.query(
            query_embeddings=[user_input_embedding],
            n_results=1
        )

        retrived_docs = retrived_results["documents"][0]

        context = "\n".join(retrived_docs)

        response = self.openai_client.responses.create(
            model=self.agent_model,
            input=f"""Answer the question based only on the context below.
                     Context:{context}
                     Question:{user_input}"""
        )

        return response.output_text





