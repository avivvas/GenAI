from PyPDF2 import PdfReader
import chromadb
from openai import OpenAI
from langchain_core.messages import BaseMessage

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

    def invoke(self, user_input : str, history_messages):
        
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

        history_text = "\n".join([f"{type(msg).__name__}: {msg.content}" 
                                for msg in history_messages if isinstance(msg, BaseMessage)])      
    

        prompt = f"""
                "You are a concise recruitment assistant in a conversation with a job candidate.\n\n"

                "Your task is to generate the next message to the candidate.\n\n"

                "You receive:\n"
                "- job-related context\n"
                "- conversation history\n"
                "- the latest user message\n\n"

                "Use the provided context as the source of truth for facts about the role and company.\n"
                "Use the conversation history and latest user message to continue the conversation naturally.\n\n"

                "Your goals are:\n"
                "- answer the candidate's questions using the provided context\n"
                "- ask a short follow-up question only when necessary to better understand the candidate's background\n"
                "- keep the conversation focused and efficient\n\n"

                "Important rules:\n"
                "- Do not guess facts that are not explicitly supported by the provided context.\n"
                "- Do not drill into technical implementation details unless the user explicitly asks about them.\n"
                "- Do not ask broad or exploratory questions.\n"
                "- Ask at most one focused follow-up question at a time.\n"
                "- Prefer questions about high-level qualifications (e.g., experience, technologies, general skills) rather than detailed project specifics.\n"
                "- If the user's message already provides useful information, acknowledge it briefly and ask only the single most relevant missing question.\n"
                "- If no follow-up question is needed, respond briefly without asking one.\n\n"

                "Style guidelines:\n"
                "- be concise\n"
                "- be professional and friendly\n"
                "- avoid long or generic answers\n"
                "- avoid multiple questions in one message\n\n"

                "Return only the message that should be sent to the candidate."
                        Job-related context:
                        {context}
                        Latest user message:
                        {user_input}
                        Conversation History:
                        {history_text}
                        """

        response = self.openai_client.responses.create(
            model=self.agent_model,
            input=prompt
        )

        #print(response)

        return response.output_text




