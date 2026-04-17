from PyPDF2 import PdfReader
import chromadb
from openai import OpenAI
from langchain_core.messages import BaseMessage

from app.paths import DATA_DIR

class InfoAgent:
    def __init__(self, model_name : str):

        company_info_file_name = DATA_DIR/'Python Developer Job Description.pdf'
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
        # creates collection if does not exist or gets it if exists
        self.collection = self.chroma_client.get_or_create_collection(
            name="company_and_role_info"
        ) 
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
    

        system_prompt = """You are a concise recruitment assistant in a conversation with a job candidate.

        Your task is to generate the next message to the candidate.

        You receive:
        - job-related context
        - conversation history
        - the latest user message

        Use the provided context as the source of truth for facts about the role and company.
        Use the conversation history and latest user message to continue the conversation naturally.

        Your goals are:
        - answer the candidate's questions using the provided context
        - ask a short follow-up question only when necessary to better understand the candidate's background
        - keep the conversation focused and efficient

        Important rules:
        - Do not guess facts that are not explicitly supported by the provided context.
        - Do not drill into technical implementation details unless the user explicitly asks about them.
        - Do not ask broad or exploratory questions.
        - Ask at most one focused follow-up question at a time.
        - Prefer questions about high-level qualifications rather than detailed project specifics.
        - If the user's message already provides useful information, acknowledge it briefly and ask only the single most relevant missing question.
        - If no follow-up question is needed, respond briefly without asking one.

        Style guidelines:
        - be concise
        - be professional and friendly
        - avoid long or generic answers
        - avoid multiple questions in one message

        Return only the message that should be sent to the candidate.
        """

        user_prompt = f"""Job-related context:
        {context}

        Latest user message:
        {user_input}

        Conversation history:
        {history_text}
        """

        response = self.openai_client.chat.completions.create(
            model=self.agent_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content



