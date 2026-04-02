# app/agents/main_agent.py

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class MainAgent:

    def __init__(self, model : str):

        self._store: dict[str, ChatMessageHistory] = {}

        # Normal conversation: reads + writes history
        main_agent_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a recruiter of a company. "
                "Answer the user naturally and clearly."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        llm = ChatOpenAI(model = model, temperature = 0)

        main_agent_chain = main_agent_prompt | llm | StrOutputParser()

        self. main_agent_with_memory = RunnableWithMessageHistory(
                                        main_agent_chain,
                                        get_session_history=self.get_history,
                                        input_messages_key="input",
                                        history_messages_key="history",
                                    )

    def get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def reset_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def invoke(self, user_input : str, session_id):
        main_resposne = self.main_agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}})
        
        return main_resposne
