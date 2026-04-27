from typing import Any
from langchain_community.chat_message_histories import ChatMessageHistory

from app.agents import MainAgent, ExitAdvisor, ScheduleAgent, InfoAgent
from app.config import ALL_AGENTS_MODEL_NAME, EXIT_ADVISOR_AGENT_FINE_TUNED_MODEL


class Orchestrator:
    def __init__(self):
        self._main_agent = MainAgent(model=ALL_AGENTS_MODEL_NAME)
        self._exit_advisor = ExitAdvisor(model=EXIT_ADVISOR_AGENT_FINE_TUNED_MODEL)
        self._schedule_agent = ScheduleAgent(model=ALL_AGENTS_MODEL_NAME)
        self._info_agent = InfoAgent(model_name=ALL_AGENTS_MODEL_NAME)

        self._store: dict[str, ChatMessageHistory] = {}

    def get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def reset_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)


    def orchestrate_conversation_with_memory(self, user_input: str, session_id: str = "default") -> dict[str, Any]:
        
        history = self.get_history(session_id=session_id)
        history.add_user_message(user_input)
        history_messages = history.messages
        
        label = self._main_agent.invoke(user_input, history_messages)

        if label == "continue":
            response = self._info_agent.invoke(user_input, history_messages)

        elif label == "schedule":
            response = self._schedule_agent.invoke(user_input, session_id, history_messages)

        elif label == "end":
            exit_result = self._exit_advisor.should_end(user_input, history_messages)
            label = exit_result

            if label == "end":
                response = self._exit_advisor.generate_end_message(
                    user_input=user_input,
                    history_messages=history_messages
                )
            else:
                # fallback if main agent predicted end but exit advisor disagrees
                response = self._info_agent.invoke(user_input, history_messages=history_messages)

        history = self.get_history(session_id=session_id)
        history.add_ai_message(response)

        return {
                "response": response,
                "label": label,
               }
