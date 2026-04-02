# app/agents/main_agent.py

from typing import Any

from langchain_core.messages import BaseMessage
from app.agents import MainAgent, ExitAdvisor, ScheduleAgent, InfoAgent
from .router import Router


class Orchestrator:
    def __init__(
        self,
        model_name: str,
    ):
        self._main_agent = MainAgent(model=model_name)
        self._exit_advisor = ExitAdvisor(model=model_name)
        self._schedule_agent = ScheduleAgent(model=model_name)
        self._info_agent = InfoAgent(model_name=model_name)

        self._router = Router(model_name)


    def orchesrate_conversation_with_memory(self, user_input: str, session_id: str = "default") -> dict[str, Any]:

        history = self._main_agent.get_history(session_id)
        full_convo = "\n".join([f"{type(msg).__name__}: {msg.content}" 
                                for msg in history.messages if isinstance(msg, BaseMessage)])

        route_result = self._router.route(user_input, history.messages)
        print(f'route result: {route_result}')
        
        if route_result == "EXIT_CANDIDATE":

            exit_result = self._exit_advisor.invoke(
                user_input=user_input,
                history_text=full_convo
            )

            if exit_result["should_exit"]:
                exit_response = exit_result["response"]

                history.add_user_message(user_input)
                history.add_ai_message(exit_response)

                return {
                    "response": exit_response,
                    "should_exit": True,
                    "route": "EXIT",
                }

        if route_result == "SCHEDULE":

            response = self._schedule_agent.invoke(
            user_input=user_input,
            history_text=full_convo,
            )

            history.add_user_message(user_input)
            history.add_ai_message(response)

            return {
                "response": response,
                "should_exit": False,
                "route": "SCHEDULE",
            }

        if route_result == "INFO":

            response = self._info_agent.invoke(user_input)

            history.add_user_message(user_input)
            history.add_ai_message(response)

            return {
                "response": response,
                "should_exit": False,
                "route": "INFO",
            }

        response = self._main_agent.invoke(user_input, session_id)
        return {
            "response": response,
            "should_exit": False,
            "route": "CONTINUE",
        }