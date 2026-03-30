# app/agents/main_agent.py

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
#from .schedule_agent import ScheduleAgent
#from .info_agent import InfoAgent
#from .exit_advisor import ExitAdvisor


class MainAgent:
    def __init__(
        self,
        model: str,
        exit_advisor,
        schedule_agent,
        info_agent,
    ):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self._store: dict[str, ChatMessageHistory] = {}

        self._exit_advisor = exit_advisor
        self._schedule_agent = schedule_agent
        self._info_agent = info_agent

        # Router: reads history, does NOT write history
        router_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a routing assistant for a company chatbot.\n"
                "Classify the user's latest message into exactly one of:\n"
                "- EXIT_CANDIDATE\n"
                "- SCHEDULE\n"
                "- INFO\n"
                "- CONTINUE\n\n"

                "Definitions:\n"
                "- EXIT_CANDIDATE: the user may be ending the conversation, saying goodbye, thanking and concluding, or indicating no further help is needed.\n"
                "- SCHEDULE: the user wants to schedule, book, arrange, move, or ask for appointment availability.\n"
                "- INFO: use this ONLY if the user is asking for factual information specifically about the company, such as the company's services, products, policies, business, team, opening hours, contact details, location, or other company-related information.\n"
                "- CONTINUE: any message that does not clearly belong to the other categories.\n\n"

                "Very important rules for INFO:\n"
                "- INFO must be chosen ONLY when the question is specifically about the company.\n"
                "- Do NOT choose INFO for general knowledge questions.\n"
                "- Do NOT choose INFO for casual conversation.\n"
                "- Do NOT choose INFO for greetings, thanks, small talk, or follow-up chat.\n"
                "- Do NOT choose INFO if the user is asking for something unrelated to the company.\n"
                "- If there is doubt between INFO and CONTINUE, choose CONTINUE.\n\n"

                "Examples:\n"
                '- "What services does your company provide?" -> INFO\n'
                '- "Where is your company located?" -> INFO\n'
                '- "What are your opening hours?" -> INFO\n'
                '- "Can you tell me about your refund policy?" -> INFO\n'
                '- "What is machine learning?" -> CONTINUE\n'
                '- "How is the weather today?" -> CONTINUE\n'
                '- "Hello" -> CONTINUE\n'
                '- "Thanks, bye" -> EXIT_CANDIDATE\n'
                '- "Do you have any appointment slots next week?" -> SCHEDULE\n\n'
                
                "Return ONLY valid JSON in this format:\n"
                '{{"route": "ONE_OF_THE_LABELS", "confidence": 0.0, "reason": "short explanation"}}'
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        self._router = router_prompt | self.llm | JsonOutputParser()

        # Normal conversation: reads + writes history
        normal_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant of a company. "
                "Answer the user naturally and clearly."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        normal_chain = normal_prompt | self.llm | StrOutputParser()

        self._normal_with_memory = RunnableWithMessageHistory(
            normal_chain,
            get_session_history=self.get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Closing response: one-off chain, no memory wrapper
        self._closing_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Write a polite closing sentence to the user at the end of a customer conversation.\n"
                "Return only the sentence that should be sent to the user.\n"
                "Do not include explanations or additional text."
            ),
            (
                "user",
                "User message: {user_input}\n"
                "Advisor result: should_exit={should_exit}, reason={reason}"
            )
        ])

        self._closing_chain = self._closing_prompt | self.llm | StrOutputParser()

    def get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def reset_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def _route(self, user_input: str, session_id: str) -> dict[str, Any]:
        history = self.get_history(session_id)
        recent_messages = history.messages[-6:]

        # Need to make this the same also in the sub agents
        # instead of passing as a string to the llm
        # we can add MessagesPlaceholder to the prompt tempalte
        # like for the router.
        # Check if it's better to pass history.messages, and not
        # just the last 6 messages. It should be a pointer, so no
        # issue od passing a large object. 
        return self._router.invoke({
            "input": user_input,
            "history": recent_messages,
        })

    def _continue_normally(self, user_input: str, session_id: str) -> str:
        return self._normal_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

    def _generate_closing_response(self, user_input: str, exit_result: dict[str, Any]) -> str:
        return self._closing_chain.invoke({
            "user_input": user_input,
            "should_exit": exit_result["should_exit"],
            "reason": exit_result["reason"],
        })

    def invoke(self, user_input: str, session_id: str = "default") -> dict[str, Any]:
        route_result = self._route(user_input, session_id)
        route = route_result["route"]
        print(f'route result: {route}')
        print(f'route result confidence: {route_result['confidence']}')
        print(f'route reason: {route_result['reason']}')

        if route == "EXIT_CANDIDATE":
            history = self.get_history(session_id)
            recent_messages = history.messages[-6:]

            exit_result = self._exit_advisor.invoke(
                user_input=user_input,
                history_messages=recent_messages
            )

            if exit_result["should_exit"]:
                response = self._generate_closing_response(user_input, exit_result)

                history.add_user_message(user_input)
                history.add_ai_message(response)

                return {
                    "response": response,
                    "should_exit": True,
                    "route": "EXIT",
                }

            response = self._continue_normally(user_input, session_id)
            return {
                "response": response,
                "should_exit": False,
                "route": "CONTINUE",
            }

        if route == "SCHEDULE":
            history = self.get_history(session_id)
            recent_messages = history.messages[-6:]

            response = self._schedule_agent.invoke(
            user_input=user_input,
            session_id=session_id,
            history_messages=recent_messages,
            )

            history = self.get_history(session_id)
            history.add_user_message(user_input)
            history.add_ai_message(response)

            return {
                "response": response,
                "should_exit": False,
                "route": "SCHEDULE",
            }

        if route == "INFO":
            response = self._info_agent.invoke(user_input)

            history = self.get_history(session_id)
            history.add_user_message(user_input)
            history.add_ai_message(response)

            return {
                "response": response,
                "should_exit": False,
                "route": "INFO",
            }

        response = self._continue_normally(user_input, session_id)
        return {
            "response": response,
            "should_exit": False,
            "route": "CONTINUE",
        }