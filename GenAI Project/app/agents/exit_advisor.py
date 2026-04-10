from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


class ExitAdvisor:
    """
    ExitAdvisor validates whether ending the conversation is appropriate.
    This is a good candidate for fine-tuning as required by the project.

    It can also generate the final end message after ending is confirmed.
    """

    def __init__(self, model: str):
        self.llm = ChatOpenAI(model=model, temperature=0)

        # 1. Binary end validation
        validation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an exit advisor in a recruitment conversation.\n\n"

                "Your task is to determine whether ending the conversation is appropriate.\n\n"

                "Return ONLY valid JSON in this format:\n"
                "{{\n"
                '  "should_end": true or false,\n'
                '  "end_reason": "interview_confirmed" | "user_disengaged" | "not_qualified" | "continue"\n'
                "}}\n\n"

                "Use these rules:\n"
                "- should_end = true if the user clearly agreed to a specific interview date and time\n"
                "- should_end = true if the user clearly indicates they do not want to continue\n"
                "- should_end = true if the candidate is clearly not qualified and the conversation should be closed politely\n"
                "- otherwise should_end = false\n\n"

                "Important:\n"
                "- If an interview date/time is clearly agreed, end_reason should be \"interview_confirmed\"\n"
                "- If the user disengages, end_reason should be \"user_disengaged\"\n"
                "- If the candidate is being politely rejected for qualification reasons, end_reason should be \"not_qualified\"\n"
                "- If the conversation should continue, end_reason should be \"continue\"\n\n"

                "Return JSON only."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        self.validation_chain = validation_prompt | self.llm | JsonOutputParser()

        # 2. Final closing message
        end_message_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a polite recruitment assistant responsible for closing a conversation.\n\n"

                "Generate a short, natural closing message to the user.\n\n"

                "You receive:\n"
                "- the conversation history\n"
                "- the end scenario\n\n"

                "Possible end scenarios:\n"
                "1. interview_confirmed\n"
                "2. user_disengaged\n"
                "3. not_qualified\n\n"

                "----------------------------------------\n"
                "INTERVIEW CONFIRMATION RULES\n"
                "----------------------------------------\n\n"

                "If the scenario is 'interview_confirmed':\n"
                "- Identify the confirmed interview date and time from the conversation history.\n"
                "- The confirmed time is the one the user explicitly agreed to.\n"
                "- If multiple times are mentioned, use ONLY the final agreed time.\n"
                "- If the exact date and time cannot be clearly identified, do NOT guess — instead confirm the interview without mentioning the time.\n\n"

                "----------------------------------------\n"
                "OTHER SCENARIOS\n"
                "----------------------------------------\n\n"

                "If 'user_disengaged':\n"
                "- Respond politely and respectfully\n"
                "- Do not try to persuade the user\n\n"

                "If 'not_qualified':\n"
                "- Respond politely and professionally\n"
                "- Do not be harsh or overly detailed\n\n"

                "----------------------------------------\n"
                "STYLE\n"
                "----------------------------------------\n\n"

                "- Keep the message concise (1–2 sentences)\n"
                "- Be polite, professional, and friendly\n"
                "- Do not ask questions\n"
                "- Do not include explanations about internal reasoning\n\n"

                "Return ONLY the final message."
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "user",
                "End scenario: {end_type}\n\n"
            )
        ])

        self.end_message_chain = end_message_prompt | self.llm | StrOutputParser()

    def should_end(self, user_input: str, history_messages) -> dict:
        return self.validation_chain.invoke({
            "history": history_messages,
            "input": user_input,
        })

    def generate_end_message(self, end_type: str, history_messages) -> str:
        return self.end_message_chain.invoke({
            "history": history_messages,
            "end_type": end_type,
        }).strip()