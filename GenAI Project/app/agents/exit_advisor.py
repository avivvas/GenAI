from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class ExitAdvisor:
    def __init__ (self, model : str):
        #self.model = model
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are an advisor that analyzes a user's message in a conversation.\n"
                "Your task is to determine whether the user intends to end the conversation.\n\n"
                "Consider the recent conversation and the latest user message.\n"
                
                "Return your answer ONLY as JSON with the following fields:\n"
                
                "\n"
                '  "should_exit": true or false,\n'
                '  "confidence": number between 0 and 1,\n'
                '  "reason": short explanation\n'
                "\n\n"

                "Meaning of fields:\n"
                "- should_exit: true if the user clearly intends to end the conversation.\n"
                "- confidence: your confidence in the decision (0 to 1).\n"
                "- reason: short explanation such as 'user_concluded', 'user_said_goodbye', "
                "'conversation_finished', or 'continue_conversation'.\n\n"

                "Examples:\n"
                'User: "Thanks, that is all I needed."\n'
                '"should_exit": true, "confidence": 0.92, "reason": "user_concluded"\n\n'

                'User: "Bye, have a nice day."\n'
                '"should_exit": true, "confidence": 0.95, "reason": "user_said_goodbye"\n\n'

                'User: "Can you explain me about your product?"\n'
                '"should_exit": false, "confidence": 0.98, "reason": "continue_conversation"'
            ),
            (
                "user",
                "Recent conversation:\n{history_text}\n\n"
                "Latest user message:\n{user_input}"
            )
        ])
        
        parser = JsonOutputParser()
        llm = ChatOpenAI(model = model, temperature = 0)
        self.chain = prompt | llm | parser

    def _format_history(self, history_messages) -> str:
        lines = []

        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
            else:
                lines.append(f"Other: {msg.content}")

        return "\n".join(lines)

    def invoke(self, user_input: str, history_messages):
        history_text = self._format_history(history_messages)

        return self.chain.invoke({
            "user_input": user_input,
            "history_text": history_text
        })