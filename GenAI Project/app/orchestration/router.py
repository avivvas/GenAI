from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

class Router:

    def __init__(self, model : str):

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
                
                "Return ONLY one of these lables:\n"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])

        llm = ChatOpenAI(model = model, temperature = 0)

        self._router = router_prompt | llm | StrOutputParser()

    def route(self, user_input: str, history) -> str:
     
        return self._router.invoke({
            "input": user_input,
            "history": history,
        })
 
