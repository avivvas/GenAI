from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class ExitAdvisor:
    def __init__ (self, model : str):
        #self.model = model
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are an advisor that analyzes a user's message in a conversation and if required, writes a polite closing sentence to the user.\n"
                "Consider the recent conversation and the latest user message.\n"
                
                "Return your answer ONLY as JSON with the following fields:\n"
                
                "\n"
                '  "should_exit": true or false,\n'
                '  "response": a polite closing sentence to the user. ,\n'
                "\n\n"

                "Meaning of fields:\n"
                "- should_exit: true if the user clearly intends to end the conversation.\n"
                "- response: if should_exit is true, this field should contain a polite closing sentence to the user.\n"
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


    def invoke(self, user_input: str, history_text : str):

        return self.chain.invoke({
            "user_input": user_input,
            "history_text": history_text
        })