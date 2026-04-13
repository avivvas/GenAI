from dotenv import load_dotenv
import os

from .orchestration.orchestrator import Orchestrator

def main():

    #get_next_three_available_slots()

    load_dotenv()  # Loads variables from .env

    openai_key = os.getenv("OPENAI_API_KEY")
    print(openai_key[:5])  # Just to check, don't print the full key!
    
    orchestrator = Orchestrator()

    session_id = "user1"

    while True:
        user_input = input("You: ")

        result = orchestrator.orchesrate_conversation_with_memory(
            user_input, session_id=session_id)

        print("Agent:", result["response"])

        if result["label"] == "end":
            break


if __name__ == '__main__':
    main()