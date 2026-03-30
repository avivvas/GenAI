from dotenv import load_dotenv
import os

from .agents import MainAgent
from .agents.main_agent import MainAgent
from .agents.exit_advisor import ExitAdvisor
from .agents.schedule_module.schedule_agent import ScheduleAgent
from app.agents.info_agent import InfoAgent
from .db.session import SessionLocal
#from .agents import get_next_three_available_slots

def main():

    #get_next_three_available_slots()

    load_dotenv()  # Loads variables from .env

    openai_key = os.getenv("OPENAI_API_KEY")
    print(openai_key[:5])  # Just to check, don't print the full key!
    
    model_name = "gpt-4o-2024-11-20"


    exit_advisor = ExitAdvisor(model=model_name)
    schedule_agent = ScheduleAgent(
        model=model_name
    )
    info_agent = InfoAgent()

    agent = MainAgent(
        model="gpt-4o-2024-11-20",
        exit_advisor=exit_advisor,
        schedule_agent=schedule_agent,
        info_agent=info_agent,
    )

    session_id = "user1"

    while True:
        user_input = input("You: ")

        result = agent.invoke(user_input, session_id=session_id)

        print("Agent:", result["response"])

        if result["should_exit"]:
            break  


if __name__ == '__main__':
    main()