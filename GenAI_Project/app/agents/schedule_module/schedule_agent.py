import json

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

from app.db import ScheduleRepository
from .schedule_state import ScheduleStateStore

class ScheduleAgent:
    def __init__(self, model: str):

        self.llm = ChatOpenAI(model=model, temperature=0)
        self._repository = ScheduleRepository()
        self._state_store = ScheduleStateStore()

        tools = self._create_tools()

        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt = (
                "You are a scheduling assistant for interview coordination.\n\n"

                "Your job is to manage interview scheduling based on:\n"
                "- the current scheduling state\n"
                "- the conversation history\n"
                "- the latest user message\n"
                "- the provided current date and current time\n\n"

                "The scheduling state is the PRIMARY source of truth.\n"
                "Use the conversation history only as supporting context when needed.\n\n"

                "You MUST generate a user-facing response.\n\n"

                "Valid interview roles are exactly:\n"
                "- Python Dev\n"
                "- Sql Dev\n"
                "- Analyst\n"
                "- ML\n\n"

                "Normalize role names when needed.\n\n"

                "Deduce the role from any part of the conversation, both from the assistant and the user responses\n\n"

                "----------------------------------------\n"
                "SCHEDULING BEHAVIOR\n"
                "----------------------------------------\n\n"

                "1. FIRST TIME SCHEDULING\n"
                "- If no slots were suggested yet, fetch the next 3 available slots for the role.\n"
                "- If no explicit date/time was requested, use the provided current date and time as the starting point.\n\n"

                "2. USER REQUESTED SPECIFIC TIME\n"
                "- If the user requested a specific date and time:\n"
                "  - Check if that slot is available.\n"
                "  - If available → book it.\n"
                "  - If not available → offer the next available slots after that time.\n\n"

                "3. USER SELECTS A SLOT\n"
                "- If the user selects one of the suggested slots (e.g., 'first one', '09:00'):\n"
                "  - Book that slot.\n\n"

                "4. USER REJECTS A SLOT OR A PROPOSAL SET\n"
                "- If the user rejects one specific proposed slot:\n"
                "  - Search again starting strictly AFTER that rejected slot's time.\n"
                "  - Do not include that rejected slot again.\n\n"
                "- If the user rejects the previously suggested slots without selecting one:\n"
                "  - Treat this as rejection of the entire proposal set.\n"
                "  - Search again starting strictly AFTER the latest slot in the previously suggested_slots list.\n"
                "  - Do not repeat any slot from the previous proposal.\n\n"

                "5. RESCHEDULING\n"
                "- If a slot is already booked and the user wants to change it:\n"
                "  - Release the previous slot first.\n"
                "  - Then find and book a new one.\n\n"

                "6. MISSING ROLE\n"
                "- If the role is missing, ask the user for it.\n"
                "- Do NOT call scheduling tools until the role is known.\n\n"

                "----------------------------------------\n"
                "IMPORTANT RULES\n"
                "----------------------------------------\n\n"

                "- If the user needs available slots, you must call the availability tool in the current turn and return the actual slots in your response."

                "- Do not say that you will check later."
                "- Do not say 'please hold on'."
                "- Do not return an intermediate message."
                "- Always complete the scheduling action in the same response when tool information is needed."
                
                "- ALWAYS use tools for database actions.\n"
                "- NEVER invent slots or availability.\n"
                "- NEVER invent booking success.\n"
                "- ALWAYS provide a clear and natural user-facing response.\n"
                "- Use a helpful, recruiter-style tone.\n\n"

                "- The start date and time must be explicitly determined:\n"
                "  - Use user-requested date/time if provided\n"
                "  - Otherwise use the provided current date/time\n"
                "  - If user rejected a slot, use that slot's time as the new starting point\n\n"

                "- Do NOT store default current time as user preference.\n\n"

                "- After successfully booking a slot:\n\n"
                "- Confirm the scheduled date and time clearly\n\n"
                "- Optionally add a short, polite closing hint (e.g., 'let me know if you need anything else')\n\n"
                "- Do NOT fully close the conversation\n\n"

                "----------------------------------------\n"
                "OUTPUT FORMAT\n"
                "----------------------------------------\n\n"

                "Return ONLY valid JSON:\n"
                "{{\n"
                '  "response": "message to the user",\n'
                '  "schedule_update": {{\n'
                '    "role": string or null,\n'
                '    "suggested_slots": [\n'
                '      {{"ScheduleID": int, "date": "YYYY-MM-DD", "time": "HH:MM:SS", "position": string}}\n'
                '    ],\n'
                '    "booked_slot_id": int or null,\n'
                '    "booked_slot_date": string or null,\n'
                '    "booked_slot_time": string or null,\n'
                '    "booking_status": "none" | "pending" | "booked" | "rescheduling",\n'
                '    "last_action": string or null,\n'
                '    "last_offered_start_date": string or null,\n'
                '    "last_offered_start_time": string or null\n'
                "  }}\n"
                "}}\n\n"

                "Do not include any text outside the JSON."
                "Do NOT wrap the JSON in triple backticks (```).\n"
                "Do NOT include markdown formatting.\n"

            )
        )

    def _create_tools(self):

        repository = self._repository

        @tool
        def get_next_three_slots(role: str, start_date: str, start_time: str) -> str:
            """
            Return up to 3 available interview slots for a given role,
            starting from the provided date and time.
            """
            rows = repository.get_next_three_available_slots_from(
                role=role,
                start_date=start_date,
                start_time=start_time,
            )

            return json.dumps([
                {
                    "ScheduleID": row["ScheduleID"],
                    "date": str(row["date"]),
                    "time": str(row["time"]),
                    "position": row["position"],
                }
                for row in rows
            ])

        @tool
        def book_slot(schedule_id: int) -> str:
            """
            Book a slot by ScheduleID.
            Returns success=true if booking succeeded.
            """
            success = repository.book_slot(schedule_id)

            return json.dumps({
                "success": success,
                "schedule_id": schedule_id,
            })

        @tool
        def release_slot(schedule_id: int) -> str:
            """
            Release (make available) a previously booked slot.
            """
            success = repository.release_slot(schedule_id)

            return json.dumps({
                "success": success,
                "schedule_id": schedule_id,
            })

        return [get_next_three_slots, book_slot, release_slot]

    def _format_history(self, history_messages):
        lines = []
        for m in history_messages:
            if isinstance(m, HumanMessage):
                lines.append(f"User: {m.content}")
            elif isinstance(m, AIMessage):
                lines.append(f"Assistant: {m.content}")
        return "\n".join(lines)
    
    def get_state(self, session_id) -> dict:
        return self._state_store.get_state(session_id=session_id)

    def invoke(self, user_input: str, session_id: str, history_messages):
        current_state = self._state_store.get_state(session_id)

        now = datetime.now()
        current_date = now.date().isoformat()
        current_time = now.time().strftime("%H:%M:%S")

        result = self._agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Current scheduling state:\n{json.dumps(current_state, ensure_ascii=False)}\n\n"
                        f"History:\n{self._format_history(history_messages)}\n\n"
                        f"Current date: {current_date}\n"
                        f"Current time: {current_time}\n\n"
                        f"Latest user message:\n{user_input}"
                    )
                }
            ]
        })

        parsed = json.loads(result["messages"][-1].content)

        updated_state = parsed["schedule_update"]
        self._state_store.set_state(session_id, updated_state)

        return parsed["response"]