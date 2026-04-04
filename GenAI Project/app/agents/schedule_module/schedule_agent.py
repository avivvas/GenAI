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

        # ---------------------------
        # PASSIVE STATE UPDATE CHAIN
        # ---------------------------
        self._state_parser = JsonOutputParser()

        state_update_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You update scheduling state for a company interview assistant.\n\n"

                "IMPORTANT:\n"
                "- This is a PASSIVE step.\n"
                "- Do NOT perform scheduling actions.\n"
                "- Do NOT assume bookings or slot availability.\n"
                "- Do NOT generate a user-facing response.\n\n"

                "Your job is ONLY to update the scheduling state based on:\n"
                "- the current state\n"
                "- the latest user message\n"
                "- optional supporting conversation context\n\n"

                "The scheduling state is the PRIMARY source of truth.\n"
                "History is only supporting context.\n\n"

                "Valid roles in the database:\n"
                "- Python Dev\n"
                "- Sql Dev\n"
                "- Analyst\n"
                "- ML\n\n"

                "Normalize role names, for example:\n"
                '- "python developer" → "Python Dev"\n'
                '- "sql developer" → "Sql Dev"\n'
                '- "data analyst" → "Analyst"\n'
                '- "ml engineer" → "ML"\n\n'

                "Rules:\n"
                "1. Only update fields that can be clearly inferred.\n"
                "2. Keep existing values unless clearly changed.\n"
                "3. Do not invent slot IDs or bookings.\n"
                "4. If user indicates rescheduling → set booking_status='rescheduling'.\n"
                "5. If user selects a slot (e.g. 'first one') → set selected_slot_id if possible.\n"
                "6. If input unrelated → return state unchanged.\n\n"

                "Return ONLY JSON:\n"
                "{{\n"
                '  "updated_state": {{\n'
                '    "role": string or null,\n'
                '    "requested_date": string or null,\n'
                '    "requested_time": string or null,\n'
                '    "requested_from_time": string or null,\n'
                '    "booking_status": "none" | "pending" | "booked" | "rescheduling",\n'
                '    "suggested_slots": [\n'
                '      {{"ScheduleID": int, "date": "YYYY-MM-DD", "time": "HH:MM:SS", "position": string}}\n'
                '    ],\n'
                '    "selected_slot_id": int or null,\n'
                '    "booked_slot_id": int or null,\n'
                '    "booked_slot_date": string or null,\n'
                '    "booked_slot_time": string or null,\n'
                '    "last_action": string or null\n'
                "  }}\n"
                "}}"
            ),
            (
                "user",
                "Current state:\n{current_state}\n\n"
                "Recent context:\n{history_text}\n\n"
                "User message:\n{user_input}"
            )
        ])

        self._state_update_chain = state_update_prompt | self.llm | self._state_parser

        # ---------------------------
        # ACTIVE SCHEDULING AGENT
        # ---------------------------
        tools = self._create_tools()

        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt = (
                "You are a scheduling assistant for interviews.\n\n"

                "IMPORTANT:\n"
                "- This is the ACTIVE scheduling step.\n"
                "- The scheduling state is the PRIMARY source of truth.\n"
                "- Use conversation history only as supporting context when needed.\n"
                "- Use tools for all database actions.\n"
                "- Do not invent slots, bookings, releases, dates, or times.\n\n"

                "Valid interview roles in the database are exactly:\n"
                "- Python Dev\n"
                "- Sql Dev\n"
                "- Analyst\n"
                "- ML\n\n"

                "If role wording differs, interpret it into the closest valid database role.\n"
                "Examples:\n"
                '- "python developer" -> "Python Dev"\n'
                '- "sql developer" -> "Sql Dev"\n'
                '- "data analyst" -> "Analyst"\n'
                '- "ml engineer" -> "ML"\n\n'

                "Scheduling rules:\n"
                "1. If the role is missing, ask the user for it.\n"
                "2. If the role exists and the user asks for available slots, fetch up to 3 relevant slots.\n"
                "3. If the user explicitly requested a date/time or a starting date/time, use that information.\n"
                "4. If the user did NOT explicitly request a date/time, use the provided current date/time as the default starting point for the search.\n"
                "5. Do NOT treat the default current date/time as user-provided information.\n"
                "6. Do NOT overwrite scheduling state fields with default current date/time unless the user explicitly requested them.\n"
                "7. If the user selected a slot, book it.\n"
                "8. If the user wants to reschedule and a slot is already booked, release the previous slot first, then book the new one.\n"
                "9. Only update state fields that changed because of the current scheduling action.\n\n"

                "Return ONLY valid JSON in this format:\n"
                "{\n"
                '  "response": "final text for the user",\n'
                '  "schedule_update": {\n'
                '    "role": string or null,\n'
                '    "requested_date": string or null,\n'
                '    "requested_time": string or null,\n'
                '    "requested_from_time": string or null,\n'
                '    "booking_status": "none" | "pending" | "booked" | "rescheduling",\n'
                '    "suggested_slots": [\n'
                '      {"ScheduleID": int, "date": "YYYY-MM-DD", "time": "HH:MM:SS", "position": string}\n'
                '    ],\n'
                '    "selected_slot_id": int or null,\n'
                '    "booked_slot_id": int or null,\n'
                '    "booked_slot_date": string or null,\n'
                '    "booked_slot_time": string or null,\n'
                '    "last_action": string or null\n'
                "  }\n"
                "}"
            )
        )

    # ---------------------------
    # TOOLS
    # ---------------------------
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

    # ---------------------------
    # HELPERS
    # ---------------------------
    def _format_history(self, history_messages):
        lines = []
        for m in history_messages:
            if isinstance(m, HumanMessage):
                lines.append(f"User: {m.content}")
            elif isinstance(m, AIMessage):
                lines.append(f"Assistant: {m.content}")
        return "\n".join(lines)

    # ---------------------------
    # PASSIVE UPDATE
    # ---------------------------
    def update_schedule_state(self, user_input: str, session_id: str, history_messages):
        current_state = self._state_store.get_state(session_id)

        result = self._state_update_chain.invoke({
            "current_state": json.dumps(current_state),
            "history_text": self._format_history(history_messages),
            "user_input": user_input,
        })

        updated_state = result["updated_state"]
        self._state_store.set_state(session_id, updated_state)

        return updated_state

    # ---------------------------
    # ACTIVE SCHEDULING
    # ---------------------------
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
                        #f"Recent conversation:\n{history_text}\n\n"
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