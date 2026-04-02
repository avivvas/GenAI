import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

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
            system_prompt=(
                "You are a scheduling assistant for a company.\n"
                "You receive:\n"
                "- the user's latest message\n"
                "- recent conversation history\n"
                "- the current scheduling state\n\n"

                "Your job is to update the scheduling state and produce the user-facing response.\n\n"

                "Rules:\n"
                "1. The role is required before suggesting or booking interview slots.\n"
                "2. If the role is missing, ask the user for it.\n"
                "3. If enough information exists, call the scheduling tool to fetch up to 3 slots.\n"
                "4. If the user selected a valid slot, call the booking tool.\n"
                "5. If the user wants to reschedule, set booking_status to 'rescheduling'.\n"
                "6. Use the current state and latest user message to determine what changed.\n"
                "7. Do not invent slots. Use tools only.\n\n"

                "Return ONLY valid JSON with this structure:\n"
                "{\n"
                '  "updated_state": {\n'
                '    "role": string or null,\n'
                '    "requested_date": string or null,\n'
                '    "requested_time": string or null,\n'
                '    "requested_from_time": string or null,\n'
                '    "booking_status": "none" | "pending" | "booked" | "rescheduling",\n'
                '    "suggested_slots": [\n'
                '      {"ScheduleID": int, "date": "YYYY-MM-DD", "time": "HH:MM:SS", "position": string}\n'
                '    ],\n'
                '    "selected_slot_id": int or null,\n'
                '    "missing_fields": [string]\n'
                "  },\n"
                '  "response": "text shown to the user",\n'
                '  "action": "ask_missing_info" | "suggest_slots" | "booked" | "rescheduling" | "continue"\n'
                "}\n\n"

                "If there are no available slots, response should clearly say so.\n"
                "If role is missing, missing_fields should include 'role'.\n"
                "If slots are suggested, put them in updated_state.suggested_slots.\n"
                "If booking succeeds, booking_status must become 'booked'."
            ),
        )

    def _create_tools(self):
        repository = self._repository

        @tool
        def get_next_three_slots(role: str, start_date: str | None = None, start_time: str | None = None) -> str:
            """
            Return up to 3 available interview slots for the given role.
            If start_date and start_time are provided, return the next 3 slots from that point onward.
            """
            if not role:
                return json.dumps([])

            if start_date and start_time:
                rows = repository.get_next_three_available_slots_from(
                    role=role,
                    start_date=start_date,
                    start_time=start_time,
                )
            else:
                rows = repository.get_next_three_available_slots(role=role)

            rows_as_dicts = []
            for row in rows:
                rows_as_dicts.append({
                    "ScheduleID": row["ScheduleID"],
                    "date": str(row["date"]),
                    "time": str(row["time"]),
                    "position": row["position"],
                })

            return json.dumps(rows_as_dicts)

        @tool
        def book_slot(schedule_id: int) -> str:
            """
            Book a specific slot by ScheduleID. Returns JSON with success true/false.
            """
            success = repository.book_slot(schedule_id)
            return json.dumps({
                "success": success,
                "schedule_id": schedule_id,
            })

        return [get_next_three_slots, book_slot]

    def invoke(self, user_input: str, session_id: str, history_text : str) -> str:
        current_state = self._state_store.get_state(session_id)

        result = self._agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Current scheduling state:\n{json.dumps(current_state, ensure_ascii=False)}\n\n"
                        f"Recent conversation:\n{history_text}\n\n"
                        f"Latest user message:\n{user_input}"
                    ),
                }
            ]
        })

        final_text = result["messages"][-1].content
        parsed = json.loads(final_text)

        self._state_store.set_state(session_id, parsed["updated_state"])

        return parsed["response"]