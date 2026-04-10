from copy import deepcopy


DEFAULT_SCHEDULE_STATE = {
    "role": None,

    "suggested_slots": [],

    "booked_slot_id": None,
    "booked_slot_date": None,
    "booked_slot_time": None,

    "booking_status": "none",  # none | pending | booked | rescheduling

    "last_action": None,

    "last_offered_start_date": None,
    "last_offered_start_time": None,
}


class ScheduleStateStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def get_state(self, session_id: str) -> dict:
        if session_id not in self._store:
            self._store[session_id] = deepcopy(DEFAULT_SCHEDULE_STATE)
        return self._store[session_id]

    def set_state(self, session_id: str, new_state: dict) -> None:
        self._store[session_id] = new_state

    def reset_state(self, session_id: str) -> None:
        self._store.pop(session_id, None)