from copy import deepcopy


DEFAULT_SCHEDULE_STATE = {
    "role": None,
    "requested_date": None,
    "requested_time": None,
    "requested_from_time": None,
    "booking_status": "none",      # none / pending / booked / rescheduling
    "suggested_slots": [],
    "selected_slot_id": None,
    "booked_slot_id": None,
    "booked_slot_date": None,
    "booked_slot_time": None,
    "last_action": None
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