from sqlalchemy import text
from .session import SessionLocal


class ScheduleRepository:
    def __init__(self):
        self._session_factory = SessionLocal

    def get_next_three_available_slots(self, role: str):
        print('invoked get_next_three_available_slots')
        db = self._session_factory()
        try:
            query = text("""
                SELECT TOP 3 ScheduleID, [date], [time], position
                FROM Schedule
                WHERE available = 1
                  AND position = :role
                ORDER BY [date], [time], ScheduleID
            """)
            result = db.execute(query, {"role": role})
            return result.mappings().all()
        finally:
            db.close()

    def get_next_three_available_slots_from(self, role: str, start_date: str, start_time: str):
        print('invoked get_next_three_available_slots_from')
        db = self._session_factory()
        try:
            query = text("""
                SELECT TOP 3 ScheduleID, [date], [time], position
                FROM Schedule
                WHERE available = 1
                  AND position = :role
                  AND (
                        [date] > :start_date
                        OR ([date] = :start_date AND [time] >= :start_time)
                  )
                ORDER BY [date], [time], ScheduleID
            """)
            result = db.execute(query, {
                "role": role,
                "start_date": start_date,
                "start_time": start_time,
            })
            return result.mappings().all()
        finally:
            db.close()

    def book_slot(self, schedule_id: int):
        db = self._session_factory()
        try:
            query = text("""
                UPDATE Schedule
                SET available = 0
                WHERE ScheduleID = :schedule_id
                  AND available = 1
            """)
            result = db.execute(query, {"schedule_id": schedule_id})
            db.commit()
            return result.rowcount == 1
        finally:
            db.close()