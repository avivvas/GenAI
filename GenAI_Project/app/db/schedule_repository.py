from sqlalchemy import text
import warnings
from sqlalchemy.exc import SAWarning
from .session import SessionLocal


class ScheduleRepository:
    def __init__(self):
        self._session_factory = SessionLocal

        # Added in order to exclude such warnings from the output
        warnings.filterwarnings(
            "ignore",
            message=".*Unrecognized server version info.*",
            category=SAWarning,
        )

    def get_next_three_available_slots_from(self, role: str, start_date: str, start_time: str):
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

    def book_slot(self, schedule_id: int) -> bool:
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

    def release_slot(self, schedule_id: int) -> bool:
        db = self._session_factory()
        try:
            query = text("""
                UPDATE Schedule
                SET available = 1
                WHERE ScheduleID = :schedule_id
            """)

            result = db.execute(query, {"schedule_id": schedule_id})
            db.commit()

            return result.rowcount == 1

        finally:
            db.close()