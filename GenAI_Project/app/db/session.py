# should we put this functionality in another file and make sched_advisor a folder (module)?
# Need to check if we need to add a column that indicates if the time slot is occupied or not    
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "Server=LAPTOP-CVQUU14B;"
    "Database=Tech;"
    "Trusted_Connection=yes;"
)

connection_url = f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}"

engine = create_engine(
    connection_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)