import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS unknown_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    time TEXT
)
""")

conn.commit()
conn.close()

print("Database Created")