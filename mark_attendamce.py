import csv
from datetime import datetime

FILE_NAME = "attendance.csv"

def mark_attendance(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(
        FILE_NAME,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [name, date, time]
        )