import sqlite3
import os
from datetime import datetime

DB_FILE = "transcriptions.db"
UPLOADS_DIR = "uploads"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    """Creates the transcriptions table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            mode TEXT NOT NULL,
            audio_path TEXT,
            transcription TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_transcription(mode, transcription, audio_data=None):
    """Adds a new transcription record to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now()
    audio_path = None

    if audio_data:
        if not os.path.exists(UPLOADS_DIR):
            os.makedirs(UPLOADS_DIR)
        
        # Create a unique filename for the audio file
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
        audio_path = os.path.join(UPLOADS_DIR, filename)
        
        # Write the audio data to the file
        with open(audio_path, "wb") as f:
            f.write(audio_data)

    cursor.execute(
        "INSERT INTO transcriptions (timestamp, mode, audio_path, transcription) VALUES (?, ?, ?, ?)",
        (timestamp, mode, audio_path, transcription)
    )
    conn.commit()
    conn.close()

def get_all_transcriptions():
    """Retrieves all transcription records from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, mode, audio_path, transcription FROM transcriptions ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return records

# --- Initialize the database and table on startup ---
create_table()
