from tinydb import TinyDB, Query
from tinydb.operations import set
import os
from datetime import datetime

DB_FILE = "transcriptions.json"
UPLOADS_DIR = "uploads"

db = TinyDB(DB_FILE)

def add_transcription(mode, transcription, audio_data=None):
    """Adds a new transcription record to the database."""
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

    db.insert({
        'timestamp': timestamp.isoformat(),
        'mode': mode,
        'audio_path': audio_path,
        'transcription': transcription
    })

def get_all_transcriptions():
    """Retrieves all transcription records from the database."""
    records = db.all()
    # Sort records by timestamp in descending order
    records.sort(key=lambda x: x['timestamp'], reverse=True)
    # Convert to a format similar to the old one for compatibility with app.py
    return [
        (r['timestamp'], r['mode'], r['audio_path'], r['transcription'])
        for r in records
    ]

def clear_history():
    """Clears all transcription records from the database."""
    db.truncate()
    # Also remove uploaded files
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            os.remove(os.path.join(UPLOADS_DIR, filename))