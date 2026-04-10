"""Database initialization and helper functions"""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'complaints.db')

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()

    # Create tables if they don't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            text         TEXT NOT NULL,
            area_name    TEXT DEFAULT '',
            latitude     REAL NOT NULL,
            longitude    REAL NOT NULL,
            category     TEXT NOT NULL,
            priority     TEXT NOT NULL,
            is_fake      INTEGER DEFAULT 0,
            department   TEXT NOT NULL,
            status       TEXT DEFAULT 'Open',
            submitter    TEXT DEFAULT 'Citizen',
            image        TEXT DEFAULT '',
            voice_note   TEXT DEFAULT '',
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.execute(
        "INSERT OR IGNORE INTO admin_users (username, password) VALUES ('admin', 'admin123')"
    )

    # --- MIGRATE OLD DATABASES ---
    # Check which columns exist and add missing ones
    existing = {row[1] for row in conn.execute("PRAGMA table_info(complaints)")}

    if 'image' not in existing:
        conn.execute("ALTER TABLE complaints ADD COLUMN image TEXT DEFAULT ''")
        print("[DB] Added column: image")

    if 'voice_note' not in existing:
        conn.execute("ALTER TABLE complaints ADD COLUMN voice_note TEXT DEFAULT ''")
        print("[DB] Added column: voice_note")

    conn.commit()
    conn.close()

def dept_for_category(cat):
    mapping = {
        'Road':        'PWD (Public Works Dept)',
        'Water':       'Water Supply Board',
        'Electricity': 'Electric Department',
        'Sanitation':  'Municipal Corporation',
        'PublicSafety':'Police Department',
    }
    return mapping.get(cat, 'General Administration')