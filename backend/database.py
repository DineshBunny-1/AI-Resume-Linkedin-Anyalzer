import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "resume_data.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            ats_score INTEGER,
            skills TEXT,
            match_score INTEGER,
            education TEXT,
            experience TEXT
        )
    """)
    conn.commit()
    return conn

def save_to_db(name, ats_score, skills, education, experience, match_score):
    conn = get_connection()
    conn.execute(
        "INSERT INTO resumes (name, ats_score, skills, match_score, education, experience) VALUES (?,?,?,?,?,?)",
        (name, ats_score, ", ".join(skills), match_score, education, experience)
    )
    conn.commit()
    conn.close()

def fetch_past_resumes():
    conn = get_connection()
    rows = conn.execute(
        "SELECT name, ats_score, skills, match_score, education FROM resumes ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return rows
