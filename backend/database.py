import mysql.connector

# ✅ MySQL Database Connection
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="ResumeDB"
    )

# ✅ Save Resume Data in MySQL
def save_to_db(name, ats_score, skills, education, experience, match_score):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO Resumes (name, ats_score, skills, education, experience, match_percentage)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (name, ats_score, ', '.join(skills), education, experience, match_score))
    conn.commit()
    cursor.close()
    conn.close()

# ✅ Fetch Past Resumes from MySQL
def fetch_past_resumes():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, ats_score, skills, match_percentage FROM Resumes ORDER BY id DESC LIMIT 5")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data
