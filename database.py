import sqlite3
import datetime
import pandas as pd
from typing import List, Dict, Optional, Tuple

DB_PATH = "attendance.db"

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number TEXT,
            class TEXT,
            section TEXT,
            registration_number TEXT,
            created_at TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            timestamp TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)
    
    conn.commit()
    conn.close()

def add_student(name: str, roll_number: str = "", class_name: str = "", 
                section: str = "", registration_number: str = "") -> int:
    """Add a new student to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    created_at = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO students (name, roll_number, class, section, registration_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, roll_number, class_name, section, registration_number, created_at))
    
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return student_id

def get_all_students() -> List[Dict]:
    """Get all students from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, roll_number, class, section, registration_number, created_at
        FROM students
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    students = []
    for row in rows:
        students.append({
            'id': row[0],
            'name': row[1],
            'roll_number': row[2],
            'class': row[3],
            'section': row[4],
            'registration_number': row[5],
            'created_at': row[6]
        })
    
    return students

def get_student_by_id(student_id: int) -> Optional[Dict]:
    """Get a student by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, roll_number, class, section, registration_number
        FROM students
        WHERE id = ?
    """, (student_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'name': row[1],
            'roll_number': row[2],
            'class': row[3],
            'section': row[4],
            'registration_number': row[5]
        }
    return None

def delete_student(student_id: int):
    """Delete a student and their attendance records"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    
    conn.commit()
    conn.close()

def mark_attendance(student_id: int, name: str):
    """Mark attendance for a student"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO attendance (student_id, name, timestamp)
        VALUES (?, ?, ?)
    """, (student_id, name, timestamp))
    
    conn.commit()
    conn.close()

def get_attendance_records(period: str = "all") -> pd.DataFrame:
    """Get attendance records with optional filtering"""
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT id, student_id, name, timestamp FROM attendance"
    
    if period == "today":
        today = datetime.date.today().isoformat()
        query += f" WHERE date(timestamp) = '{today}'"
    elif period == "week":
        week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        query += f" WHERE date(timestamp) >= '{week_ago}'"
    elif period == "month":
        month_ago = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        query += f" WHERE date(timestamp) >= '{month_ago}'"
    
    query += " ORDER BY timestamp DESC"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
    
    return df

def get_attendance_stats(days: int = 30) -> Tuple[List[str], List[int]]:
    """Get attendance statistics for the last N days"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    
    if df.empty:
        dates = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%d %b") 
                 for i in range(days-1, -1, -1)]
        return dates, [0] * days
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    last_n_days = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(days-1, -1, -1)]
    counts = [int(df[df['date'] == d].shape[0]) for d in last_n_days]
    dates = [d.strftime("%d %b") for d in last_n_days]
    
    return dates, counts

def get_total_students() -> int:
    """Get total number of students"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_today_attendance_count() -> int:
    """Get attendance count for today"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.date.today().isoformat()
    cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date(timestamp) = ?", (today,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def bulk_import_students(students_data: List[Dict]) -> Tuple[int, List[str]]:
    """Bulk import students from CSV data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    success_count = 0
    errors = []
    
    for idx, student in enumerate(students_data):
        try:
            name_val = student.get('name', '')
            name = str(name_val).strip() if pd.notna(name_val) else ''
            if not name:
                errors.append(f"Row {idx + 1}: Name is required")
                continue
            
            roll_val = student.get('roll_number', '')
            roll_number = str(roll_val).strip() if pd.notna(roll_val) else ''
            
            class_val = student.get('class', '')
            class_name = str(class_val).strip() if pd.notna(class_val) else ''
            
            section_val = student.get('section', '')
            section = str(section_val).strip() if pd.notna(section_val) else ''
            
            reg_val = student.get('registration_number', '')
            registration_number = str(reg_val).strip() if pd.notna(reg_val) else ''
            
            cursor.execute("SELECT id FROM students WHERE name = ? AND roll_number = ?", (name, roll_number))
            if cursor.fetchone():
                errors.append(f"Row {idx + 1}: Student '{name}' with roll number '{roll_number}' already exists")
                continue
            
            created_at = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO students (name, roll_number, class, section, registration_number, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, roll_number, class_name, section, registration_number, created_at))
            
            success_count += 1
            
        except Exception as e:
            errors.append(f"Row {idx + 1}: {str(e)}")
    
    conn.commit()
    conn.close()
    
    return success_count, errors

def get_class_wise_attendance(period: str = "today") -> pd.DataFrame:
    """Get class-wise attendance statistics"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT s.class, s.section, COUNT(DISTINCT a.student_id) as present,
               (SELECT COUNT(*) FROM students WHERE class = s.class AND section = s.section) as total
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
    """
    
    if period == "today":
        today = datetime.date.today().isoformat()
        query += f" AND date(a.timestamp) = '{today}'"
    elif period == "week":
        week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        query += f" AND date(a.timestamp) >= '{week_ago}'"
    elif period == "month":
        month_ago = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        query += f" AND date(a.timestamp) >= '{month_ago}'"
    
    query += " GROUP BY s.class, s.section ORDER BY s.class, s.section"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['attendance_rate'] = (df['present'] / df['total'] * 100).round(2)
    
    return df

def get_student_attendance_summary() -> pd.DataFrame:
    """Get detailed attendance summary for all students"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT s.id, s.name, s.roll_number, s.class, s.section,
               COUNT(a.id) as total_attendance,
               MAX(a.timestamp) as last_attendance
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
        GROUP BY s.id
        ORDER BY total_attendance DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df
