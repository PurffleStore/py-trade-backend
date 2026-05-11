# db.py
import os
import pyodbc

MODE = os.getenv("MODE", "local").lower()

# Local (Windows) connection
LOCAL_SQL_SERVER   = os.getenv("LOCAL_SQL_SERVER", r"localhost\SQLEXPRESS")
LOCAL_SQL_DATABASE = os.getenv("LOCAL_SQL_DATABASE", "PyTrade")
LOCAL_SQL_DRIVER   = os.getenv("LOCAL_SQL_DRIVER", "{ODBC Driver 17 for SQL Server}")

# Remote (RDS/HF) SQL Auth
RDS_SQL_SERVER   = os.getenv("RDS_SQL_SERVER", "")
RDS_SQL_DATABASE = os.getenv("RDS_SQL_DATABASE", "PyTrade")
RDS_SQL_USER     = os.getenv("RDS_SQL_USER", "")
RDS_SQL_PASSWORD = os.getenv("RDS_SQL_PASSWORD", "")
RDS_SQL_DRIVER   = os.getenv("RDS_SQL_DRIVER", "{ODBC Driver 18 for SQL Server}")
RDS_ENCRYPT      = os.getenv("RDS_ENCRYPT", "yes")
RDS_TRUST_CERT   = os.getenv("RDS_TRUST_SERVER_CERT", "yes")

def get_db_connection():
    if MODE == "local":
        return pyodbc.connect(
            f"DRIVER={LOCAL_SQL_DRIVER};"
            f"SERVER={LOCAL_SQL_SERVER};"
            f"DATABASE={LOCAL_SQL_DATABASE};"
            f"Trusted_Connection=yes;"
        )
    else:
        return pyodbc.connect(
            f"DRIVER={RDS_SQL_DRIVER};"
            f"SERVER={RDS_SQL_SERVER};"
            f"DATABASE={RDS_SQL_DATABASE};"
            f"UID={RDS_SQL_USER};PWD={RDS_SQL_PASSWORD};"
            f"Encrypt={RDS_ENCRYPT};TrustServerCertificate={RDS_TRUST_CERT};"
            f"Connection Timeout=30;"
        )

def ensure_user_table_exists():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute('''
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Users' AND xtype='U')
            CREATE TABLE Users (
                id INT IDENTITY(1,1) PRIMARY KEY,
                name NVARCHAR(120) NOT NULL,
                phone NVARCHAR(50) NOT NULL,
                email NVARCHAR(120) UNIQUE NOT NULL,
                password NVARCHAR(255) NOT NULL
            )
        ''')
        conn.commit()
    finally:
        try: cur.close()
        except: pass
        conn.close()


def ensure_community_table_exists() -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Step 1: Create table — single statement, same pattern as ensure_user_table_exists
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Community' AND xtype='U')
                CREATE TABLE Community (
                    id            INT IDENTITY(1,1) PRIMARY KEY,
                    user_id       INT NOT NULL,
                    user_name     NVARCHAR(200) NOT NULL,
                    title         NVARCHAR(300) NULL,
                    category      NVARCHAR(100) NULL,
                    tags          NVARCHAR(1000) NULL,
                    body          NVARCHAR(MAX) NOT NULL,
                    like_count    INT NOT NULL DEFAULT 0,
                    dislike_count INT NOT NULL DEFAULT 0,
                    comment_count INT NOT NULL DEFAULT 0,
                    created_at    DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
                )
        """)
        conn.commit()

        # Step 2: Indexes — each in its own execute so the table is visible
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='IX_Community_UserId' AND object_id=OBJECT_ID('Community'))
                CREATE INDEX IX_Community_UserId ON Community(user_id)
        """)
        conn.commit()

        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='IX_Community_CreatedAt' AND object_id=OBJECT_ID('Community'))
                CREATE INDEX IX_Community_CreatedAt ON Community(created_at)
        """)
        conn.commit()

        # Step 3: Add missing columns for older deployments — one execute per column
        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                           WHERE TABLE_NAME='Community' AND COLUMN_NAME='like_count')
                ALTER TABLE Community ADD like_count INT NOT NULL DEFAULT 0
        """)
        conn.commit()

        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                           WHERE TABLE_NAME='Community' AND COLUMN_NAME='dislike_count')
                ALTER TABLE Community ADD dislike_count INT NOT NULL DEFAULT 0
        """)
        conn.commit()

        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                           WHERE TABLE_NAME='Community' AND COLUMN_NAME='comment_count')
                ALTER TABLE Community ADD comment_count INT NOT NULL DEFAULT 0
        """)
        conn.commit()

    finally:
        try: cursor.close()
        except: pass
        conn.close()

