import os
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def get_db_url():
    url = os.getenv("DATABASE_URL", "mariadb://root:password@localhost:3306/fleetfast_dev")
    return url.replace('mariadb://', 'mysql+pymysql://', 1)

engine = create_engine(get_db_url())
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ DB connected")
            return True
    except Exception as e:
        print(f"❌ DB failed: {e}")
        return False

if __name__ == "__main__":
    test_db()