import sqlite3
from datetime import datetime, timedelta

def init_db():
    """データベースの初期化とテーブル作成"""
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    
    # テーブルの作成
    c.execute('''CREATE TABLE IF NOT EXISTS lifestyle_data (
                    date TEXT PRIMARY KEY,
                    sleep_hours REAL,
                    exercise_minutes REAL,
                    meal_quality INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS weather_data (
                    date TEXT PRIMARY KEY,
                    pressure REAL,
                    temperature REAL,
                    humidity REAL,
                    weather_main TEXT,
                    weather_description TEXT,
                    icon TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS daily_advice (
                    date TEXT PRIMARY KEY,
                    advice TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city_name TEXT UNIQUE,
                    is_selected INTEGER DEFAULT 0)''')
    
    conn.commit()
    conn.close()

def insert_sample_data():
    # データベースの初期化
    init_db()
    
    # データベースに接続
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    
    # サンプルデータの準備（日付を修正）
    lifestyle_data = [
        ('2024-11-29', 7.5, 45, 4),
        ('2024-11-30', 7.5, 45, 4),
        ('2024-12-01', 7.5, 45, 4),
        ('2024-12-02', 6.5, 30, 3),
        ('2024-12-03', 8.0, 60, 5),
        ('2024-12-04', 7.0, 20, 4),
        ('2024-12-05', 6.0, 40, 3),
        ('2024-12-06', 7.5, 50, 4),
        ('2024-12-07', 8.5, 70, 5),
        ('2024-12-08', 6.5, 25, 3),
    ]
    
    weather_data = [
        ('2024-11-29', 1013.2, 18.5, 65, 'Clear', '晴れ', '01d'),
        ('2024-11-30', 1013.2, 18.5, 65, 'Clear', '晴れ', '01d'),
        ('2024-12-01', 1013.2, 18.5, 65, 'Clear', '晴れ', '01d'),
        ('2024-12-02', 1012.8, 17.8, 70, 'Clouds', '曇り', '02d'),
        ('2024-12-03', 1014.5, 19.2, 60, 'Clear', '晴れ', '01d'),
        ('2024-12-04', 1011.0, 16.5, 75, 'Rain', '小雨', '10d'),
        ('2024-12-05', 1010.5, 15.8, 80, 'Rain', '雨', '09d'),
        ('2024-12-06', 1012.2, 17.5, 68, 'Clouds', '薄曇り', '03d'),
        ('2024-12-07', 1013.8, 18.8, 62, 'Clear', '快晴', '01d'),
        ('2024-12-08', 1012.5, 17.2, 72, 'Clouds', '曇り', '02d'),
    ]
    
    advice_data = [
        ('2024-12-06', '運動時間が増えているので、適切な休息も取りましょう。'),
        ('2024-12-07', '睡眠時間が十分で素晴らしいです。この調子を維持しましょう。'),
        ('2024-12-08', '睡眠時間が短めなので、今晩は早めの就寝を心がけましょう。')
    ]

    locations = [
        ('Tokyo', 1),
        ('Osaka', 0),
        ('Fukuoka', 0),
        ('Sapporo', 0)
    ]

    try:
        # 既存のデータを削除（オプション）
        c.execute("DELETE FROM lifestyle_data")
        c.execute("DELETE FROM weather_data")
        c.execute("DELETE FROM daily_advice")
        c.execute("DELETE FROM locations")
        
        # データの挿入
        c.executemany("""
            INSERT OR REPLACE INTO lifestyle_data 
            (date, sleep_hours, exercise_minutes, meal_quality) 
            VALUES (?, ?, ?, ?)
        """, lifestyle_data)

        c.executemany("""
            INSERT OR REPLACE INTO weather_data 
            (date, pressure, temperature, humidity, weather_main, weather_description, icon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, weather_data)

        c.executemany("""
            INSERT OR REPLACE INTO daily_advice 
            (date, advice)
            VALUES (?, ?)
        """, advice_data)

        c.executemany("""
            INSERT OR IGNORE INTO locations 
            (city_name, is_selected)
            VALUES (?, ?)
        """, locations)

        # 変更を確定
        conn.commit()
        print("サンプルデータの挿入が完了しました。")

    except sqlite3.Error as e:
        print(f"エラーが発生しました: {e}")
        conn.rollback()

    finally:
        # 接続を閉じる
        conn.close()

if __name__ == "__main__":
    insert_sample_data()