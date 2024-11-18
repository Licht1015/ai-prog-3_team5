import streamlit as st
import requests
import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


# .streamlit/secrets.toml からAPIキーを取得
API_KEY = st.secrets["weather_api_key"]
DB_NAME = "health_data.db"

# データベース作成関数
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 生活データ用テーブル
    c.execute('''CREATE TABLE IF NOT EXISTS lifestyle_data (
                    date TEXT PRIMARY KEY,
                    sleep_hours REAL,
                    exercise_minutes REAL,
                    meal_quality INTEGER)''')
    # 気象データ用テーブル
    c.execute('''CREATE TABLE IF NOT EXISTS weather_data (
                    date TEXT PRIMARY KEY,
                    pressure REAL)''')
    conn.commit()
    conn.close()

# OpenWeatherMap APIで気圧データ取得関数
def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        pressure = data['main']['pressure']
        return pressure
    else:
        st.error("天気データを取得できませんでした。場所を確認してください。")
        return None

# データ保存関数
def save_data(date, sleep_hours, exercise_minutes, meal_quality, pressure):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 生活データの保存
    c.execute("INSERT OR REPLACE INTO lifestyle_data (date, sleep_hours, exercise_minutes, meal_quality) VALUES (?, ?, ?, ?)",
              (date, sleep_hours, exercise_minutes, meal_quality))
    # 気象データの保存
    c.execute("INSERT OR REPLACE INTO weather_data (date, pressure) VALUES (?, ?)",
              (date, pressure))
    conn.commit()
    conn.close()

    # データ取得関数
def load_data():
    conn = sqlite3.connect(DB_NAME)
    lifestyle_df = pd.read_sql("SELECT * FROM lifestyle_data", conn)
    weather_df = pd.read_sql("SELECT * FROM weather_data", conn)
    conn.close()
    return lifestyle_df, weather_df

# Streamlitのインターフェース
def main():
    st.title("健康管理と気象条件アプリ")
    st.write("気圧と生活状況を照らし合わせて健康管理をサポートします。")

    init_db()  # データベース初期化

    # 場所の入力
    city = st.text_input("場所を入力してください（例: Tokyo）")

    # 生活状況の入力
    sleep_hours = st.slider("昨晩の睡眠時間（時間）", 0, 12, 7)
    exercise_minutes = st.slider("運動時間（分）", 0, 180, 30)
    meal_quality = st.selectbox("ご飯の量の満足度（1-5）", list(range(1, 6)))

    if st.button("データ保存"):
        if city:
            pressure = get_weather_data(city)
            if pressure:
                date = datetime.now().strftime("%Y-%m-%d")
                save_data(date, sleep_hours, exercise_minutes, meal_quality, pressure)
                st.success("データが保存されました！")

    # データの表示
    st.write("### 過去のデータ")
    lifestyle_df, weather_df = load_data()
    if not lifestyle_df.empty and not weather_df.empty:
        st.write("生活データ")
        st.dataframe(lifestyle_df)
        st.write("気象データ")
        st.dataframe(weather_df)
    else:
        st.write("まだデータがありません。")

if __name__ == "__main__":
    main()

# データ取得関数を追加
def load_lifestyle_and_weather_data():
    conn = sqlite3.connect(DB_NAME)
    lifestyle_df = pd.read_sql("SELECT * FROM lifestyle_data", conn)
    weather_df = pd.read_sql("SELECT * FROM weather_data", conn)
    conn.close()
    return lifestyle_df, weather_df

# グラフ表示用関数を追加
def display_graphs():
    st.write("### グラフ表示")
    lifestyle_df, weather_df = load_lifestyle_and_weather_data()

    # 生活データのグラフ
    if not lifestyle_df.empty:
        fig, ax = plt.subplots()
        ax.plot(lifestyle_df['date'], lifestyle_df['sleep_hours'], marker='o', color='b', label='睡眠時間 (時間)')
        ax.set_title("睡眠時間の推移")
        ax.set_xlabel("日付")
        ax.set_ylabel("time (h)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("生活データがまだありません。")

    # 気象データのグラフ
    if not weather_df.empty:
        fig, ax = plt.subplots()
        ax.plot(weather_df['date'], weather_df['pressure'], marker='x', color='r', label='気圧 (hPa)')
        ax.set_title("気圧の推移")
        ax.set_xlabel("日付")
        ax.set_ylabel("atmospheric pressure (hPa)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("気象データがまだありません。")

# main関数の修正
def main():
    st.title("健康管理と気象条件アプリ")
    st.write("気圧と生活状況を照らし合わせて健康管理をサポートします。")

    init_db()  # データベース初期化

    # 場所の入力（固有のキーを設定）
    city = st.text_input("場所を入力してください（例: Tokyo）", key="city_input")

    # 生活状況の入力（固有のキーを設定）
    sleep_hours = st.slider("昨晩の睡眠時間（時間）", 0, 12, 7, key="sleep_hours_slider")
    exercise_minutes = st.slider("運動時間（分）", 0, 180, 30, key="exercise_minutes_slider")
    meal_quality = st.selectbox("ご飯の量の満足度（1-5）", list(range(1, 6)), key="meal_quality_selectbox")

    if st.button("データ保存", key="save_button"):
        if city:
            pressure = get_weather_data(city)
            if pressure:
                date = datetime.now().strftime("%Y-%m-%d")
                save_data(date, sleep_hours, exercise_minutes, meal_quality, pressure)
                st.success("データが保存されました！")

    # データの表示
    st.write("### 過去のデータ")
    lifestyle_df, weather_df = load_data()
    if not lifestyle_df.empty and not weather_df.empty:
        st.write("生活データ")
        st.dataframe(lifestyle_df)
        st.write("気象データ")
        st.dataframe(weather_df)
    else:
        st.write("まだデータがありません。")

    # グラフの表示
    display_graphs()

if __name__ == "__main__":
    main()

