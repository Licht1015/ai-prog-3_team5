import streamlit as st
import requests
import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import requests
import os


# Ollamaのチャットモデルの初期化
chat_model = ChatOllama(
    model="llama2",
    callbacks=[StreamingStdOutCallbackHandler()],
)

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
    
    # 気象データ用テーブル（iconカラムを追加）
    c.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            date TEXT PRIMARY KEY,
            pressure REAL,
            temperature REAL,
            humidity REAL,
            weather_main TEXT,
            weather_description TEXT,
            icon TEXT
        )''')
    
    # 地点登録用テーブル
    c.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_name TEXT UNIQUE
        )''')
    
    conn.commit()
    conn.close()

# OpenWeatherMap APIで気圧データ取得関数
def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            'pressure': data['main']['pressure'],
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'weather_main': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon']  # アイコンコード
        }
        return weather_info
    else:
        st.error("天気データを取得できませんでした。場所を確認してください。")
        return None

# Ollamaによるアドバイス生成関数
def get_health_advice(weather_info, sleep_hours, exercise_minutes, meal_quality):
    prompt = f"""
    以下の情報に基づいて、健康に関するアドバイスを提供してください：

    気象条件：
    - 気温: {weather_info['temp']}℃
    - 気圧: {weather_info['pressure']}hPa
    - 湿度: {weather_info['humidity']}%
    - 天気: {weather_info['weather_main']}

    生活状況：
    - 睡眠時間: {sleep_hours}時間
    - 運動時間: {exercise_minutes}分
    - 食事の満足度: {meal_quality}/5
    """

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content

# データ保存関数
def save_data(date, sleep_hours, exercise_minutes, meal_quality, weather_info):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 生活データの保存
    c.execute("INSERT OR REPLACE INTO lifestyle_data VALUES (?, ?, ?, ?)",
              (date, sleep_hours, exercise_minutes, meal_quality))
    # 気象データの保存
    c.execute("INSERT OR REPLACE INTO weather_data VALUES (?, ?, ?, ?, ?, ?, ?)",
              (date, weather_info['pressure'], weather_info['temp'],
               weather_info['humidity'], weather_info['weather_main'],
               weather_info['weather_description'], weather_info['icon']))
    conn.commit()
    conn.close()

# データ取得関数
def load_data():
    conn = sqlite3.connect(DB_NAME)
    lifestyle_df = pd.read_sql("SELECT * FROM lifestyle_data", conn)
    weather_df = pd.read_sql("SELECT * FROM weather_data", conn)
    conn.close()
    return lifestyle_df, weather_df

# グラフ表示関数
def display_graphs():
    lifestyle_df, weather_df = load_data()
    if not lifestyle_df.empty and not weather_df.empty:
        # データを結合
        df = pd.merge(lifestyle_df, weather_df, on='date')
        fig1 = px.scatter(df, x='pressure', y='sleep_hours', title='気圧と睡眠時間の関係')
        st.plotly_chart(fig1)
        fig2 = px.scatter(df, x='pressure', y='exercise_minutes', title='気圧と運動時間の関係')
        st.plotly_chart(fig2)
        fig3 = px.scatter(df, x='pressure', y='meal_quality', title='気圧と食事の満足度の関係')
        st.plotly_chart(fig3)
    else:
        st.write("グラフを表示するためのデータがありません。")

def main():
    st.title("健康管理と気象条件アプリ")
    st.write("気圧と生活状況を照らし合わせて健康管理をサポートします。")

    init_db()

    # 地点登録機能
    st.write("### 地点の登録")
    new_city = st.text_input("新しい地点を登録（例: Tokyo）")
    if st.button("地点を登録"):
        if new_city:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            try:
                c.execute("INSERT OR IGNORE INTO locations (city_name) VALUES (?)", (new_city,))
                conn.commit()
                st.success(f"{new_city} を登録しました！")
            except Exception as e:
                st.error(f"登録中にエラーが発生しました: {e}")
            finally:
                conn.close()

    # 地点選択機能
    st.write("### 地点の選択")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT city_name FROM locations")
    locations = [row[0] for row in c.fetchall()]
    conn.close()
    selected_city = st.selectbox("登録された地点を選択してください", locations)
    
    # 生活状況の入力
    sleep_hours = st.slider("昨晩の睡眠時間（時間）", 0, 12, 7)
    exercise_minutes = st.slider("運動時間（分）", 0, 180, 30)
    meal_quality = st.selectbox("ご飯の量の満足度（1-5）", list(range(1, 6)))

    if st.button("データ保存"):
        if selected_city:
            weather_info = get_weather_data(selected_city)
            if weather_info:
                date = datetime.now().strftime("%Y-%m-%d")
                save_data(date, sleep_hours, exercise_minutes, meal_quality, weather_info)
                st.success("データが保存されました！")

                # 天気アイコンの表示
                st.write("### 天気情報")
                st.image(f"http://openweathermap.org/img/wn/{weather_info['icon']}@2x.png", 
                         caption=weather_info['weather_description'])
                st.write(f"気温: {weather_info['temp']}℃")
                st.write(f"気圧: {weather_info['pressure']}hPa")
                st.write(f"湿度: {weather_info['humidity']}%")
                
                # 健康アドバイスの表示
                advice = get_health_advice(weather_info, sleep_hours, exercise_minutes, meal_quality)
                st.write("### 本日のアドバイス")
                st.write(advice)

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