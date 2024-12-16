# Required imports
import streamlit as st
import requests
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Ollamaのチャットモデルの初期化
chat_model = ChatOllama(
    model="llama2",
    callbacks=[StreamingStdOutCallbackHandler()],
)

# .streamlit/secrets.toml からAPIキーを取得
API_KEY = st.secrets["weather_api_key"]
DB_NAME = "health_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 既存のテーブル構造を確認
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [table[0] for table in c.fetchall()]

    # 生活データ用テーブル
    c.execute('''CREATE TABLE IF NOT EXISTS lifestyle_data (
                    date TEXT PRIMARY KEY,
                    sleep_hours REAL,
                    exercise_minutes REAL,
                    meal_quality INTEGER)''')
    
    # 気象データ用テーブル
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
    
    # アドバイス保存用テーブル
    c.execute('''CREATE TABLE IF NOT EXISTS daily_advice (
                    date TEXT PRIMARY KEY,
                    advice TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    
    # 目標管理用テーブル
    c.execute('''CREATE TABLE IF NOT EXISTS health_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sleep_goal REAL,
                    exercise_goal REAL,
                    meal_quality_goal INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # locations テーブルの移行処理
    if 'locations' in existing_tables:
        c.execute("PRAGMA table_info(locations)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'is_selected' not in columns:
            c.execute('''
                CREATE TABLE locations_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city_name TEXT UNIQUE,
                    is_selected INTEGER DEFAULT 0
                )''')
            
            c.execute('''
                INSERT INTO locations_new (city_name)
                SELECT city_name FROM locations
            ''')
            
            c.execute('DROP TABLE locations')
            c.execute('ALTER TABLE locations_new RENAME TO locations')
    else:
        c.execute('''
            CREATE TABLE locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city_name TEXT UNIQUE,
                is_selected INTEGER DEFAULT 0
            )''')
    
    conn.commit()
    conn.close()

def save_goals(sleep_goal, exercise_goal, meal_quality_goal):
    """健康目標を保存"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO health_goals (sleep_goal, exercise_goal, meal_quality_goal) VALUES (?, ?, ?)",
              (sleep_goal, exercise_goal, meal_quality_goal))
    conn.commit()
    conn.close()

def get_current_goals():
    """最新の健康目標を取得"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM health_goals ORDER BY id DESC LIMIT 1")
    result = c.fetchone()
    conn.close()
    return result[1:4] if result else (8.0, 30.0, 4)  # デフォルト値を設定

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
            'icon': data['weather'][0]['icon']
        }
        return weather_info
    else:
        st.error("天気データを取得できませんでした。場所を確認してください。")
        return None

def display_weather_info(city, weather_info):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(f"http://openweathermap.org/img/wn/{weather_info['icon']}@2x.png",
                caption=weather_info['weather_description'])
    with col2:
        st.write(f"### {city}の天気")
        st.write(f"🌡️ 気温: {weather_info['temp']}℃")
        st.write(f"🌪️ 気圧: {weather_info['pressure']}hPa")
        st.write(f"💧 湿度: {weather_info['humidity']}%")

def save_advice(date, advice):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO daily_advice (date, advice) VALUES (?, ?)",
              (date, advice))
    conn.commit()
    conn.close()

def get_latest_advice():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT date, advice FROM daily_advice ORDER BY date DESC LIMIT 1")
    result = c.fetchone()
    conn.close()
    return result if result else (None, None)

def get_health_advice(weather_info, sleep_hours, exercise_minutes, meal_quality):
    sleep_goal, exercise_goal, meal_quality_goal = get_current_goals()
    
    prompt = f"""
    以下の情報から、50文字程度の簡潔な健康アドバイスを1つだけ提供してください。
    特に注意が必要な項目についてのみ言及してください。

    条件：
    気温{weather_info['temp']}℃、気圧{weather_info['pressure']}hPa、湿度{weather_info['humidity']}%
    天気：{weather_info['weather_main']}
    睡眠{sleep_hours}時間（目標{sleep_goal}時間）
    運動{exercise_minutes}分（目標{exercise_goal}分）
    食事満足度{meal_quality}/5（目標{meal_quality_goal}/5）

    形式：
    - 一文で簡潔に
    - 具体的な行動を提案
    - 理由は短く
    """

    response = chat_model.invoke([HumanMessage(content=prompt)])
    today = datetime.now().strftime("%Y-%m-%d")
    save_advice(today, response.content)
    return response.content

def save_data(date, sleep_hours, exercise_minutes, meal_quality, weather_info):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO lifestyle_data VALUES (?, ?, ?, ?)",
              (date, sleep_hours, exercise_minutes, meal_quality))
    c.execute("INSERT OR REPLACE INTO weather_data VALUES (?, ?, ?, ?, ?, ?, ?)",
              (date, weather_info['pressure'], weather_info['temp'],
               weather_info['humidity'], weather_info['weather_main'],
               weather_info['weather_description'], weather_info['icon']))
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_NAME)
    lifestyle_df = pd.read_sql("SELECT * FROM lifestyle_data", conn)
    weather_df = pd.read_sql("SELECT * FROM weather_data", conn)
    conn.close()
    return lifestyle_df, weather_df

def get_registered_locations(selected_only=False):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if selected_only:
        c.execute("SELECT city_name FROM locations WHERE is_selected = 1")
    else:
        c.execute("SELECT city_name FROM locations")
    locations = [row[0] for row in c.fetchall()]
    conn.close()
    return locations

def update_location_selection(city_name, is_selected):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE locations SET is_selected = ? WHERE city_name = ?", 
              (1 if is_selected else 0, city_name))
    conn.commit()
    conn.close()

def display_goal_progress(current_data, goals):
    """目標達成度を表示"""
    st.write("### 目標達成状況")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sleep_progress = (float(current_data['sleep_hours']) / goals[0]) * 100
        sleep_delta = float(current_data['sleep_hours']) - float(goals[0])
        st.metric("睡眠時間達成度", f"{sleep_progress:.1f}%",
                 delta=f"{sleep_delta:.1f}時間")
    
    with col2:
        exercise_progress = (float(current_data['exercise_minutes']) / goals[1]) * 100
        exercise_delta = float(current_data['exercise_minutes']) - float(goals[1])
        st.metric("運動時間達成度", f"{exercise_progress:.1f}%",
                 delta=f"{exercise_delta:.1f}分")
    
    with col3:
        meal_progress = (float(current_data['meal_quality']) / goals[2]) * 100
        meal_delta = float(current_data['meal_quality']) - float(goals[2])
        st.metric("食事品質達成度", f"{meal_progress:.1f}%",
                 delta=f"{meal_delta:.1f}")

def calculate_weekly_trends(df):
    """週間トレンドを計算"""
    # 日付を変換
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 数値列のみを選択
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[['date'] + list(numeric_columns)]
    
    # 週次平均を計算
    weekly_avg = df_numeric.set_index('date').resample('W').mean()
    
    return weekly_avg

def detect_anomalies(df, column, threshold=2):
    """異常値を検出"""
    mean = df[column].mean()
    std = df[column].std()
    anomalies = df[abs(df[column] - mean) > threshold * std]
    return anomalies

def analyze_seasonal_patterns(df):
    """季節性パターンを分析"""
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['date'].dt.month.map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                           3: 'Spring', 4: 'Spring', 5: 'Spring',
                                           6: 'Summer', 7: 'Summer', 8: 'Summer',
                                           9: 'Fall', 10: 'Fall', 11: 'Fall'})
    seasonal_stats = df.groupby('season').agg({
        'sleep_hours': 'mean',
        'exercise_minutes': 'mean',
        'meal_quality': 'mean'
    })
    return seasonal_stats

def display_enhanced_visualizations(lifestyle_df, weather_df):
    """拡張データ可視化"""
    if not lifestyle_df.empty and not weather_df.empty:
        df = pd.merge(lifestyle_df, weather_df, on='date')
        
        st.write("### 詳細分析")
        
        # トレンド分析
        trends_tab, patterns_tab, anomalies_tab = st.tabs(["トレンド", "パターン", "異常値"])
        
        with trends_tab:
            weekly_trends = calculate_weekly_trends(df)
            fig = px.line(weekly_trends,
                         y=['sleep_hours', 'exercise_minutes', 'meal_quality', 'temperature', 'pressure', 'humidity'],
                         title="週間トレンド")
            fig.update_layout(
                xaxis_title="日付",
                yaxis_title="値",
                legend_title="指標"
            )
            st.plotly_chart(fig)
            
            # データの説明を追加
            st.write("""
            #### トレンドの見方
            - 睡眠時間: 時間単位
            - 運動時間: 分単位
            - 食事満足度: 1-5の評価
            - 気温: 摂氏（℃）
            - 気圧: hPa
            - 湿度: %
            """)
        
        with patterns_tab:
            seasonal_patterns = analyze_seasonal_patterns(df)
            fig = px.bar(seasonal_patterns,
                        barmode='group',
                        title="季節別統計")
            st.plotly_chart(fig)
            
            # 気象条件との相関
            correlation_matrix = df[['sleep_hours', 'exercise_minutes', 'meal_quality',
                                   'pressure', 'temperature', 'humidity']].corr()
            fig = px.imshow(correlation_matrix,
                           title="相関ヒートマップ",
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig)
        
        with anomalies_tab:
            cols = ['sleep_hours', 'exercise_minutes', 'meal_quality']
            for col in cols:
                anomalies = detect_anomalies(df, col)
                if not anomalies.empty:
                    st.write(f"#### {col}の異常値")
                    st.dataframe(anomalies[['date', col]])

def train_linear_regression(X, y):
    """線形回帰モデルを学習し、モデルとメトリクスを返す"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse

def prepare_prediction_data(df):
    """予測のためのデータ前処理を行う"""
    df['date'] = pd.to_datetime(df['date'])
    df['date_ordinal'] = df['date'].map(datetime.toordinal)
    weather_dummies = pd.get_dummies(df['weather_main'], prefix='weather')
    features = ['pressure', 'temperature', 'humidity', 'date_ordinal']
    X = pd.concat([df[features], weather_dummies], axis=1)
    return X

def display_prediction_metrics(target_name, r2, rmse):
    """予測メトリクスを表示"""
    st.write(f"#### {target_name} 予測モデルの性能")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("決定係数 (R²)", f"{r2:.3f}")
    with col2:
        st.metric("平均二乗誤差の平方根 (RMSE)", f"{rmse:.3f}")

def main():
    st.title("健康管理と気象条件アプリ")
    
    init_db()

    tab = st.sidebar.radio(
        "機能を選択",
        ["ホーム", "データ入力・保存", "アドバイス取得", "データ閲覧・分析", "目標設定"]
    )

    with st.sidebar:
        st.write("### 地点の登録・選択")
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
                    st.error(f"登録中に���ラーが発生しました: {e}")
                finally:
                    conn.close()

        st.write("### 表示する地点の管理")
        locations = get_registered_locations()
        if locations:
            for city in locations:
                is_selected = city in get_registered_locations(selected_only=True)
                if st.checkbox(f"📍 {city}", value=is_selected, key=f"select_{city}"):
                    update_location_selection(city, True)
                else:
                    update_location_selection(city, False)
        else:
            st.info("地点が登録されていません")

        selected_city = st.selectbox(
            "データを記録する地点を選択",
            locations if locations else ["地点を登録してください"]
        )

    if tab == "ホーム":
        st.write("### 選択地点の天気情報")
        selected_locations = get_registered_locations(selected_only=True)
        
        if not selected_locations:
            st.info("表示する地点が選択されていません。サイドバーから地点を選択してください。")
        else:
            st.write("## 現在の天気")
            for i in range(0, len(selected_locations), 2):
                col1, col2 = st.columns(2)
                
                # 左カラム
                with col1:
                    weather_info = get_weather_data(selected_locations[i])
                    if weather_info:
                        st.markdown("---")
                        display_weather_info(selected_locations[i], weather_info)
                
                # 右カラム（地点が偶数個の場合のみ）
                if i + 1 < len(selected_locations):
                    with col2:
                        weather_info = get_weather_data(selected_locations[i + 1])
                        if weather_info:
                            st.markdown("---")
                            display_weather_info(selected_locations[i + 1], weather_info)
            
            # 目標達成状況の表示
            st.markdown("---")
            lifestyle_df, _ = load_data()
            if not lifestyle_df.empty:
                current_data = lifestyle_df.iloc[-1]
                display_goal_progress(current_data, get_current_goals())
            
            # 今日のアドバイスを表示
            st.markdown("---")
            latest_date, latest_advice = get_latest_advice()
            if latest_advice:
                st.write("## 👨‍⚕️ 最新のアドバイス")
                advice_container = st.container(border=True)
                with advice_container:
                    st.write(f"_{latest_date}_")
                    st.write(f"**{latest_advice}**")
            
            # 最新の健康データがあれば表示
            st.markdown("---")
            st.write("## 📊 最新の健康データ")
            lifestyle_df, _ = load_data()
            if not lifestyle_df.empty:
                latest_data = lifestyle_df.iloc[-1]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("睡眠時間", f"{latest_data['sleep_hours']}時間")
                with col2:
                    st.metric("運動時間", f"{latest_data['exercise_minutes']}分")
                with col3:
                    st.metric("食事満足度", f"{latest_data['meal_quality']}/5")
            else:
                st.info("健康データがまだ登録されていません。")

    elif tab == "データ入力・保存":
        st.write("### データ入力")
        
        # 入力方式の選択
        input_method = st.radio(
            "入力方式を選択",
            ["フォーム入力", "スライダー入力"]
        )
        
        if input_method == "フォーム入力":
            with st.form("health_data_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sleep_hours = st.number_input(
                        "睡眠時間（時間）",
                        min_value=0.0,
                        max_value=24.0,
                        value=7.0,
                        step=0.5,
                        format="%.1f"
                    )
                
                with col2:
                    exercise_minutes = st.number_input(
                        "運動時間（分）",
                        min_value=0,
                        max_value=300,
                        value=30,
                        step=5
                    )
                
                with col3:
                    meal_quality = st.number_input(
                        "食事満足度（1-5）",
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1
                    )
                
                submitted = st.form_submit_button("データを保存")
                if submitted:
                    if selected_city and selected_city != "地点を登録してください":
                        weather_info = get_weather_data(selected_city)
                        if weather_info:
                            date = datetime.now().strftime("%Y-%m-%d")
                            save_data(date, sleep_hours, exercise_minutes, meal_quality, weather_info)
                            st.success("データが保存されました！")
                            st.write("### 現在の天気")
                            display_weather_info(selected_city, weather_info)
        
        else:  # スライダー入力
            sleep_hours = st.slider(
                "昨晩の睡眠時間（時間）",
                min_value=0.0,
                max_value=12.0,
                value=7.0,
                step=0.5
            )
            
            exercise_minutes = st.slider(
                "運動時間（分）",
                min_value=0,
                max_value=180,
                value=30,
                step=5
            )
            
            meal_quality = st.select_slider(
                "食事の満足度（1-5）",
                options=list(range(1, 6)),
                value=3
            )
            
            if st.button("データを保存"):
                if selected_city and selected_city != "地点を登録してください":
                    weather_info = get_weather_data(selected_city)
                    if weather_info:
                        date = datetime.now().strftime("%Y-%m-%d")
                        save_data(date, sleep_hours, exercise_minutes, meal_quality, weather_info)
                        st.success("データが保存されました！")
                        st.write("### 現在の天気")
                        display_weather_info(selected_city, weather_info)

        # 入力ガイドの表示
        with st.expander("入力項目の説明"):
            st.write("""
            - **睡眠時間**: 前日の睡眠時間を時間単位で入力してください。
            - **運動時間**: その日の運動時間を分単位で入力してください。
            - **食事満足度**: 1(不満)から5(大変満足)で評価してください。
            """)

    elif tab == "アドバイス取得":
        st.write("### 健康アドバイスの取得")
        lifestyle_df, weather_df = load_data()
        if not lifestyle_df.empty and not weather_df.empty:
            latest_date = max(lifestyle_df['date'])
            latest_lifestyle = lifestyle_df[lifestyle_df['date'] == latest_date].iloc[0]
            latest_weather = weather_df[weather_df['date'] == latest_date].iloc[0]
            
            weather_info = {
                'temp': latest_weather['temperature'],
                'pressure': latest_weather['pressure'],
                'humidity': latest_weather['humidity'],
                'weather_main': latest_weather['weather_main']
            }
            
            if st.button("アドバイスを取得"):
                with st.spinner("アドバイスを生成中..."):
                    advice = get_health_advice(
                        weather_info,
                        latest_lifestyle['sleep_hours'],
                        latest_lifestyle['exercise_minutes'],
                        latest_lifestyle['meal_quality']
                    )
                st.write("### 健康アドバイス")
                st.write(advice)
        else:
            st.warning("アドバイスを取得するには、まずデータを入力・保存してください。")

    elif tab == "データ閲覧・分析":
        st.write("### データ閲覧")
        lifestyle_df, weather_df = load_data()
        if not lifestyle_df.empty and not weather_df.empty:
            # データ表示タブ
            view_tab = st.radio(
                "表示するデータを選択",
                ["生活データ", "気象データ", "分析と予測", "詳細分析"]
            )
            
            if view_tab == "生活データ":
                # 生活データの表示設定
                st.dataframe(
                    lifestyle_df.style.highlight_max(axis=0),
                    use_container_width=True,
                    hide_index=False
                )
                
                # CSVダウンロードボタン
                csv = lifestyle_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="生活データをCSVでダウンロード",
                    data=csv,
                    file_name='lifestyle_data.csv',
                    mime='text/csv',
                )
            elif view_tab == "気象データ":
                # 気象データの表示設定
                st.dataframe(
                    weather_df.style.highlight_max(axis=0),
                    use_container_width=True,
                    hide_index=False
                )
                
                # CSVダウンロードボタン
                csv = weather_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="気象データをCSVでダウンロード",
                    data=csv,
                    file_name='weather_data.csv',
                    mime='text/csv',
                )
            elif view_tab == "分析と予測":
                st.write("### データ分析と予測")
                if len(lifestyle_df) >= 10:
                    if st.button("予測モデルを作成"):
                        with st.spinner("予測モデルを作成中..."):
                            df = pd.merge(lifestyle_df, weather_df, on='date')
                            X = prepare_prediction_data(df)
                            
                            targets = {
                                '睡眠時間': 'sleep_hours',
                                '運動時間': 'exercise_minutes',
                                '食事満足度': 'meal_quality'
                            }
                            
                            for target_name, target_col in targets.items():
                                model, r2, rmse = train_linear_regression(X, df[target_col])
                                display_prediction_metrics(target_name, r2, rmse)
                                
                                # 特徴量の重要度を表示
                                st.write(f"#### {target_name}に影響を与える要因")
                                importance_df = pd.DataFrame({
                                    '要因': X.columns,
                                    '影響度': model.coef_
                                }).sort_values('影響度', key=abs, ascending=False)
                                st.dataframe(importance_df)
                                
                                # 予測vs実際の値のプロット
                                fig = px.scatter(df, x=target_col, y=model.predict(X),
                                               labels={'x': f'実際の{target_name}', 'y': f'予測された{target_name}'},
                                               title=f'{target_name}の予測vs実際の値')
                                fig.add_scatter(x=[df[target_col].min(), df[target_col].max()],
                                               y=[df[target_col].min(), df[target_col].max()],
                                               line=dict(dash='dash'),
                                               name='完璧な予測')
                                st.plotly_chart(fig)
                        
                        st.success("予測モデルの作成が完了しました！")
                else:
                    st.warning("予測モデルを作成するには、最低10件以上のデータが必要です。")
            else:  # 詳細分析
                display_enhanced_visualizations(lifestyle_df, weather_df)
        else:
            st.write("まだデータがありません。")

    elif tab == "目標設定":
        st.write("### 健康目標の設定")
        current_goals = get_current_goals()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_sleep_goal = st.number_input(
                "目標睡眠時間（時間）",
                min_value=4.0,
                max_value=12.0,
                value=float(current_goals[0]),
                step=0.5
            )
        
        with col2:
            new_exercise_goal = st.number_input(
                "目標運動時間（分）",
                min_value=0,
                max_value=180,
                value=int(current_goals[1]),
                step=5
            )
        
        with col3:
            new_meal_quality_goal = st.number_input(
                "目標食事満足度（1-5）",
                min_value=1,
                max_value=5,
                value=int(current_goals[2]),
                step=1
            )
        
        if st.button("目標を更新"):
            save_goals(new_sleep_goal, new_exercise_goal, new_meal_quality_goal)
            st.success("健康目標が更新されました！")
            
            # 最新のデータで目標達成状況を表示
            lifestyle_df, _ = load_data()
            if not lifestyle_df.empty:
                current_data = lifestyle_df.iloc[-1]
                display_goal_progress(current_data, (new_sleep_goal, new_exercise_goal, new_meal_quality_goal))
        
        # 目標設定のガイドライン
        with st.expander("目標設定のガイドライン"):
            st.write("""
            #### 健康的な目標設定のためのガイドライン
            
            1. **睡眠時間**
                - 成人の推奨睡眠時間は7-9時間
                - 個人差があるため、自身の生活リズムに合わせて調整
            
            2. **運動時間**
                - WHO推奨：週150分の中程度の有酸素運動
                - 1日あたり20-30分が目安
                - 激��い運動の場合は時間を短縮可能
            
            3. **食事の質**
                - バランスの取れた食事を目指す
                - 規則正しい食事時間
                - 適切な量と質を維持
            
            目標は定期的に見直し、必要に応じて調整することをお勧めします。
            """)

if __name__ == "__main__":
    main()