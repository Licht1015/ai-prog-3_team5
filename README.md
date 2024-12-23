# プロジェクト名

## チーム名とメンバー一覧

- チーム名:ai-prog-3_team5
- チームメンバー：
  - 加納 大喜
  - 永山りひと
  - 水田 彩未
  - 山﨑 千夏

## プロジェクトの概要（1-2文で簡潔に）

- 体調管理アプリ

## 開発するアプリケーションの目的

- 健康促進

## 想定されるターゲットユーザー

- 不健康な生活を正したい人向け

## 主な機能リスト

- 寝る時間、運動時間、ご飯の量で健康面をなんとかしてくれる,その日の天気（気圧）と生活を照らし合わせて注意を促すアプリ
- 記録管理
  - 体重管理: 体重を記録し、必要な栄養素を表示（ユーザーが食べたものにチェックを入れる機能）。
  - 睡眠時間管理: 希望の就寝時刻、実際の就寝・起床時刻を記録し、睡眠時間の診断（推奨睡眠時間と比較）。
  - 運動記録: 運動の内容や時間を入力し、カロリー消費を算出。
- 健康リマインダー
  - 水分補給、薬の服用、運動開始（終了）のリマインダー設定。
- グラフ化と統計管理
  - 週・月ごとの進捗をグラフで視覚化し、フィードバックを提供。
- AIアドバイス
  - 季節や天気に応じたアドバイスを提供（例：散歩のおすすめメッセージ）。
  - 運動意欲が低いユーザー向けの「なまけものスタイル」と、意欲のあるユーザー向けの「わんこスタイル」の設定。
- 位置情報の活用
  - ユーザーの位置情報を基にした天候情報の参照

## 使用予定の AI 技術や API

- Ollama
  - LLM：llama2(余裕があれば日本語対応の、Llama 3 ELYZA JP 8B,Gemma 2 2B(gemma-2-2b-jpn-it-gguf)辺りの使用検討)
- RAG
- Langchain
- OpenWeatherMap API

## 技術スタック（プログラミング言語、フレームワーク、ライブラリなど）

- Python
- フロントエンド：HTML/CSS/JS(React?)
- streamlit(または、Flask等の別のフレームワーク)
- Anaconda
- DB(sqlite3辺り？)

## 開発スケジュール（主要なマイルストーン）

## 各メンバーの役割分担

- 記録系
  - 加納大喜
  - 山﨑 千夏
- AI系
  - 廣澤 壱聖
  - 永山りひと
- 未割り当て：水田 彩未

## 想定される課題や困難点

- API, AI

## プロジェクトの成功基準

- その日の気圧と生活を照らし合わせて注意を促す

## 将来の拡張可能性

## 参考資料
