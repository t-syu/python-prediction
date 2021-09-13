import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import pickle

#アップロード機能
uploaded_file = st.file_uploader("ファイルアップロード", type="csv")
if uploaded_file is not None:
      file = pd.read_csv(uploaded_file)

      latest_iteration = st.empty() #latest_iteration には文字が入っていない
      bar = st.progress(0) #progressbarが表される。 

      for i in range(100):
        latest_iteration.text(f"少々お待ち下さい。{i+1}") # fは値を文字列に入れたい時に使う
        bar.progress(i + 1)
        time.sleep(0.01)
      
      st.write(file)
      st.write("このファイルデータでの予測結果でよろしいですか")
      st.write("注意：この予測モデルはカラムとして")
      st.write(pd.DataFrame({"カラム":["大手企業", 
                   "交通費別途支給",  
                   "残業月20時間以上", 
                   "1日7時間以下勤務OK",
                   "駅から徒歩5分以内",
                   "学校・公的機関（官公庁）", 
                   "派遣スタッフ活躍中", 
                   "大量募集", 
                   "Accessのスキルを活かす",
                   "平日休みあり",
                   "フラグオプション選択",
                   "派遣形態", 
                   "正社員登用あり",
                   "社員食堂あり",
                   "服装自由",
                   "期間・時間　勤務開始日", 
                   "PowerPointのスキルを活かす", 
                   "PCスキル不要",
                   "車通勤OK",
                   "仕事の仕方",
                   "未経験OK",
                   "土日祝休み",
                   "給与/交通費　交通費",
                   "給与/交通費　給与下限",
                   "オフィスが禁煙・分煙",
                   "残業なし"]}))
      st.write("を選んでおります。それでもよろしいですか？")

else:
  st.write("csv形式のファイルをアップロードしてください")

#モデルをロードする
# loading in the model to predict on the data
pickle_in = open('rfr.pkl', 'rb')
loaded_rfr = pickle.load(pickle_in)
# with open('rfr.pickle', mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
#     loaded_rfr = pickle.load(f)                  # オブジェクトをデシリアライズ

#データ前処理
if uploaded_file is not None:
  file = file.dropna(axis = 1).reset_index(drop=True)
  # process_file = file[["職場の様子", "大手企業", "交通費別途支給", "職種コード", "1日7時間以下勤務OK", "駅から徒歩5分以内", "学校・公的機関（官公庁）",  "派遣スタッフ活躍中", "フラグオプション選択", "期間・時間　勤務期間", "派遣形態", "16時前退社OK", "正社員登用あり", "残業月20時間未満", "英語力不要", "社員食堂あり", "10時以降出社OK", "服装自由", "PowerPointのスキルを活かす", "会社概要　業界コード", "車通勤OK", "制服あり", "仕事の仕方", "紹介予定派遣", "シフト勤務", "未経験OK", "土日祝休み", "給与/交通費　交通費", "給与/交通費　給与下限", "勤務地　市区町村コード", "残業なし"]]

  # process_file = file[["フラグオプション選択", "給与/交通費　給与下限", "正社員登用あり", "派遣形態", "勤務地　市区町村コード", "勤務地　都道府県コード", "職種コード", "会社概要　業界コード", "学校・公的機関（官公庁）", "駅から徒歩5分以内"]]

  file["期間・時間　勤務開始月"] = file["期間・時間　勤務開始日"].apply(lambda x : x.split("/")[1])
  file["期間・時間　勤務開始月"] = file["期間・時間　勤務開始月"].astype(np.int)

  process_file = file[["大手企業", 
                   "交通費別途支給",  
                   "残業月20時間以上", 
                   "1日7時間以下勤務OK",
                   "駅から徒歩5分以内",
                   "学校・公的機関（官公庁）", 
                   "派遣スタッフ活躍中", 
                   "大量募集", 
                   "Accessのスキルを活かす",
                   "平日休みあり",
                   "フラグオプション選択",
                   "派遣形態", 
                   "正社員登用あり",
                   "社員食堂あり",
                   "服装自由",
                   "期間・時間　勤務開始月", 
                   "PowerPointのスキルを活かす", 
                   "PCスキル不要",
                   "車通勤OK",
                   "仕事の仕方",
                   "未経験OK",
                   "土日祝休み",
                   "給与/交通費　交通費",
                   "給与/交通費　給与下限",
                   "オフィスが禁煙・分煙",
                   "残業なし"
                   ]]

  #データ予測
  answer = loaded_rfr.predict(process_file)
  answer_data = pd.DataFrame(data=file["お仕事No."], columns=["お仕事No."])
  answer_data["応募数 合計"] = answer

  #ダウンロード機能
  if st.button("Yes‼") == True:
    csv = answer_data.to_csv(index=False)  
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="result.csv">download</a>'
    st.markdown(f"ダウンロードするなら右をクリック👉 {href}", unsafe_allow_html=True)
    st.write("ダウンロードファイルとして、出力する結果は以下のような形になります。")
    st.write(answer_data)

    print(answer_data.describe())