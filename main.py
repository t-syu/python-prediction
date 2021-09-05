import streamlit as st
import pandas as pd
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

else:
  st.write("csv形式のファイルをアップロードしてください")

#モデルをロードする
with open('rfr.pickle', mode='rb') as f:  # with構文でファイルパスとバイナリ読み来みモードを設定
    loaded_rfr = pickle.load(f)                  # オブジェクトをデシリアライズ

#データ前処理
if uploaded_file is not None:
  file = file.dropna(axis = 1).reset_index(drop=True)
  file = file.drop(columns=file.select_dtypes(include="object").columns)
  process_file = file.drop(["お仕事No."], axis=1)

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