import pandas as pd

df = pd.read_csv("/workspace/Anotated_data/PT_2019_03_whole_body.csv")
print(df.columns)  # 列名を確認

# もし 'PatientID' でなければ、正しい列名に修正
# 例: 'patient_id' なら
df["img_path"] = df["patient_id"].apply(get_img_path)