import pandas as pd

def get_label_names(csv_path):
    """
    指定したCSVファイルの1行目・2行目をもとにラベル名リストを返す関数
    """
    df = pd.read_csv(csv_path, header=[0,1])

    def make_colname(a, b):
        if (pd.isnull(a) or str(a).strip() == '' or str(a).startswith('Unnamed')):
            return str(b)
        elif pd.isnull(b) or str(b).strip() == '':
            return str(a)
        else:
            return f"{a}_{b}"

    label_names = [make_colname(a, b) for a, b in df.columns]
    return label_names