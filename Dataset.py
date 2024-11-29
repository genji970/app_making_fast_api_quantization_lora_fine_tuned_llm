from datasets import Dataset
from datasets import load_dataset
import pandas as pd

class Data_Load():
    def data_load(self , data_name):
        data = load_dataset(data_name)

        # train-test split
        dataset = data["train"].train_test_split(test_size=0.3, seed=42)

        train_data = dataset['train']
        test_data = dataset['test']

        df_train = pd.read_csv(train_data)
        df_test = pd.read_csv(test_data)

        return df_train , df_test

    def null_process(self , df_train , df_test):
        #결측값 처리
        #결측값 처리
        df_train = df_train.fillna(0)
        df_test = df_test.fillna(0)

        # 결측값 -> 0 -> 문자열
        df_train = df_train.replace(0, "there is no input")
        df_test = df_test.replace(0, "there is no input")