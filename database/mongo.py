from pymongo import MongoClient
from bson import ObjectId
import time
import datetime
import random

import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

class Database:
    def __init__(self, client_host="mongodb://localhost:27017/", database="APC",
                 collection_name=f"bus_information ({datetime.datetime.now().strftime('%Y%m%d_%H%M%S')})"):
        self.client = MongoClient(client_host)
        self.db = self.client[database]
        self.collection_name = collection_name
        self.document = None

    def get_collection_col(self):
        """
        Return:
            self.db[self.collection_name]: Return the collection
        """
        return self.db[self.collection_name]

    def initialize_document(self, time=None):
        """
        Create a new document inside the collection

        The format is fixed in format of {"time":{time}, "in": 0, "out":0}

        Argument:
            time: A 'primary key' use to find the particular document\
            
        Return:
            time: Return the 'primary key'
        """
        if time is None:
            time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        init_data = {
            "time": time,
            "in": 0,
            "out": 0
        }       

        self.get_collection_col().insert_one(init_data)
        print(f"Document {init_data} successfully created in Collection {self.collection_name}, key: {time}")

        return time

    def increment_in(self, time_unique, amount):
        """
        Increment document 'in' by 1 with given 'primary key' named time
        """
        if not isinstance(time_unique, int):
            document_time_str = time_unique.strftime("%Y%m%d_%H%M%S")
        else:
            document_time_str = time_unique
        self.get_collection_col().update_one({"time": document_time_str}, {'$inc': {"in": amount}})
        # print(f"Document {init_data} successfully created in Collection {self.collection_name}")

    def increment_out(self, time_unique, amount):
        """
        Increment document 'out' by 1 with given 'primary key' named time
        """
        if not isinstance(time_unique, int):
              document_time_str = time_unique.strftime("%Y%m%d_%H%M%S")
        else:
            document_time_str = time_unique
        self.get_collection_col().update_one({"time": document_time_str}, {'$inc': {"out": amount}})
        # print(f"Document {init_data} successfully created in Collection {self.collection_name}
        
def main():
    # # client = MongoClient("localhost", 27017)
    # client = MongoClient("mongodb://localhost:27017/")

    # db = client["passenger"]            
    # collection_name = f"bus_information ({datetime.datetime.now().strftime('%Y%m%d_%H%M%S')})"

    # passenger_col = db[collection_name]

    mongodb = Database()

    collection_name = "bus_information (20240623_235716)"
    lists = list(mongodb.db[collection_name].find())

    cleaned_data = [{key:value for key, value in record.items() if key != '_id'} for record in lists]

    # df = pd.DataFrame(cleaned_data)
    # print(df)

    # df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d_%H%M%S').strftime("%H:%M:%S"))
    # print(df)

    # df['net'] = df['in'] - df['out']

    # print(df)

    # st.bar_chart(data=df, x='time', y='in')
    # st.bar_chart(data=df, x='time', y='out')
    # st.bar_chart(data=df, x='time', y='net')



    # while True:
    #     if not list(mongodb.get_collection_col().find()):
    #         print("NONE")
    #         mongodb.initialize_document(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    #     print("DS")
    #     time.sleep(2)    


    # while True:
    #     # current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    #     # init_data = {
    #     #     "time": current_time,
    #     #     "in": 0,
    #     #     "out": 0
    #     # }

    #     # passenger_col.insert_one(init_data)
    #     # print("Inserted")

    #     unique_time = mongodb.initialize_document(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    #     in_rand = random.randint(0, 100)
    #     out_rand = random.randint(0, 100)

    #     for _ in range(in_rand):
    #         mongodb.increment_in(unique_time)

    #     for _ in range(out_rand):
    #         mongodb.increment_out(unique_time)

    #     time.sleep(2)

if __name__== "__main__":
    main()