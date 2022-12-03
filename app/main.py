from __future__ import division
from fastapi import FastAPI
from utils.recommenders.lightgcn import lightgcn_recommend
import pandas as pd

app = FastAPI()

df_item_emb = pd.read_parquet("../data/lightgcn/item_embedding.pq")
df_user_emb = pd.read_parquet("../data/lightgcn/user_embedding.pq")
df_rest_info = pd.read_parquet("../data/cleaned/restaurants.pq")


@app.get("/")
async def root():
    return {"message": "Restaurant Recommender System"}


# lightGCN recommendation
@app.get('/lightGCN/recommendation/{item_id}')
def lightGCN_recommendation(item_id: int):
    return lightgcn_recommend(df_user_emb, df_item_emb, df_rest_info, id=item_id, TopK=20)
