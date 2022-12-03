import pandas as pd
import numpy as np
import json


# df_item_emb = pd.read_parquet("../../data/lightgcn/item_embedding.pq")
# df_user_emb = pd.read_parquet("../../data/lightgcn/user_embedding.pq")
# df_rest_info = pd.read_parquet("../../data/cleaned/restaurants.pq")

def getEmbedding(df: pd.DataFrame, id_name: str, id: int):
    emb = (df[df[id_name] == id]
           .embedding.item()
           .reshape(-1, 1))
    return emb


def getIndexTopK(df_user_emb, df_item_emb, id: int, TopK: int):
    items_emb = np.stack(df_item_emb.embedding, axis=0)
    user_emb = getEmbedding(df_user_emb, id_name="userID", id=id)
    predict_ratings = (items_emb @ user_emb).flatten()
    topK_index = (-predict_ratings).argsort()[:TopK]
    topK_itemID = df_item_emb.loc[topK_index, "itemID"].values
    return topK_itemID


def lightgcn_recommend(df_user_emb, df_item_emb, df_rest_info,  id, TopK):
    topK_itemID = getIndexTopK(df_user_emb, df_item_emb, id, TopK)
    df_recommend = df_rest_info.set_index('rest_id').loc[topK_itemID].reset_index()
    data = df_recommend.to_json(orient='records', lines=True).splitlines()
    data = [json.loads(item) for item in data]

    recommend = {'data': data}
    json_data = json.dumps(recommend)
    return json_data


# print(lightgcn_recommend(df_user_emb, df_item_emb, df_rest_info, id=20, TopK=20))