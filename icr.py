import numpy as np
import pandas as pd
import codecs
from scipy import sparse
from scipy.spatial.distance import cosine


print('Loading ...')

#データセット読み込み
df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id','item_id', 'rating', 'timestamp'])

#行列の形に変換
shape = (df.max().loc['user_id'], df.max().loc['item_id'])
R = sparse.lil_matrix(shape) 

for i in df.index:
    row = df.loc[i]
    R[row['user_id'] - 1, row['item_id'] - 1] = row['rating']


#アイテムの類似度を計算
def compute_item_similarities(R):
    # n: 映画のデータ数
    n = R.shape[1]
    sims = np.zeros((n,n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                # R[:, i] は 映画 i に関する全ユーザの評価を並べた列ベクトル
                sim = similarity(R[:,i], R[:,j])

            sims[i][j] = sim 
            sims[j][i] = sim 

    return sims 

# item1とitem2の類似度
def similarity(item1, item2):
    # item1 と item2 のどちらも評価済であるユーザの集合
    common = np.logical_and(item1 != 0, item2 != 0)

    v1 = item1[common]
    v2 = item2[common]

    sim = 0.0
    # 共通評価者が 2以上という制約にしている
    if v1.size > 1:
        sim = 1.0 - cosine(v1, v2) #コサイン類似度を計算

    return sim

sims = compute_item_similarities(R.todense())

#評価の予測
def predict(u, sims):
    # 未評価は0, 評価済は1となるベクトル。正規化の計算のために。
    x = np.zeros(u.size) 
    x[u > 0] = 1

    scores      = sims.dot(u)
    normalizers = sims.dot(x)

    prediction = np.zeros(u.size)
    for i in range(u.size):
        # 分母が 0 になるケースと評価済アイテムは予測値を 0 とする
        if normalizers[i] == 0 or u[i] > 0:
            prediction[i] = 0
        else:
            prediction[i] = scores[i] / normalizers[i]

    # ユーザ u のアイテム i に対する評価の予測
    return prediction


#アイテムIDと映画タイトルを読み込み
with codecs.open('ml-100k/u.item', 'r', 'utf-8', errors='ignore') as f:
    item_df = pd.read_table(f, delimiter='|', header=None).loc[:, 0:1]
    item_df.rename(columns={0: 'item_id', 1: 'item_title'}, inplace=True)
    item_df.set_index('item_id', inplace=True)
items = pd.Series(item_df['item_title'])


#レコメンドするユーザIDを入力
while True:
    user_id = int(input('Please input user ID(0~942)'))
    if 0 <= user_id and user_id <= 942: break

#評価値の上位10作品を求める
u = np.array(R.todense())
predicted_vector = predict(u[user_id,:], sims)
predicted_ranking = [(i, predicted_vector[i]) for i in np.argsort(predicted_vector)[::-1][:10]]

#結果
print('User ID: {}'.format(user_id))
for item_id, rating in predicted_ranking:
    # アイテムID, 映画タイトル, 予測した評価値を表示
    print('{}: {} [{}]'.format(item_id + 1, items[item_id + 1], rating))



