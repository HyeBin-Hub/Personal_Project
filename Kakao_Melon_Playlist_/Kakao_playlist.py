import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import requests

url="https://arena.kakaocdn.net/kakao_arena/melon_autoplaylist/train.json?TWGServiceId=kakao_arena&Expires=1625768419&Signature=o1CsjTyuVOcvnx1vmAZ6/4vnpx0%3D&AllowedIp=1.246.85.102&download"
r=requests.get(url)
r.encoding="utf-8"
train=r.json()

url="https://arena.kakaocdn.net/kakao_arena/melon_autoplaylist/genre_gn_all.json?TWGServiceId=kakao_arena&Expires=1625768438&Signature=MvsTYT8vukr6fxUMQVfeyWuj6Is%3D&AllowedIp=1.246.85.102&download"
r=requests.get(url)
r.encoding="utf-8"
genre=r.json()

url="https://arena.kakaocdn.net/kakao_arena/melon_autoplaylist/song_meta.json?TWGServiceId=kakao_arena&Expires=1625768456&Signature=7/p5owIBRFs8ueqHtPTWPXhJ6js%3D&AllowedIp=1.246.85.102&download"
r=requests.get(url)
r.encoding="utf-8"
song=r.json()

train_songs = []
for i in train:
    train_songs.append(i["songs"])

# Word2Vec
#   : 단어마다 차례대로 인덱싱을 하여 벡터화하지 않고,
#     유사한 단어들을 비슷한 방향과 힘의 벡터를 갖도록 단어를 벡터화 시켜주는 방법 중 하나
#
# Word2Vec의 두가지 방법
#  1. CBOW (Coutinuous Bag of Word)
#      - 주변 단어를 통해 중심 단어를 예측하는 방식
#  2. Skin-gram
#      - 중심 단어를 통해 주변 단어를 예측하는 방식
#
# Word2Vec의 형태
#  model=word2vec.Word2Vec(sentences, workers, size, min_count, window, sample,sg,iter)
#      1. sentences : 학습시킬 문장
#      2. workers : 실행할 병렬 프로세스의 수
#      3. size : 각 단어에 대한 임베딩 된 벡터차원 정의
#          -> size=2라면 문장의 벡터는[-0.1248574, 0.255778]와 같은 형태로 가지게 된다
#      4. min_count : 단어에 대한 최소 빈도수
#          -> min_count=5라면 빈도수 5이하는 무시
#      5. sample : 빠른 학습을 위해 정답 단어 라벨에 대한 다운 샘플링 비율을 지정하는 것,
#            보통 0.001이 좋은 성능을 낸다고 한다
#      6. sg : 1이면 skip-gram, 0이면 CBOW
#      7. iter : epoch와 같은 뜻으로 학습 반복 횟수 지정
#      8. window : 중심 단어를 예측하기 위해 앞,뒤로 몇개의 단어를 포함할 것인지의 범위를 window 라고한다
#
# 모델 교육을 3단계로 구분하는 이유¶
#     - 명확성과 모니터링을 위해 교육을 세 단계로 구분하는 것이 좋습니다.
#     1. Word2Vec()
#         : 이 첫 번째 단계에서는 모델의 파라미터를 하나씩 설정합니다.
#           매개 변수 문장을 제공하지 않기 때문에 모델을 초기화하지 않은 상태로 일부러 둡니다.
#     2. build_vocab()
#         : 여기 문장의 시퀀스에서 모델 초기화된 어휘들을 만든다.
#     3. train()
#         : 마지막으로 모델을 train합니다.

# Word2Vec 프로세스
# 1.어휘 빌더
#     : 문장 형태의 원시 데이터를 가져와 고유 한 단어를 추출하여 시스템의 어휘를 구축한다
# 2. 컨텍스트 빌더
#     : 단어를 벡터화한다
#         (어휘 빌더의 출력, 즉 색인과 개수를 입력으로 포함하는 어휘 객체)

from gensim.models import Word2Vec

# Word2Vec 훈련 - 대규모 곡 리스트에서 곡끼리의 연관성을 학습
item2vec=Word2Vec(vector_size=300,window=7,min_count=1,workers=10,sg=1)
# 곡 사전 빌더
songs=list(range(len(song)))
item2vec.build_vocab([songs])
# 학습 진행
item2vec.train(corpus_iterable=train_songs,epochs=10,total_words=len(song),total_examples=1)

# Word2Vec 모델 저장
item2vec.save("./item/item2vec.model2")

# Word2Vec 모델 로드
item2vec=Word2Vec.load("./item/item2vec.model2")

# 유사한 곡 10개 추출
top10=item2vec.wv.most_similar(positive=[152422],topn=10)

# 학습 데이터 생성
import random

songs_map = {s["id"]: s for s in song}

train_set = []

for i in train:
    songs = i["songs"]
    tags = i["tags"]
    gnrs = []
    for j in songs:
        gnrs.extend(songs_map[j]["song_gn_dtl_gnr_basket"])
    gnrs = set(gnrs)

    if len(gnrs) < 5:
        continue

    sample_size = len(songs) // 7

    for j in range(sample_size):
        npl = {}

        rs = random.sample(songs, 7)

        npl["s1"] = rs[0]
        npl["s2"] = rs[1]
        npl["s3"] = rs[2]
        npl["s4"] = rs[3]
        npl["s5"] = rs[4]
        npl["s6"] = rs[5]
        npl["label"] = rs[6]

        rg = random.sample(gnrs, 5)

        npl["g1"] = rg[0]
        npl["g2"] = rg[1]
        npl["g3"] = rg[2]
        npl["g4"] = rg[3]
        npl["g5"] = rg[4]

        if len(tags) > 3:

            rt = random.sample(tags, 3)

            npl["t1"] = rt[0]
            npl["t2"] = rt[1]
            npl["t3"] = rt[2]

        elif len(tags) == 0:
            npl["t1"] = ""
            npl["t2"] = ""
            npl["t3"] = ""

        elif len(tags) == 1:
            npl["t1"] = tags[0]
            npl["t2"] = ""
            npl["t3"] = ""

        elif len(tags) == 2:
            npl["t1"] = tags[0]
            npl["t2"] = tags[1]
            npl["t3"] = ""

        train_set.append(npl)

train_df = pd.DataFrame(train_set)

# 학습 데이터 저장
train_df.to_csv("./keras_train.csv",index=False)

import KerasModel
import pandas as pd

# 학습 데이터 로드
train_df=pd.read_csv("./keras_train.csv")

train_df[["t1","t2","t3"]]=train_df[["t1","t2","t3"]].fillna("-")

songs_size=len(song)

gr_nq=list(genre)

t1_nq=train_df["t1"].unique()
t2_nq=train_df["t2"].unique()
t3_nq=train_df["t3"].unique()
tg_nq=np.concatenate((t1_nq,t2_nq,t3_nq))
tg_nq=np.unique(tg_nq)

# KerasModel 모델 훈련 - test data의 곡들과 많이 자주 사용되는 곡들을 추천해 주기 위한 model
model=KerasModel.get_model(songs_size,gr_nq,tg_nq)

from tensorflow.keras.callbacks import ModelCheckpoint

# 입력 데이터 변환
sg_train=train_df[["s1","s2","s3","s4","s5","s6"]].to_numpy()
gr_train=train_df[["g1","g2","g3","g4","g5"]].to_numpy()
tg_train=train_df[["t1","t2","t3"]].to_numpy()

y_train=train_df[["label"]].to_numpy()

# 콜백함수 생성
mc=ModelCheckpoint("./model/search",moniter="val_loss",save_best_only=True,save_weights_only=True,mode="auto")

# 모델 학습
model.fit({"songs":sg_train,"tags":tg_train,"genrs":gr_train},
         y_train,
         batch_size=256,
         epochs=7,
         callbacks=[mc],
         validation_split=0.2)

# 모델 가중치를 저장
model.save_weights("./model/searchover")

import random

# 모델 가중치 로드
model.load_weights("./model/search")

# 테스트 데이터 생성
songs_map = {s["id"]: s for s in song}

for i in train[:10]:
    songs = i["songs"]
    tags = i["tags"]
    gnrs = []
    for j in songs:
        gnrs.extend(songs_map[j]["song_gn_dtl_gnr_basket"])
    gnrs = set(gnrs)

    if len(gnrs) < 5 or len(songs) < 7:
        continue

    sg_test = random.sample(songs, 6)

    gr_test = random.sample(gnrs, 5)

    tg_test = ["-", "-", "-"]
    if len(tags) > 3:
        tg_test = random.sample(tags, 3)
    else:
        for ind, t in enumerate(tags):
            tg_test[ind] = t
    tg_test = np.array(tg_test)

    # 곡들의 연관성 점수 예측 (연관성이 높을 수록 높은 수가 나오고 낮을 수록 낮은 수가 나온다)
    psongs = model.predict({"songs": np.array([sg_test]), "tags": np.array([tg_test]), "genrs": np.array([gr_test])})
    print("psong : ", psongs[0][:5])
    # 예측한 점수들을 높은 순으로 나열
    psings = np.argsort(psongs[0])
    print("psings :", psings[:5])
    # 연과성이 높은 곡들 중 실제 songs목록에 있는 곡들
    acc = [p for p in psings[:10] if p in songs]
    print("acc : ", acc)
    print()

# 학습 데이터 생성

import random

songs_map = {s["id"]: s for s in song}

train_set = []
for pl in train:
    songs = pl["songs"]
    tags = pl["tags"]
    gnrs = []
    for s in songs:
        gnr = songs_map[s]["song_gn_dtl_gnr_basket"]
        gnrs.extend(gnr)
    gnrs = set(gnrs)
    if len(gnrs) < 5:
        continue
    sample_size = len(songs) // 7

    for i in range(sample_size):
        nplT = {}

        rs = random.sample(songs, 7)
        nplT["s1"] = rs[0]
        nplT["s2"] = rs[1]
        nplT["s3"] = rs[2]
        nplT["s4"] = rs[3]
        nplT["s5"] = rs[4]
        nplT["s6"] = rs[5]
        nplT["target"] = rs[6]
        nplT["label"] = 1

        rg = random.sample(gnrs, 5)
        nplT["g1"] = rg[0]
        nplT["g2"] = rg[1]
        nplT["g3"] = rg[2]
        nplT["g4"] = rg[3]
        nplT["g5"] = rg[4]
        if len(tags) > 3:
            rt = random.sample(tags, 3)
            nplT["t1"] = rt[0]
            nplT["t2"] = rt[1]
            nplT["t3"] = rt[2]
        elif len(tags) == 0:
            nplT["t1"] = ""
            nplT["t2"] = ""
            nplT["t3"] = ""
        elif len(tags) == 1:
            nplT["t1"] = tags[0]
            nplT["t2"] = ""
            nplT["t3"] = ""
        elif len(tags) == 2:
            nplT["t1"] = tags[0]
            nplT["t2"] = tags[1]
            nplT["t3"] = ""
        train_set.append(nplT)
    while 1:
        # randint : 균일 분포의 정수 난수 1개 생성
        neg = random.randint(1, len(song))
        if neg not in songs:
            nplF = nplT.copy()
            nplF["target"] = neg
            nplF["label"] = 0
            train_set.append(nplF)
            break

train_df = pd.DataFrame(train_set)

# 학습 데이터 저장
train_df.to_csv("./keras_trainTF.csv",index=False)

# 학습 데이터 로드
train_df=pd.read_csv("./keras_trainTF.csv")

train_df[["t1","t2","t3"]]=train_df[["t1","t2","t3"]].fillna("-")

sg_size=len(song)

gr_nq=list(genre)

t1_nq=train_df["t1"].unique()
t2_nq=train_df["t2"].unique()
t3_nq=train_df["t3"].unique()
tg_nq=np.concatenate((t1_nq,t2_nq,t3_nq))
tg_nq=np.unique(tg_nq)

import KerasModel
# KerasModel 모델 훈련 - target과 곡들의 유사도를 예측해 playlist에 해당 곡을 추천해줄지 결정하는 model
model1=KerasModel.rank_model(sg_size,tg_nq,gr_nq)

import tensorflow as tf

# 입력 데이터 변환
sg_train=train_df[["s1","s3","s3","s4","s5","s6"]].to_numpy()
print(sg_train)

tg_train=train_df[["t1","t2","t3"]].astype("string").to_numpy()
print(tg_train)

gr_train=train_df[["g1","g2","g3","g4","g5"]].astype("string").to_numpy()
print(gr_train)

target_train=train_df["target"].to_numpy()

y_train=train_df["label"].to_numpy()

# 콜백함수 생성
filepath="./model/rank"
mc=tf.keras.callbacks.ModelCheckpoint(
filepath,moniter="val_loss",mode="auto",save_best_only=True,save_weights_only=True)

# 모델 학습
model1.fit({"songs":sg_train,"tags":tg_train,"genres":gr_train,"target":target_train},y_train,batch_size=256,
         epochs=3,
         callbacks=[mc],
         validation_split=0.2)

# 모델 저장
model1.save_weights("./model/rankover")

import KerasModel
model=KerasModel.get_model(sg_size,tg_nq,gr_nq)

model.save_weights("./model/search")

import random

model.load_weights("./model/search")
model1.load_weights("./model/rank")

songs_map = {s["id"]: s for s in song}

for pl in train[:30]:
    songs = pl["songs"]
    tags = pl["tags"]
    gnrs = []
    for s in songs:
        gnr = songs_map[s]["song_gn_dtl_gnr_basket"]
        gnrs.extend(gnr)

    gnrs = set(gnrs)

    if len(gnrs) < 5 or len(songs) < 7:
        continue

    sg_test = random.sample(songs, 6)
    gr_test = random.sample(gnrs, 5)
    tg_test = ["-", "-", "-"]
    if len(tags) > 3:
        tg_test = random.sample(tags, 3)
    else:
        for i, t in enumerate(tags):
            tg_test[i] = t
    tg_test = np.array(tg_test)

    # 후보군
    psongs = model.predict({"songs": np.array([sg_test]), "tags": np.array([tg_test]),
                            "genrs": np.array([gr_test])})

    candidate = np.argsort(psongs[0])[:10]
    print(psongs[0][candidate])

    # 점수화
    socre = np.zeros((10,))
    for i, cs in enumerate(candidate):
        psongs = model1.predict({"songs": np.array([sg_test]), "tags": np.array([tg_test]),
                                 "genres": np.array([gr_test]), "target": np.array([cs])})
        socre[i] = psongs[0]

    for sg in sg_test:
        for i, cs in enumerate(candidate):
            psongs = model1.predict(genre)
            socre[i] = psongs[0]

    top5 = np.argsort(socre)[-5:]
    print(socre[top5])
    print(candidate[top5])
    acc = [p for p in candidate[top5] if p in songs]
    print(acc)