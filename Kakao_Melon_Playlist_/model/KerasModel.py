import tensorflow as tf
from tensorflow.keras import layers,Input,Model,losses,optimizers

# 모델 생성
def get_model(songs_size,tg_nq,gr_nq):

    # Input()
    #    : 케라스 텐서를 인스턴스화하는 데 사용
    #    - shape : 배치 크기를 포함하지 않는 형상 튜플(정수)입니다.
    #              예를 들어, 형상=(32, )은 예상되는 입력이 32차원 벡터의 배치임을 나타냅니다.

    # songs는 6개, tags는 3개, gnrs 5개로 만들어줬으니까 입력 데이터 shape 맞춤
    songs=Input(shape=(6,),dtype="int64",name="songs")
    tags=Input(shape=(3,),dtype="string",name="tags")
    gnrs=Input(shape=(5,),dtype="string",name="genrs")

    # layers.Embedding(input_dim,output_dim)
    #     : 양의 정수(색인)를 고정된 크기의 밀집 벡터로 전환
    #     - input_dim : 어휘목록의 크기, 다시 말해 최대 정수 색인+1
    #     - output_dim : 임베딩의 차원

    # 벡터공간의 좌표로 표현하게 되면 각 대상의 좌표 사이의 거리를 계산하거나,
    # 대상들의 순서를 매기는 것과 같은 다양한 수학적 연산을 할 수 있게 되기 때문에 songs도 임베딩해줌
    # * Output shape
    #    =>3D tensor with shape: (batch_size, input_length, output_dim)
    sg_em=layers.Embedding(songs_size+1,300)(songs)  # batch,6,300

    # StringLookup() : 문자열을 어휘에서 정수 인덱스로 매핑함

    # tags는 한글이므로 정수 인덱스로 매핑
    tg_look=layers.experimental.preprocessing.StringLookup(vocabulary=tg_nq,mask_token=None)(tags)
    tg_em=layers.Embedding(len(tg_nq)+1,50)(tg_look) # batch,3,50

    # gnrs도 문자열이므로 정수 인덱스로 매핑
    gr_look=layers.experimental.preprocessing.StringLookup(vocabulary=gr_nq,mask_token=None)(gnrs)
    gr_em=layers.Embedding(len(gr_nq)+1,10)(gr_look) # batch,5,10

    # Pooling의 역할
    # 필터가 많아지면 그만큼 feature map들이 쌓이게 된다는 거고, 차원이 매우 크다는 뜻이다
    # 높은 차원을 다루려면 그 차원을 다룰 수 있는 많은 수의 파라미터 들이 필요한데
    # 파라미터가 많아지면 학습 시 overfitting이 발생할 수 있다
    # 따라서 필터에 사용된 파라미터 수를 줄여서 차원을 감소시켜주는 역할을 Pooling이 한다

    # Global Average Pooling(GAP)
    #       - Max Pooling, Average Pooling보다 더 급격하게 feature의 수를 줄인다
    #       - Max Pooling,Average Pooling과 다르게 feature를 1차원 벡터로 만든다
    #       - 채널의 feature들을 평균을 낸 다음 채널의 갯수만큼의 원소를 가지는 벡터로 만든다
    #         (height, width, channel)  -> (channel,)
    sg_gp=layers.GlobalAveragePooling1D()(sg_em) # batch, 300
    sg=layers.Dense(64,activation="relu")(sg_gp) # batch, 64

    tg_gp=layers.GlobalAveragePooling1D()(tg_em) # batch, 50
    tg=layers.Dense(16,activation="relu")(tg_gp) # batch, 16

    gr_gp=layers.GlobalAveragePooling1D()(gr_em) # batch, 10
    gr = layers.Dense(64, activation="relu")(gr_gp) # batch, 64

    concat=tf.keras.layers.Concatenate()([sg,tg,gr]) # batch, 144
    drop1=layers.Dropout(0.2)(concat)

    dn1=layers.Dense(64,activation="relu")(drop1) # batch, 64
    dn2=layers.Dense(128,activation="relu")(dn1) # batch, 128
    drop2=layers.Dropout(0.2)(dn2)
    dn3=layers.Dense(songs_size)(drop2)

    model=Model(inputs=[songs,tags,gnrs],outputs=dn3)
    model.summary()

    # 모델 학습 및 검증

    # SparseCategoricalCrossentropy()
    #    - from_logits=True라면 값을 그대로 loss의 입력으로 넣는다.
    #    - from_logits=False라면 이전 activation이 Softmax라면, 입력값을 다시 받아와서 loss의 입력으로 넣는다.
    #                           아무것도 해당되지 않는다면 log 함수를 취해서 loss의 입력으로 넣는다.
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizers.Adam(lr=0.001))

    return model

#get_model(["1","2","3"],["a","b","c"],["1","2","3"],700000)



def rank_model(sg_size,tg_nq,gr_nq):

    songs=Input(shape=(6,),dtype="int64",name="songs")
    tags=Input(shape=(3,),dtype="string",name="tags")
    gnrs=Input(shape=(5,),dtype="string",name="genres")
    target=Input(shape=(1,),dtype="int64",name="target")

    sg_em=layers.Embedding(sg_size+1,300)(songs) # batch 6 300

    target_em=layers.Embedding(sg_size+1,300)(target) # batch 1 300

    tg_look=layers.experimental.preprocessing.StringLookup(vocabulary=tg_nq,mask_token=None)(tags)
    tg_em=layers.Embedding(len(tg_nq)+1,50)(tg_look)

    gr_look=layers.experimental.preprocessing.StringLookup(vocabulary=gr_nq,mask_token=None)(gnrs)
    gr_em=layers.Embedding(len(gr_nq)+1,10)(gr_look)

    sg_gp=layers.GlobalAveragePooling1D()(sg_em)
    sg=layers.Dense(64,activation="relu")(sg_gp)

    tg_gp=layers.GlobalAveragePooling1D()(tg_em)
    tg=layers.Dense(16,activation="relu")(tg_gp)

    gr_gp=layers.GlobalAveragePooling1D()(gr_em)
    gr=layers.Dense(16,activation="relu")(gr_gp)

    concat=tf.keras.layers.Concatenate()([sg,tg,gr]) # batch 90

    drop1=layers.Dropout(0.2)(concat)

    feat_dn=layers.Dense(64,activation="relu")(drop1) # batch 64

    target_em=layers.Flatten()(target_em) # batch 1 300 => batch 300
    target_dn=layers.Dense(64,activation="relu")(target_em) # batch 1 64

    # 두 개의 텐서에서 샘플 간의 내적을 계산하는 계층
    dot=layers.Dot(axes=1,name="dot_similarity")([feat_dn,target_dn])

    rat=layers.Dense(1,activation="sigmoid")(dot)

    model=Model(inputs=[songs,tags,gnrs,target],outputs=rat)

    model.summary()

    model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(lr=0.001))

    return model


# 각각의 스펙트로그램의 특징을 추출하기 위한 model
def image_conv():

    inputs=Input(shape=(48,200,1),name="img")

    # Convolution Layer는 이미지의 특징들을 추출한다.(feature map으로 생성)
    c1=layers.Conv2D(16,(3,3),input_shape=(48,200,1))(inputs)
    drop1=layers.Dropout(0.2)(c1)
    # BatchNormalization은 batch 단위로 입력 데이터로 평균과 분산을 하여 정규화한다.
    bn1 = layers.BatchNormalization()(drop1)
    ac1=layers.Activation("relu")(bn1)
    # MaxPooling2D은 특성범위 내의 픽셀 중 가장 큰 값을 추출하여 특징을 추출한다.
    m1=layers.MaxPooling2D((2,2))(ac1)

    #         Conv2D(filter 수, (filter size))
    c2=layers.Conv2D(32,(3,3),activation="relu")(m1)
    drop2 = layers.Dropout(0.2)(c2)
    bn2 = layers.BatchNormalization()(drop2)
    ac2 = layers.Activation("relu")(bn2)
    m2=layers.MaxPooling2D((2,2))(ac2)

    # Flatten은 다차원의 입력을 단순히 일차원으로 만들기 위해 사용한다
    f=layers.Flatten()(m2)

    #         Dense(출력 뉴련의 수)
    d1=layers.Dense(128,activation="relu")(f)
    d1=layers.Dropout(0.2)(d1)
    d2=layers.Dense(64,activation="relu")(d1)
    d2=layers.Dropout(0.2)(d2)
    d3=layers.Dense(32,activation="relu")(d2)
    dr=layers.Dropout(0.2)(d3)

    return Model(inputs,dr)


# image_conv함수에서 출력된 랜덤한 임베딩 값을 dot연산해 나온 결과와 label과 비교를 위한 model
def image_sim():
    img1=Input(shape=(48,200,1),name="img1")
    img2=Input(shape=(48,200,1),name="img2")

    conv_layer=image_conv()
    conv1 = conv_layer(img1)
    conv2 = conv_layer(img2)

    # 두 개의 텐서에서 샘플 간의 내적을 계산하는 계층
    dot=layers.Dot(axes=1,name="dot_similarity")([conv1,conv2])

    d3=layers.Dense(1,activation="sigmoid")(dot)

    sim_model=Model(inputs=[img1,img2],outputs=d3)
    sim_model.summary()
    sim_model.compile(loss=losses.BinaryCrossentropy(),optimizer=optimizers.Adam(lr=0.001))
    return sim_model

from tensorflow.keras.utils import Sequence
import numpy as np


# npy라는 형태의 데이터를 활용하기 위해 커스텀하게 함수 생성
class DataGenerator(Sequence):

    def __init__(self, dataframe,batch_size=32, shuffle=True):
        # 이미지를 찾을 수 있는 경로가 dataframe에 들어있다(id로 찾을 수 있다)
        self.df=dataframe
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    # getitem의 batch에 활용되는 인덱스 목록을 리턴해주기 위한 함수
    def __len__(self):
        return int(np.ceil(len(self.df)/self.batch_size))

    # 한번의 batch안에서 필요한 아이템을 가져오는 함수
    def __getitem__(self,index):
        indexes=self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        batch_df=self.df.iloc[indexes]

        input_id=batch_df["input"].to_numpy()
        target_id=batch_df["target"].to_numpy()
        y_train=batch_df["label"].to_numpy()

        X_input=[]
        X_target=[]

        for id in input_id:
            sid=str(id)
            if id>=1000:
                img1=np.load("./arena_mel/"+sid[:-3]+"/"+sid+".npy")
            else:
                img1=np.load("./arena_mel/0/"+sid+".npy")

            img1=img1[:,:200]
            img1=np.expand_dims(img1,axis=-1)
            X_input.append(img1)

        for id in target_id:
            sid = str(id)
            if id >= 1000:
                img1 = np.load("./arena_mel/" + sid[:-3] + "/" + sid + ".npy")
            else:
                img1 = np.load("./arena_mel/0/" + sid + ".npy")

            img1 = img1[:, :200]
            # axis로 지정된 차원을 추가
            img1 = np.expand_dims(img1, axis=-1)
            X_target.append(img1)

        return  {"img1":np.array(X_input),"img2":np.array(X_target)},y_train

    # epochs마다 어떤 처리 할지 정해준다
    def on_epoch_end(self):
        # 가지고 있는 데이터 세트의 인덱스를 가져온다
        self.indexes=np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# https://hwiyong.tistory.com/335
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot?hl=ko