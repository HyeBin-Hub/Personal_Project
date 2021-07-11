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




# https://hwiyong.tistory.com/335
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot?hl=ko