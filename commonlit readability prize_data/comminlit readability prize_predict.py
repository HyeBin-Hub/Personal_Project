import pandas as pd
import nltk
import re
import tensorflow as tf
nltk.download("stopwords")

# 데이터 로드
df=pd.read_csv("./train.csv")
"""
전처리 
- NLTK(Natural Language Toolkit)
    : 언어 처리 기능을 제공하는 파이썬 라이브러리
    - 분류 토큰화(tokenization), 스테밍(stemming)와 같은 언어 전처리 모듈 부터 구문분석(parsing)과 같은 언어 분석, 클러스터링, 감정 분석(sentiment analysis), 태깅(tagging)과 같은 분석 모듈 시맨틱 추론(semantic reasoning)과 같이 보다 고차원 적인 추론 모듈도 제공하고 있다.
    - NLTK는 위와 같은 100여개 이상의 영어 단어들을 불용어로 패키지 내에서 미리 정의하고 있다
- 불용어(stopword) 제거
    : 불용어(stopword)는 분석에 큰 의미가 없는 단어를 지칭한다
    - is, the, a, will등 빈번하게 텍스테 나타나지만 분석을 하는 것에 있어서 큰 도움이 되지 않는 단어들을 뜻한다
    - 이러한 불용어들을 제거하지 않으면 그 빈번함으로 인해 오히려 중요한 단어로 인지될 수 있다
"""
# 데이터 전처리
def rmove_word(test_list):
    stop_words=set(nltk.corpus.stopwords.words("english"))
    result=[]
    for text in test_list:
        # 특수 문자 제거
        text=re.sub("[-=+,#/\?:';^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]","",text)
        words=text.split(" ")
        # 불용어 제거
        text=" ".join([w for w in words if w not in stop_words])
        result.append(text)
    return result

# 해당 text를 ndarray형태로 변형시켜 전처리 작업을 해준다
test_list=df["excerpt"].to_numpy()
result=rmove_word(test_list)


"""
Tokenizer()
    : 텍스트 corpus를 벡터화시킨다
    - 기본적으로 모든 구두점은 제거되어 텍스트가 공백으로 구분된 단어 시퀀스로 바뀐다
    - 그런 다음 이러한 시퀀스는 토큰 목록으로 분할된다
    - 그런 다음 인덱싱되거나 벡터화된다 

"""
# 모든 인지를 기본으로 하여 텍스트 corpus를 벡터화시킨다
tok=tf.keras.preprocessing.text.Tokenizer()
"""
fit_on_texts() 
    : 텍스트 목록을 기반으로 내부 어휘를 업데이트한다
    - 텍스트에 목록이 포함된 경우 목록의 각 항목은 토큰으로 가정함
    - texts_to_sequences 또는 texts_to_matrix를 사용하기 전에 필요하다.
"""

# 텍스트 데이터를 입력받아 리스트이 형태로 변환한다
tok.fit_on_texts(result)
"""
get_config()
    - tokenizer 구성을 Python dictionary로 반환한다.
"""

# 총 단어의 길이
word_size=len(tok.get_config()["word_counts"])-1
"""
texts_to_sequences()
    : 텍스트의 각 텍스트를 정수 시퀀스로 변환한다
    - 최상위 num_words-1 단어만 고려된다, tokenizer 에서 알고 있는 단어만 고려된다
"""
# 텍스트 안의 단어들을 정수형 시퀀스의 형태로 변환한다
sequences=tok.texts_to_sequences(result)


for s in sequences[:10]:
    print(len(s))

"""
pad_sequences()
    : 시퀀스를 입력하면 숫자 0을 이용해서 같은 길이의 시퀀스로 변환시킨다
    - 가장 긴 시퀀스를 기준으로 모두 같은 길이의 시퀀스를 포함하는 numpy array로 변환한다
    - 파라미터
        * sequences : 리스트의 리스트로, 각 성분이 시퀀스이다
        * maxlen : 정수, 모든 시퀀스의 최대 길이를 설정하여 제한한다.
        * dtype : 출력 시퀀스의 자료형
        * padding : 문자열이 들어가다. "pre"가 default값으로 안쪽에 0이 추가되고,
                    "post"는 뒤쪽으로 0이 추가된다
        * truncating : 문자열, "pre"는 길이가 초과됐을 때 앞쪽을 자르고, "post"는 maxlen보다 큰 시퀀스의 끝의 값들을 제거
        * value : 부동소수점 or 문자열, 패딩할 값
"""
# 딥러닝 모델에 입력 하려면 학습 데이터의 길이가 통일되어야 한다
# pad_sequences()로 리스트의 길이를 120으로 통일 시킨다 (이 과정을 패딩 (padding)이라고 한다.)
sequences=tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=120)
print(sequences)

y_train=df["target"].to_numpy()

# 모델 생성
inputs=tf.keras.Input((120,))
# 입력될 총 단어의 수는 get_config로 구해준 길이, 출력되는 벡터의 크기는 64로 지정한다
emb=tf.keras.layers.Embedding(input_dim=word_size,output_dim=64)(inputs)
b1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(emb)
dens=tf.keras.layers.Dense(64,activation="relu")(b1)
outputs=tf.keras.layers.Dense(1)(dens)

model=tf.keras.Model(inputs,outputs)
model.summary()

# 모델 학습과정 설정

# 손실함수로 MeanSquaredError()를 사용하였고, 최적화 함수로 Adam을 사용하고 learning rate는 0.001로 지정했다
model.compile(loss=tf.keras.losses.MeanSquaredError(),
             optimizer=tf.keras.optimizers.Adam(lr=0.001),
             metrics=["accuracy"])

# 모델 학습
model.fit(sequences,y_train,batch_size=32,epochs=10)


# 테스트 데이터 로드
test_df=pd.read_csv("./test.csv")

# 테스트 데이터도 해당 text를 ndarray형태로 변형시켜 전처리 작업을 해준다
test2_list=test_df["excerpt"].to_numpy()
result2=rmove_word(test2_list)

# 텍스트 데이터를 입력받아 리스트이 형태로 변환한다
tok.fit_on_texts(result2)

# 테스트 데이터의 모든 단어의 길이
word_size2=len(tok.get_config()["word_counts"])-1

# 텍스트 안의 단어들을 정수형 시퀀스의 형태로 변환한다
sequences2=tok.texts_to_sequences(result2)

# 테스트 데이터도 동일하게 리스트의 길이를 120으로 통일 시킨다
sequences2=tf.keras.preprocessing.sequence.pad_sequences(sequences2,maxlen=120)

# 모델 예측
prd=model.predict(sequences2)
print(prd)