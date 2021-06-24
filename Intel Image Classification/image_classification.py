import tensorflow as tf
"""
ImageDataGenerator():
    데이터 확대 유형 중 하나입니다. 원본 데이터를 사용하여 데이터를 생성하는 곳에서는 스케일링, 확대/축소, 중앙 집중력 증가, 잘라내기, 이미지 만들기 등의 작업을 수행합니다.
    실시간 데이터 확대를 통해 텐서 이미지 데이터 배치를 생성합니다.
    - rescale : 배율 조정
    - zoom_range : 임의의 확대/축소 범위
    - width_shift_range :  수평방향 내에서 임의로 원본 이미지 좌우로 이동
    - height_shift_range :  수직방향 내에서 임의로 원본 이미지 상하로 이동
    - validation_split : 주어진 데이터셋을 test와 training으로 나누는 비율
"""
# 다양한 방향으로 찍힌 이미지들을 잘 캐치할 수 있도록 데이터를 회전,좌우반전,이동 등으로 이미지를 바꾸기 위해 ImageDataGenerator사용(데이터 전처리)
igd=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,zoom_range=0.2,
                                                    width_shift_range=0.1,
                                                    height_shift_range=0.1,
                                                    validation_split=0.2)
"""
flow_from_directory():
     directory경로로부터 dataframe을 가져오고 augmented 또는 normalized된 data배치를 생선한다
     주요 인자
       - 첫번째 인자 : 이미지 경로를 지정
       - target_size : 패치 이미지 크기 지정
                       폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절
       - batch_size : 배치 크기를 지정
       - class_mode : 분류 방식에 대해 지정
                    - categorical : 2D 원-핫 부호화된 라벨 반환
                    - binary : 1D 이진 라벨 반환
                    - sparse : 1D 정수 라벨 반환
                    - None : 라벨이 반환되지 않음
       - save_format : 저장 형태 지정
                     - png (default)
                     - jpg
       - color_mode : 몇 개의 채널을 가질지 여부
                    - grayscale 채널 1개
                    - rbg 채널 3개 (default)
                    - rgba 채널 4개
       - subset : ImageDataGenerator에서 validation_split으로 선정한 training과 validation의 비율만큼 이미지를 가져옴
"""


# 이미지 데이터들이 들어있는 directory자체를 불러와 training set 과  validation set을 나눠주기 위해 flow_from_directory()사용
traing_g=igd.flow_from_directory("./seg_train/seg_train", target_size=(128,128),
                               color_mode="rgb", class_mode="sparse",
                               batch_size=32,save_format="jpg", subset="training")

validation_g=igd.flow_from_directory("./seg_train/seg_train", target_size=(128,128),
                               color_mode="rgb", class_mode="sparse",
                               batch_size=32,save_format="jpg", subset="validation")


for image_batch,labels_batch in traing_g:
    print(image_batch.shape)
    print(labels_batch)
    break

from tensorflow.keras import datasets,layers,models
"""
# Sequential model을 사용해서 레이어를 선현으로 연결해 보았다.
# ***Sequential은 첫번째 레이어의 입력 형태에 대한 정보를 필요로 하지만 그 후에는 저동으로 형태를 추정하여 형태 정보를 갖고 올 필요가 없다
model=models.Sequential()
model.add(layers.Conv2D(16,(3,3),activation="relu",input_shape=(128,128,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3,3),activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(6,activation="softmax"))
"""

# 모델 생성
"""
Conv2D : 특정한 패턴의 특징이 어디서 나타나는지를 확인하는 도구
       - 첫번째 인자 : filter 수 (filter 하나당 하나의 이미지가 만들어진다)
            => 여기서는 첫번째 Conv2D에서 filter 수를 16개 지정해줘서 16개의 feature map이 만들어진다
       - 두번째 인자 : filter size
       - padding : 경계 처리 방법
                 - vaild : 유효한 영역만 출력됨. 따라서, 출력이미지 사이즈는 입력이미지보다 작다
                 - same : 출력 이미지 사이즈가 입력이미지 사이즈와 동일
       - input_shape : 샘플 수를 재외한 입력 형태 정의(모델에서 첫 레이어에만 적용 (행, 열, 채널 수) )
       - activation : 활성화 함수 설정
                    - 
"""
"""
MaxPooling2D : feature map으로부터 값을 샘플링해서 정보를 압축한다
        *  MaxPooling2D 사용하는 이유
             - 이미지의 크기를 줄이면서 데이터의 손실을 막기 위해 합성곱 계층에서 스트라이드 값을 1로 지정하는 대신, 폴링 계층을 사용한다
             - 합성곱 계층의 과적합(overfittin)을 막기위해 사용한다
"""

# 이미지를 높이 128, 너비 128, 채널 3으로 형태 정보를 준다
inputs=tf.keras.Input(shape=(128,128,3),name="img")

# Conv2D, MaxPooling2D으로 주요 특징을 추출(4차원)

# filter 수를 16개 지정해줘서 16개의 feature map이 만들어진다
c1=layers.Conv2D(16,(3,3),activation="relu",input_shape=(128,128,3))(inputs)
# feature map으로부터 값을 샘플링해서 정보를 압축한다(overfittin막기 위해 사용)
m1=layers.MaxPooling2D(2,2)(c1)

# BatchNormalization으로 평균 출력을 0에 가깝게 유지하고 출력 표준 편차를 1에 가깝게 유지하는 변환을 적용
b1=layers.BatchNormalization()(m1)

# filter 수를 32개 지정해주므로 32개의 feature map이 만들어진다
c2=layers.Conv2D(32,(3,3),activation="relu")(b1)
m2=layers.MaxPooling2D(2,2)(c2)

# Conv2D는 3차원 형태의 관측치를 입력으로 받는다 ****(tensolflow 가 그렇게 정햿댜)
# 그래서 Dense layer에 전달하기 위해 2차원 자료를 1차원 자료로 변환
f=layers.Flatten()(m2)

# overfitting을 막기위해 0.2비율로 몇개의 노드 Dropout
dr=layers.Dropout(0.2)(f)

dr2=layers.Dense(64,activation="relu")(dr)
# 출력 뉴런수 6(label이 6개니까 6개로 설정)
# **** 활성화 함수로 softmax를 사용하여 해당 target의 확률 계산
outputs=layers.Dense(6,activation="softmax")(dr2)

model=tf.keras.Model(inputs=inputs,outputs=outputs)

# 모델 학습과정 설정

# loss : 현재 가중치 세트를 평가하는데 사용한 손실함수
# ****여기서는 훈련데이터의 label(target)이 정수라 SparseCategoricalCrossentropy사용

# optimizer : 최적의 가중치를 찾는데 사용되는 최적화 알고리즘으로 경사 하상법 알고리즘 중 Adam사용

# Metric : 학습 평가 기준 (학습 과정 중 제대로 학습되고 있는지 확인)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.summary()

# 모델 학습
"""
fit() 인자
   - 첫번째 인자 : training 시킬 데이터 입력
   - epochs : 전체 훈련 데이터셋의 학습 반복 횟수 지정
   - validation_data : 검증데이터셋을 제공할 제너레이터 지정
   - validation_steps : epoch종료 때마다 검증 스텝수를 지정
"""
history=model.fit(traing_g,epochs=5,validation_data=(validation_g))

# 결과 시각화

# epoch별로 loss와 val_loss의 경향, acc와 val_acc의 경향보기
import matplotlib.pyplot as plt

def show_graph(history_dict):
    accuracy = history_dict['acc']
    val_accuracy = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 1))

    plt.subplot(121)
    plt.subplots_adjust(top=2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Trainging and validation accuracy and loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)
    #     plt.legend(bbox_to_anchor=(1, -0.1))

    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)
    #     plt.legend(bbox_to_anchor=(1, 0))

    plt.show()

show_graph(history.history)

# test data도 ImageDataGenerator 생성
test_igd = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255 )
test_g = test_igd.flow_from_directory("./seg_test/seg_test",target_size=(128,128),color_mode="rgb", class_mode="sparse",
                               batch_size=32,save_format="jpg")

# 모델 평가
evaluate=model.evaluate(test_g)
dict(zip(model.metrics_names, evaluate))