# Instagram 목표 게시글 판별 딥러닝 모델 학습

이 프로젝트는 유명 딥러닝을 사용하여 Instagram 게시글이 목표 게시글인지 판별하는 프로젝트입니다.
딥 러닝을 이용한 자연어 처리 입문 (Team NLP)의 내용을 기반으로 합니다.

## 프로젝트 개요

이 프로젝트는 Instagram 게시글의 텍스트 데이터를 기반으로 딥러닝 모델을 학습시켜, 게시글이 목표 게시글인지 하는 것을 목표로 합니다.
학습 데이터는  caption(string), is_target(bool) 필드로 구성되어야 하며, is_target의 값을 예측하는 방향으로 모델이 학습됩니다.

## 설치 및 실행

1. `requirements.txt` 파일을 사용하여 필요한 라이브러리 및 패키지를 설치합니다. 아래 명령어를 사용할 수 있습니다:

   ```shell
   pip install -r requirements.txt
   ```

2. 프로젝트를 실행하기 위해 Instagram 게시글 데이터를 수집해야 합니다. 이를 위해 [scarpping-instagram](https://github.com/ukkoon/scarpping-instagram)를 사용할 수 있습니다.

3. (option)데이터 전처리를 진행합니다. caption값을 lint합니다. (목표 게시글의 성격에 따라 공백 · 숫자 등을 제거할 수 있습니다.)

4. 모델 학습을 위해 `train_model.py` 스크립트를 실행합니다. 학습에 사용되는 하이퍼파라미터와 모델 구조는 해당 스크립트 내에서 수정할 수 있습니다.

5. 학습된 모델로 테스트 예측을 수행하려면 `test_predict.py` 스크립트를 실행합니다. 이 스크립트는 입력 데이터에 대한 성격 예측 결과와 확률을  출력합니다.