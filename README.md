# LG_image_to_image

데이콘에서 실시한 LG 카메라 이미지 품질 향상 AI 경진대회 데이터를 가지고 Pix2Pix를 구현한 결과입니다.

컴퓨팅 파워가 부족하여 학습 시 많은 어려움이 있었지만 DataLoader를 수정하여 학습이 가능하게 만들었습니다. 그렇다고 좋은 성능을 내지는 못했습니다... 구현에 의의를 둔 개인 프로젝트입니다.

## 파일 설명

1. Generator.py  ,  Discriminator.py : 생성자와 판별자의 모델에 대한 파일 입니다.

2. training_code.ipynb : 학습때 사용한 노트북 파일입니다.

3. evaluation_code.ipynb : 학습을 완료한 후에 모델을 평가할 때 사용한 노트북 파일 입니다.

