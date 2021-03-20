---
layout: post
author: 한지상
title: "Kaggle(캐글) - Digit Recognizer"
date: 2021-03-07 01:20:00
categories: Machine_Learning
tags: [ML, 머신러닝, sklearn, Python]
cover: "/assets/캡처_2021_03_07_01_20_51.png"
---

# Kaggle(캐글) - Digit Recognizer_sklearn
---
> [Kaggle(캐글) - Digit Recognize](https://www.kaggle.com/c/digit-recognizer)

<br>

## 1. 문제 설명 및 구상

유명한 mnist 숫자 데이터셋의 일부를 train set으로 준다. 이미지를 28 x 28 픽셀로 만들어 label과 함께 주어진다. 

0 ~ 9까지의 숫자를 판별하면 되므로 다중분류기를 사용하면 된다.

<br>

## 2. 풀이 및 코드

sklearn의 SVC()를 사용하여 다중분류를 하였다.

![](/assets\캡처_2021_03_20_15_57_14.png)

<br>
우선 x_train에는 28 x 28픽셀 값을, y_train에는 label값을 저장한다.

<br>

![](/assets\캡처_2021_03_20_15_57_32.png)

sklearn.svm의 SVC()를 사용하면 알아서 다중분류를 해준다.

<br>

![](/assets\캡처_2021_03_20_16_07_29.png)

굉장히 간단한 문제로, 채점 정확도는 약 97%가 나왔다.
