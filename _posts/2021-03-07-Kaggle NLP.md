---
layout: post
author: 한지상
title: "Kaggle(캐글) - Natural Language Processing with Disaster Tweets"
date: 2021-03-07 01:20:00
categories: Machine_Learning
tags: [ML, 머신러닝, sklearn, Python]
cover: "/assets/캡처_2021_03_07_01_20_51.png"
---

# Kaggle(캐글) - Natural Language Processing with Disaster Tweets

> [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

## 1. 문제 설명 및 구상

train_set로 주어진 트윗 문장들을 보고 실제로 disaster을 나타내는 문장인지, disaster과 관련된 단어들이 문장에 포함되어있지만 사실 disaster을 나타내지 않는 문장인지 판단하는 문제이다. 

이 문제는 (x, y) = (문장의 단어의 포함 여부, diaster의 True/False) 를 가리는 선형 분류 문제이므로 Linear Classifier을 사용할 것이다.

<br>

## 2. 풀이 및 코드

<a href="/assets/캡처_2021_03_07_01_18_23.png">![](/assets/캡처_2021_03_07_01_18_23.png)</a>

우선 train_df와 test_df에 csv파일을 불러온다.

<br>

<a href="/assets/캡처_2021_03_07_01_18_40.png">![](/assets/캡처_2021_03_07_01_18_40.png)</a>

> 

문장을 벡터화시키기 위해 feature_extraction.text의 CountVectorizer을 사용한다. 이 메서드는 입력으로 주어진 문장들에서 출현한 단어들을 토대로 딕셔너리형태의 새로운 사전을 생성하고, 각각의 문장들에 대해서 단어가 출현되었으면 1, 출현되지 않았으면 0으로 변환한다.

<br>

<a href="/assets/캡처_2021_03_07_01_18_52.png">![](/assets/캡처_2021_03_07_01_18_52.png)</a>

train_vectors와 test_vectors에 각각 count_vectorizer 메서드를 fit_transform 및 transform시켜준다.

train_set의 평균과 표준편차를 test_set에서도 동일하게 사용하기 위해 test_set은 transform만 시켜준다.

<br>

<a href="/assets/캡처_2021_03_07_01_19_07.png">![](/assets/캡처_2021_03_07_01_19_07.png)</a>

Classifier모델 중 Ridge와 SGD를 사용하여 모델을 만들어보았다.
그리고 각각의 모델에 대한 k겹 교차검증을 한 정확도를 출력해보았다.

Ridge는 이상치 데이터를 무시하여 과대적합을 피할 수 있어서 최종적으로 RidgeClassifier모델을 사용하였다.

<a href="/assets/캡처_2021_03_07_01_19_20.png">![](/assets/캡처_2021_03_07_01_19_20.png)</a>
<a href="/assets/캡처_2021_03_07_01_38_22.png">![](/assets/캡처_2021_03_07_01_38_22.png)</a>

캐글에 제출한 결과 78%의 정확도를 얻었다. SGDClassifier모델로도 제출해 본 결과, 77.3%의 정확도를 얻었다.


