# 딥러닝을 이용한 신문 요약

  본 프로젝트는 2021년 상반기 졸업작품을 위해 개설되었으며 딥러닝 기반 자연어 처리 모델인 M3를 이용하여 신문, 뉴스 기사를 요약하는 시스템을 구축하고자 한다.

### History

| 날짜       | 내용                                   | 작성자 |
| ---------- | -------------------------------------- | ------ |
| 2021.03.15 | Readme.md 초안 작성                    | 정민수 |
| 2021.04.06 | 양식 변경, 기존 Readme는 보고서로 작성 | 정민수 |
| 2021.04.30 | generate.py 실행 방법 작성             | 정민수 |

### 개발 환경

* Ubuntu 18.04.3 LTS (GNU/Linux 4.15.0-129-generic x86_64)
* Python 3.8
* Pytorch 1.8.0
* Conda 4.9.2

## 개요

  문서 요약(Text Summarization)은 텍스트 데이터를 수집하여 필요한 정보를 압축해서 정보를 제공하는 자연어 처리(Natural Language Processing, NLP) 세부 분야 중 하나이다. 본 프로젝트에서 신문에 대하여 위 기술을 적용하여 신문을 요약하는 시스템을 구축하고자 한다.

### 디렉토리 구조

```
News-Summarization-With-Deep-Learning/
├── crawler/
│	├── config.ini
│	├── main.py
│	├── parsing.py
│	├── processing.py
│	└── crawling/
│		├── crawling_sample.json
│		├── parsing_sample.tsv
│		└── processing_sample.json
├── data/
│	└── sample_test.tsv
└── model/
	├── generate.py
	├── masked_cross_entropy.py
	├── utils.py
	├── requirements.txt
	├── tokenizer/
	│	└── sentencepiece.model
	└── ket5_finetuned/
		├── config.json
		├── pytorch_model.bin
		└── training_args.json
```

* crawler: 데이터셋을 구축하기 위해 인사이트 뉴스를 크롤링하기 위한 코드가 포함된 디렉토리
* data: 크롤링한 데이터를 정제하고 학습과 평가에 사용하기 위해 데이터를 저장하는 디렉토리
* model: 학습과 추론을 진행하는 코드가 포함된 디렉토리

## 설치

  ```sh
$ git clone https://github.com/skaeads12/News-Summarization-With-Deep-Learning
  ```

  위 명령어를 입력하면 News-Summarization-With-Deep-Learning/ 디렉토리가 생성된다. 기본적인 디렉토리 구조는 앞서 설명한 [디렉토리 구조](#디렉토리-구조)를 따른다.

## 크롤링(Crawling)

  크롤링 코드는 News-Summarization-With-Deep-Learning/crawler/ 에 샘플 코드가 작성되어 있으며 News-Summarization-With-Deep-Learning/crawler/crawling/ 내부에 결과 샘플이 있다.

* **config.ini**

```
[INDEX]
currentindex = 332644

[OUTPUT]
path = ./crawling/
```

  'config.ini' 파일은 크롤링을 시작할 인덱스와 출력될 폴더를 결정한다.



* **main.py**
  * 'config.ini' 파일에서 시작할 인덱스를 찾고 그 인덱스부터 크롤링을 시작한다.
  * 'config.ini' 파일에 설정한 폴더에 크롤링한 기사를 출력한다.
  * 100번째마다 출력하며 출력된 데이터는 폴더에 'insight_crawling_{date}.json'으로 출력된다.



<img src = "https://tva1.sinaimg.cn/large/008eGmZEgy1gpdj6o9oc2j30u00xxqv7.jpg" width=50%>

> 원문 기사: https://insight.co.kr/news/107767

**main.py 사용방법**

```sh
$ cd News-Summarization-With-Deep-Learning/crawler/
$ python main.py
```

**main.py 출력 예시**

<img src="https://tva1.sinaimg.cn/large/008eGmZEgy1gpdj9m41ybj32i60j4q8l.jpg">

> 참조: crawler/crawling/crawling_sample.json



* **processing.py**
  * 크롤링된 데이터의 후처리를 하는 파이썬 코드이다.
  * 더미 데이터가 많은 '인사'와 '엔터테인먼트' 섹션을 스킵하고 첫 줄 요약과 특수 문자('\n', '\t', '\xa0' 등)를 제거한다.
  * Json.dumps를 여러번 사용하기 때문에 미처 올바른 JSON 양식이 되지 못한 크롤링 데이터를 올바른 JSON 포맷으로 수정한다.

**processing.py 사용 방법**

```sh
$ python processing.py {open_path} {output_path}
```

> {open_path}: 후처리를 실행할 크롤링 데이터
>
> {output_path}: 출력될 파일 이름
>
> ex) python processing.py crawling_sample.json processing_sample.json

**processing.py 출력 예시**

<img src="https://tva1.sinaimg.cn/large/008eGmZEgy1gpdjmji29bj310m0u0dxv.jpg">

> 참조: crawler/crawling/processing_sample.json

* **parsing.py**
  * 후처리를 마친 코드를 TSV 파일로 파싱하는 파이썬 코드이다.
  * summary와 content 내용을 각각 {content}\t{summary}로 변환하여 한 줄에 하나의 샘플로 입력되도록 파일을 작성한다.

**parsing.py 사용 방법**

```sh
$ python parsing.py {open_path} {output_path}
```

> {open_path}: TSV로 변환할 파일
>
> {output_path}: 출력될 파일 이름
>
> ex) python parsing.py crawling/processing_sample.json crawling/parsing_sample.tsv

**parsing.py 출력 예시**

<img src="https://tva1.sinaimg.cn/large/008eGmZEgy1gpdjls65s6j31yq0u0k38.jpg">

> 참조: crawler/crawling/parsing_sample.tsv

## 평가 방법

  평가를 위한 코드는 model/ 디렉토리에 있다.

```sh
$ cd News-Summarization-With-Deep-Learning/model/
```

**라이브러리 설치**

```sh
$ conda create --name news_summarizer python=3.8
$ conda activate news_summarizer
$ pip install -r requirements.txt
```

* 설치되는 라이브러리

```
torch
tqdm
nltk
transformers
sentencepiece
```

**사전학습 모델 다운로드**



**평가**

```sh
$ python generate.py -b BATCH_SIZE -td TEST_DATASET_PATH -w FINETUNED_MODEL_DIR_PATH
```

  훈련된 샘플 모델을 `ket5_finetuned/` 디렉토리에 제공하고 있으며 평가 데이터는 `data/sample_test.tsv` 에 제공하고 있다.

```sh
$ python generate.py -b 16 -td ../data/sample_test.tsv -w ket5_finetuned/
```



