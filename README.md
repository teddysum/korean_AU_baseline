# korean_AU_baseline

본 소스코드는 '국립국어원 상시평가'의 비윤리적 문장 분류의 베이스라인 모델 및 학습과 평가를 위한 코드를 제공하고 있습니다. 

코드는 'au_main.py'이고, 'train.sh' 을 이용하면 학습에 용이하고, 학습된 모델을 이용하여 'demo.sh'를 이용하여 결과를 생성 한 뒤 'test.sh'를 실행하여 결과물에 대한 평가를 할 수 있습니다.



## 데이터
데이터는 국립국어원 모두의 말뭉치에서 다운받으실 수 있습니다. https://corpus.korean.go.kr/

데이터는 ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&'] 들로 비식별화가 돼있습니다.

#### example
``` 
{"id": "nikluge-au-2022-train-000001", "input": "보여주면서 왜 엿보냐고 비난 하는것도 웃기지만. 훔쳐 보면서 왜 보여주냐고 하는 사람 역시 우습다..", "output": 1}
{"id": "nikluge-au-2022-train-000002", "input": "왜 개인 사생활을 방송으로 보여주고 싶은지 이해도 안가지만 &location&식 프로포즈란 무슨 자로 잰 든 무릎 꿇고 반지 내밀고 나랑 결혼해줄래? 가 전부이다.", "output": 1}
{"id": "nikluge-au-2022-train-000003", "input": "이런 쓰레기같은 새끼가 아무렇지 않게 멀쩡히 돌아다닐 생각을 하니까 진짜 너무 소름돋는다.", "output": 1}
{"id": "nikluge-au-2022-train-000004", "input": "인간의 탈을 쓰고...", "output": 1}
{"id": "nikluge-au-2022-train-000005", "input": "인기글에 짱깨뭐라하니까 댓글로 ㅂㄷㅂㄷ하네요...", "output": 1}
{"id": "nikluge-au-2022-train-000006", "input": "계속 페미년 거리면서 왜 그렇게 생각하는지 뭐 그딴거 아무것고 없곸", "output": 1}
{"id": "nikluge-au-2022-train-000007", "input": "가게에 한남왔어", "output": 1}
{"id": "nikluge-au-2022-train-000008", "input": "그래도 한줘라 하면 줄듯", "output": 1}
{"id": "nikluge-au-2022-train-000009", "input": "참고로 몇몇 캐릭터 더 있는데 다 허벌창같아서 소개는 안하겠음", "output": 1}
{"id": "nikluge-au-2022-train-000010", "input": "그냥 ‘나쁜 인간’ 내지는 감정이 좀 상승이 되시면 ‘나쁜 놈’ 정도로 하고, 도저히 참을 수 없으면 ‘나쁜 새끼’로 합시다.", "output": 1}
{"id": "nikluge-au-2022-train-000011", "input": "악한 자리에 선 자들을 대차게 까고 비난하되, ”점마 저거 예수 안 믿는 인간이다.” 이런 말은 좀 자제합시다.", "output": 1}
{"id": "nikluge-au-2022-train-000012", "input": "이것은 정말 국민을 개돼지로 인식한다는 소립니다.", "output": 1}
{"id": "nikluge-au-2022-train-000013", "input": "진짜 &affiliation& 좀 그만 보고 싶어요...", "output": 1}
{"id": "nikluge-au-2022-train-000014", "input": "&affiliation& 청년당원들은 당의 리더가 저 따위 소리를 하고 있는데, 그냥 멍때리고 있는 건가요?", "output": 1}
{"id": "nikluge-au-2022-train-000015", "input": "왜 청년들 일하는 데에 끼어 드는데?", "output": 1}
{"id": "nikluge-au-2022-train-000016", "input": "왜 굳이 청년 정책 관련된 일을 하는데?", "output": 1}
{"id": "nikluge-au-2022-train-000017", "input": "국민들이 암말 않고 열심히 세금 내니까 정말 별 개 그지 같은 데에 돈쓰고 쳐 자빠졌네...", "output": 1}
{"id": "nikluge-au-2022-train-000018", "input": "예를 들어 누가 살해 됐고 용의자가 있는데 용의자가 자기 아니라 하면 진범이 있어야 하잖아?", "output": 0}
{"id": "nikluge-au-2022-train-000019", "input": "&name& 일당은 뭐야?", "output": 0}
{"id": "nikluge-au-2022-train-000020", "input": "귀신이 먹었어?", "output": 0}
```

#### 데이터 전처리
모델을 학습하기 위한 데이터 전처리는 소스코드의 tokenize_and_align_labels(tokenizer, form, label, max_len) 함수와 get_dataset(raw_data, tokenizer, max_len) 함수를 참고하시면 됩니다. tokenize_and_align_labels에서 원하는 형태의 데이터 형태로 가공하고, get_dataset에서 pytorch의 DataLoader를 이용하기 위한 TensorDataset 형태로 가공합니다.


## 모델 구성

xlm-roberta-base(https://huggingface.co/xlm-roberta-base)를 기반으로 학습하였습니다.

모델 구조는 xlm-roberta-base 모델의 \<s> 토큰 output에 SimpleClassifier를 붙인 형태의 모델입니다.

학습된 baseline 모델은 아래 링크에서 받으실 수 있습니다.

model link: https://drive.google.com/file/d/1ZVZrLtFJrx6I0NpFpPNWdFi0Z9cYB6-6/view?usp=share_link

모델 입력형태를 \<s>발화 form</s>와 같이하고, 비윤리적 표현인지에 대해 0, 1로 이진 분류를 합니다.

데이터에서는 ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&'] 태그들을 활용하여 비식별 조치를 취하였으므로 해당 태그들을 토큰으로 추가하였습니다.


입력 예시
```
<s>보여주면서 왜 엿보냐고 비난 하는것도 웃기지만. 훔쳐 보면서 왜 보여주냐고 하는 사람 역시 우습다..</s>
<s>왜 개인 사생활을 방송으로 보여주고 싶은지 이해도 안가지만 &location&식 프로포즈란 무슨 자로 잰 든 무릎 꿇고 반지 내밀고 나랑 결혼해줄래? 가 전부이다.</s>
<s>국민들이 암말 않고 열심히 세금 내니까 정말 별 개 그지 같은 데에 돈쓰고 쳐 자빠졌네...</s>
<s>예를 들어 누가 살해 됐고 용의자가 있는데 용의자가 자기 아니라 하면 진범이 있어야 하잖아?</s>
<s>&name& 일당은 뭐야?</s>
...
```

출력 예시 - 0 or 1 (윤리 or 비윤리)
```
1
1
1
0
0
...
```

### 평가
baseline 코드에서 제공된 평가 코드로 평가하였을때, 아래와 같이 결과가 나왔습니다.

train 과정에서 --do_eval을 argument로 전달하면 매 epoch마다 dev data에 대해 평가 결과를 보여줍니다.

demo.sh을 이용하여 결과물을 추출한뒤 평가 데이터를 이용하여 test.sh와 같이 평가할 수 있습니다.

평가함수는 evaluation(y_true, y_pred) 함수를 이용하면 되고, 입력 데이터는 아래와 같습니다.

모델을 이용하여 pred_data와 같은 형태의 데이터를 만들기 위한 방법은 demo.sh 파일을 참고하면 됩니다.

true_data
``` 
{"id": "nikluge-au-2022-train-000015", "input": "왜 청년들 일하는 데에 끼어 드는데?", "output": 1}
{"id": "nikluge-au-2022-train-000016", "input": "왜 굳이 청년 정책 관련된 일을 하는데?", "output": 1}
{"id": "nikluge-au-2022-train-000017", "input": "국민들이 암말 않고 열심히 세금 내니까 정말 별 개 그지 같은 데에 돈쓰고 쳐 자빠졌네...", "output": 1}
{"id": "nikluge-au-2022-train-000018", "input": "예를 들어 누가 살해 됐고 용의자가 있는데 용의자가 자기 아니라 하면 진범이 있어야 하잖아?", "output": 0}
{"id": "nikluge-au-2022-train-000019", "input": "&name& 일당은 뭐야?", "output": 0}
```


pred_data
```
{"id": "nikluge-au-2022-train-000015", "input": "왜 청년들 일하는 데에 끼어 드는데?", "output": 1}
{"id": "nikluge-au-2022-train-000016", "input": "왜 굳이 청년 정책 관련된 일을 하는데?", "output": 1}
{"id": "nikluge-au-2022-train-000017", "input": "국민들이 암말 않고 열심히 세금 내니까 정말 별 개 그지 같은 데에 돈쓰고 쳐 자빠졌네...", "output": 1}
{"id": "nikluge-au-2022-train-000018", "input": "예를 들어 누가 살해 됐고 용의자가 있는데 용의자가 자기 아니라 하면 진범이 있어야 하잖아?", "output": 0}
{"id": "nikluge-au-2022-train-000019", "input": "&name& 일당은 뭐야?", "output": 0}
```

베이스라인의 평가 결과는 아래와 같습니다.
| 모델                       | F1         |
| ---------------------------- | -------------- |
| xlm-roberta-base  |0.9078 |


## reference
xlm-roberta-base in huggingface (https://huggingface.co/xlm-roberta-base)

모두의말뭉치 in 국립국어원 (https://corpus.korean.go.kr/)
## Authors
- 정용빈, Teddysum, ybjeong@teddysum.ai
