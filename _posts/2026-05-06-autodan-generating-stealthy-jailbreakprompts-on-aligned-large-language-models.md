---
title: 'AUTODAN: GENERATING STEALTHY JAILBREAKPROMPTS ON ALIGNED LARGE LANGUAGE MODELS'
date: '2026-05-06T17:02:33+09:00'
categories:
- Paper Review
source_url: https://nulw.tistory.com/entry/Paper-Review-AUTODAN-GENERATING-STEALTHY-JAILBREAKPROMPTS-ON-ALIGNED-LARGE-LANGUAGE-MODELS
---

![Unsplash / Photo byEmiliano Bar](/assets/img/tistory/23/001.jpg)

Unsplash / Photo byEmiliano Bar

AUTODAN: GENERATING STEALTHY JAILBREAKPROMPTS ON ALIGNED LARGE LANGUAGE MODELS를 리뷰합니다.

## <strong>Previous Studies</strong>

DAN은 Do Anything Now라는 뜻으로, 사용자가 LLM에게 어떠한 제한도 무시하고 무엇이든 할 수 있는 역할을 수행하도록 요청하는 jailbreak 공격 방식입니다.

jailbreak란 prompt를 조작해서 align된 LLM이 제공해서는 안 되는 악의적인 출력을 유도하는 공격 방법을 의미합니다.

이런 jailbreak 공격에서 scalability issue, stealthiness problems가 무슨 의미일까요?

scalability issue (DAN, Do-Anything-Now)<br>1&#46; 기존 DAN 방식을 살펴보면 LLM에게 공격을 하기 위해서 사람이 직접 공격을 해야만 했습니다.<br>2&#46; 사람이 직접 prompt를 사용해서 공격을 하는 형태이고 이것이 취약점이 됩니다. 또한 모델마다 취약점이 다를 수 있기 때문에 모델마다 다르게 prompt를 구성해야합니다. 또한 LLM이 버전이 바뀔 때마다 그 부분을 보완하게 될텐데 그렇다면 기존의 prompt는 폐기되고 새로운 promp를 구성해야합니다.<br>3&#46; 대신 사람이 직접 prompt를 구성하기 때문에 자연어의 문법이나 유창성은 보장할 수 있습니다.

stealthiness issue (은밀성 문제)<br>1&#46; 사람이 직접 prompt를 구성해야해서 비용이 늘어나는 것이 문제라면 prompt를 자동으로 생성해주는 모델을 사용하면 됩니다.<br><br>2&#46; 대표적으로 GCG(Greedy Coordinate Gradient) 방식이 있습니다.<br>GCG는 LLM의 내부 구조를 알 수 있는 모델의 gradient 정보를 활용해서 jailbreak가 가능한 prompt에 들어갈 토큰을 찾아내는 과정을 안내합니다.<br>GCG의 함수를 통해서 해로운 응답을 생성하도록 유도하는 토큰의 후보를 생성합니다.

이게 잘 생성하면 좋겠지만 실질적으로 의미 있는 내용이 아닌 것입니다. 실질적으로 DAN 기반의 prompt는 자연어를 기반으로 하지만 비자연어 기반의 prompt가 LLM safety를 우회하여 jailbreak를 이끌어낼 수 있는 해답이었다는 것입니다.

그래서 DAN은 사람이 만들어내는 prompt로서 자연어 기반의 prompt, GCG는 AI가 만들어내지만 비자연어 기반의 prompt로 구성이 됩니다.

## <strong>기존연구의 limitation &#38; AutoDAN의 접근방식</strong>

GCG 기반의 prompt 생성은 자동으로 생성이 가능하고 비자연어적 요소로 인해서 jailbreak 공격이 가능하다는 것을 밝혔습니다. 대신에 결국 ‘비자연어적’ 요소로 인한 것이기 때문에 perplexity와 같은 문장의 예측 가능성을 판단할 수 있는 지표를 방어 시스템에 넣게 되면 쉽게 방어가 가능합니다.

DAN 기반의 prompt의 경우는 사람이 직접 만들어야 하는 prompt인데, 그렇기 때문에 비용도 많이 들게 되고 모델에 따라서, 그리고 각 모델의 버전에 따라서도 사용할 수 있는 prompt가 제한적입니다.

그래서 해당 논문에서는 확장성 측면에서는 자동으로 생성이 가능한 GCG 기반의 prompt 생성이 필요하지만 쉽게 걸러낼 수 있는 GCG의 특성 상 DAN의 특성을 가져야하는 것입니다.

따라서 DAN의 특성을 가지면서 자동으로 prompt 생성이 가능한 AutoDAN을 제안합니다.

### AutoDAN

- 계층적 유전 알고리즘
- 문장과 단어 수준 모두에 대해 서로 다른 교차 정책
- fine-grained space에서의 탐색 능력을 강화하는 momentum word scoring 도입
- local minimum에 빠지는 것을 방지하고 전역 최적해를 지속적으로 탐색 가능
- 사람이 만든 자연스러운 문장(DAN)에서부터 시작, 의미를 유지하면서 인공지능이 조금씩 수정해 최적의 공격 문장을 찾아냄

## <strong>기존 연구들의 한계</strong>

### Human-Aligned LLMs

Human-aligned LLM을 위한 데이터는 인간이 생성한 지시 사항에서 얻거나 심지어 다른 강력한 LLM으로부터 합성될 수도 있습니다.

예를 들어, PromptSource나 SuperNaturalInstruction 같은 방법들은 이전의 NLP 벤치마크를 자연어 지시사항으로 변환하는 반면, self-instruction 방법은 ChatGPT와 같은 모델의 인컨텍스트 학습 능력을 활용하여 새로운 지시사항을 생성합니다.

정렬을 위한 훈련 방법론 또한 지도 미세 조정(SFT)에서 인간 피드백 기반 강화 학습(RLHF)으로 진화해 왔습니다.

좋은 성능을 보이기는 하지만 Jailbreak 문제가 있습니다.

### Jailbreak Attack against LLMs

기존의 수동 제작된 탈옥 프롬프트들을 먼저 수집 및 분류한 뒤, ChatGPT를 대상으로 실증적 연구를 수행했습니다.

다른 연구에서는 접두사 주입(prefix injection) 및 거절 억제(refusal suppression)와 같은 기존 탈옥 현상을 모델의 능력과 안전 목표 사이의 경쟁 때문으로 보았습니다.

그리고 탐욕적(greedy) 탐색과 그래디언트 기반 탐색 기법을 결합하여 적대적 접미사를 자동으로 생성하는 GCG를 제안했습니다.

동시에 LLM으로부터 탈옥 프롬프트를 생성할 가능성, 수동으로 제작된 다단계 프롬프트를 통한 탈옥, 그리고 블랙박스 시나리오에서의 토큰 수준 탈옥의 효과를 조사하는 연구들도 진행되고 있습니다.

AutoDAN은 모델 훈련 과정 없이 은밀한 탈옥 프롬프트를 자동으로 생성하는 데 초점을 맞추고 있다는 점에서 이들과 차별화됩니다.

## <strong>Methods</strong>

#### Threat model

Jailbreak는 LLM alignment와 밀접한 관련이 있습니다.

공격자가 던진 악의적인 질문에 대해 모델이 거절하는 대신에 공격자가 원하는 답변을 강제합니다.

![](/assets/img/tistory/23/002.png)

악의적인 질문들의 집합 Q, 악의적인 질문들의 집합과 결합해 jailbreak를 이끌어내는 prompt J ⇒ 이 Q와 J를 결합하여 최종적으로 jailbreak를 통해 원하는 질문을 얻어낼 수 있는 입력 집합 T ⇒ T를 공격당하는 LLM 모델 M에 집어넣었을 때 응답하는 응답 집합 R

최종적으로 LLM이 답하는 응답 집합 R이 악의적인 질문 집합인 Q와 밀접하게 연관된 답변들이 되도록 하는 것

각각의 악의적인 질문에 대해 구체적인 타겟 응답을 설정하는 것은 비현실적입니다. 각각 악의적인 질문에 따른 정확한 답변을 짚어내는 것이 어려울 뿐만 아니라 일반화 성능을 떨어뜨릴 수도 있기 때문입니다.

그래서 일반적으로 타겟 응답을 ‘물론 답변이 가능합니다.’로 시작하게끔 만드는 방식으로 진행합니다.

즉, 타겟 응답을 일관된 시작 부분을 가진 텍스트로 고정함으로써 최적화에 사용되는 공격 손실 함수(attack loss function) 형태로 표현할 수 있게 됩니다.

![](/assets/img/tistory/23/003.png)

최종적으로 토큰 t로 이루어진 입력 T &#61; &#60;J, Q&#62;가 주어졌을 때 jailbreak prompts J를 최적화합니다. 이 J가 입력 토큰들에게 영향을 주게 만들어서 결과적으로 다음에 내뱉을 단어가 ‘물론 답변이 가능합니다’와 같은 단어로 시작하도록 유도하는 확률을 높이는 것입니다.

#### Genetic Algorithms

다윈의 자연 선택(natural selection)에서 영감을 받은 진화 알고리즘의 한 종류

전체적인 흐름은 population initialization → fitness evaluation → crossover → mutation → evolution → optimization 입니다.

예를 들면

1. 처음에 무리 안에 <strong>목 길이가 제각각인 기린 20마리</strong>가 존재 → <strong>Population initialization</strong>
2. 목이 긴 기린일수록 먹이를 더 잘 먹어 생존 확률이 높음 → <strong>Fitness evaluation</strong>
3. 목이 긴 기린들끼리 짝짓기해 자식을 낳음 → <strong>Crossover(교배)</strong>
4. 일부 자식에게 예상치 못한 돌연변이로 목이 더 길어지거나 짧아짐 → <strong>Mutation(돌연변이)</strong>

이 과정을 반복하다가 fitness evaluation, 즉 적합도 판정에서 threshold를 넘기면 이 반복을 종료

⇒ 수백 세대 후 무리 전체의 목이 길어짐 → <strong>evolution, optimization</strong>

#### Population Initialization

initialization은 매우 중요한 역할을 합니다. 알고리즘의 수렴 속도와 최종해의 품질에 상당한 영향을 끼치기 때문입니다. 그래서 두 가지 사항을 염두에 두어야(bear in mind) 합니다.

수동으로 제작된 프로토타입 jailbreak prompt는 특정 시나리오에서 잘 작동함이 입증된 것들이기 때문에 이로부터 크게 벗어나지 않는 것이 필수적입니다.

다양성을 확보해서 initialization하는 게 solution space의 더 넓은 탐색을 촉진하므로 중요합니다.

⇒ 즉, 기존 수동 프롬프트의 기본 특징을 유지하면서 다양화도 보장해야합니다.이를 위해서 LLM을 agent로 활용합니다. 이는 mutation LLM을 의미합니다. mutation 단계에서 랜덤하게 단어를 바꾸면 문장이 어색해질 수 있기 때문에 opensource LLM을 활용해서 “문장의 핵심은 유지하되 다르게 표현할 것”이라는 요구사항을 유지할 수 있도록 합니다.

이렇게 하면 원래 문장의 고유한 논리적 흐름과 의미를 보존하는 동시에, 단어 선택과 문장 구조에서 다양성을 도입할 수 있게 됩니다.

## FITNESS EVALUATION

Threat model 부분에서 언급했던 이 확률을 최대화하는 것이 목적입니다. 그래서 유전 알고리즘에서 fitness를 평가하기 위해서 likelihood를 계산하는 함수를 직접 사용할 수 있습니다.

loss는 낮을수록 좋으니까 확률이 높을수록 fitness가 높은 것으로 처리한다는 것을 의미합니다.

![](/assets/img/tistory/23/004.png)



![](/assets/img/tistory/23/005.png)

악의적인 질문들의 집합 Q, 악의적인 질문들의 집합과 결합해 jailbreak를 이끌어내는 prompt J ⇒ 이 Q와 J를 결합하여 최종적으로 jailbreak를 통해 원하는 질문을 얻어낼 수 있는 입력 집합 T ⇒ T를 공격당하는 LLM 모델 M에 집어넣었을 때 응답하는 응답 집합 R

최종적으로 LLM이 답하는 응답 집합 R이 악의적인 질문 집합인 Q와 밀접하게 연관된 답변들이 되도록 하는 것

악의적인 질문에 대해서 그대로 악의적인 답변을 할 수 있도록 만드는 것이 목적이라는 것입니다.

각각의 악의적인 질문에 대해 구체적인 타겟 응답을 설정하는 것은 비현실적입니다. 각각 악의적인 질문에 따른 정확한 답변을 짚어내는 것이 어려울 뿐만 아니라 일반화 성능을 떨어뜨릴 수도 있기 때문입니다.

그래서 일반적으로 타겟 응답을 ‘물론 답변이 가능합니다.’로 시작하게끔 만드는 방식으로 진행합니다. 즉, 타겟 응답을 일관된 시작 부분을 가진 텍스트로 고정함으로써 최적화에 사용되는 공격 손실 함수(attack loss function) 형태로 표현할 수 있게 됩니다.

최종적으로 토큰 t로 이루어진 입력 T &#61; &#60;J, Q&#62;가 주어졌을 때 jailbreak prompts J를 최적화합니다. 이 J가 입력 토큰들에게 영향을 주게 만들어서 결과적으로 다음에 내뱉을 단어가 ‘물론 답변이 가능합니다’와 같은 단어로 시작하도록 유도하는 확률을 높이는 것입니다.

## GENETIC POLICIES (Crossover &#38; Mutation)

### AUTODAN-GA

Genetic policies의 핵심은 crossover와 mutation을 설계하는 것입니다.

“multi-point crossover”

### AUTODAN-H(Hierarchical)GA

텍스트 데이터의 특징은 계층적 구조를 가진다는 것입니다.

텍스트의 단락이 일반적으로 문장 간 논리적인 흐름이라는 것이 있고 각 문장 내부에서는 단어의 선택이 의미를 결정합니다.

따라서 crossover를 단락 수준에서의 crossover로만 제한한다면 본질적으로 탐색 범위를 단일 계층 수준에 가두게 되어서 방대한 탐색 공간을 무시하게 됩니다.

그래서 자연어의 계층 구조를 활용하기 위해서 단락 측면과 문장 측면을 모두 고려하는 것입니다. ⇒ paragraph-level population, sentence-level population

알고리즘의 반복마다 단어 선택과 같은 문장 수준 공간을 탐색하고 나서 단락 수준 공간에서 통합하게 됩니다.

또 HGA의 핵심은 paragraph level과 sentence level을 따로 바라본다는 점입니다.

#### Paragraph-level population

예를 들어 N&#61;50이라면, 같은 Jp를 Algorithm 5에 50번 넣어 50개의 서로 다른 변형을 생성합니다. Algorithm 5의 system prompt가 "You are a helpful and creative assistant who writes well."이고 user 지시가 "길이 변경 없이 수정해줘"이므로, 매번 LLM의 sampling 확률에 의해 조금씩 다른 표현이 나옵니다.

![](/assets/img/tistory/23/006.png)

algorithm 5를 통해서 개체군을 초기화하고, 이 수식을 통해서 개체 군 내의 모든 개체의 fitness evaluation을 평가합니다.

평가를 바탕으로 cross over와 mutation을 수행할 개체들을 선택합니다.

이 때 개체는 prompt 하나하나를 의미하고, N개의 prompt가 있다고 가정합니다.

elite 비율 alpha가 주어지면 N개의 prompt에 대한 alpha만큼의 프롬프트는 수정 없이 다음 반복으로 직접 넘어가게 하고 이는 적합도 점수가 지속적으로 하락(loss function 값이 하락, 즉 개선되는 것)되도록 보장합니다.

elite prompt를 제외한 나머지 프롬프트에서 fitness score를 기반으로 softmax 확률을 적용해 점수를 매깁니다. 이 점수를 기반으로 나머지 프롬프트의 빈자리를 crossover를 적용해 넘겨줍니다.

이를 통해서 좋은 prompt는 그대로 살아남기도하고 자식도 낳게 되면서 우수한 특징을 계속 퍼뜨리게 됩니다.

> <strong>요약 예시 (N&#61;100, 엘리트 20(alpha &#61; 0.2)개인 경우)</strong>

1. 성적 순위 1&#126;20등을 <strong>그대로 복사</strong>해서 다음 세대 명단에 올립니다. (20개 확보)
2. 이제 80개의 자손을 더 만들어야 합니다.
3. 1&#126;100등 전체를 모아놓고 점수대로 확률을 줍니다. (당연히 1&#126;20등이 당첨 확률이 높겠죠?)
4. 당첨된 부모들을 섞어서 <strong>새로운 자식 80개</strong>를 만듭니다. ◦ 이때 1등과 2등이 부모로 뽑혀서 섞일 수도 있고, 1등과 50등이 섞일 수도 있습니다.
5. 복사본 20개 &#43; 새 자식 80개 &#61; 총 100개의 다음 세대 완성!

#### Sentence-level: momentum word scoring

문장 수준에서 탐색 공간은 주로 단어 선택에 집중됩니다.

이 점수를 사용해서 각 prompt의 점수를 매긴 뒤에 해당 prompt에 포함된 모든 단어에 적합도 점수를 매길 수 있습니다.

그리고 momentum scoring이라는 것을 사용합니다. 평균 점수와 EMA를 활용합니다. 바로 진짜 유의미한 단어를 가려내고 우연에 의해서 높은 점수를 얻은 단어가 아님을 확인하기 위해서입니다.

n번째 iteration에서는 매우 높은 점수를 받았지만 그 외에 iteration에서는 낮은 점수를 받았을 수도 있기 때문입니다. 그래서 평균 점수를 사용합니다. 그리고 EMA를 활용하는 이유는 점수의 기복이 심한 단어인지 아닌지를 판단하기 위해서입니다.

이를 통해서 꾸준하게 안정적으로 높은 점수를 보이는 단어를 파악할 수 있게 됩니다.

![](/assets/img/tistory/23/007.png)

algorithm 8을 보면, 일반적인 단어와 고유 명사를 우선 필터링 한 뒤에 단어 점수 dictionary를 얻게 됩니다. 일반적인 단어와 고유 명사를 나누는 이유는 be 동사나 전치사 등과 같은 의미가 없는 단어를 거르기 위함입니다. 그리고 이 dictionary에서 top-k 점수를 가진 k개의 단어들을 선택해서 다른 프롬프트에 있는 유의어(near-synonym)를 이 단어들로 교체합니다.

## Termination CRITERIA

유전 알고리즘을 다음과 같이 비유합니다.

1. 처음에 무리 안에 <strong>목 길이가 제각각인 기린 20마리</strong>가 존재 → <strong>Population initialization</strong>
2. 목이 긴 기린일수록 먹이를 더 잘 먹어 생존 확률이 높음 → <strong>Fitness evaluation</strong>
3. 목이 긴 기린들끼리 짝짓기해 자식을 낳음 → <strong>Crossover</strong>
4. 일부 자식에게 예상치 못한 돌연변이로 목이 더 길어지거나 짧아짐 → <strong>Mutation</strong>
5. 수백 세대 후 무리 전체의 목이 길어짐 → evolution, optimization

이 때 이 알고리즘의 효과성과 효율성을 보장하기 위해서 ‘최대 반복 횟수’와 ‘거절 신호 테스트’를 종료 조건으로 사용합니다.

- 알고리즘이 설정한 최대 반복 횟수를 모두 소모하거나
- LLM 응답 r 중의 상위 k개의 단어 내에서 거절 단어 리스트에 포함된 키워드가 하나도 감지되지 않으면 알고리즘은 작동을 중단
- 그 중에서 적합도 점수가 제일 높은 최적의 jailbreak prompt를 선택

---

## <strong>Experiments</strong>

### Expreimental Setup

#### Datasets

AdvBench Harmful Behaviors 사용

⇒ 비속어, 사실적인 묘사, 위협 행위, 허위 정보, 차별, 불법적인 제안을 포함하는 520개의 요청으로 구성

#### Baselines

비교 대상: GCG(Greedy Coordinate Gradient) → iteration num: 1000

#### Metrics

1&#46; Attack success rate(ASR): 키워드 기반의 공격 성공률<br>⇒ LLM의 응답에 ‘죄송합니다(I am sorry)’, ‘책임감 있는 AI로서(As a responsible AI,)’와 같은 문구가 포함되어있는지로 판단합니다. 즉, 이 단어를 내뱉지 않도록 하면 AutoDAN은 공격을 성공한 것입니다.

2&#46; GPT recheck attack success rate (REcheck)⇒ 근본적으로는 ASR로만 측정하면 PPL 측면에서 무너져도 공격은 성공할 수 있기 때문에 GPT-2를 기반으로 한 PPL을 측정합니다.<br>⇒ LLM이 악의적인 질문에 대해 직접적으로 거절하지 않을 수도 있습니다. (ASR에서는 잡지 못하는 케이스) 하지만 주제를 벗어난 내용을 전달하는 경우가 있습니다. 또는 악의적인 질문에 답변하면서도 해당 요청이 잠재적으로 불법이거나 비윤리적임을 사용자에게 경고하는 등의 조언을 붙이면서 ASR 지표에서는 잘 공격을 막아냈다고 판단할 수도 있습니다.

즉, ASR도 높으면서 PPL은 낮은 문장을 잘 생성할 수 있어야 좋은 점수를 받을 수 있습니다.

![](/assets/img/tistory/23/008.png)

### transferability

AutoDAN에 의해서 생성된 prompt가 다른 모델에 얼마나 성공적으로 jailbreak를 발생시킬 수 있는지를 의미합니다.

AutoDAN이 GCG 기반 모델보다 더 나은 transferability를 보여주는데 논문에서 추측하기로는 GCG 기반 모델은 open-source를 대상으로 whitebox 모델들의 gradient로 학습하기 때문에 open-source 모델들에 overfitting 되었을 것이라고 추측합니다.

## Limitation

- computational cost / 그래도 GCG보다는 효율적
- LLama2 모델의 경우는 Genetic Algorithm이 vanishing gradient 문제처럼 제대로 작동하지 않음
  - LLama2-7b-chat 모델의 경우는 사용성을 위해 강력하게 system prompt가 적용되어있기 때문

## Conclusion

- stealthness는 유지하면서 prompt를 자동으로 생성할 수 있음
- Genetic Algorithm을 통해서 이산 데이터에 특화된 정교한 모듈 개발

## Additional Limitation

- GPT-4와 같이 논문이 게재될 2024년 당시의 최신 모델에는 AutoDAN이 좋지 않은 성능을 보입니다.
- PPL이 낮다고 해서 완벽하게 stealthiness가 보장된다는 것은 비약적입니다.
  - 문법적으로 매끄럽다고 해서 보안 시스템이나 의미론적 의도 파악 필터를 완벽하게 속일 수 있는 실질적 은밀성을 의미하는 것은 아닙니다.
  - PPL은 fluency를 측정하는 도구이지 악의적인 의도를 측정하지는 못합니다.
  - 그래서 ASR이 있으니까 괜찮은 거 아닌가?
    - ASR은 엄밀히 따지면 I am sorry나 As a reponsible AI로 시작해서 답변을 거절하느냐 거절하지 않느냐를 판단하는 것이지 악의적인 의도를 파악한다고는 볼 수 없습니다.
    - 기존 지표(PPL &#43; ASR)가 허술한 이유
      1. <strong>PPL(Perplexity)의 맹점:</strong> PPL은 문장의 자연스러움을 측정하지만, 모델에 따라 값이 크게 요동치며 조작이 너무 쉽습니다. 예를 들어, 공격 프롬프트 뒤에 의미 없는 흔한 단어(예: "the", "and")를 반복적으로 이어 붙이기만 해도 PPL 수치가 낮아져 자연스러운 문장으로 위장될 수 있습니다.
      2. <strong>키워드 매칭 기반 ASR의 한계:</strong> "I cannot", "I'm sorry" 같은 거절 키워드가 응답에 포함되어 있는지만 검사하다 보니 오탐지(False Positive/Negative)가 잦습니다. 모델이 "해당 정보는 알려줄 수 없지만, 대신 합법적인 가이드를 제공합니다"라고 긍정적인 톤으로 거절해도 실패로 간주하거나, 반대로 악의적인 답변을 교묘하게 내놓으면서 거절 키워드를 살짝 섞으면 방어에 성공한 것으로 오해할 수 있습니다.
  - 최근 AI safety 분야의 논문들은 어떤 평가 지표를 사용하고 있는가
    - 평가용으로 강력하게 tuning되어있는 LLM(GPT-4, LLama-Guard)을 심판으로 사용합니다