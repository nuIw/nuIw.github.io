---
title: Learning rate of LoRA variation method
date: 2026-07-22 19:00:00 +0900
categories: [Development, Blog]
tags: []
description: LoRA variation method들의 laerning rate에 따른 성능 차이
pin: false
math: true
mermaid: false
comments: true
---

최근 VLM의 KV cache를 직접 양자화하면서 계속 궁금했던 점이 있습니다.

LLM 쪽에서는 KV cache를 INT4 정도로 줄여도 성능이 크게 무너지지 않는다는 결과를 자주 볼 수 있습니다. 그렇다면 visual token이 대부분을 차지하는 VLM에서도 비슷할까요? 그리고 Key와 Value, Visual token과 Text token을 정말 같은 방식으로 압축해도 괜찮을까요?

처음에는 단순히 FP16, INT8, INT4, INT2를 차례로 적용하고 accuracy만 비교하면 대략적인 답을 얻을 수 있을 것이라고 생각했습니다.

그런데 실제로 실험을 만들다 보니 bit 수보다 더 먼저 확인해야 하는 문제가 많았습니다.

과거에 이미 만들어진 KV cache만 압축하고 있는가

질문 토큰까지 실수로 같이 양자화하고 있지는 않은가

어떤 조건에서 실제로 양자화된 token 수가 0은 아닌가

INT2라고 적혀 있어도 실제 저장 비용이 정말 2 bit/value인가

accuracy가 그대로일 때도 모델 내부의 분포는 이미 달라지고 있지 않은가

이번 글에서는 이 실험을 진행하면서 확인한 것들을 정리해보려고 합니다.

이 글에서는 제가 진행 중인 전체 연구 주제는 다루지 않습니다. 여기서는 하나의 독립적인 실험으로서, VLM의 historical KV cache를 서로 다른 precision으로 압축했을 때 어떤 변화가 생기는지만 정리합니다.

## 1. 먼저 무엇을 확인하고 싶었나

실험에서 답하고 싶었던 질문은 크게 네 가지였습니다.

어느 bit까지는 비교적 안전한가

어느 구간부터 손상이 급격하게 커지는가

Key와 Value의 민감도가 다른가

이후 실험에서 사용할 수 있는 보수적인 bit와 공격적인 bit는 무엇인가

여기서 중요한 것은 처음부터 어떤 압축 정책을 잘 만드는 것이 아니었습니다.

예를 들어 중요한 token만 FP16으로 보호하거나, layer마다 다른 bit를 주거나, attention score에 따라 precision을 바꾸면 성능은 더 좋아질 수 있습니다. 하지만 그렇게 시작하면 결과가 bit-width 때문인지 정책 때문인지 알기 어려워집니다.

그래서 이번 실험에서는 가능한 한 정책을 제거했습니다.

같은 historical cache에 대해 FP16, INT8, INT4, INT2를 적용하고, Visual/Text와 K/V 축만 분리해서 먼저 기본적인 sensitivity landscape를 보는 것이 목표였습니다.

## 2. VLM의 KV cache는 어떻게 생기는가

실험은 LLaVA-OneVision 7B를 중심으로 진행했습니다.

32개의 video frame이 들어오면 각 frame은 vision encoder를 통과한 뒤 visual token으로 변환됩니다. 이 설정에서는 frame 하나가 최종적으로 약 196개의 visual token이 되고, 32 frame을 사용하면 visual block은 약 6,273 token이 됩니다.

반면 historical text는 매우 짧았습니다.

대략적인 prompt 구조를 단순화하면 다음과 같습니다.

[system / role text]
[visual tokens ........................................]
[instruction + question + choices]

LLaVA 설정에서 visual block 앞에 존재하는 historical text는 14 token뿐이었습니다.

즉 전체 history의 거의 대부분이 visual token입니다.

이후 각 transformer layer에서 hidden state로부터 Q, K, V가 만들어집니다.

hidden state
    │
    ├─ q_proj → Q
    ├─ k_proj → K → RoPE
    └─ v_proj → V

K, V → KV cache에 저장

LLaVA-OneVision 7B의 경우 28개의 attention head를 사용하지만 KV head는 4개인 GQA 구조입니다.

따라서 cache에는 layer마다 대략 다음 형태의 K/V가 저장됩니다.

[B, H_kv, T, D]
= [1, 4, T, 128]

여기서 제가 양자화한 것은 model weight도 아니고 hidden state도 아닙니다.

이미 과거를 처리한 뒤 각 layer에 저장되어 있는 K/V cache tensor만 바꿨습니다.

이 구분이 생각보다 중요했습니다.

## 3. Accuracy만으로는 거의 아무것도 보이지 않았다

평가 데이터는 50개 sample을 사용했습니다.

이 경우 accuracy는 한 sample이 바뀔 때마다 0.02씩 움직입니다.

문제는 quantization이 output을 조금씩 흔들고 있어도 최종 선택지가 그대로라면 accuracy에는 아무 변화가 없다는 점입니다.

하지만 모델의 확신도와 전체 분포는 꽤 달라졌습니다.

그래서 이 실험에서는 accuracy와 함께 NLL, 그리고 FP16 output distribution을 기준으로 한 KL divergence를 사용했습니다.

$$
D_{KL}(P_{FP16} \parallel P_{Q})
= \sum_i P_{FP16}(i)
\log \frac{P_{FP16}(i)}{P_Q(i)}
$$

여기서 중요한 것은 수식 자체보다 해석입니다.

FP16일 때의 판단 분포를 기준으로 두고, quantization 이후 그 분포가 얼마나 달라졌는지 연속적인 값으로 측정한다고 보면 됩니다.

accuracy가 그대로인 영역에서도 quantization damage를 볼 수 있기 때문에 이후 분석에는 KL이 훨씬 유용했습니다.

## 4. Key와 Value는 생각보다 많이 달랐다

이번 실험에서 가장 깔끔하게 나온 결과는 K와 V의 차이였습니다.

LLaVA에서 INT2로 visual token을 양자화했을 때

Visual K mean KL = 0.01422
Visual V mean KL = 0.00102

약 13.9배 차이가 났습니다.

Qwen2.5-VL에서도 같은 방향이었고 약 6.5배 차이가 났습니다.

이 비교가 특히 의미 있었던 이유는 K와 V 조건에서 양자화된 token 비율이 동일했기 때문입니다.

즉, 같은 token, 같은 bit, 같은 group size, 같은 layer 범위, 같은 quantized fraction을 사용하고 K냐 V냐만 바꿨습니다.

그래서 modality 비교보다 상대적으로 해석이 훨씬 깔끔합니다.
구조적으로 생각해볼 만한 이유도 있습니다.
이 모델의 cache에 저장되는 K는 RoPE가 적용된 이후의 값입니다.

반면 V에는 같은 위치 회전이 없습니다.
또한 GQA 구조에서는 저장된 소수의 KV head가 여러 query head에서 재사용됩니다. 따라서 한 KV head에 들어간 quantization error가 여러 attention head에 공유될 수 있습니다.

물론 이것만으로 "K가 민감한 이유는 RoPE 때문이다"라고 결론 내릴 수는 없습니다.
하지만 적어도 K와 V를 무조건 같은 precision으로 다루는 것이 당연한 선택은 아니다라는 점은 꽤 분명하게 보였습니다.

## 15. INT4와 INT2가 생각보다 KL divergence 값은 크게 차이가 나지만 QA accuracy는 무너지지 않는다.

사실 제일 이해하고 싶은 부분입니다. 기존의 연구들도 INT2까지는 성능 방어가 잘되고, 모델에 따라서 INT2에서 무너지더라도 Quantizer를 다르게 가져감으로 인해서 성능을 방어하는 모습을 자주 보였습니다.

연구하는 방향이 이 실험을 통해서 윤곽이 드러나기는 했으나, 점점 예상했던 범주에서 제일 벗어나는 것은 INT4, INT2가 FP16 대비 output probability의 distribution의 차이가 많이 났지만 성능 하락은 발생하지 않았다는 것입니다.
추후 연구에서는 이 부분에서 더 자세하게 다룰 필요가 있을 것 같습니다.