---
title: "Quantization Background"
date: 2026-07-20 19:00:00 +0900
categories: [Development, Blog]
tags: [Quantization, INT8, INT4, FP16, BF16]
description: INT, FP, BF
pin: false
math: true
mermaid: false
comments: true
---

일단 bit에 대한 background를 가볍게 짚고 넘어가겠습니다.

일반적으로 INT나 FP를 사용할 때 32, 16, 8, 4bit 형식을 많이 사용합니다. 여기서 숫자는 bit 수를 의미합니다.

하나의 bit는 0 또는 1로 표현되기 때문에 4bit라면 2의 네제곱인 16가지 상태를 표현할 수 있습니다. 32bit의 경우에는 $2^{32}$, 대략 42.9억 개의 상태를 표현할 수 있습니다.

여기서 중요한 점은 bit를 해석하는 방법에 따라 같은 bit 배열도 서로 다른 수를 의미한다는 것입니다.

## INT는 무엇인가?

INT는 Integer, 즉 정수를 의미합니다.

예를 들어 8bit를 unsigned integer로 사용하면 0부터 255까지 표현할 수 있습니다. 음수 표현이 필요한 signed integer에서는 가장 왼쪽 bit가 부호를 결정하는 데 사용됩니다.

따라서 일반적인 $N$bit signed INT의 범위는 다음과 같습니다.

$$
-2^{N-1} \leq x \leq 2^{N-1}-1
$$

INT는 정수 자료형이므로 실수를 직접 표현할 수 없습니다.

## INT로 실수를 어떻게 표현하는가?

실숫값과 integer 사이에 scale을 두는 방식을 사용합니다.

| Real value | INT8 |
| ---: | ---: |
| $-0.70$ | $-70$ |
| $-0.20$ | $-20$ |
| $0.10$ | $10$ |
| $0.40$ | $40$ |
| $0.90$ | $90$ |

복원할 때는 다음과 같이 계산합니다.

$$
x \approx q \times \text{scale}
$$

예를 들어 $q=37$이고 $\text{scale}=0.01$이면 실제 값은 $0.37$입니다.

즉, INT가 소수를 표현하는 것이 아니라 INT에 scale을 적용해 실수를 근사하는 것입니다.

## 8bit와 4bit의 차이

예를 들어 weight의 범위가 $-1$부터 $1$까지라면 INT8은 이를 256단계로 나누기 때문에 매우 촘촘하게 표현할 수 있습니다.

반면 INT4는 16단계밖에 없기 때문에 훨씬 더 거칠게 표현됩니다.

그래서 일반적으로 bit 수가 줄어들수록 memory와 computation은 감소하지만 Quantization Error는 커집니다.

## Floating Point

FP에서는 Floating이 핵심입니다. 즉, 소수점이 떠다닐 수 있다는 이야기입니다.

과학적 표기법의 예를 들면 $1{,}234{,}000=1.234\times10^6$이고, $0.001234=1.234\times10^{-3}$입니다.

유효 숫자와 scale의 곱으로 표현한다고 생각하면 됩니다.

## FP32의 구조

FP32는 총 32bit이며 sign 1bit, exponent 8bit, fraction 23bit로 구성됩니다.

sign은 양수와 음수의 부호를 표현합니다. exponent는 8bit이며, 쉽게 생각하면 대략 $2^{-126}$부터 $2^{127}$까지의 scale을 표현하는 역할을 합니다. fraction은 숫자를 얼마나 세밀하게 표현할 것인지를 결정하는 부분입니다.

FP16과 FP32는 표현할 수 있는 범위에 차이가 있습니다. FP16의 최댓값은 약 $65{,}504$이므로 FP32에서는 표현할 수 있는 $100{,}000$을 표현하지 못할 수 있습니다. 이런 문제를 overflow라고 합니다.

반대로 너무 작은 값도 제대로 표현하지 못할 수 있으며, 이를 underflow라고 합니다.

FP16을 사용하면 아직 학습할 수 있는 상황인데도 값이 0으로 줄어드는 치명적인 문제가 생길 수 있습니다. Loss Scaling은 이런 문제를 해결하기 위해 사용하는 기법입니다. backpropagation 전에 loss에 일정한 scale factor를 곱해 전체 gradient를 키운 뒤, 파라미터를 업데이트하기 직전에 다시 $S$로 나누어 복원합니다.

## BF16은 또 무엇일까?

BF는 BFloat, 즉 Brain Floating Point를 의미합니다. Google Brain이 딥러닝 계산을 위해 설계한 16bit Floating Point 형식입니다.

FP32와 FP16은 단순히 정밀도만 다른 것이 아니라 표현할 수 있는 범위에도 차이가 있습니다. BF16은 FP32의 범위를 거의 그대로 유지하면서 precision을 희생해 16bit로 줄이자는 아이디어입니다.

| 자료형 | Sign | Exponent | Fraction |
| --- | ---: | ---: | ---: |
| FP32 | 1 | 8 | 23 |
| FP16 | 1 | 5 | 10 |
| BF16 | 1 | 8 | 7 |

BF16은 exponent의 범위가 FP32와 같지만 fraction의 차이 때문에 정밀도는 낮을 수밖에 없습니다.

정밀도를 크게 포기하면서 range를 확보하는 이유는 실제 Neural Network에서 activation 값의 크기가 매우 다양하기 때문입니다. Neural Network를 학습하면 forward와 backward 과정에서 숫자의 크기가 굉장히 다양하게 나타납니다.

예를 들어 어떤 gradient는 $0.0000001$일 수 있고, 어떤 activation은 $50{,}000$일 수도 있습니다.

이보다 더 크거나 작은 intermediate value가 발생할 수도 있습니다. FP16은 exponent가 5bit밖에 없어서 이러한 값에서 overflow나 underflow가 발생하기 쉽습니다.
