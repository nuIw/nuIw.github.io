---
title: Normalization (ft. Covariate Shift)
date: 2026-07-18 19:00:00 +0900
categories: [Development, Blog]
tags: []
description: BatchNorm, InstanceNorm, GroupNorm, LayerNorm, RMSNorm
pin: false
math: true
mermaid: false
comments: true
---

# Normalization (ft. Covariation Shift)

## Motivation

Gradient Accumulation에 대해서 공부를 하다가, 적용을 할 때 BatchNorm의 경우 조심해서 사용해야 한다는 것을 알게 되었습니다.

그래서 정확하게 BatchNorm에 대해서 왜 조심해야 하는지에 대해서도 탐구할 겸, 다양한 Normalization들에 대해서 깊게 파고들고 왜 그렇게 다양한 Normalization 방법을 사용하는지도 탐구하고자 합니다.

---

# Background

일반적인 CNN에서의 feature map 차원은 다음과 같습니다.

$$  
X \in R^{N \times C \times H \times W}  
$$

Normalization의 차이는 결국 **평균과 분산을 어느 dimension들을 묶어서 계산하느냐**에 있습니다.

그리고 나서 일반적으로 learnable parameter를 모델에 따라, 크기에 따라 고차원으로 학습하면서 learnable parameter를 곱하거나 더해서 학습을 진행하게 되는 구조입니다.

이런 Norm들은 기본적으로 평균과 분산을 어떻게 묶어서 계산하는지, 어디서 계산하는지의 차이에 있습니다.

- BatchNorm : N, H, W 방향으로 normalize
- InstanceNorm : H, W 방향으로 normalize
- GroupNorm : 일부 C + H + W 방향으로 normalize
- LayerNorm, RMSNorm : 보통 feature dimension 방향으로 normalize

---



# Normalization Variation



## BatchNormalization

각 channel마다 normalization 하는 것입니다.

하나의 batch에서, N개의 sample에 대해서 전체적인 normalization을 하기 때문에 BatchNormalization이라는 이름이 붙은 것 같습니다.

원래 들어오는 입력의 차원이 어떤지에 따라서 BatchNorm의 코드 상 적용이 달라지는데 여기서는 BatchNorm2d로 설명하겠습니다.

$$  
\mu_c = 

\frac{1}{NHW}  
\sum_{n,h,w}  
X_{n,c,h,w}  
$$

$$  
\sigma_c^2

\frac{1}{NHW}  
\sum_{n,h,w}  
(X_{n,c,h,w}-\mu_c)^2  
$$

즉, N, H, W를 평균내고 C는 유지하는 것을 의미합니다.

대신 BatchNorm은 Training에서의 batch를 설정한 batch 값을 사용하지만, inference에서 batch size가 달라질 수 있습니다.

Training 때 방식을 그대로 사용하면 통계가 불안정하기 때문입니다.

그래서 training에서 평균과 분산을 저장하고 EMA를 통해 현재 eval에서 사용되는 batch의 평균과 분산으로 업데이트하지 않습니다.

이 부분이 LayerNorm이나 GroupNorm과의 큰 차이입니다.

어쨌든 Batch를 기준으로 다른 축을 합치는 통계를 내는 것이기 때문에 batch 수가 적다면 noise가 커질 수 있다는 점을 주의해야 합니다.

그리고 이런 점에 있어서 지금은 BatchNorm이 많이 사용되지 않는다는 추측을 해봤습니다.

Batch size가 작을수록 추출된 평균과 분산이 전체 데이터셋을 대표할 수 없는 통계가 되기 때문에 이런 점에 있어서 안정적으로 적용할 수 있는 다른 Normalization 방법들이 지금은 많이 사용되고 있는 것 같습니다.

---



## InstanceNormalization

동일하게 tensor N, C, H, W가 각 축을 담당하고 있다고 할 때 H, W만 normalize하는 것을 말합니다.

BatchNorm과 비교하자면 BatchNorm은 하나의 channel에 대해서 단 하나의 평균과 분산이 통계값으로 나오지만, InstanceNorm은 동일한 channel 안에서도 하나의 sample마다 평균과 분산을 내기 때문에 다른 데이터 sample에게 영향을 받지 않는다는 점이 주요한 장점입니다.

그리고 한 채널 안에서의 각 sample의 통계값이 독립적이라고 할 수 있기 때문에 한 channel 안에서 sample마다의 분산이 크다고 가정하면 경향성 자체는 비슷할 수 있지만 값 자체가 큰 것들을 BatchNorm에서는 알 수 없었다고 한다면 InstanceNorm에서는 알 수 있습니다.

예를 들어 서로 다른 이미지에서 밝기가 달라서 RGB에 대한 채널 값이 어떤 곳은 매우 작고, 어떤 곳은 매우 크다고 한다면 분산이 크게 나오겠지만 InstanceNorm을 사용한다면 이런 것들로 인해 통계값이 불안정해지는 것을 막을 수 있다고 볼 수 있습니다.

그래서 이런 밝기나 대조가 매우 중요한 값으로 작용했던 Style Transfer와 같은 이미지 생성 모델에서는 InstanceNorm이 많이 사용되었습니다.

---



## GroupNormalization

InstanceNorm은 샘플끼리도 따로 보고 채널끼리도 따로 통계를 냈습니다.

그래서 저장해야 할 정보가 매우 많습니다.

그래도 서로 다른 샘플에 의해서 영향을 받지 않는다는 장점을 가지고 있기 때문에 샘플은 따로 보되, 채널끼리는 group으로 통합해서 normalize하자는 아이디어가 GroupNormalization입니다.

$$  
\mu_{n,g} = 

\frac{1}{(C/G)HW}  
\sum_{c \in g,h,w}  
X_{n,c,h,w}  
$$

즉, N dimension은 normalization 과정에서 고려하지 않습니다.

주의해야 하는 점이 있다면 Channel 수만큼 group을 잘게 쪼개게 된다면 InstanceNorm과 다를 것이 없다는 점이 있습니다.

---



# LayerNormalization

LayerNorm인 이유는 단 1개의 데이터 샘플 내에서 해당 layer 전체의 뉴런들을 norm하는 방식이기 때문입니다.

그런데 이런 norm 설명은 사실 옳지는 않고, 실제로 LayerNorm을 코드에서 사용하는 것을 기준으로 했을 때는 `nn.LayerNorm()`의 괄호 안에 어떤 값을 주느냐에 따라서 normalization axis가 달라질 수 있습니다.

일반적으로는 마지막 axis에 대해서 normalization이 되고 Transformer에서는 token마다 독립적으로 hidden dimension C를 normalize하기 때문에 그 해당 Layer의 모든 dimension의 activation을 norm한다고 해서 LayerNorm인 것 같습니다.

여기까지 정리하자면,

- BatchNorm은 channel을 고정해두고 배치별 N, H, W를 normalize
- InstanceNorm은 N과 Channel을 하나씩 고정해두고 H, W만 normalize
- GroupNorm은 N을 고정해두고 Channel을 group으로 묶고 H, W를 normalize
- LayerNorm은 일반적으로 feature dimension 방향으로 normalize

---



# RMSNormalization

원래 Transformer 기반의 모델들에서 LayerNorm을 사용해왔는데 그렇다면 지금은 왜 RMSNorm을 선택해서 사용하는 것일까요?

일단 LayerNorm에서 언급했던 것처럼 일반적으로 feature dimension 방향으로 구한 평균과 분산을 가지고 다음과 같이 norm을 합니다.

$$  
LayerNorm(x) = 

\gamma  
\frac{x-\mu}  
{\sqrt{\sigma^2+\epsilon}}  

- 

\beta  
$$

이때 LayerNorm의 역할은 x에서 평균을 빼서 centering을 하게 되고, 분산에 epsilon을 더한 값에 root를 씌워 scaling을 하는 두 가지 역할이 있습니다.

$$  
RMS(x) = 

\sqrt{  
\frac{1}{d}  
\sum_i x_i^2  

- 

\epsilon  
}  
$$

$$  
RMSNorm(x) = 

\gamma  
\frac{x}{RMS(x)}  
$$

반면 RMSNorm에서 분모에 들어가는 값은 입력의 각 원소를 제곱해서 평균을 낸 후에 root를 씌운 값입니다.

왜 이렇게 사용하냐면, Normalization 성공의 핵심적인 역할은 평균을 맞추는 것이 아니라 활성화 값의 크기를 일정하게 유지하는 것이 밝혀져서 RMS는 평균을 구하는 연산을 완전히 생략하고 오직 데이터의 전반적인 크기만 측정하기 위해 사용하는 것입니다.

애당초에 LayerNorm에서는 magnitude가 너무 커지거나 작아지는 것을 막아서 optimization을 안정화시키는 것이 목적이었다면, 왜 한쪽에 hidden state 값들을 zero-mean으로 만드는 것보다 scale을 제어하는 것이 더 중요하다고 하는 것입니다.

값을 0으로 끌어당긴다는 것뿐만 아니라 그 scale 자체를 일정하게 유지하는 것이 중요하다는 것입니다.

만약 0으로 끌어당긴다면 [0, 1, 2]라는 값과 [100, 101, 102]라는 게 전부 다 [0, 1, 2]라는 값으로 처리되는 것이고 아예 동일한 정보 취급이 되기 때문에 엄밀히 다른 값임에도 불구하고 동일하게 여겨질 수 있다는 것입니다.

차라리 상대적인 표현량 자체를 어느 정도 유지하면서 scale을 제어하는 것이 더 도움이 될 수 있다는 관점입니다.

---



# RMSNorm의 제일 중요한 성질: Scale Invariance

hidden state를 상수 a배했다고 합시다.

$$  
x' = ax  
$$

RMS는

$$  
RMS(ax)

=

|a|RMS(x)  
$$

이므로 $a>0$일 때

$$  
\frac{ax}{RMS(ax)}

=

\frac{ax}{aRMS(x)}

=

\frac{x}{RMS(x)}  
$$

입니다.

즉,

$$  
RMSNorm(ax)  
\approx  
RMSNorm(x)  
$$

입니다.

입력 hidden state의 전체 scale이 달라져도 normalized representation은 거의 변하지 않습니다.

이것이 optimizer 관점에서 매우 유용합니다.

LayerNorm에서도 물론 scale invariance가 적용이 되지만, 유용한 scale invariance에 대한 normalization 성질을 유지하면서 re-centering에 대한 연산을 줄였다는 contribution도 있어 널리 사용되고 있는 것 같습니다.

---



# BatchNorm이 Internal Covariance Shift를 해결해준다는 것은 사실 거짓이다?

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (ICML 2015)** 논문에서는 원래 training하면서 앞 layer의 parameter가 계속 변하기 때문에 동일한 데이터가 들어와도 동일한 layer에서 들어오는 activation의 distribution이 계속 변하게 되고, 이게 계속 한쪽으로 치우치게 된다는 것이 Internal Covariate Shift이고 이것을 해결해주기 때문에 BatchNorm이 성능이 좋다고 주장했습니다.

하지만 **How Does Batch Normalization Help Optimization? (NeurIPS 2018)** 논문에서는 일부러 BatchNorm 적용 이후에 covariance shift를 적용시켜서 distribution을 흔들어버리면 BatchNorm의 장점이 없어져야겠지만 비슷하게 잘 학습한다는 것이 밝혀진 것입니다.

그리고 오히려 BN이 distribution의 variance를 그렇게 안정화시키지도 않는다는 것이 밝혀졌고 BatchNorm이 그 당시에 성능을 좋게 만들어줬던 것은 loss landscape를 smooth하게 해줬기 때문이라고 밝혀졌습니다.

이를 통해 더 큰 learning rate를 사용할 수 있다는 것이 BatchNorm의 중요한 장점이라는 것으로 설명되었습니다.