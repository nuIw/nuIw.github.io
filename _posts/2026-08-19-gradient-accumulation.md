---
title: GitHub 기술 블로그 시작
date: 2026-07-17 19:00:00 +0900
categories: [Development, Blog]
tags: [github-pages, jekyll, chirpy]
description: GitHub Pages와 Chirpy를 사용해 기술 블로그를 구축한 과정을 기록합니다.
pin: false
math: true
mermaid: false
comments: true
---

# Normalization (ft. Covariate Shift)

## Motivation

Gradient Accumulation에 대해서 공부하다가 실제 학습에 적용할 때 **BatchNorm을 사용할 경우 주의해야 한다**는 점을 알게 되었습니다.

그래서 정확하게 BatchNorm을 왜 조심해서 사용해야 하는지 알아보는 것을 시작으로, 다양한 Normalization 방법에 대해 깊게 파고들고, 왜 서로 다른 Normalization 방법들이 사용되는지 정리해보고자 합니다.

---

# Background

일반적인 CNN의 feature map을 다음과 같이 두겠습니다.

$$
X \in \mathbb{R}^{N \times C \times H \times W}
$$

각 dimension은 다음을 의미합니다.

- $N$: Batch size
- $C$: Channel
- $H$: Height
- $W$: Width

Normalization 방법들의 차이는 결국

> **평균과 분산을 어느 dimension들을 묶어서 계산하느냐**

에 있습니다.

대부분의 Normalization은 기본적으로 다음과 같은 형태를 가집니다.

$$

\hat{x}

\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

그리고 일반적으로 learnable parameter $\gamma$, $\beta$를 이용해

$$
y = \gamma \hat{x} + \beta
$$

와 같은 affine transformation을 추가합니다.

물론 RMSNorm처럼 centering을 하지 않거나 bias $\beta$를 사용하지 않는 예외도 있습니다.

CNN의 $N \times C \times H \times W$ 관점에서 각 방법을 먼저 요약하면 다음과 같습니다.

- **BatchNorm**: $N, H, W$ 방향으로 statistics 계산
- **InstanceNorm**: $H, W$ 방향으로 statistics 계산
- **GroupNorm**: 일부 $C$와 $H, W$ 방향으로 statistics 계산
- **LayerNorm**: 지정한 feature dimension 방향으로 statistics 계산
- **RMSNorm**: LayerNorm과 비슷한 feature dimension에 적용하지만 mean centering 없이 RMS를 이용해 scale만 정규화

---

# Normalization Variations



## Batch Normalization

BatchNorm은 CNN의 경우 **각 channel마다 독립적으로 normalization**을 수행합니다.

Feature map이

$$
X \in \mathbb{R}^{N \times C \times H \times W}
$$

일 때 특정 channel $c$에 대한 평균은

$$

\mu_c

\frac{1}{NHW}
\sum_{n,h,w}
X_{n,c,h,w}
$$

이고, 분산은

$$

\sigma_c^2

\frac{1}{NHW}
\sum_{n,h,w}
\left(
X_{n,c,h,w}-\mu_c
\right)^2
$$

입니다.

즉,

$$
\boxed{
\text{BatchNorm: } N,H,W \text{를 평균내고 } C \text{는 유지}
}
$$

한다고 볼 수 있습니다.

특정 channel 하나를 고정한 뒤, 해당 channel에 속하는 **batch의 모든 sample과 모든 spatial position**을 이용해 하나의 평균과 분산을 구하는 것입니다.

예를 들어

$$
X \in \mathbb{R}^{32 \times 64 \times 28 \times 28}
$$

이라면 channel $c$ 하나에 대해

$$
32 \times 28 \times 28
$$

개의 값을 사용하여 하나의 $\mu_c$, $\sigma_c^2$를 계산합니다.

따라서 총 64개의 channel에 대해 각각 평균과 분산이 존재합니다.

---



### BatchNorm이라는 이름이 붙은 이유

BatchNorm의 중요한 특징은 **한 sample의 normalization 결과가 다른 sample의 activation에도 영향을 받는다는 점**입니다.

예를 들어 sample A를 normalize할 때도 같은 batch에 포함된 B, C, D 등의 값이 평균과 분산 계산에 함께 들어갑니다.

즉,

$$
x_A
$$

의 normalized value가

$$
x_B,;x_C,;x_D,\ldots
$$

의 영향을 받을 수 있습니다.

이 점이 이후 설명할 LayerNorm, GroupNorm, InstanceNorm과의 중요한 차이입니다.

---



## Training과 Inference에서의 BatchNorm

BatchNorm은 training과 inference에서 동작 방식이 다릅니다.

Training에서는 현재 mini-batch로부터

$$
\mu_{\text{batch}},
\qquad
\sigma_{\text{batch}}^2
$$

를 계산합니다.

하지만 inference에서는 batch size가 training 때와 다를 수 있고, 심지어 batch size가 1일 수도 있습니다.

이 상태에서 training 때와 동일하게 현재 batch statistics를 사용하면 통계적으로 매우 불안정할 수 있습니다.

따라서 BatchNorm은 training 과정에서 `running_mean`, `running_variance`를 유지합니다.

예를 들어 running mean은 대략 다음과 같이 업데이트됩니다.

$$
\mu_{\text{running}}
\leftarrow
(1-m)\mu_{\text{running}}
+
m\mu_{\text{batch}}
$$

따라서

- **Training**: 현재 mini-batch statistics 사용
- **Inference**: training 중 축적된 running statistics 사용

이라는 차이가 발생합니다.

이것은 LayerNorm이나 GroupNorm과의 큰 차이입니다.

---



## Batch size가 작을 때 BatchNorm이 문제가 되는 이유

BatchNorm은 batch statistics에 의존하기 때문에 batch size가 작아지면 추정되는 평균과 분산의 noise가 커질 수 있습니다.

예를 들어 batch size가 충분히 크다면

$$
\mu_{\text{batch}}
\approx
\mu_{\text{population}}
$$

라고 기대할 수 있지만, batch size가 매우 작다면 mini-batch에서 추정한 평균과 분산이 전체 data distribution을 제대로 대표하지 못할 수 있습니다.

특히 Object Detection이나 Segmentation처럼 image resolution이 커서 GPU 하나에

$$
\text{batch size}=1\sim4
$$

정도밖에 넣지 못하는 상황에서는 이 문제가 더 중요해집니다.

이러한 이유로 batch statistics에 의존하지 않는 GroupNorm 등의 방법이 사용되기도 합니다.

---



## Gradient Accumulation과 BatchNorm

Gradient Accumulation을 사용할 때도 이 부분을 주의해야 합니다.

예를 들어

```text
micro batch size = 4
gradient accumulation steps = 8
```

이라면 optimizer 관점의 effective batch size는

$$
4 \times 8 = 32
$$

입니다.

하지만 BatchNorm 관점에서는 다릅니다.

각 forward pass에서 BatchNorm이 보는 batch size는 여전히

$$
4
$$

입니다.

즉 Gradient Accumulation으로 gradient를 8번 누적한다고 해서 BatchNorm이 32개의 sample을 한꺼번에 이용해 statistics를 계산하는 것은 아닙니다.

$$
\boxed{
\text{Effective Batch Size} = 32
\quad\neq\quad
\text{BatchNorm Statistics Batch Size} = 4
}
$$

따라서 Gradient Accumulation으로 effective batch size를 크게 만들었다고 하더라도, BatchNorm은 여전히 **micro batch size에 영향을 받습니다.**

이것이 Gradient Accumulation과 BatchNorm을 함께 사용할 때 주의해야 하는 중요한 이유입니다.

---



# Instance Normalization

InstanceNorm은 동일하게

$$
X \in \mathbb{R}^{N \times C \times H \times W}
$$

라는 tensor가 있다고 할 때, 각 sample과 channel을 고정하고 $H,W$에 대해서만 normalization을 수행합니다.

평균은

$$

\mu_{n,c}

\frac{1}{HW}
\sum_{h,w}
X_{n,c,h,w}
$$

이고 분산은

$$

\sigma_{n,c}^2

\frac{1}{HW}
\sum_{h,w}
\left(
X_{n,c,h,w}-\mu_{n,c}
\right)^2
$$

입니다.

즉

$$
\boxed{
\text{InstanceNorm: } H,W \text{를 normalize하고 } N,C \text{는 유지}
}
$$

합니다.

---



## BatchNorm과 InstanceNorm의 차이

BatchNorm에서는 특정 channel $c$에 대해 batch 전체가 하나의 평균과 분산을 공유합니다.

$$

\mu_c

\operatorname{mean}_{N,H,W}(X)
$$

반면 InstanceNorm은 sample별로 statistics를 따로 계산합니다.

$$

\mu_{n,c}

\operatorname{mean}_{H,W}(X)
$$

예를 들어 동일한 channel $c$라도

```text
Image 1, Channel c → mean / variance 1
Image 2, Channel c → mean / variance 2
Image 3, Channel c → mean / variance 3
```

처럼 서로 다른 statistics를 사용합니다.

따라서 InstanceNorm에서는 다른 sample의 영향을 받지 않습니다.

---



## InstanceNorm과 Style

이러한 성질 때문에 InstanceNorm은 image의 global appearance statistics를 제거하는 데 유용할 수 있습니다.

예를 들어 두 feature map이

$$
[10,11,10,12,\ldots]
$$

와

$$
[100,101,100,102,\ldots]
$$

처럼 전체적인 intensity 수준은 다르지만 spatial pattern이 비슷하다고 생각해볼 수 있습니다.

InstanceNorm은 각 sample과 channel 내부에서 평균과 분산을 따로 계산하기 때문에 absolute intensity나 contrast의 차이를 상당 부분 제거할 수 있습니다.

따라서

- brightness
- contrast
- style statistics
- texture statistics

등에 덜 민감한 representation을 만들 수 있습니다.

이러한 특성 때문에 InstanceNorm은 특히 **Style Transfer**와 초기 image generation architecture에서 널리 사용되었습니다.

---



# Group Normalization

InstanceNorm은 sample마다, 그리고 channel마다 각각 statistics를 계산합니다.

반면 GroupNorm은 channel들을 여러 개의 group으로 묶은 뒤, **각 sample 안에서 group 단위로 normalization**합니다.

Channel 수가 $C$이고 group 수를 $G$라고 하면, group 하나에는

$$
\frac{C}{G}
$$

개의 channel이 들어갑니다.

예를 들어

$$
C=64,\qquad G=8
$$

이면 group 하나에는

$$
\frac{64}{8}=8
$$

개의 channel이 포함됩니다.

각 sample $n$과 group $g$에 대한 평균은

$$

\mu_{n,g}

\frac{1}{(C/G)HW}
\sum_{c \in g,h,w}
X_{n,c,h,w}
$$

입니다.

분산 역시 같은 group 내부의 channel과 spatial position 전체에서 계산합니다.

즉,

$$
\boxed{
\text{GroupNorm: } N\text{은 유지하고 group 내부 }C,H,W\text{를 normalize}
}
$$

합니다.

중요한 것은 **batch dimension $N$을 statistics 계산에 사용하지 않는다**는 것입니다.

따라서 GroupNorm은 BatchNorm과 달리 batch size에 직접적으로 의존하지 않습니다.

---



## GroupNorm과 InstanceNorm의 관계

GroupNorm에서 group 수를 channel 수와 같게 설정하면

$$
G=C
$$

가 됩니다.

그러면 한 group에는 정확히 channel 하나만 들어갑니다.

따라서 normalization dimension은

$$
H,W
$$

만 남게 됩니다.

즉,

$$
\boxed{
\text{GroupNorm}(G=C)
\approx
\text{InstanceNorm}
}
$$

으로 볼 수 있습니다.

반대로

$$
G=1
$$

이면 하나의 sample에서 모든 channel과 spatial dimension인

$$
C,H,W
$$

를 함께 normalization하게 됩니다.

따라서 GroupNorm은 group 수에 따라 InstanceNorm과 LayerNorm-like normalization 사이를 연결하는 형태로 이해할 수 있습니다.

---



# Layer Normalization

LayerNorm이라는 이름만 보고 "항상 layer 전체 activation을 normalize한다"고 이해하면 정확하지 않습니다.

실제로 LayerNorm은 **지정된** `normalized_shape`**에 해당하는 마지막 dimension들**을 normalize합니다.

일반적인 fully-connected feature가

$$
X \in \mathbb{R}^{N \times D}
$$

라면 한 sample의 feature vector

$$
x_n=[x_1,x_2,\ldots,x_D]
$$

에 대해

$$

\mu_n

\frac{1}{D}
\sum_{d=1}^{D}X_{n,d}
$$

를 계산합니다.

즉 sample끼리는 statistics를 공유하지 않습니다.

---



## CNN에서의 LayerNorm

CNN feature map에

```python
nn.LayerNorm([C, H, W])
```

를 적용하면 sample 하나의

$$
C\times H\times W
$$

전체를 normalization합니다.

평균은

$$

\mu_n

\frac{1}{CHW}
\sum_{c,h,w}
X_{n,c,h,w}
$$

가 됩니다.

따라서 이 경우에는

$$
\boxed{
\text{LayerNorm: } C,H,W\text{를 normalize하고 }N\text{은 유지}
}
$$

합니다.

---



## Transformer에서의 LayerNorm

Transformer에서는 tensor가 일반적으로

$$
X \in \mathbb{R}^{N \times L \times D}
$$

형태입니다.

- $N$: Batch
- $L$: Sequence length
- $D$: Hidden dimension

LayerNorm은 일반적으로 마지막 hidden dimension $D$에 적용됩니다.

즉 각 token마다

$$

\mu_{n,l}

\frac{1}{D}
\sum_{d=1}^{D}
X_{n,l,d}
$$

를 계산합니다.

따라서

$$
\boxed{
\text{Transformer LayerNorm: 각 token마다 }D\text{를 normalize}
}
$$

합니다.

다른 batch sample뿐만 아니라 다른 token의 statistics도 사용하지 않습니다.

---



# 여기까지 정리

CNN의

$$
X \in \mathbb{R}^{N\times C\times H\times W}
$$

관점에서 생각하면 다음과 같습니다.

### BatchNorm

Channel을 고정하고

$$
N,H,W
$$

를 normalization합니다.

$$
\boxed{
\mu_c=\operatorname{mean}_{N,H,W}(X)
}
$$

### InstanceNorm

Sample과 Channel을 하나씩 고정하고

$$
H,W
$$

를 normalization합니다.

$$
\boxed{
\mu_{n,c}=\operatorname{mean}_{H,W}(X)
}
$$

### GroupNorm

Sample을 고정하고 Channel을 group으로 묶은 뒤

$$
C_{\text{group}},H,W
$$

를 normalization합니다.

$$

\boxed{
\mu_{n,g}

\operatorname{mean}_{C/G,H,W}(X)
}
$$

### LayerNorm

일반적으로 지정된 feature dimension을 normalization합니다.

Transformer의

$$
X\in\mathbb{R}^{N\times L\times D}
$$

에서는 각 token마다

$$
D
$$

를 normalization합니다.

---



# RMS Normalization

Transformer 기반 모델에서는 오랫동안 LayerNorm이 널리 사용되어 왔습니다.

그렇다면 최근 LLaMA, Mistral 등의 모델은 왜 RMSNorm을 사용하는 것일까요?

이를 이해하려면 먼저 LayerNorm의 연산을 다시 살펴볼 필요가 있습니다.

LayerNorm은

$$

\operatorname{LayerNorm}(x)

\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

로 표현할 수 있습니다.

여기에서 LayerNorm은 크게 두 가지 작업을 수행합니다.

### 1. Re-centering

$$
x \rightarrow x-\mu
$$

즉 feature들의 평균을 0으로 이동시킵니다.

### 2. Re-scaling

$$
x-\mu
\rightarrow
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

즉 activation의 scale을 조절합니다.

---



# RMSNorm

RMSNorm은 LayerNorm에서 **re-centering을 제거하고 re-scaling에 집중한 형태**라고 볼 수 있습니다.

먼저 RMS는 다음과 같습니다.

$$

\operatorname{RMS}(x)

\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}x_i^2
+
\epsilon
}
$$

그리고 RMSNorm은

$$

\operatorname{RMSNorm}(x)

\gamma
\frac{x}
{\operatorname{RMS}(x)}
$$

로 정의됩니다.

즉 LayerNorm처럼

$$
x-\mu
$$

를 계산하지 않습니다.

또한 일반적인 RMSNorm 구현에서는 LayerNorm의 bias에 해당하는 $\beta$도 사용하지 않습니다.

따라서

$$

\boxed{
\text{LayerNorm}

\text{Centering}
+
\text{Scaling}
}
$$

인 반면,

$$

\boxed{
\text{RMSNorm}

\text{Scaling}
}
$$

이라고 볼 수 있습니다.

---



# 왜 Mean을 제거하지 않아도 되는가?

Normalization의 중요한 역할 중 하나는 activation의 magnitude가 지나치게 커지거나 작아지는 것을 막아 optimization을 안정화하는 것입니다.

예를 들어

$$
x=[1,2,3]
$$

이라는 hidden state가 layer를 거치면서

$$
x'=[100,200,300]
$$

이 되었다고 생각해보겠습니다.

두 vector는

$$
x'=100x
$$

이므로 방향은 동일하지만 magnitude가 100배 차이납니다.

RMSNorm은

$$
\frac{x}
{\sqrt{\operatorname{mean}(x^2)}}
$$

를 통해 이러한 전체 scale 변화의 영향을 제거합니다.

즉 Transformer의 optimization에서 중요한 것이 반드시

$$
\mu=0
$$

을 만드는 것이 아니라, hidden state의 magnitude를 안정적인 범위로 유지하는 것이라면 mean subtraction은 생략할 수 있습니다.

이것이 RMSNorm의 핵심 아이디어입니다.

---



# LayerNorm과 RMSNorm이 정보를 다루는 차이

예를 들어

$$
x_1=[0,1,2]
$$

와

$$
x_2=[100,101,102]
$$

를 생각해봅시다.

두 vector에 LayerNorm을 적용하면 각각 평균을 먼저 제거합니다.

첫 번째 vector는

$$
\mu_1=1
$$

이므로

$$

x_1-\mu_1

[-1,0,1]
$$

이 됩니다.

두 번째 vector는

$$
\mu_2=101
$$

이므로

$$

x_2-\mu_2

[-1,0,1]
$$

이 됩니다.

즉 두 vector는 LayerNorm 이후 동일한 normalized pattern을 갖게 됩니다.

LayerNorm은 translation에 대해 invariant한 성질을 가지기 때문입니다.

반면 RMSNorm은 mean을 제거하지 않기 때문에 이러한 global offset 정보를 직접 제거하지 않습니다.

대신 activation vector 전체의 scale을 제어합니다.

이 점에서 RMSNorm은 LayerNorm보다 단순한 normalization이라고 볼 수 있습니다.

---



# RMSNorm의 중요한 성질: Scale Invariance

RMSNorm의 중요한 성질 중 하나는 **Scale Invariance**입니다.

hidden state를 양의 상수 $a$배했다고 합시다.

$$
x'=ax
$$

RMS의 정의에 의해

$$

\operatorname{RMS}(ax)

|a|\operatorname{RMS}(x)
$$

입니다.

$a>0$이라면

$$

\frac{ax}
{\operatorname{RMS}(ax)}

\frac{ax}

{a\operatorname{RMS}(x)}

\frac{x}
{\operatorname{RMS}(x)}
$$

가 됩니다.

따라서

$$
\boxed{
\operatorname{RMSNorm}(ax)
\approx
\operatorname{RMSNorm}(x)
}
$$

입니다.

$\epsilon$이 존재하기 때문에 실제 구현에서는 엄밀하게 완전히 동일하지 않을 수 있지만, 충분한 magnitude에서는 거의 동일하게 동작합니다.

즉 입력 hidden state의 전체 scale이 달라져도 normalized representation은 거의 변하지 않습니다.

이러한 성질은 optimization 관점에서 유용합니다.

---



## LayerNorm도 Scale Invariance를 가지지 않는가?

그렇습니다.

Scale Invariance 자체가 RMSNorm만의 독점적인 성질은 아닙니다.

LayerNorm 역시 양의 scale 변화에 대해 기본적으로 invariant합니다.

따라서 RMSNorm의 핵심적인 contribution을

> "LayerNorm에는 없던 Scale Invariance를 만들었다."

라고 설명하는 것은 정확하지 않습니다.

보다 정확하게는

> **LayerNorm의 유용한 scale normalization 성질은 유지하면서 re-centering 연산을 제거해 더 단순한 normalization을 만든 것**

으로 이해할 수 있습니다.

---



# 왜 현대 LLM에서 RMSNorm을 많이 사용하는가?

LayerNorm이

$$
\text{Centering}+\text{Scaling}
$$

을 모두 수행했다면 RMSNorm은

$$
\text{Scaling}
$$

만 수행합니다.

그럼에도 Transformer에서 경쟁력 있는 성능과 안정적인 optimization을 제공할 수 있습니다.

따라서 RMSNorm은

- mean 계산 제거
- mean subtraction 제거
- 일반적으로 bias $\beta$ 제거
- scale normalization 유지

라는 단순화가 가능합니다.

현대 LLM에서는 normalization이 수많은 Transformer block에서 반복적으로 수행되므로 이러한 단순화가 시스템적으로도 유용할 수 있습니다.

다만 RMSNorm 하나만으로 전체 LLM의 계산량이 크게 감소한다고 보는 것은 과도한 해석입니다.

Transformer의 대부분의 연산량은 Attention과 MLP의 matrix multiplication에서 발생하기 때문입니다.

---



# BatchNorm이 Internal Covariate Shift를 해결한다는 것은 사실일까?

BatchNorm의 역사에서 흥미로운 부분 중 하나는 **BatchNorm이 왜 잘 작동하는가에 대한 설명 자체가 시간이 지나며 바뀌었다는 점**입니다.

---



## 초기 설명: Internal Covariate Shift

Ioffe와 Szegedy의 2015년 논문

> **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**

에서는 BatchNorm의 효과를 **Internal Covariate Shift, ICS**를 줄이는 것으로 설명했습니다.

Deep Network가

$$
x
\rightarrow
f_1
\rightarrow
h_1
\rightarrow
f_2
\rightarrow
h_2
\rightarrow
\cdots
$$

와 같이 구성되어 있다고 생각해봅시다.

Layer $l$의 입력은

$$
h_{l-1}
$$

입니다.

그런데 training 과정에서 앞 layer의 parameter가 계속 update됩니다.

$$
W_{l-1}^{(t)}
\rightarrow
W_{l-1}^{(t+1)}
$$

그러면 동일한 데이터가 입력되더라도 activation이 달라집니다.

$$
h_{l-1}^{(t)}
\neq
h_{l-1}^{(t+1)}
$$

따라서 뒤쪽 layer의 관점에서는 자신의 input distribution이 지속적으로 변하게 됩니다.

$$
P_t(h_{l-1})
\rightarrow
P_{t+1}(h_{l-1})
$$

원 논문에서는 이러한 현상을 **Internal Covariate Shift**라고 설명했습니다.

따라서 BatchNorm으로 activation을

$$

\hat{x}

\frac{x-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

와 같이 normalization하면 mini-batch 수준에서

$$
E[\hat{x}]
\approx0
$$

$$
\operatorname{Var}(\hat{x})
\approx1
$$

이 되므로 layer input distribution이 안정화되고, 그 결과 optimization이 쉬워진다는 설명이었습니다.

즉 초기 설명은

$$
\boxed{
\text{BatchNorm}
\rightarrow
\text{ICS 감소}
\rightarrow
\text{Optimization 안정화}
}
$$

였습니다.

---



# 2018년: Internal Covariate Shift 설명에 대한 반박

2018년 Santurkar et al.은 NeurIPS에

> **How Does Batch Normalization Help Optimization?**

이라는 논문을 발표했습니다.

이 논문은 다음 질문을 직접적으로 검증합니다.

> BatchNorm이 실제로 잘 동작하는 핵심적인 이유가 정말 Internal Covariate Shift를 감소시키기 때문인가?

저자들의 결론은 부정적이었습니다.

---



# Noisy BatchNorm Experiment

가장 직관적인 실험은 BatchNorm 이후 activation distribution을 **의도적으로 다시 흔들어버리는 것**이었습니다.

기본 구조가

```text
Linear / Convolution
        ↓
    BatchNorm
        ↓
    Activation
```

이라면, 실험에서는 BatchNorm 이후 추가적인 noise를 주어 activation의 distribution을 변화시킵니다.

개념적으로

$$

x'

a_t x_{\mathrm{BN}}
+
b_t
$$

와 같이 step마다 변하는 random scale과 offset을 추가한다고 생각할 수 있습니다.

따라서 BatchNorm이 activation을 안정화하더라도 다시

$$
P_t(x')
\neq
P_{t+1}(x')
$$

가 되도록 만든 것입니다.

만약 BatchNorm의 성능 향상이 정말로

$$
\text{ICS 감소}
$$

때문이었다면 이러한 distributional instability를 다시 추가한 **Noisy BatchNorm의 성능은 크게 떨어져야 합니다.**

그러나 실험 결과는 그렇지 않았습니다.

Noisy BatchNorm에서도 BatchNorm이 제공하던 optimization상의 이점이 상당 부분 유지되었습니다.

즉,

$$
\boxed{
\text{ICS가 크게 존재하는 상황에서도 BatchNorm은 잘 학습될 수 있다.}
}
$$

는 결과가 나온 것입니다.

이는 적어도

$$
\boxed{
\text{ICS 감소가 BatchNorm 성능 향상의 필수 조건은 아니다.}
}
$$

라는 강한 증거가 됩니다.

---



# BatchNorm이 Distribution을 그렇게 안정화시키지도 않았다

Santurkar et al.은 BatchNorm을 사용한 network와 사용하지 않은 network의 activation distribution이 training 동안 얼마나 변화하는지도 분석했습니다.

기존의 ICS 가설이 강하게 맞다면

$$
\Delta P_{\mathrm{BN}}
\ll
\Delta P_{\mathrm{NonBN}}
$$

와 같은 결과를 기대할 수 있습니다.

하지만 실제로 측정한 distributional stability의 차이는 BatchNorm이 제공하는 optimization 성능 차이를 충분히 설명하지 못했습니다.

즉,

$$
\boxed{
\text{Distribution Stability}
\not\Rightarrow
\text{Optimization Improvement}
}
$$

라고 볼 수 있는 결과였습니다.

---



# Gradient 관점에서의 Internal Covariate Shift

단순히 activation의 평균과 분산 변화만으로 ICS를 정의하는 것이 충분하지 않을 수도 있습니다.

Optimization의 관점에서 더 중요한 것은 앞 layer가 update되었을 때 현재 layer가 보게 되는 **gradient가 얼마나 변하는가**일 수 있습니다.

현재 layer $i$의 gradient를

$$

G_{t,i}

\nabla_{W_i}L
$$

이라고 하겠습니다.

앞 layer를 update한 뒤 동일한 $W_i$에 대해 다시 계산한 gradient를

$$
G'_{t,i}
$$

라고 하면

$$
\left|
G_{t,i}-G'_{t,i}
\right|_2
$$

를 통해 앞 layer의 update가 현재 layer의 optimization problem을 얼마나 변화시켰는지 측정할 수 있습니다.

gradient direction의 변화 역시 cosine similarity

$$
\cos(G_{t,i},G'_{t,i})
$$

등으로 측정할 수 있습니다.

하지만 이러한 관점에서도 BatchNorm이 항상 더 작은 변화량을 보여주는 것은 아니었습니다.

즉 BatchNorm을 사용한 network가 비슷하거나 더 큰 shift를 보이는 경우에도 훨씬 잘 학습될 수 있었습니다.

---



# 그렇다면 BatchNorm은 왜 잘 작동하는가?

Santurkar et al.이 제시한 중요한 설명은 다음과 같습니다.

$$
\boxed{
\text{BatchNorm makes the optimization landscape smoother.}
}
$$

즉 BatchNorm의 주요 효과를 activation distribution의 고정이 아니라 **optimization geometry의 개선**에서 찾았습니다.

---



# Smooth Optimization Landscape

Loss function을

$$
L(W)
$$

라고 하겠습니다.

현재 parameter $W$에서 gradient는

$$
g=\nabla L(W)
$$

입니다.

SGD에서는

$$

W'

W-\eta g
$$

와 같이 parameter를 update합니다.

만약 loss landscape가 매우 sharp하다면 parameter를 조금만 움직여도 gradient가 크게 달라질 수 있습니다.

$$
\nabla L(W+\Delta W)
\not\approx
\nabla L(W)
$$

이 경우 현재의 gradient direction은 조금 뒤의 optimization landscape를 제대로 예측하지 못합니다.

따라서 큰 learning rate를 사용하면 쉽게 overshooting이나 unstable training이 발생할 수 있습니다.

반대로 landscape가 smooth하다면

$$
\nabla L(W+\Delta W)
\approx
\nabla L(W)
$$

가 어느 정도 유지됩니다.

현재 계산한 gradient가 더 넓은 parameter neighborhood에서도 유효한 descent direction으로 작동할 수 있는 것입니다.

---



# Gradient Lipschitzness

이러한 smoothness는 gradient의 Lipschitz continuity로 표현할 수 있습니다.

$$

\left|
\nabla L(W_1)

\nabla L(W_2)
\right|
\le
\beta
\left|
W_1-W_2
\right|
$$

여기에서 $\beta$가 작을수록 parameter를 조금 변화시켰을 때 gradient가 천천히 변화합니다.

즉 loss landscape가 더 smooth하다고 볼 수 있습니다.

Santurkar et al.은 BatchNorm이 이러한 optimization smoothness를 개선하며, gradient를 더 predictable하게 만들어 optimization을 쉽게 만든다고 설명했습니다.

따라서 BatchNorm에서는 상대적으로 큰 learning rate도 안정적으로 사용할 수 있습니다.

---



# Normalization을 어떻게 이해해야 하는가?

Normalization의 역사를 보면 초기에는 다음과 같이 생각했습니다.

$$
\text{Normalization}
\rightarrow
\text{Distribution Stabilization}
\rightarrow
\text{Training Stabilization}
$$

하지만 이후 연구를 통해 activation distribution 자체를 일정하게 만드는 것만으로 Normalization의 효과를 설명하기 어렵다는 것이 드러났습니다.

현대적인 관점에서는 Normalization의 역할을 보다 넓게

- activation scale control
- gradient scale control
- optimization conditioning
- parameterization
- optimization landscape smoothing
- training stability

등과 연결해서 바라보는 것이 더 적절합니다.

RMSNorm의 성공도 이러한 관점과 잘 연결됩니다.

RMSNorm은 activation의 mean을 0으로 만드는 centering조차 수행하지 않지만 현대 Transformer에서 매우 잘 작동합니다.

즉 Normalization에서 중요한 것은 반드시

$$
\mu=0
$$

을 만드는 것이 아니라,

$$
|x|
$$

와 같은 representation scale을 안정적으로 제어하여 optimization을 쉽게 만드는 것일 수 있다는 점을 보여줍니다.

---



# 정리

Normalization 방법들의 가장 직접적인 차이는 **어떤 dimension을 사용하여 statistics를 계산하는가**입니다.

CNN feature map

$$
X\in\mathbb{R}^{N\times C\times H\times W}
$$

을 기준으로 보면

$$
\boxed{
\text{BatchNorm}
:
\operatorname{Norm}_{N,H,W}
}
$$

$$
\boxed{
\text{InstanceNorm}
:
\operatorname{Norm}_{H,W}
}
$$

$$
\boxed{
\text{GroupNorm}
:
\operatorname{Norm}_{C/G,H,W}
}
$$

입니다.

Transformer에서는

$$
X\in\mathbb{R}^{N\times L\times D}
$$

에 대해 LayerNorm과 RMSNorm이 일반적으로 hidden dimension $D$를 대상으로 동작합니다.

LayerNorm은

$$

\boxed{
\operatorname{LayerNorm}(x)

\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+
\beta
}
$$

로 centering과 scaling을 모두 수행하는 반면,

RMSNorm은

$$

\boxed{
\operatorname{RMSNorm}(x)

\gamma
\frac{x}
{\sqrt{
\frac{1}{d}
\sum_i x_i^2+\epsilon
}}
}
$$

와 같이 scaling에 집중합니다.

그리고 BatchNorm의 역사에서 중요하게 기억할 부분은

$$
\boxed{
\text{BatchNorm의 성능 향상}
\neq
\text{단순한 Internal Covariate Shift 감소}
}
$$

라는 점입니다.

2015년에는 BatchNorm의 효과를 ICS 감소로 설명했지만, 2018년 Santurkar et al.의 연구를 통해 **ICS 감소가 BatchNorm의 optimization improvement를 설명하기에는 부족하다**는 강한 실험적 근거가 제시되었습니다.

그 이후에는 BatchNorm과 Normalization을 activation distribution의 고정이라는 관점만이 아니라 **scale control과 optimization geometry의 개선이라는 관점에서 이해하는 것이 더 적절합니다.**

---



# References

1. Sergey Ioffe and Christian Szegedy.
  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.**
   ICML, 2015.
2. Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry.
  **How Does Batch Normalization Help Optimization?**
   NeurIPS, 2018.
3. Biao Zhang and Rico Sennrich.
  **Root Mean Square Layer Normalization.**
   NeurIPS, 2019.
4. Yuxin Wu and Kaiming He.
  **Group Normalization.**
   ECCV, 2018.
5. Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky.
  **Instance Normalization: The Missing Ingredient for Fast Stylization.**
   2016.

