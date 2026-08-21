---
title: Effective Batch Size와 Gradient Accumulation
date: 2026-07-18 19:00:00 +0900
categories: [Development, Blog]
tags: []
description: Effective Batch Size, Gradient Accumulation
pin: false
math: true
mermaid: false
comments: true
---

논문 reproduce도 해보고 개인적인 연구를 할 때 학교에서 사용하는 GPU는 VRAM이 24GB이고 연구실에서 가끔씩 자리가 난다면 빌려서 사용할 수 있는 VRAM 48GB GPU가 있습니다. 그래서 연구를 하거나 reproduce할 때 항상 한계에 부딪히고는 합니다. 학습할 때 VRAM이 가장 중요하게 여겨지는 상황은 일단 모델 가중치가 업로드 되고 나서 이후 공간에 batch size를 얼마만큼 적용해서 학습할 수 있는가인 것 같습니다.

일반적으로 논문을 reproduce할 때 논문에서 effective batch size라고 해서 학습할 때 적정한 batch size를 제시해줍니다.

batch size라는 개념은 가중치가 실제로 1회 업데이트 될 때 모델이 학습에 반영하는 데이터 sample의 수를 의미합니다.

이 때 effective batch size는 GPU 당 batch size × GPU 개수 × Gradient Accumulation의 총합을 의미합니다.

LoRA variation method들을 reproduce할 때 대표적으로 PiSSA 같은 경우 Effective Batch size를 128로 지정해주는데, 저는 사용할 수 있는 GPU가 제한적이라 일반적으로 한 개에서 두 개 정도를 사용할 수 있습니다. 그러면 나머지 batch size와 Gradient Accumulation 수치를 가지고 128을 맞춰야 하는데, Mistral-7B 모델의 가중치를 FP32로 올리는 순간 이미 28GB가 되어버려서 남은 공간은 20GB 정도 남게 되고, batch size는 128로 학습을 해야하지만 남은 공간에 다 올릴 수 없는 극한의 상황이 됩니다. 그렇다고 batch size가 지나치게 작으면 파라미터가 안정적으로 수렴하지 못하고 엉뚱한 방향으로 학습이 되거나 발산할 수도 있는 문제가 생깁니다.

그래서 이런 문제를 완화하고자 사용되는 게 Gradient Accumulation이라는 것입니다. 편하게 생각하면 약간의 시간을 더 들이면서 더 큰 batch size로 학습하는 효과를 내보겠다는 뜻입니다.

Gradient Accumulation은 Gradient를 구할 때 연산이 선형성임을 이용해서 나눠서 계산하도록 하는 것입니다.

$$
\nabla L = \nabla\left(\frac{l_1 + l_2 + l_3 + l_4\dots}{4}\right) = \frac{\nabla l_1}{4} + \frac{\nabla l_2}{4} + \frac{\nabla l_3}{4} + \frac{\nabla l_4}{4}\dots
$$

이렇게 수학적으로 동일한 결과를 얻을 수 있다는 것입니다.

하지만 이 때 BatchNorm을 사용하는 모델의 경우는 Gradient Accumulation을 주의해야합니다. 왜냐하면 BatchNorm은 중간에 minibatch를 축으로 평균과 표준 편차를 계산하기 때문에 어떤 sample을 가지고 계산을 하느냐가 Gradient Accumulation에 치명적인 영향을 줄 수 있습니다.

코드에서 사용할 때에는 Accerlate를 사용하면 되고, 일반적으로 Acclerator() 안에 gradient_accmulation_steps를 몇으로 할당할 것인지를 정하면 됩니다.

그렇다면 자연스럽게 이런 생각을 할 수 있습니다.

“batch size를 작게하고 gradient accumulation step을 늘리면 학습이 가능하겠다.”

BatchNorm에서는 이 논리가 들어맞지 않지만 그래도 LayerNorm이나 RMSNorm 같은 경우에는 한 token의 feature dimension만 평균과 표준편차를 계산하는데 사용하기 때문에 다른 sample은 사용하지 않아서 수학적인 이론상으로는 잘 학습이 될 것이라고 기대할 수 있습니다. 대신 Mistral 모델은 Transformer 기반 모델이고 애당초 Attention is All you need 논문에서 나온 Vanilla Transformer도 LayerNorm이었고 현재는 RMSNorm을 많이 사용하는 추세라서 제가 학습하는 환경에서는 무리없이 사용하고 있습니다.

하지만 Floating point 연산의 결합법칙 (associativity) 문제 때문에 FP16, BF16, FP32에서 $a+(b+c)$와 $(a+b)+c$ 연산이 정확하게 같지 않을 수 있습니다.

또한 response token 수가 다른 batch가 있다고 하면 단순하게 결합법칙에 대한 선형성이라는 Gradient Accumulation의 장점이 적용되지 않을 수 있다는 것도 고려해야 합니다.

일단 기본적으로 Gradient Accumulation이 도입된 이유는 batch size를 무한정으로 키워서 학습할 수 없기 때문이며, batch size는 그래도 기본적으로 커야 GPU라는 병렬화 작업에 특화된 하드웨어를 활용할 수 있고 학습 속도도 빠르게 진행될 수 있기 때문에 현재 VRAM 사정에 맞게 진행하면서 batch size는 크게 사용할 수 있는 그 어딘가의 지점을 찾는 것이 중요합니다.

상식적으로 parameter를 업데이트하지 않고 gradient를 ‘쌓아두기만 한다면’ batch size × accumulation’이 동일한 값이면 이론적으로 같은 gradient를 만들어낼 수 있는 것은 사실입니다.

VRAM 48GB에서 FP32로 Mistral-7B 모델의 가중치를 올리고 난 뒤에 학습 공간이 부족해서 일단 batch size를 4로 두고 학습을 해보고 있는데 엄청 느립니다... 이런 병렬적인 효과, 학습적인 효율을 극대화하기 위해서 batch size 최대로 키우는 것이 좋다는 것을 직접 느끼고 있는 중입니다.