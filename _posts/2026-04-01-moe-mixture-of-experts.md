---
title: 'MoE: Mixture of Experts'
date: '2026-04-01T18:02:14+09:00'
categories:
- 공부 일기
source_url: https://nulw.tistory.com/entry/MoE-Mixture-of-Experts
---

![Unsplash / Photo by Lucas Law](/assets/img/tistory/22/001.jpg)

Unsplash / Photo by Lucas Law

MoE: Mixture of Experts에 대한 내용을 다룹니다.

GraLoRA라는 아이디어에 MoE를 적용시키려고 하고 사전 지식 습득을 위해 아이디어 측면에서 다룬 내용입니다.

기존의 Dense Neural Network에서는 입력 데이터가 들어올 때마다 모델 내부의 모든 파라미터가 연산 과정에서 활용되었습니다. 이 때 본질적으로 조건부 연산을 시행해서 필요한 파라미터만 연산을 하게 되면 좋을 것이라는 아이디어에 출발하게 됩니다. 그래서 거대한 network를 여러 개의 전문가, expert로 쪼개서 입력 데이터의 특성에 따라서 일부 expert만 활성화시키게 되면 연산량이 줄어들면서 핵심적인 연산만 할 수 있게 됩니다.

MoE architecture 자체는 2017년부터 논의되기 시작했고 현재까지도 오픈소스, private model 상관없이 다양하게 사용되고 있는 추세입니다.

하지만 MoE에는 문제점이 있습니다.

이전 연구들에서는 성능적인 측면에서만 연구가 많이 되어왔지만 실질적으로 GPU 메모리, 연산량, latency 측면에서 문제점이 많았다는 것입니다.

Transformer에서 MoE를 사용할 때, Experts의 위치는 FFN layer에 위치합니다.

<strong>Self-attention ⇒ FFN Up-projection ⇒ Non-linearity Activation ⇒ FFN Down-projection</strong>

이 과정을 일반적으로 거치게 되는데, self-attention은 건드리지 않고 FFN 부분만 Expert를 사용하게 됩니다. 대신 하나의 전문가에 대한 표현력을 유지 시켜야 성능이 나올 것이기 때문에 FFN layer에 해당하는 가중치를 8개를 복사해서 각기 다른 expert로 작동하게끔 만들어줍니다.

다만 이게 너무 용량을 많이 차지하는 문제가 있습니다. 8배를 늘린다고 해서 필요한 파라미터가 8배가 늘어나는 것은 아니기는 합니다. 다만 13B 파라미터를 사용했던 것이 MoE를 사용하게 되면 47B 정도의 파라미터를 요구하게 되는 것입니다. 그러면 그냥 expert를 한 번에 다 올리는 게 아니라 필요한 expert만 그 때 그 때 올리면 되지 않을까라는 생각을 할 수 있습니다.

결론부터 말하자면 불가능합니다.

애당초에 8개의 expert를 만들어서 특정 task에 fit하도록 만들어 성능을 끌어올리는 것이 목적인데, 그 때 8개의 expert 중 확률이 제일 높은 expert 두 개를 뽑아서 연산하게 됩니다. 이 확률이 높은 expert를 뽑는 과정을 MoE Routing이라고 합니다.

Attention layer 통과하고 나온 token이 1차원으로 존재하는데, 이 토큰을 평가하기 위한 Router 가중치 행렬이 8개 있습니다. 해당 토큰 벡터와 가중치 행렬을 dot product를 실행하고 그 결과로 크기가 8인 벡터가 나오게됩니다. 이 값들을 가지고 softmax를 취해서 각 expert에 대한 확률을 구합니다. 이 때 제일 확률이 높은 두 개의 expert에 할당을 해서 학습을 진행하게 됩니다.

처음에는 가중치 행렬이 random initialization이지만 학습하는 과정에서 특정 토큰에 대해 어느 방향으로 학습을 하는 것이 성능에 도움이 되는지 조절되는 것입니다.

그렇기 때문에 들어오는 토큰을 보고 softmax를 통해서 적용을 해야하는, 동적 routing 시스템으로 작동합니다.

이 때 동적 routing 시스템 상으로 두 개의 expert를 고를 때 다른 메모리에서 그 때 그 때 expert를 가져와서 사용하게 되면 memory bandwidth 문제가 있어서 실질적인 시간이 너무 오래 걸린다는 문제가 있습니다. 따라서 메모리를 조금 포기하더라도 연산량 자체는 어차피 뽑힌 2개의 expert만 활용하기 때문에 큰 차이가 없고, 성능은 오르기 때문에 사용하는 것입니다.

MoE의 장점을 정리하자면

- 동일한 연산량, 즉 컴퓨팅 자원을 크게 소모하지 않으면서 모델 전체가 기억할 수 있는 파라미터의 총량을 획기적으로 늘릴 수 있습니다. MoE의 sparse activation 구조 덕분에 훨씬 더 적은 계산 비용으로 더 낮은 loss에 도달할 수 있게 됩니다.
- expert 구조를 통해서 데이터를 강제로 나눠주지 않더라도 Router와 Expert가 loss를 줄이는 방향으로 학습되면 특정 domain에 대해 최적화가 되는 것입니다.
- 파라미터 수에 비해서 실제 연산에 참여하는 비율이 10%&#126;20% 정도이기 때문에 메모리 대역폭 문제만 해결되면 동일 파라미터 크기의 Dense 모델보다 연산 속도가 압도적으로 빠릅니다

반면에 단점을 정리하자면

- 초기 무작위 상태에서 성능이 좋게 나온 특정 expert에게만 router가 모든 토큰을 몰아주는 routing collapse가 발생하게 됩니다. 그래서 모든 expert가 비슷한 양의 토큰을 받도록 규제하는 방식이 있는데 이 방식은 자유도를 제한할 뿐더러 특정 task에 최적화되는 방식에 정면으로 충돌하게 되기 때문에 결과적으로 모델의 성능이 최적이 되지 않게 됩니다.
- Communication Overhead가 있습니다. GPU가 고정된 크기의 연속적인 행렬 연산에 최적화되어있는데 MoE의 동적 routing 시스템으로 인해서 여러 GPU에 expert를 분산시키게 됩니다. 따라서 딥러닝 학습 시간의 절반 이상을 GPU 간 데이터를 주고받는 통신에 허비하게 됩니다.
- 실제로 Fine tuning에 MoE를 적용했을 때에도 일반화 성능이 너무 떨어지는 문제가 있습니다. 소량의 데이터로 훈련했을 때 Router의 밸런스가 무너지고 Expert들의 전문성이 파괴되기도 합니다.

그래서 다양한 문제가 있는데 deepseek의 경우는 모델의 파라미터를 늘리는 게 아니라 고정시킨 상태에서 하나의 expert에게 제한된 파라미터를 주고 많은 expert를 배치하는 방식을 선택했고, LoRA에 MoE를 적용한 MoELoRA도 존재합니다.