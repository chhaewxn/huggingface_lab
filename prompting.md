<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# LLM 프롬프팅 가이드 LLM prompting guide

[[Colab에서 열기 / open-in-colab]]

Large Language Models such as Falcon, LLaMA, etc. are pretrained transformer models initially trained to predict the 
next token given some input text. They typically have billions of parameters and have been trained on trillions of 
tokens for an extended period of time. As a result, these models become quite powerful and versatile, and you can use 
them to solve multiple NLP tasks out of the box by instructing the models with natural language prompts.
Falcon, LLaMA 등의 대규모 언어 모델은 주어진 입력 텍스트에 대해 다음 토큰을 예측하도록 초기에 훈련된 사전 훈련된 트랜스포머 모델입니다. 이들은 일반적으로 수십억 개의 매개변수를 가지고 있으며 장기간에 걸쳐 수조 개의 토큰으로 훈련되었습니다. 그 결과, 이 모델들은 매우 강력하고 다재다능해져서 자연어 프롬프트로 모델에 지시하여 여러 NLP 작업을 즉시 해결할 수 있습니다.

Designing such prompts to ensure the optimal output is often called "prompt engineering". Prompt engineering is an 
iterative process that requires a fair amount of experimentation. Natural languages are much more flexible and expressive 
than programming languages, however, they can also introduce some ambiguity. At the same time, prompts in natural language 
are quite sensitive to changes. Even minor modifications in prompts can lead to wildly different outputs.
최적의 출력을 보장하기 위해 이러한 프롬프트를 설계하는 것을 종종 "프롬프트 엔지니어링"이라고 합니다. 프롬프트 엔지니어링은 상당한 실험이 필요한 반복적인 과정입니다. 자연어는 프로그래밍 언어보다 훨씬 더 유연하고 표현력이 풍부하지만, 동시에 약간의 모호성을 도입할 수 있습니다. 또한, 자연어 프롬프트는 변화에 매우 민감합니다. 프롬프트의 사소한 수정도 매우 다른 출력을 초래할 수 있습니다.

While there is no exact recipe for creating prompts to match all cases, researchers have worked out a number of best 
practices that help to achieve optimal results more consistently. 
모든 경우에 맞는 프롬프트를 만들기 위한 정확한 레시피는 없지만, 연구자들은 더 일관되게 최적의 결과를 얻는 데 도움이 되는 여러 모범 사례를 개발했습니다.

This guide covers the prompt engineering best practices to help you craft better LLM prompts and solve various NLP tasks. 
You'll learn:
이 가이드는 더 나은 LLM 프롬프트를 만들고 다양한 NLP 작업을 해결하는 데 도움이 되는 프롬프트 엔지니어링 모범 사례를 다룹니다. 다음 내용을 배우게 됩니다:

- [Basics of prompting](#basics-of-prompting)
- [Best practices of LLM prompting](#best-practices-of-llm-prompting)
- [Advanced prompting techniques: few-shot prompting and chain-of-thought](#advanced-prompting-techniques)
- [When to fine-tune instead of prompting](#prompting-vs-fine-tuning)
프롬프팅의 기초
LLM 프롬프팅의 모범 사례
고급 프롬프팅 기법: 퓨샷 프롬프팅과 사고 체인
프롬프팅 대신 미세 조정을 해야 할 때

<Tip>

Prompt engineering is only a part of the LLM output optimization process. Another essential component is choosing the 
optimal text generation strategy. You can customize how your LLM selects each of the subsequent tokens when generating 
the text without modifying any of the trainable parameters. By tweaking the text generation parameters, you can reduce 
repetition in the generated text and make it more coherent and human-sounding. 
프롬프트 엔지니어링은 LLM 출력 최적화 과정의 일부일 뿐입니다. 또 다른 중요한 요소는 최적의 텍스트 생성 전략을 선택하는 것입니다. 학습 가능한 매개변수를 전혀 수정하지 않고도 텍스트 생성 시 후속 토큰을 선택하는 방식을 사용자 정의할 수 있습니다. 텍스트 생성 매개변수를 조정함으로써 생성된 텍스트의 반복을 줄이고 더 일관성 있고 인간다운 소리가 나도록 만들 수 있습니다.

Text generation strategies and parameters are out of scope for this guide, but you can learn more about these topics in 
the following guides: 
텍스트 생성 전략과 매개변수는 이 가이드의 범위를 벗어나지만, 다음 가이드에서 이러한 주제에 대해 더 자세히 알아볼 수 있습니다:
 
* [Generation with LLMs](../llm_tutorial)
* [Text generation strategies](../generation_strategies)
LLM을 이용한 생성
텍스트 생성 전략

</Tip>

## 프롬프팅의 기초 Basics of prompting

### 모델의 유형 Types of models 

The majority of modern LLMs are decoder-only transformers. Some examples include: [LLaMA](../model_doc/llama), 
[Llama2](../model_doc/llama2), [Falcon](../model_doc/falcon), [GPT2](../model_doc/gpt2). However, you may encounter
encoder-decoder transformer LLMs as well, for instance, [Flan-T5](../model_doc/flan-t5) and [BART](../model_doc/bart).
현대의 대부분의 LLM은 디코더 전용 트랜스포머입니다. 예를 들어 LLaMA, Llama2, Falcon, GPT2 등이 있습니다. 그러나 Flan-T5와 BART와 같은 인코더-디코더 트랜스포머 LLM을 접할 수도 있습니다.

Encoder-decoder-style models are typically used in generative tasks where the output **heavily** relies on the input, for 
example, in translation and summarization. The decoder-only models are used for all other types of generative tasks.
인코더-디코더 스타일 모델은 일반적으로 출력이 입력에 크게 의존하는 생성 작업에 사용됩니다. 예를 들어, 번역과 요약 작업에 사용됩니다. 디코더 전용 모델은 다른 모든 유형의 생성 작업에 사용됩니다.

When using a pipeline to generate text with an LLM, it's important to know what type of LLM you are using, because 
they use different pipelines. 
파이프라인을 사용하여 LLM으로 텍스트를 생성할 때, 어떤 유형의 LLM을 사용하고 있는지 아는 것이 중요합니다. 왜냐하면 이들은 서로 다른 파이프라인을 사용하기 때문입니다.

Run inference with decoder-only models with the `text-generation` pipeline:
디코더 전용 모델로 추론을 실행하려면 text-generation 파이프라인을 사용하세요:

```python
>>> from transformers import pipeline
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT

>>> generator = pipeline('text-generation', model = 'openai-community/gpt2')
>>> prompt = "Hello, I'm a language model"

>>> generator(prompt, max_length = 30)
[{'generated_text': "Hello, I'm a language model programmer so you can use some of my stuff. But you also need some sort of a C program to run."}]
```

To run inference with an encoder-decoder, use the `text2text-generation` pipeline:
인코더-디코더로 추론을 실행하려면 text2text-generation 파이프라인을 사용하세요:

```python
>>> text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-base')
>>> prompt = "Translate from English to French: I'm very happy to see you"

>>> text2text_generator(prompt)
[{'generated_text': 'Je suis très heureuse de vous rencontrer.'}]
```

### 기본 모델 vs 지시/채팅 모델 Base vs instruct/chat models

Most of the recent LLM checkpoints available on 🤗 Hub come in two versions: base and instruct (or chat). For example, 
[`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b) and [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct).
🤗 Hub에서 최근 사용 가능한 대부분의 LLM 체크포인트는 두 가지 버전으로 제공됩니다: 기본 버전과 지시(또는 채팅) 버전입니다. 예를 들어, tiiuae/falcon-7b와 tiiuae/falcon-7b-instruct가 있습니다.

Base models are excellent at completing the text when given an initial prompt, however, they are not ideal for NLP tasks 
where they need to follow instructions, or for conversational use. This is where the instruct (chat) versions come in. 
기본 모델은 초기 프롬프트가 주어졌을 때 텍스트를 완성하는 데 탁월하지만, 지시를 따라야 하거나 대화형 사용이 필요한 NLP 작업에는 이상적이지 않습니다. 이때 지시(채팅) 버전이 필요합니다.

These checkpoints are the result of further fine-tuning of the pre-trained base versions on instructions and conversational data. 
이러한 체크포인트는 사전 훈련된 기본 버전을 지시사항과 대화 데이터로 추가 미세 조정한 결과입니다.

This additional fine-tuning makes them a better choice for many NLP tasks.
이 추가적인 미세 조정으로 인해 많은 NLP 작업에 더 적합한 선택이 됩니다.  

Let's illustrate some simple prompts that you can use with [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct) 
to solve some common NLP tasks.
tiiuae/falcon-7b-instruct를 사용하여 일반적인 NLP 작업을 해결하는 데 사용할 수 있는 몇 가지 간단한 프롬프트를 살펴보겠습니다.

### NLP 작업 NLP tasks 

First, let's set up the environment: 
먼저, 환경을 설정해 보겠습니다:

```bash
pip install -q transformers accelerate
```

Next, let's load the model with the appropriate pipeline (`"text-generation"`): 
다음으로, 적절한 파이프라인("text-generation")을 사용하여 모델을 로드하겠습니다:

```python
>>> from transformers import pipeline, AutoTokenizer
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> model = "tiiuae/falcon-7b-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(model)
>>> pipe = pipeline(
...     "text-generation",
...     model=model,
...     tokenizer=tokenizer,
...     torch_dtype=torch.bfloat16,
...     device_map="auto",
... )
```

<Tip>

Note that Falcon models were trained using the `bfloat16` datatype, so we recommend you use the same. This requires a recent 
version of CUDA and works best on modern cards.
Falcon 모델은 bfloat16 데이터 타입을 사용하여 훈련되었으므로, 같은 타입을 사용하는 것을 권장합니다. 이를 위해서는 최신 버전의 CUDA가 필요하며, 최신 그래픽 카드에서 가장 잘 작동합니다.

</Tip>

Now that we have the model loaded via the pipeline, let's explore how you can use prompts to solve NLP tasks.
이제 파이프라인을 통해 모델을 로드했으니, 프롬프트를 사용하여 NLP 작업을 해결하는 방법을 살펴보겠습니다.

#### 텍스트 분류 Text classification

One of the most common forms of text classification is sentiment analysis, which assigns a label like "positive", "negative", 
or "neutral" to a sequence of text. Let's write a prompt that instructs the model to classify a given text (a movie review). 
We'll start by giving the instruction, and then specifying the text to classify. Note that instead of leaving it at that, we're 
also adding the beginning of the response - `"Sentiment: "`:
텍스트 분류의 가장 일반적인 형태 중 하나는 감정 분석입니다. 이는 텍스트 시퀀스에 "긍정적", "부정적" 또는 "중립적"과 같은 레이블을 할당합니다. 주어진 텍스트(영화 리뷰)를 분류하도록 모델에 지시하는 프롬프트를 작성해 보겠습니다. 먼저 지시사항을 제공한 다음, 분류할 텍스트를 지정하겠습니다. 여기서 주목할 점은 단순히 거기서 끝내지 않고, 응답의 시작 부분 - "Sentiment: "을 추가한다는 것입니다:

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Classify the text into neutral, negative or positive. 
... Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
... Sentiment:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
결과: 텍스트를 중립적, 부정적 또는 긍정적으로 분류하세요. 
텍스트: 이 영화는 확실히 이 장르에서 내가 가장 좋아하는 영화 중 하나입니다. 존경할 만하고 도덕적으로 강한 인물들 간의 상호작용은 기사도와 도둑과 경찰 사이의 명예 코드에 대한 찬사입니다.
감정:
긍정적
```

As a result, the output contains a classification label from the list we have provided in the instructions, and it is a correct one!
결과적으로, 출력에는 우리가 지시사항에서 제공한 목록에서 분류 레이블이 포함되어 있으며, 이는 정확한 것입니다!

<Tip>

You may notice that in addition to the prompt, we pass a `max_new_tokens` parameter. It controls the number of tokens the 
model shall generate, and it is one of the many text generation parameters that you can learn about 
in [Text generation strategies](../generation_strategies) guide.

프롬프트 외에도 max_new_tokens 매개변수를 전달하는 것을 볼 수 있습니다. 이는 모델이 생성할 토큰의 수를 제어하며, 텍스트 생성 전략 가이드에서 배울 수 있는 여러 텍스트 생성 매개변수 중 하나입니다.

</Tip>

#### 개체명 인식 Named Entity Recognition

Named Entity Recognition (NER) is a task of finding named entities in a piece of text, such as a person, location, or organization.
Let's modify the instructions in the prompt to make the LLM perform this task. Here, let's also set `return_full_text = False` 
so that output doesn't contain the prompt:
개체명 인식(Named Entity Recognition, NER)은 텍스트에서 인물, 장소, 조직과 같은 명명된 개체를 찾는 작업입니다.
프롬프트의 지시사항을 수정하여 LLM이 이 작업을 수행하도록 해보겠습니다. 여기서는 return_full_text = False로 설정하여 출력에 프롬프트가 포함되지 않도록 하겠습니다:

```python
>>> torch.manual_seed(1) # doctest: +IGNORE_RESULT
>>> prompt = """Return a list of named entities in the text.
... Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
... Named entities:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=15,
...     return_full_text = False,    
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
- Golden State Warriors
- San Francisco
```

As you can see, the model correctly identified two named entities from the given text.
보시다시피, 모델이 주어진 텍스트에서 두 개의 명명된 개체를 정확하게 식별했습니다.

#### 번역 Translation

Another task LLMs can perform is translation. You can choose to use encoder-decoder models for this task, however, here,
for the simplicity of the examples, we'll keep using Falcon-7b-instruct, which does a decent job. Once again, here's how 
you can write a basic prompt to instruct a model to translate a piece of text from English to Italian: 
LLM이 수행할 수 있는 또 다른 작업은 번역입니다. 이 작업을 위해 인코더-디코더 모델을 사용할 수 있지만, 여기서는 예시의 단순성을 위해 꽤 좋은 성능을 보이는 Falcon-7b-instruct를 계속 사용하겠습니다. 다시 한 번, 모델에게 영어에서 이탈리아어로 텍스트를 번역하도록 지시하는 기본적인 프롬프트를 작성하는 방법은 다음과 같습니다:

```python
>>> torch.manual_seed(2) # doctest: +IGNORE_RESULT
>>> prompt = """Translate the English text to Italian.
... Text: Sometimes, I've believed as many as six impossible things before breakfast.
... Translation:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=20,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
A volte, ho creduto a sei impossibili cose prima di colazione.
```

Here we've added a `do_sample=True` and `top_k=10` to allow the model to be a bit more flexible when generating output.
여기서는 모델이 출력을 생성할 때 조금 더 유연해질 수 있도록 do_sample=True와 top_k=10을 추가했습니다.

#### 텍스트 요약 Text summarization

Similar to the translation, text summarization is another generative task where the output **heavily** relies on the input, 
and encoder-decoder models can be a better choice. However, decoder-style models can be used for this task as well.
Previously, we have placed the instructions at the very beginning of the prompt. However, the very end of the prompt can 
also be a suitable location for instructions. Typically, it's better to place the instruction on one of the extreme ends.  
번역과 마찬가지로, 텍스트 요약은 출력이 입력에 크게 의존하는 또 다른 생성 작업이며, 인코더-디코더 모델이 더 나은 선택일 수 있습니다. 그러나 디코더 스타일 모델도 이 작업에 사용될 수 있습니다. 이전에는 프롬프트의 맨 처음에 지시사항을 배치했습니다. 하지만 프롬프트의 맨 끝도 지시사항을 위한 적절한 위치가 될 수 있습니다. 일반적으로 지시사항을 양 극단 중 하나에 배치하는 것이 더 좋습니다.

```python
>>> torch.manual_seed(3) # doctest: +IGNORE_RESULT
>>> prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
... Write a summary of the above text.
... Summary:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
Permaculture is an ecological design mimicking natural ecosystems to meet basic needs and prepare for climate change. It is based on traditional knowledge and scientific understanding.
```

#### 질문 답변 Question answering

For question answering task we can structure the prompt into the following logical components: instructions, context, question, and 
the leading word or phrase (`"Answer:"`) to nudge the model to start generating the answer:
질문 답변 작업을 위해 프롬프트를 다음과 같은 논리적 구성요소로 구조화할 수 있습니다: 지시사항, 맥락, 질문, 그리고 모델이 답변 생성을 시작하도록 유도하는 선도 단어나 구문("답변:"):

```python
>>> torch.manual_seed(4) # doctest: +IGNORE_RESULT
>>> prompt = """Answer the question using the context below.
... Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
... Question: What modern tool is used to make gazpacho?
... Answer:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Modern tools often used to make gazpacho include
```

#### 추론 Reasoning

Reasoning is one of the most difficult tasks for LLMs, and achieving good results often requires applying advanced prompting techniques, like 
[Chain-of-though](#chain-of-thought).

Let's try if we can make a model reason about a simple arithmetics task with a basic prompt: 
추론은 LLM에게 가장 어려운 작업 중 하나이며, 좋은 결과를 얻기 위해서는 종종 사고 체인과 같은 고급 프롬프팅 기법을 적용해야 합니다.
간단한 산술 작업에 대해 기본적인 프롬프트로 모델이 추론할 수 있는지 시도해 보겠습니다:

```python
>>> torch.manual_seed(5) # doctest: +IGNORE_RESULT
>>> prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
There are a total of 5 groups, so there are 5 x 4=20 students in the class.
```

Correct! Let's increase the complexity a little and see if we can still get away with a basic prompt:
정확합니다! 복잡성을 조금 높여보고 기본적인 프롬프트로도 여전히 해결할 수 있는지 확인해 보겠습니다:

```python
>>> torch.manual_seed(6) # doctest: +IGNORE_RESULT
>>> prompt = """I baked 15 muffins. I ate 2 muffins and gave 5 muffins to a neighbor. My partner then bought 6 more muffins and ate 2. How many muffins do we now have?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
The total number of muffins now is 21
```

This is a wrong answer, it should be 12. In this case, this can be due to the prompt being too basic, or due to the choice 
of model, after all we've picked the smallest version of Falcon. Reasoning is difficult for models of all sizes, but larger 
models are likely to perform better. 
이는 잘못된 답변입니다. 정답은 12여야 합니다. 이 경우, 프롬프트가 너무 기본적이거나 모델 선택의 문제일 수 있습니다. 결국 우리는 Falcon의 가장 작은 버전을 선택했습니다. 추론은 모든 크기의 모델에게 어려운 작업이지만, 더 큰 모델들이 더 나은 성능을 보일 가능성이 높습니다.

## LLM 프롬프트 작성의 모범 사례 Best practices of LLM prompting

In this section of the guide we have compiled a list of best practices that tend to improve the prompt results:
이 섹션에서는 프롬프트 결과를 향상시키는 경향이 있는 모범 사례 목록을 작성했습니다:

* When choosing the model to work with, the latest and most capable models are likely to perform better. 
* Start with a simple and short prompt, and iterate from there.
* Put the instructions at the beginning of the prompt, or at the very end. When working with large context, models apply various optimizations to prevent Attention complexity from scaling quadratically. This may make a model more attentive to the beginning or end of a prompt than the middle.
* Clearly separate instructions from the text they apply to - more on this in the next section. 
* Be specific and descriptive about the task and the desired outcome - its format, length, style, language, etc.
* Avoid ambiguous descriptions and instructions.
* Favor instructions that say "what to do" instead of those that say "what not to do".
* "Lead" the output in the right direction by writing the first word (or even begin the first sentence for the model).
* Use advanced techniques like [Few-shot prompting](#few-shot-prompting) and [Chain-of-thought](#chain-of-thought)
* Test your prompts with different models to assess their robustness. 
* Version and track the performance of your prompts. 
작업할 모델을 선택할 때 최신 및 가장 강력한 모델이 더 나은 성능을 발휘할 가능성이 높습니다.
간단하고 짧은 프롬프트로 시작하고 거기서부터 반복하십시오.
지시사항은 프롬프트의 시작 부분이나 끝 부분에 두십시오. 대용량 컨텍스트로 작업할 때, 모델은 Attention 복잡성이 제곱으로 확장되는 것을 방지하기 위해 다양한 최적화를 적용합니다. 이는 모델이 프롬프트의 중간보다는 시작이나 끝 부분에 더 집중하게 할 수 있습니다.
지시사항을 적용할 텍스트와 명확하게 분리하십시오 - 이에 대해서는 다음 섹션에서 더 자세히 다룹니다.
작업과 원하는 결과에 대해 구체적이고 설명적으로 작성하십시오 - 형식, 길이, 스타일, 언어 등을 명확하게 작성하십시오.
모호한 설명과 지시사항을 피하십시오.
"하지 말라"는 지시보다는 "무엇을 해야 하는지"를 말하는 지시를 선호하십시오.
첫 번째 단어를 쓰거나 첫 번째 문장을 시작하여 출력을 올바른 방향으로 "유도"하십시오.
Few-shot 프롬프팅 및 Chain-of-thought와 같은 고급 기술을 사용하십시오.
프롬프트의 강건성을 평가하기 위해 다른 모델로 테스트하십시오.
프롬프트의 성능을 버전 관리하고 추적하십시오.

## Advanced prompting techniques

### Few-shot prompting

The basic prompts in the sections above are the examples of "zero-shot" prompts, meaning, the model has been given 
instructions and context, but no examples with solutions. LLMs that have been fine-tuned on instruction datasets, generally 
perform well on such "zero-shot" tasks. However, you may find that your task has more complexity or nuance, and, perhaps, 
you have some requirements forㄴ the output that the model doesn't catch on just from the instructions. In this case, you can 
try the technique called few-shot prompting. 

In few-shot prompting, we provide examples in the prompt giving the model more context to improve the performance. 
The examples condition the model to generate the output following the patterns in the examples.

Here's an example: 

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
... Date: 04/12/1961
... Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
... Date:"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=8,
...     do_sample=True,
...     top_k=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date: 09/28/1960
```

In the above code snippet we used a single example to demonstrate the desired output to the model, so this can be called a 
"one-shot" prompting. However, depending on the task complexity you may need to use more than one example. 

Limitations of the few-shot prompting technique: 
- While LLMs can pick up on the patterns in the examples, these technique doesn't work well on complex reasoning tasks
- Few-shot prompting requires creating lengthy prompts. Prompts with large number of tokens can increase computation and latency. There's also a limit to the length of the prompts.  
- Sometimes when given a number of examples, models can learn patterns that you didn't intend them to learn, e.g. that the third movie review is always negative.

### Chain-of-thought

Chain-of-thought (CoT) prompting is a technique that nudges a model to produce intermediate reasoning steps thus improving 
the results on complex reasoning tasks. 

There are two ways of steering a model to producing the reasoning steps:
- few-shot prompting by illustrating examples with detailed answers to questions, showing the model how to work through a problem.
- by instructing the model to reason by adding phrases like "Let's think step by step" or "Take a deep breath and work through the problem step by step."

If we apply the CoT technique to the muffins example from the [reasoning section](#reasoning) and use a larger model, 
such as (`tiiuae/falcon-180B-chat`) which you can play with in the [HuggingChat](https://huggingface.co/chat/), 
we'll get a significant improvement on the reasoning result:

```text
Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
Therefore, you now have 12 muffins.
```

## Prompting vs fine-tuning

You can achieve great results by optimizing your prompts, however, you may still ponder whether fine-tuning a model 
would work better for your case. Here are some scenarios when fine-tuning a smaller model may be a preferred option:

- Your domain is wildly different from what LLMs were pre-trained on and extensive prompt optimization did not yield sufficient results. 
- You need your model to work well in a low-resource language.
- You need the model to be trained on sensitive data that is under strict regulations. 
- You have to use a small model due to cost, privacy, infrastructure or other limitations. 

In all of the above examples, you will need to make sure that you either already have or can easily obtain a large enough 
domain-specific dataset at a reasonable cost to fine-tune a model. You will also need to have enough time and resources 
to fine-tune a model.

If the above examples are not the case for you, optimizing prompts can prove to be more beneficial.   


