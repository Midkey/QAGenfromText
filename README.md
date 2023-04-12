# 从文本中生成QA对

这是一个问题生成脚本。借助LLM的能力，我们可以直接从文本中得到问题以及对应的答案。

这个工程计划服务于 驼铃B ：ChatHarryPotter。了解驼铃B请点击：[TuoLingB](https://github.com/LC1332/CamelBell-Chinese-LoRA/blob/main/data/HarryPotter/ShortReport.md)


## 当前工程生成问题思路

### 1. 文本获取

文本获取方式十分多样，并且没有定论。在我们工程目标想打造哈利波特世界的训练数据，因此也就从哈利波特的wiki中提取文档，然后作为本工程的输入。输出的结果组织成问题与答案文本对的json文件，能够直接用于finetune其他的LLM模型。
```
python download_html.py
```
由于获取内容多样，并且这个文件类似爬虫，就暂时置空（可能大概就是繁忙与懒惰吧）

### 2. 问题答案对生成

实际过程类似使用ChatGPT获取文档段落的问题和答案对（pair），在本工程中，直接使用了[Alpaca-Lora](https://github.com/tloen/alpaca-lora) 的模型作为LLM，提取问题与答案。

```
python generate_QAPairs.py --document_file ./HarryPotterDoc/Harry Potter self_v0.1.txt
```

这个脚本将每一行文本作为一段输入，向LLM询问5个问题（实际测试询问10个问题效果感觉不太稳定）。然后将同样的文本输入作为history，让LLM提供每个问题的答案。

本脚本暂时额外做了一些简单的文本清洗：
对于生成的特别短的问题（len(q) < 3），直接跳过忽略。后续考虑更多的清洗条件。但是实际使用的时候，应用场景的不同，清洗的方式肯定也很不同，需要一个通用的方式进行清洗。因此并不想写很多if else去处理这个问题。待定。

## 应用方式探讨

本项目想通过得到的QA对，然后能通过finetune的方式长文本的信息注入到LLM模型中。
1. 先通过本工程得到QA对，作为训练数据集
2. 利用lora等技术，将信息finetune到模型中

实际上，这种的信息注入方式十分粗浅，但是相对简洁。

类似更高明的方法可以见类似这个文档的做法: [使用 GPT 基于个人文档建立聊天机器人的逐步教程 ](https://juejin.cn/post/7216968724938129465)

不过由于需要管理一个外部的知识库，然后每次运行的时候送入指定信息，以实现问题的回答。

但是对于一些专有领域的文档，模型先熟读一遍文档，然后需要的时候再查阅具体段落，会更符合人类的逻辑。

这就可以考虑这两种方式进行结合。不过感觉信息通过QA方式的进行注入，效率还是比较低。
这个可以进一步探讨。

## TODO:

- [ ]  多文档（文件夹）QA提取
- [ ]  多GPU并行QA生成
- [ ]  文档清洗方式

## 类似代码

[question_generation](https://github.com/deeppavlov/question_generation) 
