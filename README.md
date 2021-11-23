# Transformers

<p align="center">
<img src="https://i.pinimg.com/640x/5f/da/a1/5fdaa1f698b0ef1f33527915172b4f22.jpg" width=80% height=500>
</p>

1- What is a Transformer ?
2- Transformers VS RNNs ?
3- Applications

# What is a Transformer? 

The Transformer was proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) in 2017. It uses attention mechanism, as the title indicates. Like LSTM, Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the previously described/existing sequence-to-sequence models because it does not imply any Recurrent Networks (GRU, LSTM, etc.).

To have a deeper look at the transformers, please refer to this [blog post](http://jalammar.github.io/illustrated-transformer/)


## What is Attention?
<p align="center">
  <img src="https://i.stack.imgur.com/jWduk.png" width=50% />
</p>

Here are some notes about what an attention layer is: 
- Attention is an added layer that lets the model focus on what’s important
- An attention function can be described as mapping a query and a set of key-value pairs to an output
- Imagine that you are translating English into German. You can represent the word embeddings in the English language as keys and values. The queries will then be the German equivalent. You can then calculate the dot product between the query and the key. The intuition here is that you want to identify the corresponding words in the queries that are similar to the keys. This would allow your model to "look" or focus on the right place when translating each word. We then run a softmax
softmax(Q KT) (1)
- That allows us to get a distribution of numbers between 0 and 1.  One more step is required. We then would multiply the output by V. Remember V in this example was the same as our keys, corresponding to the English word embeddings. Hence the equation becomes
softmax(Q KT)V (2)
- This tells us how much of each word, or which combination of words we will be feeding into our decoder to predict the next German word. This is called scale dot product attention.



# Transformers VS RNN

The conventional encoder-decoder architecture which is based on RNNs and used to compute T sequential steps. In contrast, transformers are based on attention and don't require any sequential computation per layer, only one single step is needed. Additionally, the gradient steps that need to be taken from the last output to the first input in a transformer is just one.

Transformers don't suffer from vanishing gradients problems that are related to the length of the sequences. Transformer differs from sequence to sequence by using multi-head attention layers instead of recurrent layers.

Issues with RNNs are :
- No parallel computing
- Loss of information
- Vanishing Gradient

In contrast, transformers:
- Don't require any sequential computation per layer
- Don't suffer from vanishing gradients problems

# Applications

Transformer based architectures were not only used for NLP, there are some other applications for computer vision and speech recognition tasks. We can cite [Bert](https://arxiv.org/abs/1810.04805) for NLP and [ViT](https://arxiv.org/abs/2010.11929) for computer vision and [Speech Transformer](https://ieeexplore.ieee.org/document/8682586).

## Bert Applications

### Sentiment Analysis <a href="https://colab.research.google.com/drive/1PSn5L0o6OcXAiAvHcE5JkLld8k9X4NUM?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this project, we will use a fine-tuned Bert model for sentiment analysis. We explain how a pytorch Bert model can be used and adjusted for multi-class classification. 
