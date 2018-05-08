# Generative Text Simplification

This project aims to adopt generative approaches dealing with (neural)
text generation and transformation for the task of text simplification.

## Problem description

Starting with the simple wikipedia data set as trainging data the 
project goal is to automatize text simplification.

Data set: http://www.cs.pomona.edu/~dkauchak/simplification/

The data set contains aligned sentences, so we have pairs of complex
and simplified expressions.

As a first approach we'd like to adopt Shen et al.'s (2017) approach
to text transformation. They were originally dealing with style 
transfer. We argue that textsimplification can basically be understood
as transferring natural language expressions from a more complex to a
more simplistic style.

## Proposed procedure:

Adopt Shen et al.'s implementation and adapt it to our use case. In
addition we propose to tackle the large vocabulary problem by only 
adopting the most frequent words of the vocabulary and maybe exclude
non-atomic words and in addition train a network that encodes words
starting at character level constrainted by the already existing word
level embeddings.

If Shen et al.'s approach is not successful, there might be Hu et al. 
(2018) as an alternative.

