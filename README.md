# CMUDaaN
CMU Dictionary as a Network (CMUDaaN) is a lightweight library that reframes the CMU Dictionary as network consisting of weighted edges by proportional matching and words as nodes.

## How it works
Each word given by the CMU Dict is broken down into syllables. In Pythonic terms, for each word, there is a list of syllables that represent that word. Therefore, for ``n`` words, we now have ``(n * n-1) / 2`` combinations without replacement of words to be studied against using similarity metrics. The metric chosen for this was Jaro Similarity since this produced a number between 0 and 1 and made relative sense with respect to this topic.

## Example
Using the function ``graph(n = 10, show = True)``, you can create a network like the following:
<p align="center">
  <img src="images/example.png">
</p>
This selects 10 random words and establishes connections to them through Jaro Similarity. If the similarity is 0, no connection is made.
