# Log of Thoughts
In this document the train of thoughts, searches on the web and learnings are
documented with their corresponding timestamps in order to give reviewers a
window to the developers mind.


[Fri 20 Jun 2020 21:12:17 CES]
What is POS Tagging?
[Wikipedia](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
This looks like a supervised imbalanced multi-class classification task.
Data is sequential i.e. the class of the object depends also on the objects
before and after it. A `no-skill` classifier can reach an accurary of 90%.

> ... merely assigning the most common tag to each known word and the tag "proper noun" to all unknowns will approach 90% accuracy.
>
> -- <cite>[Part-of-speech_tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)</cite>

[Fri 20 Jun 2020 21:28:31 CES]
Based on further readings a probabilistic approach seems to be a better choice.
Although Hidden Markov Models seem to be a good baseline to start from, the type
of problem looks like a good fit for recurrent neural networks.

[Fri 20 Jun 2020 21:43:31 CES]
LSTM architecture looks even more promising and relatively easy to impelent in
one day.

> - The most commonly used LSTM architecture (vanilla LSTM) performs reasonably
well on various datasets…
> - Learning rate and network size are the most crucial tunable LSTM 
hyperparameters
> - This implies that the hyperparameters can be tuned independently. In 
particular, the learning rate can be calibrated first using a fairly small 
network, thus saving a lot of experimentation time.
>
> -- <cite>[machinelearningmastery.com LSTM Tutorial](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)</cite>

[Fri 20 Jun 2020 21:59:06 CES]
[Peilu Wang's paper](https://arxiv.org/pdf/1510.06168.pdf) seems to verify my 
intuition.

[Fri 20 Jun 2020 22:13:31 CES]
This [GitHub repository](https://github.com/aneesh-joshi/LSTM_POS_Tagger) hast 
tried to implement the above paper. The architecture is also not very complex.

[Fri 20 Jun 2020 22:41:09 CES]
Found a better [NLP tutorial](https://nlpforhackers.io/lstm-pos-tagger-keras/)

[Fri 20 Jun 2020 23:19:11 CES]
A Proof of Concept is implemented using the above tutorial.
See `scripts/make_gum.py`. An Accuracy of 97.855 is reahced.

[Fri 20 Jun 2020 23:28:57 CES]
Adding some TODOs for tommorow and calling it a day:
- [ ] Create project
- [ ] Add train of thoughts and learnings of yesterday evening.
- [ ] Write the first test
- [ ] Choose a (simple) baseline algorithm (decison tree, hmm)
- [ ] Implement LSTM in PyTorch
- [ ] Choose hyperparameters (learning rate, network size)
- [ ] Plot loss/accuracy over time (watch out for overfitting)
- [ ] Evaluate against unseen data
- [ ] How does it compare to state of the art implementations (97% accuracy)
- [ ] Evaluate model using bigger corpi (optional)
- [ ] decide on embeddings (pre-trained like (Glove, or implement)
- [ ] Deploy
    - [ ] How big is the trained model? Deploy with serverless?
    - [ ] How to calculate Word2Vec? Are the public APIs for word embeddings?
    - [ ] Add CNAME
- [ ] Save model or only save weights?

[Sat 20 Jun 2020 17:44:31 CEST]
Tested the tagging of one sentene.
```
How  :  SCONJ
do  :  AUX
people  :  NOUN
look  :  VERB
at  :  ADP
and  :  CCONJ
experience  :  VERB
art  :  NOUN
```

[Sat 20 Jun 2020 19:06:54 CEST] After adding embeddings from the Glove dataset
the accuracy increased. I still think there should be other ways to increase
the performance on unseen words.

[Sat 20 Jun 2020 21:46:04 CEST]
Branching out to implement the same network in PyTorch

[Sun 21 Jun 2020 09:12:13 CEST]
The [implementation in PyTorch](https://github.com/arashzandi/chewing_gum/blob/feature/use_pytorch/scripts/torch_gum.py) has a lower performance compared to the
implementation in Keras. Both networks are identical but is it possilbe that
the metrics are calculated slightly different? Keras should be used rather for
prototyping but a more carefull and unit-tested implementation in Pytorch is
more favourable.

[Sun 21 Jun 2020 09:14:33 CEST]
Time to wrap up and wirte some tests and move some parts of code to classes.

[Sun 21 Jun 2020 21:31:26 CEST]
Unfortunately I have to make compromises because of lack of time. Tuning 
the learning rate and the LSTM network size or the size of the Glove Embedding
vectors can probably enhance the model performance slightly.

[Sun 21 Jun 2020 21:34:50 CEST]
Adding TODOs to the source code for where improvements can be acheived easily.