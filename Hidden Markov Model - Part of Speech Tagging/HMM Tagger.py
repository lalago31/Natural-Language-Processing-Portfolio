import matplotlib.pyplot as plt
import numpy as np

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

# load data
data = Dataset("data//tags-universal.txt", "data//brown-universal.txt", train_test_split=0.8)

print("There are {} sentences in the corpus.".format(len(data)))
print("There are {} sentences in the training set.".format(len(data.training_set)))
print("There are {} sentences in the testing set.".format(len(data.testing_set)))

# step 1: Most Frequent Class Tagger
'''
Perhaps the simplest tagger (and a good baseline for tagger performance) is to simply choose the tag most frequently assigned to each word. 
This "most frequent class" tagger inspects each observed word in the sequence and assigns it the label that was most often assigned to that word in the corpus.
'''
def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    
    counts = defaultdict(Counter)
    for t, w in zip(chain(*sequences_A), chain(*sequences_B)):
        counts[t][w] += 1
    return counts

# Calculate C(t_i, w_i)
emission_counts = pair_counts(data.training_set.Y, data.training_set.X)
''' example:
{'NOUN': {'Mr.': 844,
  'Podger': 21,
  'use': 352,
  'advice': 50,
  'difference': 148
  '''

'''
Next Use the `pair_counts()` function and the training dataset to find the most frequent class label for each word in the training data, 
and populate the `mfc_table` below. The table keys should be words, and the values should be the appropriate tag string.

The `MFCTagger` class is provided to mock the interface of Pomegranite HMM models so that they can be used interchangeably.
'''

# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word
from collections import namedtuple

FakeState = namedtuple("FakeState", "name")

class MFCTagger:
    # NOTE: You should not need to modify this class or any of its methods
    missing = FakeState(name="<MISSING>")
    
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
        
    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))


# calculate the frequency of each tag being assigned to each word (hint: similar, but not
# the same as the emission probabilities) and use it to fill the mfc_table

word_counts = pair_counts(data.training_set.X, data.training_set.Y)
# get the most frequent tag for each word
mfc_table = {word: max(word_counts[word], key=word_counts[word].get) for word in word_counts.keys()}

mfc_model = MFCTagger(mfc_table) # Create a Most Frequent Class tagger instance

# Making Predictions with the baseline model
# The helper functions provided below interface with Pomegranate network models & the mocked MFCTagger 
# to take advantage of the [missing value](http://pomegranate.readthedocs.io/en/latest/nan.html) 
# functionality in Pomegranate through a simple sequence decoding function.
def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]  # do not show the start/end state predictions

# evaluate baseline model
def accuracy(X, Y, model):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.
    
    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.
    
    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [(), (), ...]
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions

mfc_training_acc = accuracy(data.training_set.X, data.training_set.Y, mfc_model)
print("training accuracy mfc_model: {:.2f}%".format(100 * mfc_training_acc))

mfc_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, mfc_model)
print("testing accuracy mfc_model: {:.2f}%".format(100 * mfc_testing_acc))

#********************************************************************************
# Step 2: Build an HMM tagger
'''
The HMM tagger has one hidden state for each possible tag, and parameterized by two distributions: 
the emission probabilties giving the conditional probability of observing a given **word** from each hidden state, 
and the transition probabilities giving the conditional probability of moving between **tags** during the sequence.

We will also estimate the starting probability distribution (the probability of each **tag** being the first tag in a sequence), 
and the terminal probability distribution (the probability of each **tag** being the last tag in a sequence).

'''
# 2.1 unigram count func: get frequency of individual tags
def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.
    
    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    
    unigram_counts = {}
    for sentence in sequences:
        for tag in sentence:
            unigram_counts[tag]= unigram_counts.get(tag,0)+1
    return unigram_counts

#  call unigram_counts with a list of tag sequences from the training set
tag_unigrams = unigram_counts(data.training_set.Y)
'''
{'ADV': 44877,
 'NOUN': 220632,
 '.': 117757,
 'VERB': 146161,
 'ADP': 115808,
 'ADJ': 66754,
 'CONJ': 30537,
 'DET': 109671,
 'PRT': 23906,
 'NUM': 11878,
 'PRON': 39383,
 'X': 1094}
'''

# 2.2 bigram count func: get frequency of two tags appearing together
def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
    bigram_counts = {}
    for sentence in sequences:
        for i in range(len(sentence)-1):
            j = i +1
            bigram_counts[(sentence[i], sentence[j])]= bigram_counts.get((sentence[i], sentence[j]),0)+1
    return bigram_counts
    

# call bigram_counts with a list of tag sequences from the training set
tag_bigrams = bigram_counts(data.training_set.Y)
'''
{('ADV', 'NOUN'): 1478,
 ('NOUN', '.'): 62639,
 ('.', 'ADV'): 5124,
 ('ADV', '.'): 7577,
 ('.', 'VERB'): 9041,
 ('VERB', 'ADP'): 24927,
 ('ADP', 'ADJ'): 9533,
 ('ADJ', 'NOUN'): 43664,
 ('NOUN', 'CONJ'): 13185
 '''

 # 2.3 sequence starting count: estimate the probabilities of a sequence starting with each tag
def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.
    
    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    starting_counts = {}
    for sent in sequences:
        start_tag = sent[0]
        starting_counts[start_tag] = starting_counts.get(start_tag, 0) +1
    return starting_counts

#  Calculate the count of each tag starting a sequence
tag_starts = starting_counts(data.training_set.Y)

# 2.4 Sequence Ending Counts
def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.
    
    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """
    ending_counts = {}
    for sent in sequences:
        end_tag = sent[-1]
        ending_counts[end_tag] = ending_counts.get(end_tag, 0) +1
    return ending_counts

# TODO: Calculate the count of each tag ending a sequence
tag_ends = ending_counts(data.training_set.Y)

# 2.5 basic Hidden Markov Model
'''
Use the tag unigrams and bigrams calculated above to construct a hidden Markov tagger.

- Add one state per tag
    - The emission distribution at each state should be estimated with the formula: P(w|t) = C(t, w)}/{C(t)}
- Add an edge from the starting state `basic_model.start` to each tag
    - The transition probability should be estimated with the formula: P(t|start) = C(start, t)}/{C(start)}
- Add an edge from each tag to the end state `basic_model.end`
    - The transition probability should be estimated with the formula: P(end|t) = C(t, end)}/{C(t)}
- Add an edge between _every_ pair of tags
    - The transition probability should be estimated with the formula: P(t_2|t_1) = C(t_1, t_2)}/{C(t_1)}
'''
basic_model = HiddenMarkovModel(name="base-hmm-tagger")

#Part1: emission probability
# create states with emission probability distributions P(word | tag) and add to the model
# P(word | tag), given the tag, what is the prob of the word = count of word with the tag/ count of all words of that tag

emission_tag_word_counts = pair_counts(data.training_set.Y, data.training_set.X) #{'NOUN':{'word': count}}
tag_unigrams = unigram_counts(data.training_set.Y) # count of occurrance of a tag
tag_bigrams = bigram_counts(data.training_set.Y)
starting_tag_count = starting_counts(data.training_set.Y) #the number of times a tag occured at the start
ending_tag_count = ending_counts(data.training_set.Y)

# Initialize a dictionary to store the states object by tag key
states = dict()
for tag, words_dict in emission_tag_word_counts.items():
    tag_count = tag_unigrams[tag]
    emission_distribution = {word: word_count/tag_count for word, word_count in words_dict.items()}
    prob_word_given_tag = DiscreteDistribution(emission_distribution)
    word_given_tag_state = State(prob_word_given_tag, name=tag)
    basic_model.add_states(word_given_tag_state)
    states[tag] = word_given_tag_state

#Part2: transition probability of start and end state

# add start state prob = p(tag| start state) = p(start state,tag)/p(start state)
# = num of times a tag appears in the start/ number of first tags in whole dataaset
# add end state prob = p(tag| end state) = p(end state,tag)/p(tag) ****** this is diff from the start state prob!!!
# = num of times a tag appears in the end/ number of that tag in whole dataaset
for tag in data.training_set.tagset:
    
    tag_start_prob = starting_tag_count[tag]/sum(starting_tag_count.values())
    basic_model.add_transition(basic_model.start, states[tag], tag_start_prob)
    tag_end_prob = ending_tag_count[tag]/tag_unigrams[tag]
    basic_model.add_transition(states[tag], basic_model.end, tag_end_prob)


#Part3: transition probability between states

# add edges between states for the observed transition frequencies P(tag_i | tag_i-1) = P(tag_i , tag_i-1) / P(tag_i-1)
for tag_1, tag_2 in tag_bigrams.keys():
    transition_tag_prob = tag_bigrams[(tag_1, tag_2)]/ tag_unigrams[tag_1]
    basic_model.add_transition(states[tag_1], states[tag_2], transition_tag_prob)

# evaluate the HMM tagger performance
basic_model.bake()
hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc)) # 97.54%

hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc)) #95.95%

# example: actual vs. predicted labels
for key in data.testing_set.keys[:3]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, basic_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")