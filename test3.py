from collections import Counter
import pandas
import * from scipy
headlines = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
]

# Find all the unique words in the headlines.
unique_words = list(set(" ".join(headlines).split(" ")))
def make_matrix(headlines, vocab):
    matrix = []
    for headline in headlines:
        # Count each word in the headline, and make a dictionary.
        counter = Counter(headline)
        # Turn the dictionary into a matrix row using the vocab.
        row = [counter.get(w, 0) for w in vocab]
        matrix.append(row)
    df = pandas.DataFrame(matrix)
    df.columns = unique_words
    return df

print (make_matrix(headlines, unique_words))

print (len(unique_words))
