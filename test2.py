import string

def better_tokenize(st):
    exclude = set(string.punctuation)
    st=st.lower()
    return ''.join(ch for ch in st if ch not in exclude).split()

t = "the dog jumped over the short fence, then walked: into the riv3r"

print (better_tokenize(t))
