from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf
from cosine_similarity import cosine_similarity

# Sample corpus
documents = [
    "the cat sat on the mat", # considered as a document, document 1
    "the dog sat on the log", # document 2
    "cats and dogs are great pets" # document 3
]

print("Documents:")
for doc in documents:
    print(doc)

# Tokenize and apply lowercase the documents into words
tokenized_docs = [doc.lower().split() for doc in documents]

print("Documents:")
for i, doc in enumerate(tokenized_docs):
    print(f"Document {i+1}: {doc}")

# Create a set of unique words (vocabulary)
vocab = set(word for doc in tokenized_docs for word in doc)

print("\nVocabulary:")
print(vocab)

# Compute the term frequency for each document
tf_vectors =  [ compute_tf(doc, vocab) for doc in tokenized_docs ]

print("\nTerm Frequency Vectors:")
for i, tf_vector in enumerate(tf_vectors):
    print(f"Document {i+1}: {tf_vector}")

# Compute the Inverse Document Frequency (IDF)
idf = compute_idf(tokenized_docs, vocab)
print("\nInverse Document Frequency:")
for term, idf_value in idf.items():
    print(f"{term}: {idf_value}")

tfidf_vectors = [ compute_tfidf(tf, idf, vocab) for tf in tf_vectors ]
print("\nTF-IDF Vectors:") 
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"Document {i+1}: {tfidf_vector}")

# Compute the Cosine Similarity between the first two documents
similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1], vocab)
print("\nCosine Similarity between Document 1 and Document 2:")
print(similarity)
