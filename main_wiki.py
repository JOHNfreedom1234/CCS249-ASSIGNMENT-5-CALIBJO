import wikipedia
import re
from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf
import pandas as pd
from cosine_similarity import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

titles = [
    "Artificial intelligence",
    "Structured prediction",
    "Neural network (machine learning)",
    "Deep learning",
    "Speech synthesis",
]

documents = []
successful_titles = []
for title in titles:
    try: 
        text = wikipedia.page(title).content
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        documents.append(tokens)
        successful_titles.append(title)
    except wikipedia.exceptions.PageError:
        print(f"Page not found for title: {title}")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for title: {title}. Options: {e.options}")

print("Successfully retrieved documents for titles:")
for title in successful_titles:
    print(title)

vocab = sorted(set(term for doc in documents for term in doc))

idf = compute_idf(documents, vocab)

tf_matrix = []
tfidf_matrix = []

for doc in documents:
    tf = compute_tf(doc, vocab)
    tfidf = compute_tfidf(tf, idf, vocab)
    tf_matrix.append(tf)
    tfidf_matrix.append(tfidf)

df_tf = pd.DataFrame(tf_matrix, index=titles)
df_tfidf = pd.DataFrame(tfidf_matrix, index=titles)

print("=== Term Frequency Matrix (truncated) ===")
print(df_tf.iloc[:, :10])  # Show first 10 columns

print("\n=== TF-IDF Matrix (truncated) ===")
print(df_tfidf.iloc[:, :10])

# Compute the Cosine Similarity between the first two documents
similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[3], vocab)
print("\nCosine Similarity between Document 1 and Document 4:")
print(similarity)

similarity_table = []

for i, vec1 in enumerate(tfidf_matrix):
    row = []
    for j, vec2 in enumerate(tfidf_matrix):
        sim = cosine_similarity(vec1, vec2, vocab)
        row.append(sim)
    similarity_table.append(row)

titles = [
    "Artificial intelligence",
    "Structured prediction",
    "Neural network (machine learning)",
    "Deep learning",
    "Speech synthesis"
]

df_sim = pd.DataFrame(similarity_table, index=titles, columns=titles)

print("\n=== Cosine Similarity Matrix ===")
print(df_sim.round(3))

def document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

w2v_model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1)
X = np.array([document_vector(doc, w2v_model) for doc in documents])
y = [1, 0, 1, 1, 0]

clf = LogisticRegression()
clf.fit(X, y)
print(classification_report(y, clf.predict(X)))

# TF-IDF (sparse)
df_tfidf_matrix = pd.DataFrame(tfidf_matrix).fillna(0)
pca_tfidf = PCA(n_components=2)
tfidf_reduced = pca_tfidf.fit_transform(df_tfidf_matrix)

# Word2Vec (dense)
pca_w2v = PCA(n_components=2)
w2v_reduced = pca_w2v.fit_transform(X)

# Plot both side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, label in enumerate(successful_titles):
    axes[0].scatter(*tfidf_reduced[i], color='blue')
    axes[0].text(*tfidf_reduced[i], label, fontsize=9)
    axes[1].scatter(*w2v_reduced[i], color='green')
    axes[1].text(*w2v_reduced[i], label, fontsize=9)

axes[0].set_title("TF-IDF (Sparse)")
axes[1].set_title("Word2Vec (Dense)")
plt.tight_layout()
plt.show()

X_3d = X[:, :3]
np.save("w2v_vectors_3d.npy", X_3d)
np.save("titles.npy", np.array(successful_titles))