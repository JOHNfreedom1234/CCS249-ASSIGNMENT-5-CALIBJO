import wikipedia
import re
from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf
import pandas as pd
from cosine_similarity import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

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

plt.figure(figsize=(10, 8))
sns.heatmap(df_sim, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Document Similarity (Cosine TF-IDF)")
plt.show()