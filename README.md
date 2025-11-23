A movie recommendation engine that uses Word2Vec embeddings and cosine similarity to find movies similar to a given movie based on their descriptions.

**How It Works:**

- Data Processing: Loads movie metadata and cleans/preprocesses text descriptions
- Word Embeddings: Trains a Word2Vec model to learn word relationships from movie descriptions
- Vector Averaging: Converts each movie description into a single 300-dimensional vector
- Similarity Matching: Calculates cosine similarity between movies to find the most similar ones
- Recommendations: Returns top 5 movies most similar to your query
