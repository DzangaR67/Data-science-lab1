# Data-science-lab1

# Reflections

**Content-based filtering**
- It worked better overall, as it provided more relevant and personalized recommendations by leveraging genere metadata, especially for users with little rating history. For example, User 1 received suggestions like Inside Out and Kung-Fu Panda, which aligned well with their preference for animation. This method also handled the cold-start problem(the challenge of making accurate recommendations when there is little to no historical data to work with) more gracefully, making it ideal for new users or niche tastes.

**Collaborative Filtering**
- While conceptually powerful, struggled due to sparse user overlap in the dataset. The similarity matrix often returned zeros between users, leading to weak or random recommendations. It became more effective only after increasing rating overlap across users, but didn’t capture the subtle, context-rich aspects of user preferences compared to genre-based matching.

**Challenges faced**

- Missing Data: Sparse ratings across users made it difficult for collaborative filtering to compute meaningful similarities.

- Cold Start Problem: New users with few ratings couldn’t benefit from collaborative filtering, while content-based filtering could still recommend based on genre.

- Inconsistent Formatting: Issues like trailing spaces in column names and movie titles caused errors during pivoting and indexing.

- Genre Metadata: Initially missing or inconsistent genre entries led to TF-IDF failures until cleaned and standardized.
