
from vector_store import VectorStore

def main():
    index_store = "./new/db/expl/texts.index"
    tweet_store = "./new/db/expl/texts.npy"
    explanation_store = "./new/db/expl/explanations.npy"
    sentiment_store = "./new/db/expl/sentiments.npy"
    vs = VectorStore(index_store=index_store, tweet_store=tweet_store, explanation_store=explanation_store, sentiment_store=sentiment_store)
    
    print(f"Length of vector store is {len(vs)}")
    query = "job cuts due to coronavirus"
    
    distance, expl, tweet, sentiments, _, _ = vs.find_nearest(query)
    print(distance)
    print(expl)
    print(tweet)
    print(sentiments)


        
if __name__ == "__main__":
    main()