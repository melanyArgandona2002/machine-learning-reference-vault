from capstone.mpg_regression import MpgRegressionCapstone
from capstone.tumor_classification import TumorClassificationCapstone
from capstone.text_embedding import TextEmbedding

def main() -> None:
    # Run the capstone projects
    mpg_capstone = MpgRegressionCapstone()
    mpg_capstone.run()

    tumor_capstone = TumorClassificationCapstone()
    tumor_capstone.run()

    # Create an instance of TextEmbedding
    text_embedding = TextEmbedding()

    # Define your texts to compare
    samples = [
        "In other words, hopefully, a model that has neither high bias nor high variance...\n",
        "Notice how some of these crosses get classified among the circles...\n",
        "ties, all distinct, the treap associated with these nodes is unique...\n",
    ]

    # Compare the texts and get similarities
    similarities, embeddings = text_embedding.compare_samples(samples)
    
    print("Embeddings:", embeddings)
    print("Cosine Similarities:", similarities)

if __name__ == "__main__":
    main()
