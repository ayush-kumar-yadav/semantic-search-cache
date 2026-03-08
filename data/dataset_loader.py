from sklearn.datasets import fetch_20newsgroups


def load_dataset():

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data

    cleaned_docs = []

    for doc in documents:
        doc = doc.strip()

        if len(doc) > 50:
            cleaned_docs.append(doc)

    print("Total cleaned documents:", len(cleaned_docs))

    return cleaned_docs