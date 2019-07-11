import pickle
from collections import Counter

from matplotlib.figure import Figure, Axes
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from wikipedia import loadf, dumpf, load_iterable


def main():
    user_sets = list(x[1] for x in load_iterable(
        './wiktionary_article_users_map.p'))

    print(len(user_sets))

    user_sets = [Counter(s) for s in user_sets]

    vectoriser = DictVectorizer()
    group_vectors = vectoriser.fit_transform(user_sets)
    users = vectoriser.feature_names_
    print(len(users))

    # cross_correlation(group_vectors)

    user_embeddings_pipeline = make_user_embeddings_pipeline()
    user_embeddings = user_embeddings_pipeline.fit_transform(
        group_vectors.T)
    del group_vectors
    dumpf('wiktionary_user_embeddings.p', user_embeddings)
    user_embeddings = loadf('wiktionary_user_embeddings.p')

    # list_nn(user_embeddings, users)

    users_2d = user_2d_clustering(user_embeddings)
    del user_embeddings
    figure = plt.figure(figsize=[20, 20])
    plot_2d_users(figure, users_2d)
    plt.show()


def make_user_embeddings_pipeline(n_components=50):
    return Pipeline([
        ('scale', StandardScaler(with_mean=False)),
        ('svd', TruncatedSVD(n_components=n_components))
    ])


def cross_correlation(group_vectors):
    cc = group_vectors.T @ group_vectors
    return cc


def user_2d_clustering(group_vectors):
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('projection', TSNE(n_components=2))
    ])
    return pipeline.fit_transform(group_vectors)


def plot_2d_users(fig: Figure, reduced_vectors):
    assert reduced_vectors.shape[1] == 2
    ax: Axes = fig.add_subplot()
    ax.plot(reduced_vectors[:, 0], reduced_vectors[:, 1], 'rx')


def list_nn(vectors, labels, metric='cosine'):
    nn = NearestNeighbors(metric=metric)
    nn.fit(vectors)
    distances, related_users = nn.kneighbors(vectors)
    for i in range(len(labels)):
        label = labels[i]
        neighbors = sorted(
            zip(related_users[i, :], distances[i, :]),
            key=lambda x: x[1])
        neighbors = [(labels[j], d) for j, d in neighbors]

        print(f'{label}: {neighbors}')


def load_wiki_data():
    with open('wiki.p', 'rb') as f:
        return pickle.load(f)[1]


if __name__ == '__main__':
    main()
