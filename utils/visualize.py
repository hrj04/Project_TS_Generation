import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from einops import reduce
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_pca(ori_data, fake_data, n_sample=3000, savefig=None):
    n_data = len(ori_data)
    n_sample = min(n_sample, n_data)
    idx = np.random.permutation(n_data)[:n_sample]
    
    # Data preprocessing
    ori_data, fake_data = ori_data[idx], fake_data[idx]
    ori_data = reduce(ori_data, "batch seq_len feature_dim -> batch seq_len", "mean")
    fake_data = reduce(fake_data, "batch seq_len feature_dim -> batch seq_len", "mean")

    # PCA Analysis
    concat_data = np.concatenate([ori_data, fake_data], axis=0)
    pca = PCA(n_components=2)
    pca.fit(concat_data)
    pca_ori = pca.transform(ori_data)
    pca_fake = pca.transform(fake_data)

    # Plotting
    plt.figure()
    plt.scatter(pca_ori[:, 0], 
                pca_ori[:, 1], 
                c="red", 
                alpha=0.2, 
                label="Original")
    plt.scatter(pca_fake[:, 0], 
                pca_fake[:, 1], 
                c="blue", 
                alpha=0.2, 
                label="Synthetic")
    plt.legend()
    plt.title('PCA Comparison of Original and Synthetic Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def visualize_tsne(ori_data, fake_data, n_sample=3000, savefig=None):
    n_data = len(ori_data)
    n_sample = min(n_sample, n_data)
    idx = np.random.permutation(n_data)[:n_sample]
    
    # Data preprocessing
    ori_data, fake_data = ori_data[idx], fake_data[idx]
    ori_data = reduce(ori_data, "batch seq_len feature_dim -> batch seq_len", "mean")
    fake_data = reduce(fake_data, "batch seq_len feature_dim -> batch seq_len", "mean")

    # TSNE Anlaysis
    concat_data = np.concatenate((ori_data, fake_data), axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(concat_data)

    # Plotting
    plt.figure()
    plt.scatter(tsne_results[:n_sample, 0], 
                tsne_results[:n_sample, 1],
                c="red", 
                alpha=0.2, 
                label="Original")
    plt.scatter(tsne_results[n_sample:, 0], 
                tsne_results[n_sample:, 1],
                c="blue", 
                alpha=0.2, 
                label="Synthetic")
    plt.legend()
    plt.title('t-SNE Comparison of Original and Synthetic Data')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')

    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def visualize_kernel(ori_data, fake_data, n_sample=3000, savefig=None):
    n_data = len(ori_data)
    n_sample = min([n_sample, n_data])
    idx = np.random.permutation(n_data)[:n_sample]
    colors_ori, colors_fake = ["red"]*n_sample, ["blue"]*n_sample
    
    # data preprocessing
    ori_data, fake_data = ori_data[idx], fake_data[idx]
    ori_data = reduce(ori_data, "batch seq_len feature_dim -> batch seq_len", "mean")
    fake_data = reduce(fake_data, "batch seq_len feature_dim -> batch seq_len", "mean")

    # Visualization parameter
    f, ax = plt.subplots(1)
    sns.distplot(ori_data, hist=False, kde=True, kde_kws={'linewidth': 5}, label='Original', color="red")
    sns.distplot(fake_data, hist=False, kde=True, kde_kws={'linewidth': 5, 'linestyle':'--'}, label='Synthetic', color="blue")
    # Plot formatting

    # plt.legend(prop={'size': 22})
    plt.legend()
    plt.xlabel('Data Value')
    plt.ylabel('Data Density Estimate')
    # plt.rcParams['pdf.fonttype'] = 42

    # plt.ylim((0, 12))
    plt.show()
    

    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)
