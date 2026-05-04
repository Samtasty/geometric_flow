import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embeddings(model, dataset, emb_type='item', n_components=2):
    if emb_type == 'item':
        emb = model.item_emb.weight.detach().cpu().numpy()
        labels = dataset.items
        title = 'Item Embeddings'
    else:
        emb = model.student_emb.weight.detach().cpu().numpy()
        labels = dataset.students
        title = 'Student Embeddings'
    if emb.shape[1] > n_components:
        emb = PCA(n_components=n_components).fit_transform(emb)
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], alpha=0.7)
    for i, label in enumerate(labels):
        plt.text(emb[i,0], emb[i,1], str(label), fontsize=8, alpha=0.6)
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.show()
