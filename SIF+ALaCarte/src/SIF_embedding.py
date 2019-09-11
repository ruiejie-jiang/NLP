import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        set =  [We[k] for k in x[i]]
        emb[i] = np.dot(w[i], set) / np.sum(w[i])
    return emb


def get_weighted_average_rare(We, x, w, win, A):
    """
    Compute the weighted average vectors
    :param A the linear regression model which has been trained to predict rare-words embedding
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        set = list()
        for j in range(len(x[i])):
            if x[i][j] != len(We) - 1:
                set.append(We[x[i][j]])
            else:
                if j > win - 1 and j < len(x[i]) - win:
                    num = 2 * win
                    #window size, used to rescale
                    averg = (np.sum(We[np.array(x[i][j - win:j + win + 1])], axis=0) - We[x[i][j]]) / num
                    act = A.predict([averg])
                    set.append(act.reshape(300,))
                elif j > win-1:
                    num = len(x[i]) - j + win - 1
                    averg = (np.sum(We[np.array(x[i][j - win:])], axis=0) - We[x[i][j]]) / num
                    act = A.predict([averg])
                    set.append(act.reshape(300,))
                elif j < len(x[i]) - win and j < win:
                    num = win + j
                    averg = (np.sum(We[np.array(x[i][:j + win + 1])], axis=0) - We[x[i][j]]) / num
                    act = A.predict([averg])
                    set.append(act.reshape(300,))
                else:
                    num = len(x[i])
                    averg = (np.sum(We[np.array(x[i][:])], axis=0) - We[x[i][j]]) / num
                    act = A.predict([averg])
                    set.append(act.reshape(300,))

        emb[i] = np.dot(w[i], set) / np.count_nonzero(w[i])

    return emb


def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if  params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb

def SIF_embedding_pc(We, x, w):
    """
    Compute the scores between pairs of sentences using weighted average
    don't remove the projection on the first principal component
    """
    emb = get_weighted_average(We, x, w)

    return emb

def SIF_embedding_alac(We, x, w, win, A, params):
    """
    Compute the scores between pairs of sentences using weighted average
    :param win: the length of the window which is used to predict the rare-words embedding
    :param A: the linear regression model which has been trained to predict rare-words embedding
    """
    emb = get_weighted_average_rare(We, x, w, win, A)
    if  params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb
