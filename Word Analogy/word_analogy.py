import numpy as np

def cosine_sim(u, v):
    dot_prod = np.dot(u, v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    sim = dot_prod/ (u_norm * v_norm)
    return sim

def read_embeddings():
    file = open('data/glove.6B.50d.txt', 'r', encoding="utf-8")
    glove_emb = {}
    for line in file.readlines():
        word = line.strip().split()[0]
        emb = line.split()[1:]
        glove_emb[word] = np.array(emb, dtype = np.float64)
    file.close()
    return glove_emb


if __name__ == '__main__':
    str = input(' enter 3 words separated by space').split(' ')
    a = str[0].lower()
    b = str[1].lower()
    c = str[2].lower()

    embedding = read_embeddings()
    words = list(embedding)

    a_emb = embedding[a]
    b_emb = embedding[b]
    c_emb = embedding[c]
    u = a_emb - b_emb

    best_word = None
    dist = -999
    for d in words:
        if d in [a,b,c]:
            continue
        cos_sim = cosine_sim(u, c_emb - embedding[d])
        if dist < cos_sim:
            dist = cos_sim
            best_word = d

    print('{0} is to {1}'.format(a, b))
    print('as')
    print('{0} is to {1}'.format(c, best_word))