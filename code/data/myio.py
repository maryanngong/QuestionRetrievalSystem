import sys
import os
import gzip
import random
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from zipfile import ZipFile
from tqdm import tqdm
import cPickle as pickle

def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()

def getEmbeddingTensor(embedding_path):
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    v_len = 0
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb ]
        v_len = len(vector)
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    print("new embedding...")
    embedding_tensor.append( np.zeros( v_len ))
    word_to_indx["<unk>"] = len(embedding_tensor) - 1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

def getGloveEmbeddingTensor(prune=False, corpuses=None):
    if prune:
        embedding_path="../data/glove.840B.300d.zip"
        embeddings_file = "../data/glove_embedding_tensor_pruned.npy"
        word_to_indx_file = "../data/glove_word_to_indx_pruned" 
        all_tokens = get_all_tokens(corpuses)
    else:       
        embedding_path="../data/glove.840B.300d.zip"
        embeddings_file = "../data/glove_embedding_tensor.npy"
        word_to_indx_file = "../data/glove_word_to_indx"
    if os.path.exists(embeddings_file) and os.path.exists(word_to_indx_file):
        print("Loading Glove embeddings from file...")
        embedding_tensor = np.load(embeddings_file)
        with open(word_to_indx_file, 'rb') as f:
            word_to_indx = pickle.load(f)
        return embedding_tensor, word_to_indx


    print("Reading Glove embeddings from zipfile. This will take a few moments...")
    with ZipFile(embedding_path) as fin:
        content = fin.read('glove.840B.300d.txt')
        lines = content.splitlines()

        embedding_tensor = []
        word_to_indx = {}
        v_len = 0
        indx = 0
        for l in tqdm(lines, total=len(lines)):
            l = l.rstrip()
            word, emb = l.split()[0], l.split()[1:]
            if prune:
                if word not in all_tokens: continue
            vector = [float(x) for x in emb ]
            v_len = len(vector)
            if indx == 0:
                embedding_tensor.append( np.zeros( len(vector) ) )
            embedding_tensor.append(vector)
            word_to_indx[word] = indx+1
            indx += 1

        embedding_tensor.append( np.zeros( v_len ))
        word_to_indx["<unk>"] = len(embedding_tensor) - 1
        embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
        print("saving embeddings to file...")
        np.save(embeddings_file, embedding_tensor)
        with open(word_to_indx_file, 'wb') as f:
            pickle.dump(word_to_indx, f)
    return embedding_tensor, word_to_indx

# raw_corpuses is a list of raw_corpus dictionary objects
def get_all_tokens(raw_corpuses):
    all_tokens = set()
    for raw_corpus in raw_corpuses:
        for pair in raw_corpus.values():
            for x in pair[0]:
                all_tokens.add(x)
                all_tokens.add(x.lower())
            for x in pair[1]:
                all_tokens.add(x)
                all_tokens.add(x.lower())
    return all_tokens


# Helper function that constructs and index tensor given the list of text tokens
def getIndicesTensor(text_arr, word_to_indx, max_length=100):
    nil_indx = word_to_indx["<unk>"]
    # lowercase the corpus now
    text_indx = []
    count_upper = 0
    count_total = 0
    count_unk = 0
    for x in text_arr:
        x_indx = nil_indx
        if x in word_to_indx:
            if x.lower() in word_to_indx:
                x_indx = word_to_indx[x.lower()]
            else:
                x_indx = word_to_indx[x]
                count_upper += 1
        else:
            count_unk += 1
        count_total += 1
        text_indx.append(x_indx)
    # print("Number of words that only exist in uppercase, unk, total:", count_upper, count_unk, count_total)
    return text_indx[:max_length]


def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print id
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus


def read_corpus_documents(path):
    '''Reads corpus but leaves titles and bodies as strings (documents) rather than lists
    '''
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print id
                empty_cnt += 1
                continue
            title = title.strip()
            body = body.strip()
            raw_corpus[id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus


def read_corpus_flat(path):
    fopen = gzip.open if path.endswith(".gz") else open
    corpus_text = []
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            corpus_text.append(title)
            corpus_text.append(body)
    return corpus_text


def map_corpus(raw_corpus, word_to_indx, max_len=100):
    ids_corpus = { }
    for id, pair in raw_corpus.iteritems():
        item = (getIndicesTensor(pair[0], word_to_indx), getIndicesTensor(pair[1], word_to_indx, max_length=max_len))
        #if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item
    return ids_corpus

def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = [ ]
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))

    return lst


# Reads annotations from txt file
# Processes each line which contains two ids
# first id - query question
# second id - corresponding neg/pos example (depending on the file type)
# Returns:
#   pid_to_qids - dictionary mapping each query id to its set of pos/neg ids.
def read_annotations_android(path):
    pid_to_qids = {}
    with open(path) as fin:
        for line in fin:
            parts = line.rstrip().split()
            pid, qid = parts[:2]
            if pid not in pid_to_qids:
                pid_to_qids[pid] = set([qid])
            else:
                pid_to_qids[pid].add(qid)
    return pid_to_qids

def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True):
    if perm is None:
        perm = range(len(data))
        random.shuffle(perm)

    N = len(data)
    cnt = 0
    pid2id = {}
    titles = [ ]
    bodies = [ ]
    triples = [ ]
    batches = [ ]
    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus: continue
        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus: continue
                pid2id[id] = len(titles)
                t, b = ids_corpus[id]
                titles.append(t)
                bodies.append(b)
        pid = pid2id[pid]
        pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
        neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
        triples += [ [pid,x]+neg for x in pos ]

        if cnt == batch_size or u == N-1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            triples = create_hinge_batch(triples)
            batches.append((titles, bodies, triples))
            titles = [ ]
            bodies = [ ]
            triples = [ ]
            pid2id = {}
            cnt = 0
    return batches

def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = [ ]
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
    return lst

def create_tfidf_batches(raw_corpus, data):
    lst = [ ]
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        for id in [pid]+qids:
            t, b = raw_corpus[id]
            titles.append(t)
            bodies.append(b)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
    return lst

# Creates batches for evaluation of the following form
# list of tuples
#    - tuples of (titles, bodies, qlabels) -> title tensor data, body tensor data, and qlabel indicator if aligned candidate is pos or neg
# Parameters:
#   - ids_corpus: map from query id to emedding indexed tensor
#   - pos_data : dictionary mapping query id to set of positive query ids
#   - neg_data: dictionary mapping query id to set of negative query ids
def create_eval_batches_android(ids_corpus, pos_data, neg_data, padding_id=0, pad_left=False):
    all_pids = set(pos_data.keys() + neg_data.keys())
    lst = []
    for pid in all_pids:
        t, b = ids_corpus[pid]
        titles = [t]
        bodies = [b]
        qlabels = []

        for qid in neg_data[pid]:
            t, b = ids_corpus[qid]
            titles.append(t)
            bodies.append(b)
            qlabels.append(0)
        for qid in pos_data[pid]:
            t, b = ids_corpus[qid]
            titles.append(t)
            bodies.append(b)
            qlabels.append(1)

        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels)))
    return lst

def create_tfidf_batches_android(raw_corpus, pos_data, neg_data):
    all_pids = set(pos_data.keys() + neg_data.keys())
    lst = []
    for pid in all_pids:
        t, b = raw_corpus[pid]
        titles = [t]
        bodies = [b]
        qlabels = []

        for qid in pos_data[pid]:
            t, b = raw_corpus[qid]
            titles.append(t)
            bodies.append(b)
            qlabels.append(1)
        for qid in neg_data[pid]:
            t, b = raw_corpus[qid]
            titles.append(t)
            bodies.append(b)
            qlabels.append(0)

        lst.append((titles, bodies, np.array(qlabels)))
    return lst

def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        padded_titles = [ torch.from_numpy(np.pad(x,(max_title_len-len(x),0),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in titles]
        padded_bodies = [ torch.from_numpy(np.pad(x,(max_body_len-len(x),0),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in bodies]
        return np.stack(padded_titles), np.stack(padded_bodies)
    else:
        padded_titles = [ torch.from_numpy(np.pad(x,(0,max_title_len-len(x)),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in titles]
        padded_bodies = [ torch.from_numpy(np.pad(x,(0,max_body_len-len(x)),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in bodies]
        return torch.stack(padded_titles), torch.stack(padded_bodies)
    return titles, bodies

def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                        for x in triples ])
    return triples

if __name__ == '__main__':
    raw_android_corpus = read_corpus('../../Android/corpus.tsv.gz')
    raw_ubuntu_corpus = read_corpus('../../askubuntu/text_tokenized.txt.gz')
    print("starting load")
    embedding_tensor, word_to_indx = getGloveEmbeddingTensor(prune=True, corpuses=[raw_ubuntu_corpus, raw_android_corpus])
    print "done"
<<<<<<< HEAD
    print len(word_to_indx)
    print word_to_indx.keys()[:10]
=======
    # ids_android_corpus = map_corpus(raw_android_corpus, word_to_indx, max_len=100)
>>>>>>> master
