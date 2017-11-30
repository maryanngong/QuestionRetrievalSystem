import gzip
import numpy as np
import torch
import cPickle as pickle
import pandas as pd
import os
import random


def read_train_batches_from_file():
    batches_filename = "train_batches.pkl"
    with open(batches_filename, 'rb') as f:
        batches = pickle.load(f)
        return batches

def read_test_batches_from_file():
    batches_filename = "test_batches.pkl"
    with open(batches_filename, 'rb') as f:
        batches = pickle.load(f)
        return batches

def read_dev_batches_from_file():
    batches_filename = "dev_batches.pkl"
    with open(batches_filename, 'rb') as f:
        batches = pickle.load(f)
        return batches        


def getEmbeddingTensor():
    embedding_path='../../askubuntu/vector/vectors_pruned.200.txt.gz'
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

def getIndicesTensor(text_arr, word_to_indx, max_length=1000):
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    # if len(text_indx) < max_length:
    #     text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    # x =  torch.LongTensor(text_indx)
    x = text_indx

    return x

def map_corpus(raw_corpus, embedding_layer, max_len=100):
    ids_corpus = { }
    for id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=True),
                          embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len])
        #if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item
    return ids_corpus


# taken from paper github (add proper citation)
def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        padded_titles = [ torch.from_numpy(np.pad(x,(max_title_len-len(x),0),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in titles]
        padded_bodies = [ torch.from_numpy(np.pad(x,(max_body_len-len(x),0),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in bodies]
        # l = len(padded_titles[0])
        return np.stack(padded_titles), np.stack(padded_bodies)
    else:
        padded_titles = [ torch.from_numpy(np.pad(x,(0,max_title_len-len(x)),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in titles]
        padded_bodies = [ torch.from_numpy(np.pad(x,(0,max_body_len-len(x)),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in bodies]
        # l = len(padded_titles[0])
        return torch.stack(padded_titles), torch.stack(padded_bodies)
    return titles, bodies

def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                        for x in triples ])
    return triples

# reads the file of all text query data
# returns a pandas dataframe indexed by query id, with a column of title text tokens and a column of body text tokens
def getTokenizedTextDataFrame(word_to_indx):
    all_data_path='../../askubuntu/text_tokenized.txt.gz'
    lines = []
    with gzip.open(all_data_path) as file:
        lines = file.readlines()
        file.close()

    data_dict = {'id':[], 'title':[], 'body':[]}
    for line in lines:
        query_id, title, body = line.split('\t')
        data_dict['id'].append(query_id)
        data_dict['title'].append(getIndicesTensor(title.split(), word_to_indx))
        data_dict['body'].append(getIndicesTensor(body.split(), word_to_indx))
    return pd.DataFrame(data=data_dict, index=data_dict['id'])

# reads the file of all train query ids and its simlar ids and negative ids
# returns a pandas dataframe indexed by query id, with the columns
#   - id : query id
#   - similar_ids : query ids for similar queries
#   - negative_ids: query ids for negative (dissimilar) randomly drawn queries
#   - candidate_ids: the union between similar_ids and negative_ids
def getTrainingDataIds():
    train_data_path='../../askubuntu/train_random.txt'
    lines = []
    with open(train_data_path) as f:
        lines = f.readlines()

    data_dict = {'id':[], 'similar_ids':[], 'negative_ids':[], 'candidate_ids':[]}
    for line in lines:
        query_id, similar_ids, negative_ids = line.split('\t')
        data_dict['id'].append(query_id)
        data_dict['similar_ids'].append(similar_ids.split())
        data_dict['negative_ids'].append(negative_ids.split())
        data_dict['candidate_ids'].append(list(set(data_dict['similar_ids'][-1] + data_dict['negative_ids'][-1])))
    return pd.DataFrame(data=data_dict, index=data_dict['id'])

# reads dev.txt or test.txt file for dev or test set query ids and their similar query ids and their set of total candidate ids
# returns pandas dataframe indexed by query id with the columns
#   - id: query id
#   - similar_ids: query ids for similar queries
#   - candidate_ids: query ids for all queries in the candidate set (similar ids is a subset of candidate_ids)
def getDevTestDataIds(filename):
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    data_dict = {'id':[], 'similar_ids':[], 'candidate_ids':[], 'BM25':[]}
    for line in lines:
        query_id, similar_ids, candidate_ids, scores = line.split('\t')
        data_dict['id'].append(query_id)
        data_dict['similar_ids'].append(similar_ids.split())
        data_dict['candidate_ids'].append(candidate_ids.split())
        data_dict['BM25'].append(scores.split())
    return pd.DataFrame(data=data_dict, index=data_dict['id'])


# build dataset
def load_dataset():
    print("\nLoading data...")
    embeddings, word_to_indx = getEmbeddingTensor()
    embedding_dim = embeddings.shape[1]
    print "embedding_dim", embedding_dim

    # df_fname = "all_data.csv"
    # dev_fname = "dev_ids.csv"
    # train_fname = "train_ids.csv"
    # test_fname = "test_ids.csv"
    # if os.path.exists(df_fname) and os.path.exists(dev_fname) and os.path.exists(train_fname) and os.path.exists(test_fname):
    #     # load these
    #     train = pd.read_csv(train_fname)
    #     dev = pd.read_csv(dev_fname)
    #     test = pd.read_csv(test_fname)
    #     dataframe = pd.read_csv(df_fname)
    # else:
    dataframe = getTokenizedTextDataFrame(word_to_indx)
    train = getTrainingDataIds()
    dev = getDevTestDataIds('../../askubuntu/dev.txt')
    test = getDevTestDataIds('../../askubuntu/test.txt')

        # # save for easy access later
        # dataframe.to_csv(df_fname)
        # train.to_csv(train_fname)
        # dev.to_csv(dev_fname)
        # test.to_csv(test_fname)

    return train, dev, test, dataframe, embeddings

# Dataset class
# On intiialization this dataset loads the data from file
# To access the training, dev, or test sets of data, call the following functions:
#   get_train_data()
#   get_dev_data()
#   get_test_data()
# The first time these funcitons are ever called, they will need to parse the data into the correct format before returning, so it may take a while
# but after parsing, it writes to file and all subsequent calls can just read from file (much faster)
# Train data will be returned in the format of a pandas dataframe indexed by query id with the following columns:
#   - id : query id
#   - query: list of query text tokens
#   - similars: list of lists of similary query text tokens
#   - negatives: list of lists of negative (not similar) query text tokens
#   - similar_ids: lists of similar query ids
#   - negative_ids: list of negative query ids
# Dev and test data will be returned in the format of a pandas dataframe indexed by query id with the following columns:
#   - id : query id
#   - query: list of query text tokens
#   - similars: list of lists of similary query text tokens
#   - candidates: list of lists of candidate query text tokens
#   - similar_ids: lists of similar query ids
#   - candidate_ids: list of negative query ids
#   - BM25: list of bm25 scores of candidate queries
class Dataset():
    def __init__(self, batch_size=32):
        trainIds, devIds, testIds, allData, embeddings = load_dataset()
        self.train_batches_filename = "train_batches.pkl"
        self.dev_batches_filename = "dev_batches.pkl"
        self.test_batches_filename = "test_batches.pkl"
        self.allData = allData
        self.trainIds = trainIds
        self.devIds = devIds
        self.testIds = testIds
        self.embeddings = embeddings
        self.trainData = None
        self.devData = None
        self.testData = None
        self.padding_id = 0 # get padding id from embeddings?
        self.pad_left = False
        self.batch_size = batch_size

        self.get_train_data()
        self.get_test_data()
        self.get_dev_data()

    def get_embeddings(self):
        return self.embeddings

    def get_question_from_id(self, query_id):
        query = self.allData.loc[query_id]
        return query['title'], query['body']

    def convertGroupings(self, query_group, data_type='train'):
        title, body = self.get_question_from_id(query_group['id'])
        query_group['title'] = title
        query_group['body'] = body
        similars = []
        for sim_id in query_group['similar_ids']:
            # print "sim-id", sim_id
            similars.append(self.get_question_from_id(sim_id))
        query_group['similars'] = similars
        if data_type=='train':
            negatives = []
            for neg_id in query_group['negative_ids']:
                negatives.append(self.get_question_from_id(neg_id))
            query_group['negatives'] = negatives
            # return query, similars, negatives
        else:
            negatives = []
            for neg_id in query_group['candidate_ids']:
                negatives.append(self.get_question_from_id(neg_id))
            query_group['candidates'] = negatives
        return query_group

    def get_train_batches(self, perm=None):
        batches_filename = self.train_batches_filename
        # if os.path.exists(batches_filename):
        #     print("reading train batches from file...")
        #     with open(batches_filename, 'rb') as f:
        #         batches = pickle.load(f)
        #     return batches
        data = self.trainData
        if perm is None:
            perm = range(len(data))
            # random.shuffle(perm)

        N = len(data)
        cnt = 0
        id_to_index = {}
        titles = [ ]
        bodies = [ ]
        triples = [ ]
        batches = [ ]
        for u in xrange(N):
            i = perm[u]
            pid = data.iloc[i]['id']
            if pid not in id_to_index:
                id_to_index[pid] = len(titles)
                titles.append(data.iloc[i]['title'])
                bodies.append(data.iloc[i]['body'])
            positive_ids = data.iloc[i]['similar_ids']
            negative_ids = data.iloc[i]['negative_ids']
            positive_text_tokens = data.iloc[i]['similars']
            negative_text_tokens = data.iloc[i]['negatives']
            # qids = data.iloc[i]['candidate_ids'] qlabels = data.iloc[i]
            # if pid not in ids_corpus: continue
            cnt += 1
            # print "positive_ids", type(positive_ids)
            # print len(positive_ids)
            for j,id in enumerate(positive_ids):# + negative_ids:
                if id not in id_to_index:
                    # if id not in ids_corpus: continue
                    id_to_index[id] = len(titles)
                    title = positive_text_tokens[0][j]
                    body = positive_text_tokens[1][j]
                    titles.append(title)
                    bodies.append(body)

            for j,id in enumerate(negative_ids):# + negative_ids:
                if id not in id_to_index:
                    # if id not in ids_corpus: continue
                    id_to_index[id] = len(titles)
                    title = negative_text_tokens[0][j]
                    body = negative_text_tokens[1][j]
                    titles.append(title)
                    bodies.append(body)

            p_index = id_to_index[pid]
            positive_indices = [id_to_index[p] for p in positive_ids]
            negative_indices = [id_to_index[p] for p in negative_ids]

            # pid = pid2id[pid]
            # pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
            # neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
            triples += [ [p_index, x] + negative_indices for x in positive_indices ]

            padding_id = self.padding_id
            pad_left = self.pad_left
            if cnt == self.batch_size or u == N-1:
                # assert len(titles) == len(bodies)
                # assert max(id_to_index.values()) <= len(titles)
                titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
                triples = create_hinge_batch(triples)
                batches.append((titles, bodies, triples))
                titles = [ ]
                bodies = [ ]
                triples = [ ]
                pid2id = {}
                id_to_index = {}
                cnt = 0
        # with open(batches_filename, 'wb') as f:
        #     print("pickle dumping train batches...")
        #     pickle.dump(batches, f)
        return batches

    def create_eval_batches(self, data_set):
        if "test" in data_set:
            batches_filename = self.test_batches_filename
        else:
            batches_filename = self.dev_batches_filename
        # if os.path.exists(batches_filename):
        #     print("reading batches from file...")
        #     with open(batches_filename, 'rb') as f:
        #         batches = pickle.load(f)
        #     return batches
        if "test" in data_set:
            data = self.testData
        else:
            data = self.devData
        padding_id = self.padding_id
        pad_left = self.pad_left
        lst = [ ]
        for i in range(len(data)):
            titles = [ ]
            bodies = [ ]
            pid = data.iloc[i]['id']
            titles.append(data.iloc[i]['title'])
            bodies.append(data.iloc[i]['body'])
            positive_ids_set = set(data.iloc[i]['similar_ids'])
            candidate_ids = data.iloc[i]['candidate_ids']
            candidate_text_tokens = data.iloc[i]['candidates']
            qlabels = [int(c_id in positive_ids_set) for c_id in candidate_ids]

            titles, bodies = candidate_text_tokens

            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
        # with open(batches_filename, 'wb') as f:
        #     pickle.dump(lst, f)
        return lst

    def get_dev_batches(self, perm=None):
        batches_filename = self.dev_batches_filename
        # if os.path.exists(batches_filename):
        #     print("reading train batches from file...")
        #     with open(batches_filename, 'rb') as f:
        #         batches = pickle.load(f)
        #     return batches
        data = self.devData
        if perm is None:
            perm = range(len(data))
            # random.shuffle(perm)

        N = len(data)
        cnt = 0
        id_to_index = {}
        titles = [ ]
        bodies = [ ]
        triples = [ ]
        batches = [ ]
        for u in xrange(N):
            i = perm[u]
            pid = data.iloc[i]['id']
            if pid not in id_to_index:
                id_to_index[pid] = len(titles)
                titles.append(data.iloc[i]['title'])
                bodies.append(data.iloc[i]['body'])
            positive_ids = data.iloc[i]['similar_ids']
            positive_ids_set = set(positive_ids)
            candidate_ids = data.iloc[i]['candidate_ids']
            candidate_text_tokens = data.iloc[i]['candidates']
            negative_ids = []

            # negative_ids = data.iloc[i]['negative_ids']
            # positive_text_tokens = data.iloc[i]['similars']
            # negative_text_tokens = data.iloc[i]['negatives']
            # qids = data.iloc[i]['candidate_ids'] qlabels = data.iloc[i]
            # if pid not in ids_corpus: continue
            cnt += 1
            # print "positive_ids", type(positive_ids)
            # print len(positive_ids)
            for j,id in enumerate(candidate_ids):# + negative_ids:
                if id not in id_to_index:
                    # if id not in ids_corpus: continue
                    id_to_index[id] = len(titles)
                    title = candidate_text_tokens[0][j]
                    body = candidate_text_tokens[1][j]
                    titles.append(title)
                    bodies.append(body)

                    if id not in positive_ids_set:
                        negative_ids.append(id)

            # for j,id in enumerate(negative_ids):# + negative_ids:
            #     if id not in id_to_index:
            #         # if id not in ids_corpus: continue
            #         id_to_index[id] = len(titles)
            #         title = negative_text_tokens[0][j]
            #         body = negative_text_tokens[1][j]
            #         titles.append(title)
            #         bodies.append(body)

            p_index = id_to_index[pid]
            positive_indices = [id_to_index[p] for p in positive_ids]
            negative_indices = [id_to_index[p] for p in negative_ids]

            # pid = pid2id[pid]
            # pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
            # neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
            triples += [ [p_index, x] + negative_indices for x in positive_indices ]

            padding_id = self.padding_id
            pad_left = self.pad_left
            if cnt == self.batch_size or u == N-1:
                # assert len(titles) == len(bodies)
                # assert max(id_to_index.values()) <= len(titles)
                titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
                triples = create_hinge_batch(triples)
                batches.append((titles, bodies, triples))
                titles = [ ]
                bodies = [ ]
                triples = [ ]
                pid2id = {}
                id_to_index = {}
                cnt = 0
        # with open(batches_filename, 'wb') as f:
        #     print("pickle dumping train batches...")
        #     pickle.dump(batches, f)
        return batches

    def get_train_data(self):
        train_data_file = "train_embedded_dataframe.pkl"
        if self.trainData is not None or os.path.exists(train_data_file):
            print "reading train data..."
            train_df = pd.read_pickle(train_data_file)
            self.trainData = train_df
            return self.trainData

        print "generating train data..."
        queries = []
        similar_groups = []
        negative_groups = []
        train_df = self.trainIds.groupby(['id']).apply(self.convertGroupings, 'train')
        # save it for later!
        train_df.to_pickle(train_data_file)
        self.trainData = train_df
        return train_df

    def get_dev_data(self):
        dev_data_file = "dev_embedded_dataframe.pkl"
        if self.devData is not None or os.path.exists(dev_data_file):
            print "reading dev data..."
            dev_df = pd.read_pickle(dev_data_file)
            self.devData = dev_df
            return self.devData

        print "generating dev data..."
        queries = []
        similar_groups = []
        negative_groups = []
        dev_df = self.devIds.groupby(['id']).apply(self.convertGroupings, 'dev')
        dev_df.to_pickle(dev_data_file)
        self.devData = dev_df
        return dev_df

    def get_test_data(self):
        test_data_file = "test_embedded_dataframe.pkl"
        if self.testData is not None or os.path.exists(test_data_file):
            test_df = pd.read_pickle(test_data_file)
            self.testData = test_df
            return self.testData

        print "generating test data..."
        queries = []
        similar_groups = []
        negative_groups = []
        test_df = self.testIds.groupby(['id']).apply(self.convertGroupings, 'test')
        test_df.to_pickle(test_data_file)
        self.testData = test_df
        return test_df

if __name__ == '__main__':
    dataset = Dataset()
    train = dataset.get_train_data()
    dev = dataset.get_dev_data()
    test = dataset.get_test_data()
    print "train.."
    # print train.head()
    print ""
    print ""
    print "dev..."
    print dev.head()
    print ""
    print ""
    print "test..."
    print test.head()
