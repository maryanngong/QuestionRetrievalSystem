import gzip
import numpy as np
import torch
import cPickle as pickle
import pandas as pd
import os
import random

# Helper function that constructs embedding tensor and word_to_indx mapping
# this code is adapted from Adam Yala's example project github
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

# Helper function that constructs and index tensor given the list of text tokens
def getIndicesTensor(text_arr, word_to_indx, max_length=100):
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    x = text_indx
    return x

# Creates and pads one batch of data ready for training
# Parameters:
#   - titles: list of title index tokens
#   - bodies: list of body index tokens
#   - padding_id: index to use for padding
#   - pad_left: boolean flag whether to pad on left or right of sequence
#   - cnn: boolean flag whether this batch is to be used in cnn, if so, then the masking layer is constructed differently
# Returns:
#   - titles tensor that has been padded to same size
#   - bodies tensor that has been padded to same size but truncated at 100 sequence length
#   - masking matrix for title
#   - masking matrix for body
def create_one_batch(titles, bodies, padding_id, pad_left, cnn=False):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))

    if cnn:
        mask_title = torch.zeros(len(titles), max_title_len-2)
        mask_body = torch.zeros(len(bodies), max_body_len-2)
    else:
        mask_title = torch.zeros(len(titles), max_title_len)
        mask_body = torch.zeros(len(bodies), max_body_len)
    for i,x in enumerate(titles):
        if len(x) > 0:
            l = len(x)
            # removing this because they still use part of the info from non padded,
            # also getting error result of slicing is empty tensor error
            # if cnn:
            #     l -= 1
            mask_title[i,:l] = 1.0 / max(1,l)
    for i,x in enumerate(bodies):
        if len(x) > 0:
            l = len(x)
            # if cnn:
            #     l -= 1
            mask_body[i,:l] = 1.0 / max(1,l)
    if pad_left:
        padded_titles = [ torch.from_numpy(np.pad(x,(max_title_len-len(x),0),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in titles]
        padded_bodies = [ torch.from_numpy(np.pad(x,(max_body_len-len(x),0),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in bodies]
        return np.stack(padded_titles), np.stack(padded_bodies), mask_title, mask_body
    else:
        padded_titles = [ torch.from_numpy(np.pad(x,(0,max_title_len-len(x)),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in titles]
        padded_bodies = [ torch.from_numpy(np.pad(x,(0,max_body_len-len(x)),'constant',
                                constant_values=padding_id).astype(np.int64)) for x in bodies]
        return torch.stack(padded_titles), torch.stack(padded_bodies), mask_title, mask_body

# formats training ids and pads by repeating the last question id until it meets the max length so that all
# training groups have the same number of samples wehn the flag train is set to true
# otherwise, these training ids are not padded and are allowed to be different lengths
def create_hinge_batch(triples, train=True):
    if train:
        max_len = max(len(x) for x in triples)
        triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                            for x in triples ])
    else:
        triples = triples
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

    dataframe = getTokenizedTextDataFrame(word_to_indx)
    train = getTrainingDataIds()
    dev = getDevTestDataIds('../../askubuntu/dev.txt')
    test = getDevTestDataIds('../../askubuntu/test.txt')

    return train, dev, test, dataframe, embeddings

# Dataset class
# On intiialization this dataset loads the data from file
# To access the training data in batches form, call:
#   get_train_batches()
# To access the training data, dev data, or test data for evaluation, call:
#   create_eval_batches_train()
#   create_eval_batches("dev")
#   create_eval_batches("test")
#
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
    def __init__(self, batch_size=32, debug_mode=False, cnn=False):
        trainIds, devIds, testIds, allData, embeddings = load_dataset()
        self.train_batches_filename = "train_batches_2.pkl"
        self.dev_batches_filename = "dev_batches_2.pkl"
        self.test_batches_filename = "test_batches_2.pkl"
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
        self.limit=20
        self.batch_size = batch_size
        self.train_batches = None
        self.dev_batches = None
        self.test_batches = None
        self.cnn = cnn

        self.get_train_data()
        self.get_test_data()
        self.get_dev_data()

    def get_embeddings(self):
        return self.embeddings

    # given question id, returns the title text tokens and body text tokens
    def get_question_from_id(self, query_id):
        query = self.allData.loc[query_id]
        return query['title'], query['body']

    # helper function to construct minibatch of questions, mapping question ids to actual text tokens
    def convertGroupings(self, query_group, data_type='train'):
        title, body = self.get_question_from_id(query_group['id'])
        query_group['title'] = title
        query_group['body'] = body
        similars = []
        for sim_id in query_group['similar_ids']:
            similars.append(self.get_question_from_id(sim_id))
        query_group['similars'] = similars
        if data_type=='train':
            negatives = []
            for neg_id in query_group['negative_ids']:
                negatives.append(self.get_question_from_id(neg_id))
            query_group['negatives'] = negatives
        else:
            negatives = []
            for neg_id in query_group['candidate_ids']:
                negatives.append(self.get_question_from_id(neg_id))
            query_group['candidates'] = negatives
        return query_group

    # This function is adapted from Tao Lei's github
    # Constructs batches of data for training of model
    # Permutes the data so that each call to get_train_batches will return different question groups.
    # Formats the data into groups of self.batch_size, where each group has self.batch_size minibatches
    # Returns list of batches
    #   - each batch consists of a tuple of (titles, bodies, triples, t_mask, b_mask)
    #       - titles: titles of all questions in batch
    #       - bodies: bodies of all questions in batch
    #       - triples: training group ids organized into minibatches
    #       - t_mask: masking matrix for titles
    #       - b_mask: masking matrix for bodies
    def get_train_batches(self):
        ### This code below is for saving train batches to file if you want to use the same one always ####
        # if self.train_batches is not None:
        #     print("returning train batches...")
        #     return self.train_batches
        # batches_filename = self.train_batches_filename
        # if os.path.exists(batches_filename):
        #     print("reading train batches from file...")
        #     with open(batches_filename, 'rb') as f:
        #         batches = pickle.load(f)
        #     return batches
        data = self.trainData
        # generate permutation of data order
        perm = range(len(data))
        random.shuffle(perm)

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
            cnt += 1

            for j,id in enumerate(positive_ids):
                if id not in id_to_index:
                    id_to_index[id] = len(titles)
                    title = positive_text_tokens[0][j]
                    body = positive_text_tokens[1][j]
                    titles.append(title)
                    bodies.append(body)

            for j,id in enumerate(negative_ids):
                if id not in id_to_index:
                    id_to_index[id] = len(titles)
                    title = negative_text_tokens[0][j]
                    body = negative_text_tokens[1][j]
                    titles.append(title)
                    bodies.append(body)

            p_index = id_to_index[pid]
            positive_indices = [id_to_index[p] for p in positive_ids]
            negative_indices = [id_to_index[p] for p in negative_ids]

            triples += [ [p_index, x] + random.sample(negative_indices, self.limit) for x in positive_indices ]

            padding_id = self.padding_id
            pad_left = self.pad_left
            if cnt == self.batch_size or u == N-1:
                titles, bodies, t_mask, b_mask = create_one_batch(titles, bodies, padding_id, pad_left, cnn=self.cnn)
                triples = create_hinge_batch(triples)
                batches.append((titles, bodies, triples, t_mask, b_mask))
                titles = [ ]
                bodies = [ ]
                triples = [ ]
                pid2id = {}
                id_to_index = {}
                cnt = 0
        return batches

    # This function is adapted from Tao Lei's github
    # Constructs batches of data for evaluation of model using the training data
    # Formats the data into groups of self.batch_size, where each group has self.batch_size minibatches
    # Uses only a subset of the training data, truncates to the first 100 samples, and only 30 questions for minibatch group
    # Returns batches of one minibatch each. Each minibatch begins with the query question, followed by the candidate questions
    #   - each minibatch consists of a tuple of (titles, bodies, qlabels, t_mask, b_mask)
    #       - titles: titles of all questions in batch
    #       - bodies: bodies of all questions in batch
    #       - qlabels: indicator representing whether the aligned question is positive (1) or negative (0). (-1) is used for the query question 
    #       - t_mask: masking matrix for titles
    #       - b_mask: masking matrix for bodies
    def create_eval_batches_train(self):
        data = self.trainData
        padding_id = self.padding_id
        pad_left = self.pad_left
        lst = [ ]
        l = min(100, len(data))
        for i in range(l):
            titles = [ ]
            bodies = [ ]
            pid = data.iloc[i]['id']
            titles.append(data.iloc[i]['title'])
            bodies.append(data.iloc[i]['body'])
            positive_ids_set = set(data.iloc[i]['similar_ids'])
            positive_ids = data.iloc[i]['similar_ids']
            negative_ids = data.iloc[i]['negative_ids']
            positive_text_tokens = data.iloc[i]['similars']
            negative_text_tokens = data.iloc[i]['negatives']
            qlabels = [-1] + [1 for j in positive_ids] + [0 for j in negative_ids]
            qlabels = qlabels[:30]
            for k in range(len(positive_ids)):
                if len(titles) >= 30:
                    break
                titles.append(positive_text_tokens[0][k])
                bodies.append(positive_text_tokens[1][k])

            for k in range(len(negative_ids)):
                if len(titles) >= 30:
                    break
                titles.append(negative_text_tokens[0][k])
                bodies.append(negative_text_tokens[1][k])
            titles, bodies, t_mask, b_mask = create_one_batch(titles, bodies, padding_id, pad_left, cnn=self.cnn)
            lst.append((titles, bodies, np.array(qlabels, dtype="int32"), t_mask, b_mask))
        return lst

    # This function is adapted from Tao Lei's github
    # Constructs batches of data for evaluation of model using either dev or test data
    # Formats the data into groups of self.batch_size, where each group has self.batch_size minibatches
    # Parameters:
    #   - data_set: string representing dataset type "test" or "dev"
    # Returns batches of one minibatch each. Each minibatch begins with the query question, followed by the candidate questions
    #   - each minibatch consists of a tuple of (titles, bodies, qlabels, t_mask, b_mask)
    #       - titles: titles of all questions in batch
    #       - bodies: bodies of all questions in batch
    #       - qlabels: indicator representing whether the aligned question is positive (1) or negative (0). (-1) is used for the query question 
    #       - t_mask: masking matrix for titles
    #       - b_mask: masking matrix for bodies
    def create_eval_batches(self, data_set):
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
            qlabels = [-1] + [int(c_id in positive_ids_set) for c_id in candidate_ids]
            for j in range(len(candidate_ids)):
                titles.append(candidate_text_tokens[0][j])
                bodies.append(candidate_text_tokens[1][j])
            titles, bodies, t_mask, b_mask = create_one_batch(titles, bodies, padding_id, pad_left, cnn=self.cnn)
            lst.append((titles, bodies, np.array(qlabels, dtype="int32"), t_mask, b_mask))
        return lst

    # Helper function for fetching the embedded training data dataframe
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

    # Helper function for fetching the embedded dev data dataframe
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

    # Helper function for fetching the embedded test data dataframe
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
    dev = dataset.create_eval_batches('dev')

