import gzip
import numpy as np
import torch
import cPickle as pickle
import pandas as pd 
import os


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


# reads the file of all text query data
# returns a pandas dataframe indexed by query id, with a column of title text tokens and a column of body text tokens
def getTokenizedTextDataFrame():
    all_data_path='../../askubuntu/text_tokenized.txt.gz'
    lines = []
    with gzip.open(all_data_path) as file:
        lines = file.readlines()
        file.close()

    data_dict = {'id':[], 'title':[], 'body':[]}
    for line in lines:
        query_id, title, body = line.split('\t')
        data_dict['id'].append(query_id)
        data_dict['title'].append(title.split())
        data_dict['body'].append(body.split())
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
    dataframe = getTokenizedTextDataFrame()
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
    def __init__(self):
        trainIds, devIds, testIds, allData, embeddings = load_dataset()
        self.allData = allData
        self.trainIds = trainIds
        self.devIds = devIds
        self.testIds = testIds
        self.embeddings = embeddings
        self.trainData = None
        self.devData = None
        self.testData = None

    def get_embeddings(self):
        return self.embeddings

    def get_question_from_id(self, query_id):
        query = self.allData.loc[query_id]
        return query['title'] + query['body']

    def convertGroupings(self, query_group, data_type='train'):
        query = self.get_question_from_id(query_group['id'])
        query_group['query'] = query
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


    def get_train_data(self):
        train_data_file = "train_dataframe.csv"
        if self.trainData or os.path.exists(train_data_file):
            print "reading train data..."
            train_df = pd.read_csv(train_data_file)
            self.trainData = train_df
            return self.trainData

        print "generating train data..."
        queries = []
        similar_groups = []
        negative_groups = []
        train_df = self.trainIds.groupby(['id']).apply(self.convertGroupings, 'train')
        # save it for later!
        train_df.to_csv(train_data_file)
        self.trainData = train_df
        return train_df

    def get_dev_data(self):
        dev_data_file = "dev_dataframe.csv"
        if self.devData or os.path.exists(dev_data_file):
            print "reading dev data..."
            dev_df = pd.read_csv(dev_data_file)
            self.devData = dev_df
            return self.devData
            
        print "generating dev data..."
        queries = []
        similar_groups = []
        negative_groups = []
        dev_df = self.devIds.groupby(['id']).apply(self.convertGroupings, 'dev')
        dev_df.to_csv(dev_data_file)
        self.devData = dev_df
        return dev_df

    def get_test_data(self):
        test_data_file = "test_dataframe.csv"
        if self.testData or os.path.exists(test_data_file):
            test_df = pd.read_csv(test_data_file)
            self.testData = test_df
            return self.testData
            
        print "generating test data..."
        queries = []
        similar_groups = []
        negative_groups = []
        test_df = self.testIds.groupby(['id']).apply(self.convertGroupings, 'test')
        test_df.to_csv(test_data_file)
        self.testData = test_df
        return test_df

if __name__ == '__main__':
    dataset = Dataset()
    train = dataset.get_train_data()
    dev = dataset.get_dev_data()
    test = dataset.get_test_data()
    print "train.."
    print train.head()
    print ""
    print ""
    print "dev..."
    # print dev.head()
    print ""
    print ""
    print "test..."
    # print test.head()
        