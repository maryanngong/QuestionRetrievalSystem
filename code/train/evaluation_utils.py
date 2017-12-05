# Data format:
# List of lists of I(question id) e.g. ordered by rank of score
# I(qid) is an indicator function for relvance
#   [[q1 q2 ...], [q1 q2 ...], ...]

def mean_average_precision(results):
    '''Calculates the Mean of Average Precisions across a list of document retreival results

    The MAP is the mean of average precisions. The Average Prevision for each result is calculated as:
        AP = Sum from k=1...n (Precision@k * I(k)) / (# of documents)
        where I() = 1 if that query was relevant, 0 otherwise

    Args:
        results list(list(1|0)): list of each result where an entry in result is 1 if query i is relevant, 0 otherwise

    Returns:
        Mean Average Precision (scalar)

    >>> mean_average_precision([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
    0.7222222222222222
    '''
    map_terms = []
    for r in results:
        i, count = 0, 0
        precisions = []
        for query_correct in r:
            if query_correct == 1:
                count += 1.0
                precisions.append(count / (i+1))
            i += 1
        if len(precisions) > 0:
            avg_precision = sum(precisions) / len(precisions)
            map_terms.append(avg_precision)
    map = 0.0
    if len(map_terms) > 0:
        map = sum(map_terms) / len(map_terms)
    return map


def mean_reciprocal_rank(results):
    '''Calculates the Mean Reciprocal Rank across a list of document retreival results

    The MRR is the Average of the Reciprocal Ranks calculated as:
        MRR = 1 / |Q| * Sum(1 / rank_i)
        where rank_i is the rank of the first relevant query

    Args:
        results list(list(1|0)): list of each result where an entry in result is 1 if query i is relevant, 0 otherwise

    Returns:
        Mean Reciprocal Rank (scalar)

    >>> mean_reciprocal_rank([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
    0.7777777777777777
    '''
    mrr_terms = []
    for r in results:
        # print("type r", type(r))
        # assert type(r) == list
        # if sum(r) > 0:
        if 1 in r :
            # print("we have one!!")
            mrr_terms.append(1.0 / (r.index(1) + 1))
    mrr = 0
    if len(mrr_terms) > 0:
        mrr = sum(mrr_terms) / len(mrr_terms)
    return mrr


def precision_at_k(results, k=1):
    '''Calculates the Precision at k across a list of document retreival results

    Precision@k = # of relevant / # of queries

    Args:
        results list(list(1|0)): list of each result where an entry in result is 1 if query i is relevant, 0 otherwise
        k (int): number of retrieved queries to consider when calculating precision

    Returns:
        Precision at K (scalar)

    >>> precision_at_k([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
    0.6666666666666666
    >>> precision_at_k([[0, 0, 1], [1, 0, 1], [1, 1, 1]], 2)
    0.5
    '''
    precisions = []
    for r in results:
        k_queries_correct = r[:k]
        if any(x==1 for x in r):
            if len(k_queries_correct) > 0:
                precisions.append(sum(k_queries_correct)*1.0 / len(k_queries_correct))
            else:
                precisions.append(0)
    precision = 0.0
    if len(precisions) > 0:
        precision = sum(precisions) / len(precisions)
    return precision


if __name__ == '__main__':
    import doctest
    doctest.testmod()
