import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate
import datetime
import pdb
import numpy as np
import score_utils as score_utils
import evaluation_utils as eval_utils
from itertools import ifilter
from evaluation import Evaluation
import data.data_utils as data_utils
from meter import AUCMeter
from sklearn.metrics import accuracy_score

# Takes in raw dataset and masks out padding and then takes sum average for LSTM
def average_without_padding_lstm(x, ids, eps=1e-8, padding_id=0, cuda=True):
    #normalize x
    x = F.normalize(x,dim=2)
    mask = ids != 0
    mask = mask.type(torch.FloatTensor)
    mask = autograd.Variable(mask.unsqueeze(2), requires_grad=False)
    if cuda:
        mask = mask.cuda()

    s = torch.sum(x*mask, 1) / (torch.sum(mask, 1)+eps)
    return s

# Takes in raw dataset and masks out padding and then takes sum average for CNN
def average_without_padding_cnn(x, ids, eps=1e-8, padding_id=0, cuda=True):
    # normalize x
    x = F.normalize(x, dim=1)
    mask = ids != 0
    mask = mask[:,:-2]
    mask = mask.type(torch.FloatTensor)
    mask = autograd.Variable(mask.unsqueeze(1), requires_grad=False)
    if cuda:
        mask = mask.cuda()

    s = torch.sum(x*mask, 2) / (torch.sum(mask, 2)+eps)
    return s

def mask(x, ids, padding_id=0, cuda=True):
    mask = ids != padding_id
    mask = mask.type(torch.FloatTensor)
    mask = autograd.Variable(mask.unsqueeze(2), requires_grad=False)
    if cuda:
        mask = mask.cuda()
    masked_x = x*mask
    return masked_x

def evaluate(model_data=None, model=None, args=None, vectorizer=None, vectorizer_data=None, embedding_model=None):
    results = []
    meter = AUCMeter()
    if model is not None:
        model.eval()
        print("Computing Model Evaluation Metrics...")
        results = compute_model_rankings(model_data, model, args, meter, embedding_model)
    if vectorizer is not None:
        if vectorizer_data is None:
            print("No vectorizer compatible data. Aborting...")
            return 0, 0, 0, 0
        print("Computing TFIDF Evaluation Metrics...")
        results = compute_tfidf_rankings(vectorizer_data, vectorizer, meter)
    e = Evaluation(results)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    auc5 = meter.value(max_fpr=0.05)
    return MAP, MRR, P1, P5, auc5

def compute_model_rankings(data, model, args, meter, embedding_model=None):
    res = []
    for idts, idbs, labels in tqdm(data):
        titles, bodies = autograd.Variable(idts), autograd.Variable(idbs)
        if args.cuda:
            titles, bodies = titles.cuda(), bodies.cuda()
        if embedding_model is not None:
            titles = embedding_model(titles)
            bodies = embedding_model(bodies)
        encode_titles = model(titles)
        encode_bodies = model(bodies)

        if 'lstm' in args.model_name:
            titles_encodings = average_without_padding_lstm(encode_titles, idts, cuda=args.cuda)
            bodies_encodings = average_without_padding_lstm(encode_bodies, idbs, cuda=args.cuda)
        else:
            titles_encodings = average_without_padding_cnn(encode_titles, idts, cuda=args.cuda)
            bodies_encodings = average_without_padding_cnn(encode_bodies, idbs, cuda=args.cuda)

        text_encodings = (titles_encodings + bodies_encodings) * 0.5
        scores = score_utils.batch_cosine_similarity_eval(text_encodings, cuda=args.cuda).data.cpu().numpy()
        meter.add(scores, labels)
        assert len(scores) == len(labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    return res


def compute_tfidf_rankings(data, vectorizer, meter):
    res = []
    for titles, bodies, labels in tqdm(data):
        encoded_titles = vectorizer.transform(titles)
        encoded_bodies = vectorizer.transform(bodies)
        text_encodings = (encoded_titles + encoded_bodies) * 0.5
        # print('TFIDF ENCODINGS')
        # print(text_encodings)
        scores = score_utils.batch_cosine_similarity_tfidf(text_encodings).numpy()
        meter.add(scores, labels)
        assert len(scores) == len(labels)
        if sorted(scores, reverse=True)[0] < 0.01:
            print("ZERO BEST")
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    return res

def get_embedding_model(embeddings, cuda):
    vocab_size, embed_dim = embeddings.shape
    embedding_layer = nn.Embedding(vocab_size, embed_dim)
    embedding_layer.weight.data = torch.from_numpy(embeddings)
    embedding_layer.weight.requires_grad = False
    embedding_model = nn.Sequential(embedding_layer)
    if cuda:
        embedding_model = embedding_model.cuda()
    return embedding_model

def train_gan(transformer, discriminator, encoder, transformer_batches, discriminator_batches, encoder_batches, encoder_batches_vanilla, dev_data, test_data, args, embeddings, results_lock=None):
    embedding_model = get_embedding_model(embeddings, args.cuda)
    ones = autograd.Variable(torch.ones(discriminator_batches[0][0].size()[0]), requires_grad=False).unsqueeze(1)
    zeros = autograd.Variable(torch.zeros(discriminator_batches[0][0].size()[0]), requires_grad=False).unsqueeze(1)
    if args.cuda:
        transformer = transformer.cuda()
        discriminator = discriminator.cuda()
        encoder = encoder.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    parameters_t = ifilter(lambda p: p.requires_grad, transformer.parameters())
    parameters_e = ifilter(lambda p: p.requires_grad, encoder.parameters())
    if args.wgan:
        optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)
        optimizer_t = torch.optim.RMSprop(parameters_t, lr=5e-5)
        optimizer_e = torch.optim.RMSprop(parameters_e, lr=5e-5)
    else:
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)
        optimizer_t = torch.optim.Adam(parameters_t, lr=args.lr_t)
        optimizer_e = torch.optim.Adam(parameters_e, lr=args.lr)
    best_epoch = 0
    best_AUC05_dev = 0.0
    best_AUC05_test = 0.0
    for epoch in range(1, args.epochs + 1):
        print("-------------------\nEpoch {}:\n".format(epoch))
        # Adverserial Training
        discriminator.train()
        transformer.train()
        encoder.train()
        losses_d = []
        losses_t = []
        losses_e = []
        index_d = 0
        for num_batch in xrange(len(transformer_batches)):
            print("ADVERSERIAL BATCH #{} of {}".format(num_batch, len(transformer_batches)))
            # k Discriminator batches
            for p in discriminator.parameters():
                p.requires_grad = True
            dk = args.dk
            if args.wgan:
                if (epoch == 0 and num_batch < 20) or (epoch * num_batch % 500 == 0):
                    dk = 100
            for num_batch_d in xrange(dk):
                if args.wgan and (not args.grad_penalty):
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                data = discriminator_batches[index_d + num_batch_d]
                titles_s, bodies_s, titles_t, bodies_t = [autograd.Variable(x) for x in data]
                if args.cuda:
                    titles_s, bodies_s, titles_t, bodies_t = [x.cuda() for x in (titles_s, bodies_s, titles_t, bodies_t)]
                # TODO test if changing variable names matters
                # Train on real target data
                embedded_titles_t = embedding_model(titles_t)
                embedded_bodies_t = embedding_model(bodies_t)
                is_target_titles_t = discriminator(embedded_titles_t)
                is_target_bodies_t = discriminator(embedded_bodies_t)
                loss_real = (F.binary_cross_entropy_with_logits(is_target_titles_t, ones) + F.binary_cross_entropy_with_logits(is_target_bodies_t, ones)) * 0.5
                # Train on fake (transformed source) data
                transformed_titles_s = mask(transformer(titles_s), titles_s.cpu().data, cuda=args.cuda)
                transformed_bodies_s = mask(transformer(bodies_s), bodies_s.cpu().data, cuda=args.cuda)
                is_target_titles_s = discriminator(transformed_titles_s)
                is_target_bodies_s = discriminator(transformed_bodies_s)
                loss_fake = (F.binary_cross_entropy_with_logits(is_target_titles_s, zeros) + F.binary_cross_entropy_with_logits(is_target_bodies_s, zeros)) * 0.5
                # Compute gradients and backprop
                total_discriminator_loss = loss_real + loss_fake
                if args.wgan:
                    loss_t = (torch.mean(is_target_titles_t) - torch.mean(is_target_titles_s))
                    loss_b = (torch.mean(is_target_bodies_t) - torch.mean(is_target_bodies_s))
                    total_discriminator_loss = (loss_t + loss_b) * 0.5
                    if args.grad_penalty:
                        # print("EMBED")
                        # print(embedded_titles_t)
                        dist_titles = ((embedded_titles_t-transformed_titles_s)**2).sum(2)**0.5
                        dist_bodies = ((embedded_bodies_t-transformed_bodies_s)**2).sum(2)**0.5
                        # print("DIST")
                        # print(dist)
                        lipschitz_est_titles = (is_target_titles_t-is_target_titles_s).abs()/(dist_titles+1e-8)
                        lipschitz_est_bodies = (is_target_bodies_t-is_target_bodies_s).abs()/(dist_bodies+1e-8)
                        # print("LIP")
                        # print(lipschitz_est)
                        lipschitz_est_titles = F.normalize(lipschitz_est_titles, p=2, dim=1)
                        lipschitz_est_bodies = F.normalize(lipschitz_est_bodies, p=2, dim=1)
                        lipschitz_loss_titles = args.gam*((1.0-lipschitz_est_titles)**2).mean(0).mean(0).view(1)
                        lipschitz_loss_bodies = args.gam*((1.0-lipschitz_est_bodies)**2).mean(0).mean(0).view(1)
                        lipschitz_loss = (lipschitz_loss_titles + lipschitz_loss_bodies) * 0.5
                        total_discriminator_loss += lipschitz_loss
                losses_d.append(total_discriminator_loss.cpu().data[0])
                optimizer_d.zero_grad()
                total_discriminator_loss.backward()
                optimizer_d.step()
            # Update index for disc batches
            index_d += dk
            # Transformer batch
            for p in discriminator.parameters():
                p.requires_grad = False
            data = transformer_batches[num_batch]
            titles_s, bodies_s, _, _ = [autograd.Variable(x) for x in data]
            if args.cuda:
                titles_s, bodies_s = [x.cuda() for x in (titles_s, bodies_s)]
            is_target_titles = discriminator(mask(transformer(titles_s), titles_s.cpu().data, cuda=args.cuda))
            is_target_bodies = discriminator(mask(transformer(bodies_s), bodies_s.cpu().data, cuda=args.cuda))
            total_transformer_loss = (F.binary_cross_entropy_with_logits(is_target_titles, ones) + F.binary_cross_entropy_with_logits(is_target_bodies, ones)) * 0.5
            if args.wgan:
                total_transformer_loss = (torch.mean(is_target_titles) + torch.mean(is_target_bodies)) * 0.5
            losses_t.append(total_transformer_loss.cpu().data[0])
            optimizer_t.zero_grad()
            total_transformer_loss.backward()
            optimizer_t.step()
            if args.verbose:
                print("LAST DISCRIMINATOR LOSS: {}".format(losses_d[-1]))
                print("LAST TRANSFORMER LOSS: {}".format(losses_t[-1]))
        # Encoder Training
        for num_batch in xrange(len(encoder_batches)):
            print("ENCODER BATCH #{} of {}".format(num_batch, len(encoder_batches)))
            titles_s, bodies_s, train_group_ids = encoder_batches[num_batch]
            titles_sv, bodies_sv, train_group_ids_v = encoder_batches_vanilla[num_batch]
            titles_s, bodies_s = [autograd.Variable(x) for x in (titles_s, bodies_s)]
            titles_sv, bodies_sv = [autograd.Variable(x) for x in (titles_sv, bodies_sv)]
            if args.cuda:
                titles_s, bodies_s = [x.cuda() for x in (titles_s, bodies_s)]
                titles_sv, bodies_sv = [x.cuda() for x in (titles_sv, bodies_sv)]

            encoded_titles = encoder(mask(transformer(titles_s), titles_s.cpu().data, cuda=args.cuda))
            encoded_bodies = encoder(mask(transformer(bodies_s), bodies_s.cpu().data, cuda=args.cuda))
            encoded_titles_v = embedding_model(titles_sv)
            encoded_bodies_v = embedding_model(bodies_sv)
            encoded_titles_v = encoder(encoded_titles_v)
            encoded_bodies_v = encoder(encoded_bodies_v)

            avg_encoded_titles = average_without_padding_cnn(encoded_titles, titles_s.cpu().data, cuda=args.cuda)
            avg_encoded_bodies = average_without_padding_cnn(encoded_bodies, bodies_s.cpu().data, cuda=args.cuda)
            avg_encoded_text = (avg_encoded_titles + avg_encoded_bodies) * 0.5
            avg_encoded_titles_v = average_without_padding_cnn(encoded_titles_v, titles_sv.cpu().data, cuda=args.cuda)
            avg_encoded_bodies_v = average_without_padding_cnn(encoded_bodies_v, bodies_sv.cpu().data, cuda=args.cuda)
            avg_encoded_text_v = (avg_encoded_titles_v + avg_encoded_bodies_v) * 0.5

            scores, target_indices = score_utils.batch_cosine_similarity(avg_encoded_text, train_group_ids, args.cuda)
            scores_v, target_indices_v = score_utils.batch_cosine_similarity(avg_encoded_text_v, train_group_ids_v, args.cuda)
            total_encoder_loss = F.multi_margin_loss(scores, target_indices, margin=args.margin)
            total_encoder_loss_v = F.multi_margin_loss(scores_v, target_indices_v, margin=args.margin)
            losses_e.append(total_encoder_loss.cpu().data[0])
            losses_e.append(total_encoder_loss_v.cpu().data[0])

            # Step for transformed encodings
            optimizer_e.zero_grad()
            total_encoder_loss.backward()
            optimizer_e.step()
            # Step for source encodings
            optimizer_e.zero_grad()
            total_encoder_loss_v.backward()
            optimizer_e.step()
            if args.verbose:
                print("LAST ENCODER LOSSES (transf, vanilla): \n{}\n{}".format(losses_e[-2], losses_e[-1]))
        # Evaluation
        encoder.eval()
        print("Dev data performance")
        _, _, _, _, AUC05_dev = evaluate(model_data=dev_data, model=encoder, args=args, embedding_model=embedding_model)
        print(tabulate([[AUC05_dev]], headers=['AUC_0.5']))
        print("\nTest data performance")
        _, _, _, _, AUC05_test = evaluate(model_data=test_data, model=encoder, args=args, embedding_model=embedding_model)
        print(tabulate([[AUC05_test]], headers=['AUC_0.5']))
        if AUC05_dev > best_AUC05_dev:
            best_AUC05_dev = AUC05_dev
            best_AUC05_test = AUC05_test
            best_epoch = epoch
            # TODO save models
        print("Best epoch so far:", best_epoch, best_AUC05_dev)
        print()
    if results_lock:
        results_lock.acquire()
        try:
            data_utils.record_best_results(args.results_path, args.save_path+"_args_"+serialize_model_name(args), [best_AUC05_dev], [best_AUC05_test], best_epoch)
        finally:
            results_lock.release()
    else:
        data_utils.record_best_results(args.results_path, args.save_path+"_args_"+serialize_model_name(args), [best_AUC05_dev], [best_AUC05_test], best_epoch)


def train_model(model, train, dev_data, test_data, ids_corpus, batch_size, args, model_2=None, train_batches_2=None, results_lock=None):
    if args.cuda:
        print ('Available devices ', torch.cuda.device_count())
        print ('Current cuda device ', torch.cuda.current_device())
        print("SETTING DEVICE TO " + str(args.gpuid))
        torch.cuda.set_device(args.gpuid)
        print ('Current cuda device ', torch.cuda.current_device())
    is_training=True
    best_metrics_dev = []
    best_metrics_test = []
    if args.cuda:
        model = model.cuda()
    parameters = ifilter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters , lr=args.lr)
    model.train()

    if args.domain_adaptation:
        if args.cuda:
            model_2 = model_2.cuda()
        parameters_2 = ifilter(lambda p: p.requires_grad, model_2.parameters())
        optimizer_2 = torch.optim.Adam(parameters_2, lr=args.lr2)
        model_2.train()

    best_epoch = 0
    best_MRR = 0
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        # randomize new dataset each time
        train_batches = data_utils.create_batches(ids_corpus, train, batch_size, 0, pad_left=False)

        N =len(train_batches)
        losses = []
        all_results = []
        for i in xrange(N):
            # get current batch
            t, b, train_group_ids = train_batches[i]
            titles, bodies = autograd.Variable(t), autograd.Variable(b)
            if args.cuda:
                titles, bodies = titles.cuda(), bodies.cuda()
            if is_training:
                model.train()
                optimizer.zero_grad()
                model.train()
            # Encode all of the title and body text using model
            # squeeze dimension differs for lstm and cnn models

            encode_titles = model(titles)
            encode_bodies = model(bodies)

            if 'cnn' in args.model_name:
                titles_encodings = average_without_padding_cnn(encode_titles, t, cuda=args.cuda)
                bodies_encodings = average_without_padding_cnn(encode_bodies, b, cuda=args.cuda)
            else:
                titles_encodings = average_without_padding_lstm(encode_titles, t, cuda=args.cuda)
                bodies_encodings = average_without_padding_lstm(encode_bodies, b, cuda=args.cuda)

            text_encodings = (titles_encodings + bodies_encodings) * 0.5

            # Calculate Loss = Multi-Margin-Loss(train_group_ids, text_encodings)
            scores, target_indices = score_utils.batch_cosine_similarity(text_encodings, train_group_ids, args.cuda)
            loss = F.multi_margin_loss(scores, target_indices, margin=args.margin)

            # Batch 2
            if args.domain_adaptation:
                t2, b2, d = train_batches_2[i]
                titles_2, bodies_2, domains = autograd.Variable(t2), autograd.Variable(b2), autograd.Variable(torch.FloatTensor(d), requires_grad=False)
                if args.cuda:
                    titles_2, bodies_2, domains = titles_2.cuda(), bodies_2.cuda(), domains.cuda()
                if is_training:
                    model_2.train()
                    optimizer_2.zero_grad()
                    model_2.train()
                encode_titles_2 = model(titles_2)
                encode_bodies_2 = model(bodies_2)
                if 'cnn' in args.model_name:
                    titles_encodings_2 = average_without_padding_cnn(encode_titles_2, t2, cuda=args.cuda)
                    bodies_encodings_2 = average_without_padding_cnn(encode_bodies_2, b2, cuda=args.cuda)
                else:
                    titles_encodings_2 = average_without_padding_lstm(encode_titles_2, t2, cuda=args.cuda)
                    bodies_encodings_2 = average_without_padding_lstm(encode_bodies_2, b2, cuda=args.cuda)
                encoded_text = (titles_encodings_2 + bodies_encodings_2) * 0.5
                # Run through discriminators
                labeled_encodings_2 = model_2(encoded_text)
                labeled_encodings_2 = torch.squeeze(labeled_encodings_2)
                # Calculate loss 2
                loss_2 = F.binary_cross_entropy_with_logits(labeled_encodings_2, domains)

                if args.show_discr_loss:
                    print("discriminator loss: ", loss_2)
                    preds = labeled_encodings_2 >= 0.5
                    acc = accuracy_score(d, preds.cpu().data.numpy())
                    print("Discriminator accuracy:", acc)
                # Calculate total cost
                total_cost = loss - (args.lam * loss_2)
                total_cost.backward()
                optimizer.step()
                optimizer_2.step()
                losses.append(total_cost.cpu().data[0])
            else:
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().data[0])
            print("BATCH LOSS "+str(i+1)+" out of "+str(N)+": ")
            print(losses[-1])
            rankings = compile_rankings(train_group_ids, scores.cpu().data.numpy())
            results = strip_ids_and_scores(rankings)
            all_results += results

        if epoch % 2 == 0 and len(args.save_path) > 0:
            torch.save(model, args.save_path+"_args_"+serialize_model_name(args)+"_epoch"+str(epoch)+'.pt')
            if args.domain_adaptation:
                torch.save(model_2, args.save_path+"_args_"+serialize_model_name(args)+"_epoch"+str(epoch)+"_discriminator.pt")
        # Evaluation Metrics
        precision_at_1 = eval_utils.precision_at_k(all_results, 1)*100
        precision_at_5 = eval_utils.precision_at_k(all_results, 5)*100
        MAP = eval_utils.mean_average_precision(all_results)*100
        MRR = eval_utils.mean_reciprocal_rank(all_results)*100
        print(tabulate([[MAP, MRR, precision_at_1, precision_at_5]], headers=['MAP', 'MRR', 'P@1', 'P@5']))
        avg_loss = np.mean(losses)
        print('Train max-margin loss: {:.6f}'.format( avg_loss))

        print("Dev data Performance")
        MAP, MRR, P1, P5, auc5 = evaluate(model_data=dev_data, model=model, args=args)
        print(tabulate([[MAP, MRR, P1, P5, auc5]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        print()
        print("Test data Performance")
        mapt, mrrt, p1t, p5t, auc5t = evaluate(model_data=test_data, model=model, args=args)
        print(tabulate([[mapt, mrrt, p1t, p5t, auc5t]], headers=['MAP', 'MRR', 'P@1', 'P@5', 'AUC0.05']))
        print()

        # for android dataset, validate using AUC0.05 metric
        if args.android:
            MRR = auc5
        if MRR > best_MRR:
            best_MRR = MRR
            best_metrics_dev = [MAP, MRR, P1, P5, auc5]
            best_metrics_test = [mapt, mrrt, p1t, p5t, auc5t]
            best_epoch = epoch
            # Save model
            torch.save(model, args.save_path+"_args_"+serialize_model_name(args)+"_epoch_best.pt")
            if args.domain_adaptation:
                torch.save(model_2, args.save_path+"_args_"+serialize_model_name(args)+"_epoch_best_discriminator.pt")


        print("Best EPOCH so far:", best_epoch, best_MRR)
        print()
    if results_lock:
        results_lock.acquire()
        try:
            data_utils.record_best_results(args.results_path, args.save_path+"_args_"+serialize_model_name(args), best_metrics_dev, best_metrics_test, best_epoch)
        finally:
            results_lock.release()
    else:
        data_utils.record_best_results(args.results_path, args.save_path+"_args_"+serialize_model_name(args), best_metrics_dev, best_metrics_test, best_epoch)

# same as compile_rankings function except it already has positive and negative labels passed as qlabels
# also only handles one minibatch
def compile_rankings_eval(qlabels, scores):
    all_rankings = []
    for i in range(len(qlabels)):
        entry = (None, scores[i], qlabels[i])
        all_rankings.append(entry)
    return sorted(all_rankings, key=lambda x: x[1], reverse=True)

def compile_rankings(train_group_ids, scores_variable):
    '''Compiles a list of lists ranking the queries by their scores

    Args:
        train_group_ids
        scores

    Returns:
        rankings list(list(tuple(id, score, relevant)))

    >>> compile_rankings([[123, 200, 567, 876, 987], [123, 201, 567, 876, 987], [134, 243, 902, 581, 939]], [[0.91, 0.1, 0.25, 0.3], [0.97, 0.1, 0.25, 0.3], [0.84, 0.41, 0.15, 0.2001]])
    [[(201, 0.97, 1), (200, 0.91, 1), (987, 0.3, 0), (876, 0.25, 0), (567, 0.1, 0)], [(243, 0.84, 1), (902, 0.41, 0), (939, 0.2001, 0), (581, 0.15, 0)]]

    '''
    scores = scores_variable #.data
    all_rankings, rankings = [], []
    last_question_id = None
    for i, group in enumerate(train_group_ids):
        if group[0] != last_question_id:
            if len(rankings) > 0:
                all_rankings.append(sorted(rankings, key=lambda q: q[1], reverse=True))
            rankings = []
            last_question_id = group[0]
            # Append negative queries just once
            for j, negative_id in enumerate(group[2:]):
                # index = group_index - 1 = j + 2 - 1 = j + 1
                entry = (negative_id, scores[i][j+1], 0)
                rankings.append(entry)
        # Append this entry's positive query
        entry = (group[1], scores[i][0], 1)
        rankings.append(entry)
    all_rankings.append(sorted(rankings, key=lambda q: q[1], reverse=True))
    return all_rankings

def strip_ids_and_scores(rankings):
    '''Strips the ids and scores from a list of rankings

    Args:
        rankings: (see compile_rankings for description)

    Returns:
        list of lists or relevance values: 1 if that query was relevant, 0 otherwise

    >>> strip_ids_and_scores([[(243, 0.8, 1), (989, 0.45, 0), (544, 0.47, 1)],[(12, 0.99, 0), (912, 0.34, 1), (453, 0.6532, 1)],[(888, 0.3, 0), (97, 0.1, 0), (334, 0.2, 0)]])
    [[1, 0, 1], [0, 1, 1], [0, 0, 0]]

    '''
    results = []
    for ranking in rankings:
        result = [entry[2] for entry in ranking]
        results.append(result)
    return results

def serialize_model_name(args):
    name = "_da-" + str(args.domain_adaptation) + "_lr-" + str(args.lr) + "_hidden-" + str(args.num_hidden) + "_drop-" + str(args.dropout) + "_marg" + str(args.margin)
    if args.domain_adaptation:
        name += "_lam" + str(args.lam)
    name += "_"
    return name

if __name__ == '__main__':
    import doctest
    doctest.testmod()
