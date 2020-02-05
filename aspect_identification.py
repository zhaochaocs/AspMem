#!/usr/bin/env python

import pickle
import argparse
from tqdm import tqdm
import numpy as np
from numpy.random import permutation

import matplotlib
matplotlib.use('agg')

import torch
torch.manual_seed(1)  # cpu
torch.cuda.manual_seed(1)  # gpu
np.random.seed(1)  # numpy
torch.backends.cudnn.deterministic = True  # cudnn

from model import AspectMemoryEncoder, OrthogonalityLoss
from data_utils import batch_generator, batch_test_generator, load_aspects
from utils import evaluate

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="Dataset name (without extension)", type=str)
    parser.add_argument('--test_data', help="pkl file for test", type=str, default='')
    parser.add_argument('--dev_data', help="pkl file for train", type=str, default='')
    parser.add_argument('--domain', help="domain of the review data", type=str, default='')
    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int,
                        default=2)
    parser.add_argument('--aspects', help="Number of aspects (default: 10)", type=int, default=10)
    parser.add_argument('--extend_aspects', help="Number of aspects to be extended (default: 0)", type=int, default=0)
    parser.add_argument('--aspect_seeds', help='file that contains aspect seed words (overrides number of aspects)',
                        type=str, default='')
    parser.add_argument('--epochs', help="Number of epochs (default: 15)", type=int, default=15)
    parser.add_argument('--lr', help="Learning rate (default: 0.001)", type=float, default=0.001)
    parser.add_argument('--l', help="Orthogonality loss coefficient (default: 100)", type=float, default=100)
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')
    args = parser.parse_args()

    if not args.quiet:
        print('Loading data...')

    # load tow dicts
    id2word = {}
    word2id = {}
    with open(args.data + '_word_mapping.txt', 'r') as fvoc:
        for line in fvoc:
            word, id = line.split()
            id2word[int(id)] = word
            word2id[word] = int(id)

    with open(args.data + '.pkl', 'rb') as frb:
        dataset = pickle.load(frb)

    w_emb_array = dataset['w2v']
    w_emb = torch.from_numpy(w_emb_array)
    vocab_size, emb_size = w_emb.size()

    with open(args.test_data + '.pkl', 'rb') as frb:
        dataset_test = pickle.load(frb)
    with open(args.dev_data + '.pkl', 'rb') as frb:
        dataset_dev = pickle.load(frb)

    a_emb = load_aspects(args.aspect_seeds, word2id, w_emb_array)
    args.aspects = a_emb.shape[0]
    print("Overwrite the number of aspects as {}".format(args.aspects))

    if not args.quiet:
        print('Building model..')

    net = AspectMemoryEncoder(vocab_size, emb_size, w_emb=w_emb, a_emb=torch.from_numpy(a_emb), extend_aspect=args.extend_aspects)
    if torch.cuda.is_available():
        net = net.cuda()

    orth_loss = OrthogonalityLoss()
    if args.extend_aspects:
        params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

    if not args.quiet:
        print('Starting training...')
    dev_perf, test_perf = 0, 0

    if not args.extend_aspects:
        args.epochs = 1

    for epoch in range(args.epochs):

        if args.extend_aspects:
            # model training
            net.train()
            if not args.quiet:
                print('Epoch {}'.format(epoch + 1))
            epoch_loss = {'prob_loss': 0, 'orth_loss': 0}

            # generate the batches
            batch_size = 300
            num_batches_per_epoch = int((len(dataset['data']) - 1) / batch_size) + 1
            batched_data = batch_generator(dataset, batch_size)

            with tqdm(total=num_batches_per_epoch) as pbar:
                for inputs, _ in batched_data:
                    pbar.update(1)
                    if inputs.shape[1] < args.min_len:
                        continue
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()

                    enc_out, attention = net(inputs)
                    a_probs = net.predict_by_aspect(enc_out)
                    if a_probs.size(0) > 0:
                        pred_prob, pred_label = a_probs.max(-1)
                        loss = -torch.logsumexp(a_probs, dim=1).sum()
                    epoch_loss['prob_loss'] -= loss.item()

                    aspects = net.get_aspects()
                    orth_loss_asp = args.l * orth_loss(aspects)
                    epoch_loss['orth_loss'] += orth_loss_asp.item()
                    loss += orth_loss_asp

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print('Train Epoch: {} Prob Loss: {:.6f} Orth Loss: {:.6f}'.format(
                epoch, epoch_loss['prob_loss'], epoch_loss['orth_loss']))

        # model eval
        net.eval()

        batch_size = 100
        num_batches_per_epoch = int((len(dataset_dev['data']) - 1) / batch_size) + 1
        batched_data = batch_test_generator(dataset_dev, batch_size)
        true_labels_dev, pred_labels_dev = dataset_dev['labels'], []  # true labels: list of list

        for inputs in batched_data:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            with torch.no_grad():
                enc_out, attention = net(inputs)
                a_probs = net.predict_by_aspect(enc_out)
            if a_probs.size(0) > 0:
                pred_prob, pred_label = a_probs.max(-1)
                pred_labels_dev.append(pred_label.data.cpu().numpy())

        pred_labels_dev = np.concatenate(pred_labels_dev, axis=0)
        f1 = evaluate(true_labels_dev, pred_labels_dev, args.domain, 'multi_label')
        if f1 > dev_perf:
            dev_perf = f1
            print('New best f_1 on dev: {}'.format(dev_perf))

            batch_size = 100
            num_batches_per_epoch = int((len(dataset_test['data']) - 1) / batch_size) + 1

            batched_data = batch_test_generator(dataset_test, batch_size)
            true_labels_test, pred_labels_test = dataset_test['labels'], []  # true labels: list of list

            for inputs in batched_data:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                with torch.no_grad():
                    enc_out, attention = net(inputs)
                    a_probs = net.predict_by_aspect(enc_out)
                if a_probs.size(0) > 0:
                    pred_prob, pred_label = a_probs.max(-1)
                    pred_labels_test.append(pred_label.data.cpu().numpy())

            pred_labels_test = np.concatenate(pred_labels_test, axis=0)
            test_perf = evaluate(true_labels_test, pred_labels_test, args.domain, 'multi_label')
            print("Micro f_1 scores on test: {}".format(test_perf))

    print("Final f_1 scores on test: {}".format(test_perf))
