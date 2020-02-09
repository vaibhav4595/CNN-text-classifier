import os
import sys
import pickle
from utils import *
from build_vocab import VocabEntry
import argparse
from cnn import CNN
from pdb import set_trace as bp
import numpy as np
import torch
import time

def train(args):

    device = args.device
    train_lines = open(args.train_file).readlines()
    val_lines = open(args.val_file).readlines()

    log_every = args.log_every
    valid_iter = args.valid_iter
    train_iter = 0
    cum_loss = 0
    avg_loss = 0
    valid_num = 0
    patience = 0
    num_trial = 0
    hist_valid_scores = []
    begin_time = time.time()

    vocab = get_vocab(args.vocab_file)
    model = CNN(args, vocab)
    if args.use_embed == 1:
        model.load_vector(args, vocab)
    if args.device == 'cuda':
        model.cuda()

    lr = args.lr
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    model.train()

    for ep in range(args.max_epochs):

        train_iter = 0
        val_iter = 0

        for examples, labels in batch_iter(train_lines, vocab, args.batch_size, \
                                               args.max_sent_len, shuffle=True):

            train_iter += 1
            optim.zero_grad()
            labels = torch.tensor(labels).to(device=device)
            examples = torch.tensor(examples).to(device=device)
            output = model(examples)

            loss = criterion(output, labels)
            avg_loss += loss.item()
            cum_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.clip_grad)
            optim.step()

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg.loss %.2f, time elapsed %.2f'\
                     % (ep + 1, train_iter, avg_loss / log_every, time.time() - begin_time), file=sys.stderr)

                begin_time = time.time()
                avg_loss = 0

            if train_iter % valid_iter == 0:

                print('epoch %d, iter %d, cum.loss %.2f, time elapsed %.2f'\
                     % (ep + 1, train_iter, cum_loss / valid_iter, time.time() - begin_time), file=sys.stderr)


                cum_loss = 0
                valid_num += 1

                print("Begin Validation ", file=sys.stderr)

                model.eval()
                acc = test(val_lines, model, vocab, args)
                model.train()

                print('validation: iter %d, acc %f' % (train_iter, acc), file=sys.stderr)

                is_better = (len(hist_valid_scores) == 0) or (acc > max(hist_valid_scores))
                hist_valid_scores.append(acc)

                if is_better:
                    patience = 0
                    print("Save the current model and optimiser state")
                    torch.save(model, args.model_save_path)

                    torch.save(optim.state_dict(), args.model_save_path + '.optim')

                elif patience < args.patience:

                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == args.max_num_trials:
                            print('early stop!', file=sys.stderr)
                            return

                        lr = lr * args.lr_decay

                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
                        model = load(args.model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)

                        optim = torch.optim.Adam(list(model.parameters()), lr=lr)
                        optim.load_state_dict(torch.load(args.model_save_path + '.optim'))
                        for state in optim.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)
                        for group in optim.param_groups:
                            group['lr'] = lr

                        patience = 0

    print("Training Finished", file=sys.stderr) 

def test(data, model, vocab, args, typer='val'):

    softmax = torch.nn.Softmax(dim=1)

    total = 0
    correct = 0

    outputs = []

    for examples, labels in batch_iter(data, vocab, args.batch_size, \
                                args.max_sent_len, shuffle=False):

        examples = torch.tensor(examples).to(args.device)
        output = torch.argmax(softmax(model(examples)), dim=1)

        total += examples.shape[0]

        if typer == 'test':
            labels = torch.tensor(labels).to(args.device)
            correct = correct +  torch.sum((labels == output)).item()

        for each in output:
            outputs.append(vocab.id2class(int(each.item())))


    if typer == 'test':
        fp = open(args.write_output_path, 'w')
        for each in outputs:
            fp.write(each + '\n')
        fp.close()

    return correct / total

def load(model_path):

    model = torch.load(model_path)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--train_file', type=str, default='../data/topicclass/topicclass_train.txt')
    parser.add_argument('--val_file', type=str, default='../data/topicclass/topicclass_valid.txt')
    parser.add_argument('--test_file', type=str, default='../data/topicclass/topicclass_valid.test')
    parser.add_argument('--embed_file', type=str, default='../data/glove.6B.300d.txt')
    parser.add_argument('--use_embed', type=int, default=1)
    parser.add_argument('--fine_tune', type=int, default=1)
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--model_path', type=str, default='../models/model.pkl')
    parser.add_argument('--model_save_path', type=str, default='../models/model.pkl')
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--max_sent_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--valid_iter', type=int, default=2500)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.25)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_num_trials", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--write_output_path", type=str, default='../outputs/valid_output.txt')
    args = parser.parse_args()


    if args.mode == 'train':
        train(args)

    if args.mode == 'test':
        test_lines = open(args.test_file).readlines()
        model = load(args.model_path)
        model.eval()
        vocab = get_vocab(args.vocab_file)
        acc = test(test_lines, model, vocab, args, typer="test")
        print("Acc = ", acc)
