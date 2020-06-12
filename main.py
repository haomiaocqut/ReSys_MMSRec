import os
from time import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from evaluate import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# ============Arguments ==================
dataName = 'Yelp'
top_N = 10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', nargs='?', default=dataName,
                    help='Choose a dataset.')
# parser.add_argument('--train_dir', required=True)
parser.add_argument('--train_dir', default='train_dir')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=60, type=int)
parser.add_argument('--hidden_units', default=30, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=5, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)
with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

data_set = data_partition(args.dataset)
[user_train, user_valid, user_test, user_num, item_num] = data_set
num_batch = int(len(user_train) / args.batch_size)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

if __name__ == '__main__':
    sampler = WarpSampler(user_train, user_num, item_num, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = Model(user_num, item_num, args)
    sess.run(tf.global_variables_initializer())

    model_out_file = 'modelSave/%s_%d.h5' % (args.dataset, time())
    print('=' * 50 + 'Evaluating' + '=' * 50)
    try:
        for epoch in range(1, args.num_epochs + 1):
            t1 = time()
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                         model.is_training: True})

            t2 = time()
            t_valid = evaluate_valid(model, data_set, args, sess, top_N)
            t_test = evaluate_test(model, data_set, args, sess, top_N)
            print('epoch:%d, time: %f(s), valid (HR@10: %.4f, NDCG@10: %.4f), test (HR@10: %.4f, NDCG@10: %.4f)' % (
                epoch, t2 - t1, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            if epoch == 1:
                best_hr1, best_ndcg2, best_iter = t_test[0], t_test[1], epoch
            if t_test[0] > best_hr1:
                best_hr1, best_ndcg1, best_iter1 = t_test[0], t_test[1], epoch
            if t_test[1] > best_ndcg2:
                best_hr2, best_ndcg2, best_iter2 = t_test[0], t_test[1], epoch
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
        print("End. Best_HR Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter1, best_hr1, best_ndcg1))
        print("End. Best_NDCG Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter2, best_hr2, best_ndcg2))
    finally:
        sampler.close()
        f.close()
        exit(0)

    f.close()
    sampler.close()
    print("Done")
