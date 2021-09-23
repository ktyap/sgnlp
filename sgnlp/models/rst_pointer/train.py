import csv
import os
import copy
import pickle
import torch
import random
import logging
import numpy as np
from typing import List
from torch.backends import cudnn

from sgnlp.models.rst_pointer.config import RstPointerSegmenterConfig
from sgnlp.models.rst_pointer.preprocess import RSTPreprocessor
from .modeling import RstPointerParserModel, RstPointerParserConfig, RstPointerSegmenterModel
from .utils import parse_args_and_load_config
from .data_class import RstPointerParserTrainArgs, RstPointerSegmenterTrainArgs
from .modules.type import DiscourseTreeNode, DiscourseTreeSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup(seed):
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Parser training code
def get_span_dict(discourse_tree_splits: List[DiscourseTreeSplit]):
    span_dict = {}

    for split in discourse_tree_splits:
        left_span_key, left_node_value = get_span_key_and_node_value(split.left)
        span_dict[left_span_key] = left_node_value
        right_span_key, right_node_value = get_span_key_and_node_value(split.right)
        span_dict[right_span_key] = right_node_value

    return span_dict


def get_span_key_and_node_value(node: DiscourseTreeNode):
    span_key = f'{node.span[0]}-{node.span[1]}'
    node_value = [node.label, node.ns_type]
    return span_key, node_value


def get_measurement(discourse_tree_splits_1, discourse_tree_splits_2):
    span_dict1 = get_span_dict(discourse_tree_splits_1)
    span_dict2 = get_span_dict(discourse_tree_splits_2)
    num_correct_relations = 0
    num_correct_nuclearity = 0
    num_spans_1 = len(span_dict1)
    num_spans_2 = len(span_dict2)

    # no of right spans
    matching_spans = list(set(span_dict1.keys()).intersection(set(span_dict2.keys())))
    num_matching_spans = len(matching_spans)

    # count matching relations and nuclearity
    for span in matching_spans:
        if span_dict1[span][0] == span_dict2[span][0]:
            num_correct_relations += 1
        if span_dict1[span][1] == span_dict2[span][1]:
            num_correct_nuclearity += 1

    return num_matching_spans, num_correct_relations, num_correct_nuclearity, num_spans_1, num_spans_2


def get_batch_measure(input_splits_batch, golden_metric_batch):
    num_matching_spans = 0
    num_correct_relations = 0
    num_correct_nuclearity = 0
    num_spans_input = 0
    num_spans_golden = 0

    for input_splits, golden_splits in zip(input_splits_batch, golden_metric_batch):
        if input_splits and golden_splits:
            # if both splits have values in the list
            _num_matching_spans, _num_correct_relations, _num_correct_nuclearity, _num_spans_input, _num_spans_golden \
                = get_measurement(input_splits, golden_splits)

            num_matching_spans += _num_matching_spans
            num_correct_relations += _num_correct_relations
            num_correct_nuclearity += _num_correct_nuclearity
            num_spans_input += _num_spans_input
            num_spans_golden += _num_spans_golden

        elif input_splits and not golden_splits:
            # each split has 2 spans
            num_spans_input += len(input_splits) * 2

        elif not input_splits and golden_splits:
            num_spans_golden += len(golden_splits) * 2

    return num_matching_spans, num_correct_relations, num_correct_nuclearity, num_spans_input, num_spans_golden


def get_micro_measure(correct_span, correct_relation, correct_nuclearity, no_system, no_golden):
    # Compute Micro-average measure
    # Span
    precision_span = correct_span / no_system
    recall_span = correct_span / no_golden
    f1_span = (2 * correct_span) / (no_golden + no_system)

    # Relation
    precision_relation = correct_relation / no_system
    recall_relation = correct_relation / no_golden
    f1_relation = (2 * correct_relation) / (no_golden + no_system)

    # Nuclearity
    precision_nuclearity = correct_nuclearity / no_system
    recall_nuclearity = correct_nuclearity / no_golden
    f1_nuclearity = (2 * correct_nuclearity) / (no_golden + no_system)

    return (precision_span, recall_span, f1_span), (precision_relation, recall_relation, f1_relation), \
           (precision_nuclearity, recall_nuclearity, f1_nuclearity)


def get_batch_data_training(input_sentences, edu_breaks, decoder_input, relation_label,
                            parsing_breaks, golden_metric, parents_index, sibling, batch_size):
    # change them into np.array
    input_sentences = np.array(input_sentences, dtype="object")
    edu_breaks = np.array(edu_breaks, dtype="object")
    decoder_input = np.array(decoder_input, dtype="object")
    relation_label = np.array(relation_label, dtype="object")
    parsing_breaks = np.array(parsing_breaks, dtype="object")
    golden_metric = np.array(golden_metric, dtype="object")
    parents_index = np.array(parents_index, dtype="object")
    sibling = np.array(sibling, dtype="object")

    if len(decoder_input) < batch_size:
        batch_size = len(decoder_input)

    sample_indices = random.sample(range(len(decoder_input)), batch_size)
    # Get batch data
    input_sentences_batch = copy.deepcopy(input_sentences[sample_indices])
    edu_breaks_batch = copy.deepcopy(edu_breaks[sample_indices])
    decoder_input_batch = copy.deepcopy(decoder_input[sample_indices])
    relation_label_batch = copy.deepcopy(relation_label[sample_indices])
    parsing_breaks_batch = copy.deepcopy(parsing_breaks[sample_indices])
    golden_metric_batch = copy.deepcopy(golden_metric[sample_indices])
    parents_index_batch = copy.deepcopy(parents_index[sample_indices])
    sibling_batch = copy.deepcopy(sibling[sample_indices])

    # Get sorted
    lengths_batch = np.array([len(sent) for sent in input_sentences_batch])
    idx = np.argsort(lengths_batch)
    idx = idx[::-1]

    # Convert them back to list
    input_sentences_batch = input_sentences_batch[idx].tolist()
    edu_breaks_batch = edu_breaks_batch[idx].tolist()
    decoder_input_batch = decoder_input_batch[idx].tolist()
    relation_label_batch = relation_label_batch[idx].tolist()
    parsing_breaks_batch = parsing_breaks_batch[idx].tolist()
    golden_metric_batch = golden_metric_batch[idx].tolist()
    parents_index_batch = parents_index_batch[idx].tolist()
    sibling_batch = sibling_batch[idx].tolist()

    return input_sentences_batch, edu_breaks_batch, decoder_input_batch, relation_label_batch, \
           parsing_breaks_batch, golden_metric_batch, parents_index_batch, sibling_batch


def get_batch_data(input_sentences, edu_breaks, decoder_input, relation_label,
                   parsing_breaks, golden_metric, batch_size):
    # change them into np.array
    input_sentences = np.array(input_sentences, dtype="object")
    edu_breaks = np.array(edu_breaks, dtype="object")
    decoder_input = np.array(decoder_input, dtype="object")
    relation_label = np.array(relation_label, dtype="object")
    parsing_breaks = np.array(parsing_breaks, dtype="object")
    golden_metric = np.array(golden_metric, dtype="object")

    if len(decoder_input) < batch_size:
        batch_size = len(decoder_input)
    sample_indices = random.sample(range(len(decoder_input)), batch_size)

    # Get batch data
    input_sentences_batch = copy.deepcopy(input_sentences[sample_indices])
    edu_breaks_batch = copy.deepcopy(edu_breaks[sample_indices])
    decoder_input_batch = copy.deepcopy(decoder_input[sample_indices])
    relation_label_batch = copy.deepcopy(relation_label[sample_indices])
    parsing_breaks_batch = copy.deepcopy(parsing_breaks[sample_indices])
    golden_metric_batch = copy.deepcopy(golden_metric[sample_indices])

    # Get sorted
    lengths_batch = np.array([len(sent) for sent in input_sentences_batch])
    idx = np.argsort(lengths_batch)
    idx = idx[::-1]

    # Convert them back to list
    input_sentences_batch = input_sentences_batch[idx].tolist()
    edu_breaks_batch = edu_breaks_batch[idx].tolist()
    decoder_input_batch = decoder_input_batch[idx].tolist()
    relation_label_batch = relation_label_batch[idx].tolist()
    parsing_breaks_batch = parsing_breaks_batch[idx].tolist()
    golden_metric_batch = golden_metric_batch[idx].tolist()

    return input_sentences_batch, edu_breaks_batch, decoder_input_batch, relation_label_batch, parsing_breaks_batch, golden_metric_batch


def get_accuracy(model, preprocessor, input_sentences, edu_breaks, decoder_input, relation_label,
                 parsing_breaks, golden_metric, batch_size):
    num_loops = int(np.ceil(len(edu_breaks) / batch_size))

    loss_tree_all = []
    loss_label_all = []
    correct_span = 0
    correct_relation = 0
    correct_nuclearity = 0
    no_system = 0
    no_golden = 0

    for loop in range(num_loops):
        start_idx = loop * batch_size
        end_idx = (loop + 1) * batch_size
        if end_idx > len(edu_breaks):
            end_idx = len(edu_breaks)

        input_sentences_batch, edu_breaks_batch, _, \
        relation_label_batch, parsing_breaks_batch, golden_metric_splits_batch = \
            get_batch_data(input_sentences[start_idx:end_idx],
                           edu_breaks[start_idx:end_idx],
                           decoder_input[start_idx:end_idx],
                           relation_label[start_idx:end_idx],
                           parsing_breaks[start_idx:end_idx],
                           golden_metric[start_idx:end_idx], batch_size)

        input_sentences_ids_batch, sentence_lengths = preprocessor(input_sentences_batch)

        model_output = model.forward(
            input_sentence=input_sentences_ids_batch,
            edu_breaks=edu_breaks_batch,
            label_index=relation_label_batch,
            parsing_index=parsing_breaks_batch,
            sentence_lengths=sentence_lengths,
            generate_splits=True
        )

        loss_tree_all.append(model_output.loss_tree_batch)
        loss_label_all.append(model_output.loss_label_batch)
        correct_span_batch, correct_relation_batch, correct_nuclearity_batch, \
        no_system_batch, no_golden_batch = get_batch_measure(model_output.split_batch,
                                                             golden_metric_splits_batch)

        correct_span = correct_span + correct_span_batch
        correct_relation = correct_relation + correct_relation_batch
        correct_nuclearity = correct_nuclearity + correct_nuclearity_batch
        no_system = no_system + no_system_batch
        no_golden = no_golden + no_golden_batch

    span_points, relation_points, nuclearity_points = get_micro_measure(
        correct_span, correct_relation, correct_nuclearity, no_system, no_golden)

    return np.mean(loss_tree_all), np.mean(loss_label_all), span_points, relation_points, nuclearity_points


def learning_rate_adjust(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=50):
    if (epoch % lr_decay_epoch == 0) and (epoch != 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay


def log_metrics(log_prefix, loss_tree, loss_label, f1_span, f1_relation, f1_nuclearity):
    logger.info(f'{log_prefix} \n'
                f'\t'
                f'loss_tree: {loss_tree:.3f}, loss_label: {loss_label:.3f} \n'
                f'\t'
                f'f1_span: {f1_span:.3f}, f1_relation: {f1_relation:.3f}, f1_nuclearity: {f1_nuclearity:.3f}')


def train_parser(cfg: RstPointerParserTrainArgs) -> None:
    logger.info(f'===== Training RST Pointer Parser =====')

    # Setup
    setup(seed=cfg.seed)

    train_data_dir = cfg.train_data_dir
    test_data_dir = cfg.test_data_dir
    save_dir = cfg.save_dir
    batch_size = cfg.batch_size
    hidden_size = cfg.hidden_size
    rnn_layers = cfg.rnn_layers
    dropout_e = cfg.dropout_e
    dropout_d = cfg.dropout_d
    dropout_c = cfg.dropout_c
    atten_model = cfg.atten_model
    classifier_input_size = cfg.classifier_input_size
    classifier_hidden_size = cfg.classifier_hidden_size
    classifier_bias = cfg.classifier_bias
    elmo_size = cfg.elmo_size
    seed = cfg.seed
    eval_size = cfg.eval_size
    epochs = cfg.epochs
    lr = cfg.lr
    lr_decay_epoch = cfg.lr_decay_epoch
    weight_decay = cfg.weight_decay
    highorder = cfg.highorder

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:" + str(cfg.gpu_id) if USE_CUDA else "cpu")
    logger.info(f'Using CUDA: {USE_CUDA}')

    logger.info('Loading training and test data...')
    # Load Training data
    tr_input_sentences = pickle.load(open(os.path.join(train_data_dir, "tokenized_sentences.pickle"), "rb"))
    tr_edu_breaks = pickle.load(open(os.path.join(train_data_dir, "edu_breaks.pickle"), "rb"))
    tr_decoder_input = pickle.load(open(os.path.join(train_data_dir, "decoder_input_index.pickle"), "rb"))
    tr_relation_label = pickle.load(open(os.path.join(train_data_dir, "relation_index.pickle"), "rb"))
    tr_parsing_breaks = pickle.load(open(os.path.join(train_data_dir, "splits_order.pickle"), "rb"))
    tr_golden_metric = pickle.load(open(os.path.join(train_data_dir, "discourse_tree_splits.pickle"), "rb"))
    tr_parents_index = pickle.load(open(os.path.join(train_data_dir, "parent_index.pickle"), "rb"))
    tr_sibling_index = pickle.load(open(os.path.join(train_data_dir, "sibling_index.pickle"), "rb"))

    # Load Testing data
    test_input_sentences = pickle.load(open(os.path.join(test_data_dir, "tokenized_sentences.pickle"), "rb"))
    test_edu_breaks = pickle.load(open(os.path.join(test_data_dir, "edu_breaks.pickle"), "rb"))
    test_decoder_input = pickle.load(open(os.path.join(test_data_dir, "decoder_input_index.pickle"), "rb"))
    test_relation_label = pickle.load(open(os.path.join(test_data_dir, "relation_index.pickle"), "rb"))
    test_parsing_breaks = pickle.load(open(os.path.join(test_data_dir, "splits_order.pickle"), "rb"))
    test_golden_metric = pickle.load(open(os.path.join(test_data_dir, "discourse_tree_splits.pickle"), "rb"))

    # To save model and data
    file_name = f'seed_{seed}_batchSize_{batch_size}_elmo_{elmo_size}_attenModel_{atten_model}' \
                f'_rnnLayers_{rnn_layers}_rnnHiddenSize_{hidden_size}_classifierHiddenSize_{classifier_hidden_size}'

    model_save_dir = os.path.join(save_dir, file_name)

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting model training...')
    logger.info('--------------------------------------------------------------------')
    # Initialize model
    # model_config = RstPointerParserConfig.from_pretrained(cfg.model_config_path)
    model_config = RstPointerParserConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        decoder_input_size=hidden_size,
        atten_model=atten_model,
        classifier_input_size=classifier_input_size,
        classifier_hidden_size=classifier_hidden_size,
        highorder=highorder,
        classes_label=39,
        classifier_bias=classifier_bias,
        rnn_layers=rnn_layers,
        dropout_e=dropout_e,
        dropout_d=dropout_d,
        dropout_c=dropout_c,
        elmo_size=elmo_size
    )

    model = RstPointerParserModel(model_config)
    model = model.to(device)
    # model.embedding.to(device)  # Elmo layer doesn't get put onto device automatically

    preprocessor = RSTPreprocessor(device=device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, betas=(0.9, 0.9), weight_decay=weight_decay)

    num_iterations = int(np.ceil(len(tr_parsing_breaks) / batch_size))

    try:
        os.mkdir(model_save_dir)
    except:
        pass

    best_f1_relation = 0
    best_f1_span = 0
    for current_epoch in range(epochs):
        learning_rate_adjust(optimizer, current_epoch, 0.8, lr_decay_epoch)

        for current_iteration in range(num_iterations):
            input_sentences_batch, edu_breaks_batch, decoder_input_batch, \
            relation_label_batch, parsing_breaks_batch, _, parents_index_batch, \
            sibling_batch = get_batch_data_training(
                tr_input_sentences, tr_edu_breaks,
                tr_decoder_input, tr_relation_label,
                tr_parsing_breaks, tr_golden_metric,
                tr_parents_index, tr_sibling_index, batch_size)

            model.zero_grad()

            input_sentences_ids_batch, sentence_lengths = preprocessor(input_sentences_batch)

            loss_tree_batch, loss_label_batch = model.forward_train(
                input_sentence_ids_batch=input_sentences_ids_batch,
                edu_breaks_batch=edu_breaks_batch,
                label_index_batch=relation_label_batch,
                parsing_index_batch=parsing_breaks_batch,
                decoder_input_index_batch=decoder_input_batch,
                parents_index_batch=parents_index_batch,
                sibling_index_batch=sibling_batch,
                sentence_lengths=sentence_lengths
            )

            loss = loss_tree_batch + loss_label_batch
            loss.backward()

            cur_loss = float(loss.item())

            logger.info(f'Epoch: {current_epoch + 1}/{epochs}, '
                        f'iteration: {current_iteration + 1}/{num_iterations}, '
                        f'loss: {cur_loss:.3f}')

            # To avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

        # Convert model to eval
        model.eval()

        # Eval on Testing data
        loss_tree_test, loss_label_test, span_points_test, relation_points_test, nuclearity_points_test = \
            get_accuracy(model, preprocessor, test_input_sentences, test_edu_breaks,
                         test_decoder_input, test_relation_label,
                         test_parsing_breaks, test_golden_metric, batch_size)

        # Unfold numbers
        # Test
        precision_span, recall_span, f1_span = span_points_test
        precision_relation, recall_relation, f1_relation = relation_points_test
        precision_nuclearity, recall_nuclearity, f1_nuclearity = nuclearity_points_test

        # Relation will take the priority consideration
        if f1_relation > best_f1_relation:
            best_epoch = current_epoch
            # relation
            best_f1_relation = f1_relation
            best_precision_relation = precision_relation
            best_recall_relation = recall_relation
            # span
            best_f1_span = f1_span
            best_precision_span = precision_span
            best_recall_span = recall_span
            # nuclearity
            best_f1_nuclearity = f1_nuclearity
            best_precision_nuclearity = precision_nuclearity
            best_recall_nuclearity = recall_nuclearity

        # Saving data
        save_data = [current_epoch, loss_tree_test, loss_label_test, f1_span, f1_relation, f1_nuclearity]

        # Log evaluation and test metrics
        log_metrics(log_prefix='Metrics on test data --',
                    loss_tree=loss_tree_test, loss_label=loss_label_test,
                    f1_span=f1_span, f1_relation=f1_relation, f1_nuclearity=f1_nuclearity)

        logger.info(f'End of epoch {current_epoch + 1}')
        file_name = f'span_bs_{batch_size}_es_{eval_size}_lr_{lr}_' \
                    f'lrdc_{lr_decay_epoch}_wd_{weight_decay}.txt'

        with open(os.path.join(model_save_dir, file_name), 'a+') as f:
            f.write(','.join(map(str, save_data)) + '\n')

        # Saving model
        if best_epoch == current_epoch:
            model.save_pretrained(model_save_dir)

        # Convert back to training
        model.train()

    logger.info('--------------------------------------------------------------------')
    logger.info('Model training completed!')
    logger.info('--------------------------------------------------------------------')
    logger.info(f'The best F1 points for Relation is: {best_f1_relation:.3f}.')
    logger.info(f'The best F1 points for Nuclearity is: {best_f1_nuclearity:.3f}')
    logger.info(f'The best F1 points for Span is: {best_f1_span:.3f}')

    # Save result
    with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=['best_epoch', 'f1_relation', 'precision_relation', 'recall_relation',
                                               'f1_span', 'precision_span', 'recall_span',
                                               'f1_nuclearity', 'precision_nuclearity', 'recall_nuclearity'])
        writer.writerow({
            'best_epoch': best_epoch,
            'f1_relation': best_f1_relation,
            'precision_relation': best_precision_relation,
            'recall_relation': best_recall_relation,
            'f1_span': best_f1_span,
            'precision_span': best_precision_span,
            'recall_span': best_recall_span,
            'f1_nuclearity': best_f1_nuclearity,
            'precision_nuclearity': best_precision_nuclearity,
            'recall_nuclearity': best_recall_nuclearity
        })


# Segmenter training code
def sample_a_sorted_batch_from_numpy(input_x, output_y, batch_size, use_cuda):
    input_x = np.array(input_x, dtype="object")
    output_y = np.array(output_y, dtype="object")

    if batch_size is not None:
        select_index = random.sample(range(len(output_y)), batch_size)
    else:
        select_index = np.array(range(len(output_y)))

    batch_x = copy.deepcopy(input_x[select_index])
    batch_y = copy.deepcopy(output_y[select_index])

    all_lens = np.array([len(x) for x in batch_x])

    idx = np.argsort(all_lens)
    idx = idx[::-1]  # decreasing

    batch_x = batch_x[idx]

    batch_y = batch_y[idx]

    # decoder input
    batch_x_index = []

    for i in range(len(batch_y)):
        cur_y = batch_y[i]

        temp = [x + 1 for x in cur_y]
        temp.insert(0, 0)
        temp.pop()
        batch_x_index.append(temp)

    all_lens = all_lens[idx]

    return batch_x, batch_x_index, batch_y, all_lens


def get_batch_test(x, y, batch_size):
    x = np.array(x, dtype="object")
    y = np.array(y, dtype="object")

    if batch_size is not None:
        select_index = random.sample(range(len(y)), batch_size)
    else:
        select_index = np.array(range(len(y)))

    batch_x = copy.deepcopy(x[select_index])
    batch_y = copy.deepcopy(y[select_index])

    all_lens = np.array([len(x) for x in batch_x])

    return batch_x, batch_y, all_lens


class TrainSolver(object):
    def __init__(self, model, train_x, train_y, dev_x, dev_y, save_path, batch_size, eval_size, epoch, lr,
                 lr_decay_epoch, weight_decay, use_cuda):

        self.lr = lr
        self.model = model
        self.num_epochs = epoch
        self.train_x = train_x
        self.train_y = train_y
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.lr_decay_epoch = lr_decay_epoch
        self.eval_size = eval_size
        self.dev_x, self.dev_y = dev_x, dev_y
        self.model = model
        self.save_path = save_path
        self.weight_decay = weight_decay

    def sample_dev(self):

        select_index = random.sample(range(len(self.train_y)), self.eval_size)

        train_x = np.array(self.train_x, dtype="object")
        train_y = np.array(self.train_y, dtype="object")
        test_tr_x = train_x[select_index]
        test_tr_y = train_y[select_index]

        return test_tr_x, test_tr_y

    def get_batch_micro_metric(self, pre_b, ground_b):
        All_C = []
        All_R = []
        All_G = []
        for i in range(len(ground_b)):
            index_of_1 = np.array(ground_b[i])
            index_pre = pre_b[i]

            index_pre = np.array(index_pre)

            END_B = index_of_1[-1]
            index_pre = index_pre[index_pre != END_B]
            index_of_1 = index_of_1[index_of_1 != END_B]

            no_correct = len(np.intersect1d(list(index_of_1), list(index_pre)))
            All_C.append(no_correct)
            All_R.append(len(index_pre))
            All_G.append(len(index_of_1))

        return All_C, All_R, All_G

    def get_batch_metric(self, pre_b, ground_b):
        b_pr = []
        b_re = []
        b_f1 = []
        for i, cur_seq_y in enumerate(ground_b):
            index_of_1 = np.where(cur_seq_y == 1)[0]
            index_pre = pre_b[i]

            no_correct = len(np.intersect1d(index_of_1, index_pre))

            cur_pre = no_correct / len(index_pre)
            cur_rec = no_correct / len(index_of_1)
            cur_f1 = 2 * cur_pre * cur_rec / (cur_pre + cur_rec)

            b_pr.append(cur_pre)
            b_re.append(cur_rec)
            b_f1.append(cur_f1)

        return b_pr, b_re, b_f1

    def check_accuracy(self, x, y):
        num_loops = int(np.ceil(len(y) / self.batch_size))

        all_ave_loss = []
        all_start_boundaries = []
        all_end_boundaries = []
        all_index_decoder_y = []
        all_x_save = []

        all_C = []
        all_R = []
        all_G = []
        for i in range(num_loops):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            if end_idx > len(y):
                end_idx = len(y)

            batch_x, batch_y, all_lens = get_batch_test(x[start_idx:end_idx], y[start_idx:end_idx], None)

            output = self.model(batch_x, all_lens, batch_y)
            batch_ave_loss = output.batch_loss
            batch_start_boundaries = output.batch_start_boundaries
            batch_end_boundaries = output.batch_end_boundaries
            # batch_ave_loss, batch_start_boundaries, batch_end_boundaries = self.model(batch_x, all_lens, batch_y)

            all_ave_loss.extend([batch_ave_loss.cpu().data.numpy()])
            all_start_boundaries.extend(batch_start_boundaries)
            all_end_boundaries.extend(batch_end_boundaries)

            ba_C, ba_R, ba_G = self.get_batch_micro_metric(batch_end_boundaries, batch_y)

            all_C.extend(ba_C)
            all_R.extend(ba_R)
            all_G.extend(ba_G)

        ba_pre = np.sum(all_C) / np.sum(all_R)
        ba_rec = np.sum(all_C) / np.sum(all_G)
        ba_f1 = 2 * ba_pre * ba_rec / (ba_pre + ba_rec)

        return np.mean(all_ave_loss), ba_pre, ba_rec, ba_f1, \
               (all_x_save, all_index_decoder_y, all_start_boundaries, all_end_boundaries)

    def adjust_learning_rate(self, optimizer, epoch, lr_decay=0.5, lr_decay_epoch=50):
        if (epoch % lr_decay_epoch == 0) and (epoch != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def train(self):
        test_train_x, test_train_y = self.sample_dev()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                     weight_decay=self.weight_decay)

        num_iterations = int(np.round(len(self.train_y) / self.batch_size))

        os.makedirs(self.save_path, exist_ok=True)

        best_epoch = 0
        best_f1 = 0

        for current_epoch in range(self.num_epochs):

            self.adjust_learning_rate(optimizer, current_epoch, 0.8, self.lr_decay_epoch)

            track_epoch_loss = []
            for current_iter in range(num_iterations):
                batch_x, batch_x_index, batch_y, all_lens = sample_a_sorted_batch_from_numpy(
                    self.train_x, self.train_y, self.batch_size, self.use_cuda)

                self.model.zero_grad()

                neg_loss = self.model.neg_log_likelihood(batch_x, batch_x_index, batch_y, all_lens)
                neg_loss_v = float(neg_loss.data)

                track_epoch_loss.append(neg_loss_v)
                logger.info(f'Epoch: {current_epoch + 1}/{self.num_epochs}, '
                            f'iteration: {current_iter + 1}/{num_iterations}, '
                            f'loss: {neg_loss_v:.3f}')

                neg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

            self.model.eval()

            logger.info('Running end of epoch evaluations on sample train data and test data...')
            tr_batch_ave_loss, tr_pre, tr_rec, tr_f1, tr_visdata = self.check_accuracy(test_train_x,
                                                                                       test_train_y)

            dev_batch_ave_loss, dev_pre, dev_rec, dev_f1, dev_visdata = self.check_accuracy(self.dev_x, self.dev_y)
            _, _, _, all_end_boundaries = dev_visdata

            logger.info(f'train sample -- loss: {tr_batch_ave_loss:.3f}, '
                        f'precision: {tr_pre:.3f}, recall: {tr_rec:.3f}, f1: {tr_f1:.3f}')
            logger.info(f'test sample -- loss: {dev_batch_ave_loss:.3f}, '
                        f'precision: {dev_pre:.3f}, recall: {dev_rec:.3f}, f1: {dev_f1:.3f}')

            if best_f1 < dev_f1:
                best_f1 = dev_f1
                best_rec = dev_rec
                best_pre = dev_pre
                best_epoch = current_epoch

            save_data = [current_epoch, tr_batch_ave_loss, tr_pre, tr_rec, tr_f1,
                         dev_batch_ave_loss, dev_pre, dev_rec, dev_f1]

            save_file_name = f'bs_{self.batch_size}_es_{self.eval_size}_lr_{self.lr}_lrdc_{self.lr_decay_epoch}_' \
                             f'wd_{self.weight_decay}_epoch_loss_acc_pk_wd.txt'
            with open(os.path.join(self.save_path, save_file_name), 'a+') as f:
                f.write(','.join(map(str, save_data)) + '\n')

            if current_epoch == best_epoch:
                logger.info('Saving best model...')
                self.model.save_pretrained(self.save_path)

                with open(os.path.join(self.save_path, 'best_segmentation.pickle'), 'wb') as f:
                    pickle.dump(all_end_boundaries, f)

            self.model.train()

        return best_epoch, best_pre, best_rec, best_f1


def train_segmenter(cfg: RstPointerSegmenterTrainArgs) -> None:
    logger.info(f'===== Training RST Pointer Segmenter =====')

    setup(seed=cfg.seed)

    train_data_dir = cfg.train_data_dir
    test_data_dir = cfg.test_data_dir
    save_dir = cfg.save_dir

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    logger.info(f'Using CUDA: {use_cuda}')
    iscudnn = cfg.iscudnn == 'True'

    cudnn.enabled = iscudnn
    hidden_dim = cfg.hdim
    rnn_type = cfg.rnn
    rnn_layers = cfg.rnnlayers
    lr = cfg.lr
    dout = cfg.dout
    wd = cfg.wd
    myseed = cfg.seed
    batch_size = cfg.bsize
    lrdepoch = cfg.lrdepoch
    elmo_size = cfg.elmo_size

    is_bidirectional = cfg.isbi == 'True'
    finetune = cfg.fine == 'True'
    is_batch_norm = cfg.isbarnor == 'True'

    tr_x = pickle.load(open(os.path.join(train_data_dir, "tokenized_sentences.pickle"), "rb"))
    tr_y = pickle.load(open(os.path.join(train_data_dir, "edu_breaks.pickle"), "rb"))

    dev_x = pickle.load(open(os.path.join(test_data_dir, "tokenized_sentences.pickle"), "rb"))
    dev_y = pickle.load(open(os.path.join(test_data_dir, "edu_breaks.pickle"), "rb"))

    filename = 'elmoLarge_dot_' + str(myseed) + 'seed_' + str(hidden_dim) + 'hidden_' + \
               str(is_bidirectional) + 'bi_' + rnn_type + 'rnn_' + str(finetune) + 'Fined_' + str(rnn_layers) + \
               'rnnlayers_' + str(lr) + 'lr_' + str(dout) + 'dropout_' + str(wd) + 'weightdecay_' + str(
        batch_size) + 'bsize_' + str(lrdepoch) + 'lrdepoch_' + \
               str(is_batch_norm) + 'barnor_' + str(iscudnn) + 'iscudnn'

    model_config = RstPointerSegmenterConfig(hidden_dim=hidden_dim,
                                             is_bi_encoder_rnn=is_bidirectional,
                                             rnn_type=rnn_type, rnn_layers=rnn_layers,
                                             dropout_prob=dout, use_cuda=use_cuda, with_finetuning=finetune,
                                             is_batch_norm=is_batch_norm, elmo_size=elmo_size)
    model = RstPointerSegmenterModel(model_config)
    model.to(device=device)

    # Arbitrary eval_size
    eval_size = len(dev_x) * 2 // 3

    save_path = os.path.join(save_dir, filename)
    mysolver = TrainSolver(model, tr_x, tr_y, dev_x, dev_y, save_path,
                           batch_size=batch_size, eval_size=eval_size, epoch=cfg.epochs, lr=lr, lr_decay_epoch=lrdepoch,
                           weight_decay=wd,
                           use_cuda=use_cuda)

    best_epoch, best_pre, best_rec, best_f1 = mysolver.train()

    with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=["best_epoch", "precision", "recall", "f1"])
        writer.writerow({
            'best_epoch': best_epoch,
            'precision': best_pre,
            'recall': best_rec,
            'f1': best_f1
        })


if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    if isinstance(cfg, RstPointerSegmenterTrainArgs):
        train_segmenter(cfg)
        print(cfg)
    if isinstance(cfg, RstPointerParserTrainArgs):
        train_parser(cfg)
        print(cfg)
