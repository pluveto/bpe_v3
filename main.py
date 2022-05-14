import argparse
from asyncio.log import logger
import logging

import numpy as np

from bpe_cn import BpeCn
from max_match_token import Tokenizer, load_word_freq_map
from owned_mutex import WithMutex
from paralled_task import ParalledTask


def parse_args():
    """解析命令行参数"""
    argparser = argparse.ArgumentParser(
        description='BPE Chinese Implementation')
    argparser.add_argument(
        '-i', '--input', help='input training file', default='data/train_BPE.txt', dest='train_file')
    argparser.add_argument(
        '-o', '--output', help='output vocab file', default='output/vocab_BPE.txt', dest='vocab_file')
    argparser.add_argument(
        '-t', '--test-input', help='input test file (to be tokenized)', default='data/test_BPE.txt', dest='test_file')
    argparser.add_argument(
        '-O', '--test-output', help='output test file (tokenized)', default='output/tokenized.txt', dest='test_output')
    argparser.add_argument(
        '-n', '--nworker', help='number of workers', default=4, dest='nworker', type=int)
    # mode, default is train and test
    argparser.add_argument(
        '-m', '--mode', help='mode: train_test | train | test', default='train_test', dest='mode')
    return argparser.parse_args()


def init_logger(logger):
    """初始化日志记录器"""
    GRAY_COLOR = '\033[1;30m'
    RESET_COLOR = '\033[0m'
    formatter = logging.Formatter(
        GRAY_COLOR + '[%(levelname)s] %(filename)s:%(lineno)d '+RESET_COLOR+'%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def tok_worker(ctx: dict):
    bar :WithMutex = ctx.get('bar')
    worker_id = ctx.get('worker_id')
    tokenizer = ctx.get('tokenizer')
    dataset: list[list[str]] = ctx.get('datasets')[worker_id]
    result_lines = []
    for line in dataset:
        result = tokenizer.tokenize(line)
        result_lines.append(result)
        with bar.mutex:
            bar.obj()
    return result_lines


def tok_reducer(results: dict[list]):
    logger = logging.getLogger("default")
    logger.info("Reducing...")
    lines = []
    for k, v in results.items():
        lines.extend(v)
    logger.info("Reduced.")
    return lines


def main():
    args = parse_args()
    init_logger(logging.getLogger("default"))
    if args.mode == 'train_test' or args.mode == 'train':
        bpeCn = BpeCn(args)
        bpeCn.train()
    if args.mode == 'train_test' or args.mode == 'test':
        logger.info("Training done.")
        logger.info("Start testing...")
        test_file = open(args.test_file, 'r', encoding='utf-8').readlines()
        test_lines = [line.split() for line in test_file]
        tokenizer = Tokenizer()\
            .add_dict(load_word_freq_map("output/vocab_BPE.txt_1"))\
            .add_dict(load_word_freq_map("output/vocab_BPE.txt_2"))\
            .add_dict(load_word_freq_map("output/vocab_BPE.txt_3"))\
            .add_dict(load_word_freq_map("output/vocab_BPE.txt_4"))

        datasets = np.array_split(test_lines, args.nworker)
        result_lines = ParalledTask.create('-- test')\
            .set_nworker(args.nworker)\
            .set_worker_func(tok_worker)\
            .set_reducer_func(tok_reducer)\
            .set_progress_goal(len(test_lines))\
            .set_worker_args({'datasets': datasets, 'tokenizer': tokenizer})\
            .execute()\
            .get_results()

        with open(args.test_output, 'w', encoding='utf-8') as f:
            for line in result_lines:
                f.write(' '.join(line) + '\n')
        logger.info("Test done. Saved to %s" % args.test_output)


if __name__ == '__main__':
    main()
