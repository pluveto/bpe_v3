import collections
import logging
import os
from typing import List

import numpy as np
from locker import MutexMap
from owned_mutex import WithMutex

from paralled_task import ParalledTask


logger = logging.getLogger("default")


class BpeCn:
    def __init__(self, options):
        """
        options:
            - train_file: 训练文件。
            - vocab_file: 输出文件。
            - nworkers: 并行协程数
        """
        # 加载内部训练数据源
        logger.info("Loading train data from {} ...".format(
            options.train_file))
        self._train_lines = open(
            options.train_file, 'r', encoding='utf-8').readlines()
        logger.info(
            "Loading train data (total {} lines) done.".format(len(self._train_lines)))
        self.nworker = options.nworker
        self.vocab_file = options.vocab_file
        # 自动进入预处理阶段
        self._preproc()

    def partition(self, lines: np.ndarray, nworker: int) -> List[List[str]]:
        """将训练数据分割为 nworker 份。"""
        # 分割数据
        datasets = []
        for i in range(nworker):
            datasets.append([])
        for i in range(len(lines)):
            datasets[i % nworker].append(lines[i])
        return datasets

    def _preproc(self):
        """进行数据的预处理。"""
        logger.info("Preprocessing train data ...")
        # 每行切分为字符列表
        train_lines = [line.strip().split() for line in self._train_lines]
        self._train_lines = None
        self._train_lines_np = np.array(train_lines, dtype=object)

        puncs_zh = ['。', '，', '？', '！', '；', '：', '、', '（', '）', '「',
                    '」', '“', '”', '‘', '’', '《', '》', '【', '】', '…', '—', '～', '　']
        puncs_en = ['.', ',', '?', '!', ';', ':',
                    '(', ')', '"', '"', '\'', '\'', '<', '>', '[', ']', '...', '~']
        puncs = [*puncs_zh, *puncs_en]
        # 替换标点符号为 `#`

        def _replace_worker(ctx: dict):
            task = ctx.get('task')
            bar = ctx.get('bar')
            worker_id = ctx.get('worker_id')
            line_strs: List[List[str]] = ctx.get('datasets')[worker_id]
            for line in line_strs:
                for i in range(len(line)):
                    if line[i] in puncs:
                        line[i] = '#'
                with bar.mutex:
                    bar.obj()

        logger.info("--removing duplication.")
        self._train_lines_np = np.unique(self._train_lines_np)
        logger.info("--removing duplication done.")

        self.datasets = self.partition(self._train_lines_np, self.nworker)
        ParalledTask.create('-- removing punc')\
            .set_nworker(self.nworker)\
            .set_worker_func(_replace_worker)\
            .set_progress_goal(len(self._train_lines_np))\
            .set_worker_args({'datasets': self.datasets})\
            .execute()
        logger.info("Preprocessing train data done.")

    def train(self):
        """
        训练BPE模型
        Args:
            out_file (str): 输出文件
        """
        # 创建字典，键为词，如“电脑”，值为词频，如“100”
        self.vocab_map = MutexMap(collections.defaultdict(int))
        # 创建并行数据集
        ds = self.datasets
        # 词频统计总表。用于多个执行器间共享
        shared_freq_stat_map = None
        min_thold = len(self._train_lines_np) / 2000
        logger.info("--min_thold: {}".format(min_thold))

        def _train_worker(ctx: dict):
            """单行原始数据的处理执行器
            Args:
                line_strs (list): 单行数据，已经切分为字符列表。例如：['我', '爱', '北', '京', '天安', '门']
            """
            inner_freq_stat_map = collections.defaultdict(int)
            task = ctx.get('task')
            bar: WithMutex = ctx.get('bar')
            worker_id = ctx.get('worker_id')
            line_strs: List[List[str]] = ctx.get('datasets')[worker_id]

            for line in line_strs:
                # 对每个 Byte Pair 进行处理
                for i in range(len(line) - 1):
                    # 如果是 `#`，则跳过
                    if line[i] == '#' or line[i+1] == '#':
                        continue
                    # 获取当前词和下一个词，如 '天安', '门'
                    cur_word: str = line[i]
                    next_word: str = line[i + 1]
                    # 当前词和下一个词拼接，如 '天安门'
                    cur_word_next_word: str = cur_word + next_word
                    # 当前词和下一个词拼接的词频
                    freq = inner_freq_stat_map[cur_word_next_word]
                    # 当前词和下一个词拼接的词频加1
                    inner_freq_stat_map[cur_word_next_word] = freq + 1
                with bar.mutex:
                    bar.obj()

            # 将当前行的词频统计表加入总表
            with shared_freq_stat_map.mutex:
                for key, value in inner_freq_stat_map.items():
                    shared_freq_stat_map.obj[key] += value

        def _connect_worker(ctx: dict):
            """分词执行器
            Args:
                ctx (dict):
                * `task`
                * `bar`
                * `worker_id`
                * `line_strs`
                * `thold`
            """
            bar: WithMutex = ctx.get('bar')
            worker_id = ctx.get('worker_id')
            line_strs: List[List[str]] = ctx.get('datasets')[worker_id]
            thold = ctx.get('thold')

            for line in line_strs:
                # 对每个 Byte Pair 进行处
                i = 0
                while(i < len(line) - 1):
                    # 如果是 `#`，则跳过
                    if line[i] == '#' or line[i+1] == '#':
                        i += 1
                        continue
                    # 获取当前词和下一个词，如 '天安', '门'
                    cur_word: str = line[i]
                    next_word: str = line[i + 1]
                    # 当前词和下一个词拼接，如 '天安门'
                    cur_word_next_word: str = cur_word + next_word
                    # 比较是否大于阈值
                    if shared_freq_stat_map.obj[cur_word_next_word] > thold:
                        if shared_freq_stat_map.obj[cur_word] / shared_freq_stat_map.obj[cur_word_next_word]\
                                > 1.1:
                            i += 1
                            continue
                        # 如果大于阈值，则连接
                        line[i] = cur_word_next_word
                        # 删除下一个词
                        line.pop(i + 1)
                        i += 1
                    i += 1
                with bar.mutex:
                    bar.obj()

        round_num = 1
        max_round_num = 4  # self.max_round_num
        nline = len(self._train_lines_np)
        baseline = int(nline / 12)
        ntok_tholds = [int(baseline*1.5), int(baseline*1.3),
                       int(baseline*1.1), baseline]
        logger.info("ntok_tholds: {}".format(ntok_tholds))

        while round_num <= max_round_num:
            logger.info("Round {} start...".format(round_num))
            # 初始化
            shared_freq_stat_map = WithMutex(collections.defaultdict(int))

            ParalledTask.create("--freq stat round={}".format(round_num))\
                .set_nworker(self.nworker)\
                .set_worker_args({'datasets': ds})\
                .set_worker_func(_train_worker)\
                .set_progress_goal(len(self._train_lines_np))\
                .execute()

            # 从高频到低频排序
            sorted_freq_stat_ls = sorted(
                shared_freq_stat_map.obj.items(), key=lambda x: x[1], reverse=True)

            # 用 n 个进行分词
            ntok_thold = ntok_tholds[round_num - 1]
            thold = sorted_freq_stat_ls[0:ntok_thold][-1][1]
            thold = max(min_thold, thold)
            logger.info("Thold: {}".format(thold))

            ParalledTask.create("--connect round={}".format(round_num))\
                .set_nworker(self.nworker)\
                .set_worker_args({'datasets': ds, 'thold': thold})\
                .set_worker_func(_connect_worker)\
                .set_progress_goal(len(self._train_lines_np))\
                .execute()

            logger.info("Round {} done.".format(round_num))
            self.dump(self.vocab_file + "_" + str(round_num),
                      sorted_freq_stat_ls, 10000)

            round_num += 1

        # preview 1000 words
        for i in range(500):
            print(sorted_freq_stat_ls[i][0])

    def dump(self, filename, freq_map, max_num):
        # 如果文件夹不存在，则创建
        output_dir = os.path.dirname(filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存所有词到文件
        with open(filename, 'w', encoding="utf-8") as f:
            i = 0
            for word, freq in freq_map:
                if i >= max_num:
                    break
                f.write(word + '\t' + str(freq) + '\n')
                i += 1
        logger.info("Saved to {}.".format(filename))
