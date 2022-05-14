import logging
import concurrent.futures
from asyncio.log import logger
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from alive_progress import alive_bar

from owned_mutex import WithMutex

logger = logging.getLogger("default")

class ParalledTask:
    """实现了对并行协程的封装
    """
    @staticmethod
    def create(task_name) -> 'ParalledTask':
        """
        创建并行任务
        """
        instance = ParalledTask()
        instance.task_name = task_name
        instance.results = {} # key: worker_id, value: worker return value
        return instance

    def set_nworker(self, nworker: int) -> 'ParalledTask':
        """
        设置并行协程数
        Args:
            nworker (int): 并行协程数
        """
        self.nworker = nworker
        return self

    def set_worker_func(self, worker_func) -> 'ParalledTask':
        """
        设置并行协程执行器
        Args:
            worker_func (callable): 协程执行器
        """
        self.worker_func = worker_func
        return self

    def set_worker_args(self, worker_args) -> 'ParalledTask':
        """
        设置协程执行器的参数
        Args:
            worker_args (list): 协程执行器的参数
        """
        self.worker_args = worker_args
        return self

    def set_worker_arg_provider_func(self, worker_arg_provider_func):
        """
        设置参数提供函数
        函数原型为：worker_arg_provider_func(worker_id=worker_id, nworker=nworker)
        Args:
            worker_arg_provider_func (callable): 参数提供函数
        """
        self.worker_arg_provider_func = worker_arg_provider_func
        return self

    def set_reducer_func(self, reducer_func) -> 'ParalledTask':
        """
        设置并行任务执行结果合并器
        Args:
            reducer_func (callable): 合并器
        """
        self.reducer_func = reducer_func
        return self

    def set_progress_goal(self, goal: int) -> 'ParalledTask':
        self.progress_goal = goal
        return self

    def execute(self) -> 'ParalledTask':
        """
        执行并行任务
        """
        logger.info(f'{self.task_name} start')
        with ExitStack() as stack, \
                ThreadPoolExecutor(max_workers=self.nworker) as executor:
            ctxs = []
            if hasattr(self, 'progress_goal'):
                goal = self.progress_goal
                # 创建进度条
                bar = stack.enter_context(alive_bar(goal))
                # 使用互斥锁封装进度条
                bar_with_mutex = WithMutex(bar)
            for worker_id in range(self.nworker):
                # if has worker_arg_provider_func attr
                if(hasattr(self, 'worker_arg_provider_func')):
                    worker_arg = self.worker_arg_provider_func(
                        worker_id, self.nworker)
                else:
                    worker_arg = self.worker_args

                worker_ctx = {
                    **worker_arg,
                    'worker_id': worker_id,
                    'task': self,
                    'bar': bar_with_mutex,
                }
                ctxs.append(worker_ctx)
            # 提交任务到执行器
            futures = [executor.submit(self.worker_func, ctxs[i]) for i in range(self.nworker)]
            # 等待完成，并收集执行结果
            for future in concurrent.futures.as_completed(futures):
                workder_id = futures.index(future)
                self.results[workder_id] = future.result()
            # 根据执行器 id 排序，避免乱序
            self.results = {k: v for k, v in sorted(self.results.items(), key=lambda item: item[0])}
        logger.info(f'{self.task_name} done')
        return self

    def get_results(self):        
        """
        获取并行任务执行结果
        """
        # 如果有合并器，则进行合并，否则直接返回结果
        if(self.reducer_func is None):
            return self.results

        return self.reducer_func(self.results)
