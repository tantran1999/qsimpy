from abc import ABC, abstractmethod

from gymenv_qsimpy import QSimPyEnv


class BaseScheduler(ABC):
    def __init__(self):
        ...
    
    @abstractmethod
    def select_qnode(self) -> int:
        ...


class GreedyScheduler(BaseScheduler):
    def __init__(self, env: QSimPyEnv):
        self.env = env

    def select_qnode(self) -> int:
        qnode_with_min = min(self.env.qnodes, key=lambda x: x.next_available_time)
        is_statisfied, _ = self.env.broker.check_qtask_constraints(
            qtask=self.env.current_qtask,
            qnode=self.env.qnodes[qnode_with_min.id]
        )

        if is_statisfied:
            return qnode_with_min.id

        # Get the QNode with largest qubit_number
        max_qubit_number = max(self.env.qnodes, key=lambda x: x.qubit_number).qubit_number
        qnodes_with_max_qubit = [qnode for qnode in self.env.qnodes if qnode.qubit_number == max_qubit_number]
        select_qnode = min(qnodes_with_max_qubit, key=lambda x: x.next_available_time)

        return select_qnode.id


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, env: QSimPyEnv):
        self.env = env
        self.current_executing_qnode = 0

    def _step(self):
        self.current_executing_qnode += 1
        if self.current_executing_qnode > (self.env.n_qnodes - 1):
            self.current_executing_qnode %= self.env.n_qnodes

    def select_qnode(self) -> int:
        qnode =  self.current_executing_qnode

        self._step()

        return qnode