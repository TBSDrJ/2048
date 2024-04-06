import random
import tensorflow as tf
import numpy as np

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import ArraySpec
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import TimeStep

from Game2048 import Game2048

class PyEnv2048(PyEnvironment):
    """A TF-Agents Python Environment for my implementation of 2408."""
    def __init__(self, width=4, height=4, prob_4=0.1):
        self.width = width
        self.height = height
        self.prob_4 = prob_4
        self.discount = 0.966 # 1 point 20 rounds later is worth 0.5 pts now.
        self.penalty = 1000 # Subtracted from reward if game ends
        super().__init__()

    def observation_spec(self):
        """Spec for an observation of the 2048 board.
        
        I've chosen to *not* use a BoundedArraySpec to allow the value of the
        tiles to grow essentially without limit."""
        return ArraySpec(shape=(self.width, self.height), dtype=np.int64,
                name='observation')

    def action_spec(self):
        """Spec for an action.  Using integers 0-3 to match underlying class."""
        return BoundedArraySpec(shape=(), dtype=np.int64, minimum=0,
                maximum=3, name='action')

    def _reset(self):
        """Return a time step with a new game of 2048."""
        self.game = Game2048(
                width=self.width, 
                height=self.height, 
                prob_4=self.prob_4,
        )
        return TimeStep(
                step_type = 0,
                reward = 0,
                discount = self.discount,
                observation = self.game.board,
        )

    def _step(self, action):
        """Return a time step in the game with the action applied to it."""
        score_before = self.game.score
        self.game.one_turn(action)
        if not self.game.game_over:
            return TimeStep(
                    step_type = 1, 
                    reward = self.game.score - score_before,
                    discount = self.discount,
                    observation = self.game.board,
            )
        else:
            return TimeStep(
                    step_type = 2,
                    reward = self.game.score - score_before - self.penalty,
                    discount = self.discount,
                    observation = self.game.board,
            )



if __name__ == "__main__":
    env = PyEnv2048()
    print(env.time_step_spec())
