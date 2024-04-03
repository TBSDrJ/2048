# 2048

This repo contains the following:
1. [2048_old_py27.py](2048_old_py27.py): An old Python 2.7 implementation of 2048 that I wrote many years ago.  My idea at the time was that it would be more fun to write a program that would beat me at 2048 than it would be to play 2048 myself.  I was correct, as far as I was concerned.  I wrote this a looong time ago in terms of my knowledge of Python and good coding practices, so it is a bit rough around the edges.  It has three main components:
    1. A way to just play 2048 with a text-based interface.
    1. A basic greedy look-ahead algorithm: check all possible moves n steps ahead several times, make the computer's next move be the first move of the sequence with the highest average result.
    1. A genetic look-ahead algorithm: generate several random sequences of moves, check which are better, keep those, toss the rest, randomly modify the better ones, check which are better, keep those, toss the rest, etc..  After a specified number of generations, make the computer's next move the first move of the sequence with the highest average result.
1. [Game2048.py](Game2048.py): This is a Python 3 class that implements 2048 as a class (`Game2048`) with a simple API.  This has much better coding practices and documentation (including this README). This implementation does not include any rendering, it is just a game engine.  The file also contains a text-based rendering engine as a class (`Text2048`).  My goal is to create a `tf-agents` environment out of this and train an AI-powered agent to beat me at 2048.

## `Game2048` Documentation

Documentation for the `Game2048` class:

### Attributes:

1. `.width`: `int`: The number of columns in the game board.
1. `.height`: `int`: The number of rows in the game board.
1. `.prob_4`: `float`: The probability that a newly generated tile has a value of 4.  All other tiles have a value of 2.
1. `.game_over`: `bool`: `True` if the game is over, `False` if there is still at least one legal move.
1. `.score`: `int`: The current score of the game.
1. `.board`: `list[list[int]]`: The game board, with each entry representing the value of the tile at that location. Empty tiles have a value of 0.  Notice that `len(self.board) = self.height` and `len(self.board[0]) = self.width`.

### `Game2048` constructor

The constructor takes three optional arguments, which can be provided positionally in the order listed here, or as keyword arguments:
1. `width`: `int`, default: `4`.  Number of columns in the game board.
1. `height`: `int`, default: `4`.  Number of rows in the game board.
1. `prob_4`: `float`, default: `0.1`.  Probability that a new tile is a 4.  Otherwise, any new tile will be a 2.

Examples:

#### Example 1

```python
from Game2048 import Game2048

game = Game2048()
```

Then `game` is an instance of 2048 with the usual setup: 4x4 grid, 10% of new tiles have a value of 4, 90% have a value of 2.

#### Example 2

```python
from Game2048 import Game2048

game = Game2048(5, 3, 0.25)
```
Then `game` is an instance of 2048 with 5 columns, 3 rows, and where 25% of new tiles have a value of 4, and 75% have a value of 2.

#### Example 3

```python
from Game2048 import Game2048

game = Game2048(prob_4 = 0.05)
```
Then `game` is an instance of 2048 with 4 columns, 4 rows, where 5% of new tiles have a value of 4, and 95% have a value of 2.

### `.one_turn()` 

The `one_turn()` method takes one required argument:
1. `move`: `int`.  This is the direction of the attempted move: 0=Up, 1=Left, 2=Down, 3=Right (following WASD order).

The `one_turn()` method has one return value:
1. `bool`.  This is `True` if the game is over at the of this turn, and `False` if the game is not over at the end of the turn.

#### Implementation Detail

If a move is supplied, and the board does not change at all after that move is applied, then a new tile is not added to the board -- something must move in order for the game to consider a turn to have passed.

#### Example 1

```python
from Game2048 import Game2048

game = Game2048()
changed = game.one_turn(0)
```

Then, `game` is a 2048 instance with default settings, where both of the initial tiles have been moved up.  If either of the initial tiles moved during the up move, then `changed` is `True`, and a third tile will have been randomly generated and placed on the board.  If both of the initial tiles were already in the top row before the move, then `changed` is `False` and there will still be only two tiles on the board.

#### Example 2

```python
from random import randrange
from Game2048 import Game2048

game = Game2048()
while not game.game_over:
    game.one_turn(randrange(4))
print(game.score)
```

Then, `game` is a 2048 instance and then the program will run random moves over and over until the game is over, then it will print the final score.

#### Example 3

```python
from Game2048 import Game2048

game = Game2048()
while not game.game_over:
    changed_0 = game.one_turn(0)
    changed_1 = game.one_turn(1)
    if not changed_0 and not changed_1:
        changed_2 = game.one_turn(2)
    if not changed_0 and not changed_1 and not changed_2:
        game.one_turn(3)
print(game.score)
```

Then, `game` is a 2048 instance and then the program will run Up, Left, Up, Left, ... moves. If neither Up nor Left move accomplishes anything , it will move Down, and if that doesn't acomplish anything, it will move Right.  It will continue to do this until the game ends, then it will print the final score.  

This is not a great strategy, but it is not actually terrible either.  In my limited testing, it seems to be a bit more than twice as good as random moves.

The remaining methods are just components of `.one_turn()`, but are exposed in case they are useful:

### `.merge_up()`, `.merge_left()`, `.merge_down()`, `.merge_right()`

These carry out the merge step of the move.

The `.merge_x()` methods all have one optional argument:
1. `execute`: `bool`, default: `True`.  If `execute` is `True`, then `self.board` will be changed to reflect the move.  If `execute` is `False`, then `self.board` will be unchanged, but the return value will indicate whether or not the board would have been changed if this move were carried out.  The `False` case was included to make it easy to test if the game is over or not -- the game is over when none of the four possible moves changes the board.

These have one return value:
1. `bool`: Whether or not the board was/would have been changed by carrying out this move.

### `.game_over_check()`

This method just checks if the game is over or not, by checking if there is a legal move remaining.  It does not change the board at all.

This takes no arguments, and has one return:
1. `bool`: `True` if the game is over, and `False` if there is at least one legal move remaining.

### `.add_tile()`

This adds one tile to the board.  The location is randomly generated, using a (approx.) uniform distribution across only the empty locations.  The value of the tile is also randomly determined: either a 4, using `self.prob_4` as the probability this occurs, or a 2, otherwise.

### Private methods

There are also a few private methods that are not intended for outside access, you can use introspection or read the source code if you want to cause trouble with these.