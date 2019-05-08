# lumpi
## MLP Chess Engine

Lumpi is a multilayer perceptron (MLP) based chess evaluation tool. A neural net is trained on positions scored by a
conventional chess engine. Given a particular board state, the model predicts the corresponding centipawn value.

External python libraries used:
python-chess,
scikit-learn,
numpy,
tensorflow

Chess analysis engine:
Stockfish (https://stockfishchess.org/)

Parallelism Tools:
GNU-parallel (https://www.gnu.org/software/parallel/)

## TODO:
### High Priority:
![Distribution of Stockfish evaluations from games between master-level players.](https://github.com/casey-martin/lumpi/blob/master/figures/stockfish_eval_dist.png)

Subsample training data. Over the course of a game between two master-level players, the vast majority of chess positions will have roughly an even board position. Training on this full dataset results in the network minimizing error by randomly guessing values around 0. Reducing the relative proportion of roughly equal positions should help mitigate this effect. However, this means we must throw out a majority of the training data for network initialization.

### Long Term:
* Chess 960 compatibility.
  * I suspect that increasing the diversity of board states in the training data will help the network learn more generalizable policies.
* Game tree search.
  * Current plan is a greedy policy: choose the highest evaluated move a half ply deep. 
  * Beta-pruning and MCTS implementation? Python performance will be a limiting factor. Cython implementation of python-chess?
* Amplification reinforcement
  * Train the evaluation function to approximate the output of the evaluation function coupled with a tree search.
* UCI support
