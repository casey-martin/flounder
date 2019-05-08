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

TODO:
Chess 960 compatibility,
Game tree search,
UCI support,
Amplification reinforcement
