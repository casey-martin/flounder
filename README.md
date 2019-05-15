# lumpi
## ANN Chess Engine

Lumpi is an artificial neural network (ANN) based chess evaluation tool. A neural net is trained on positions scored by a
conventional chess engine. Given a particular board state, the model predicts the corresponding centipawn value.

Chess analysis engine used for generation of training data:
Stockfish (https://stockfishchess.org/)

Parallelism Tools:
GNU-parallel (https://www.gnu.org/software/parallel/)

## Preliminary Findings:
### Filter your training data:
![Distribution of Stockfish evaluations from games between master-level players.](https://github.com/casey-martin/lumpi/blob/master/figures/stockfish_eval_dist.png)  

* Increase evenness of the distribution of the training labels.
  * Over the course of many games between two master-level players, >70% of moves will have roughly an even board position (  centipawn difference ~ 0). Training on this full dataset results in the network minimizing error by randomly guessing values around 0. Reducing the relative proportion of roughly equal positions mitigates this effect. However, this means we must throw out a majority of the training data for network initialization. 
<br/><br/>
### Model performance after 30 epochs training (1.4M positions) on downsampled data.
![Initial network performance after fitting on downsampled training data.](https://github.com/casey-martin/lumpi/blob/master/figures/cp-0020.ckpt.png)  
<br/><br/>
  Downsampling is successful and model has predictive power. Initial results are promising. Will continue with a larger dataset (~10M positions).

## TODO:
### Immediate:
* Memory Usage:
  * Implement tensorflow's dataset pipeline in evaluator_training.py so training data isn't loaded into RAM all at once.
  * Redo downsample_training script so board states aren't loaded into memory. 

### Long Term:
* Chess 960 compatibility.
  * I suspect that increasing the diversity of board states in the training data will help the network learn more generalizable policies.
* Game tree search.
  * Beta-pruning and MCTS implementation? Python performance will be a limiting factor. Cython implementation of python-chess? 
* Amplification reinforcement
  * Train the evaluation function to approximate the output of the evaluation function coupled with a tree search.
* UCI support (http://wbec-ridderkerk.nl/html/UCIProtocol.html)

## Sabatelli et. al.
https://pdfs.semanticscholar.org/5171/32097f4de960f154185a8a8fec4178a15665.pdf  
https://github.com/paintception/DeepChess

* Achieved an MSE of 0.0016 on ~3M position dataset. 
  * Did not report on how positions were selected or number of epochs for convergence. 
  * Rescaled centipawn scale to 0:1, with 0 being a board position winning for black and 1 is a win for white.
  * Did not mention the centipawn value used for mate/mate-in-N.

* Claim to have a centipawn error of 0.04, which does not correspond to its estimated FIDE ELO of ~2000.
  * With mate==10000 centipawns, an MSE of 0.0016 is an 80 centipawn error which places its ELO estimation closer to 1600. (https://chess-db.com/public/research/qualityofplay.html)
* Using a similarly sized dataset with similar network architecture (no dropout) and learning parameters, I can achieve MSE of 0.0062 on validation data (0.0051 test data) after 50 epochs of training. 
