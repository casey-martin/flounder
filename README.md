<img src="https://github.com/casey-martin/flounder/blob/master/figures/flounder.svg" alt="drawing" width="150">

# flounder:
 1. *(v) to proceed clumsily or ineffectually*
 2. *(n) a European, marine flatfish, Platichthys flesus, used for food.*
 
Flounder is an artificial neural network (ANN) based chess evaluation tool. Like its piscine namesake, Flounder is a bottom feeder; it learns from positions labeled by an existing (and likely much better) chess engine. Given a particular board state, the model predicts the corresponding evaluation. The goal of this training schematic is to create an evaluation function that approximates the tree search of the engine that is training Flounder.

Chess analysis engine used for generation of training data:
Stockfish (https://stockfishchess.org/)

Parallelism Tools:
GNU-parallel (https://www.gnu.org/software/parallel/)

## Preliminary Findings:
### Limitations and Modifications:
I was unable to replicate the performance of Sabatelli et al's model, which was trained on 3M positions.
* Flounder was initially trained on 20M positions with a learning rate of 0.001, nesterov momentum = 0.7, as per  
  Sabatelli et al. Model had prohibitively low convergence. 
  * Increasing learning rate to 0.01 allowed for feasible training. 20M positions proved to be insufficient and
    the training dataset was increased to ~500M positions.
* Weighting the rarer and more extreme positions resulted in greatly improved performance.

### Model performance on unweighted validation data after weighted training.
![alt text](https://github.com/casey-martin/flounder/blob/master/figures/sample_weight_effects.svg)

Evaluation function coupled with alpha-beta search 3 moves deep results in flawed and highly aggressive play.

## TODO:
### High Priority:
#### 1. PyPy
* install minimal packages via PyPy for increased game-tree search performance.
#### 2. Training Scheduling:
* Create a scheduled host process to check if training folder has been populated since last training run.
    * Train for N epochs, update saved weights, and record size of training batch and network performance.
    * Schedule minimatch with Stockfish if there's time between batches?
    * Move processed board states and labels for archival storage.
* Create a client supervisor process that creates a queue of jobs for a worker pool and write retrieved results to file. Supervisor will also scp the labeled data to the host every hour and remove the file once transferred.
    * If scp failure. Retain data, and next batch of data to be appended to the queue. 
    * Fail state(?) if repeated scp failure condition met, halt until connection successfully reestablished?
   

### Long Term Flounder:
* Game tree search.
  * Beta-pruning is very slow. Need efficient way of exploring the gametree and sending data to the gpu in large chunks.
  * Initial breadth first search, followed by beta-pruning?
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

* Claim to have an evaluation error of 0.04 pawns, which does not correspond to its estimated FIDE ELO of ~2000.
  * Is this the result of disproprotionate effect of the tails? Misevaluation of blunders, i.e. ruin from 'rare' events.
  * With mate==10000 centipawns, an MSE of 0.0016 is an 80 centipawn error which places its ELO estimation closer to 1600. (https://chess-db.com/public/research/qualityofplay.html)

# Future Engines:
* Movement policy output. Train from high quality engine play.
* Create hash for position:{possible move vector} where each element (move) is a probability of choice.
  * Weight by outcome of game and engine elo
* Couple with board evaluation ala Flounder to make an alpha/leela-like network.
