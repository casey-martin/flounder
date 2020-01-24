<img src="https://github.com/casey-martin/flounder/blob/master/figures/flounder.svg" alt="drawing" width="150">

# Flounder
Flounder is an artificial neural network (ANN) based chess evaluation tool. Like its piscine namesake, Flounder is a bottom feeder; it learns from positions labeled by an existing (and likely much better) chess engine. Given a particular board state, the model predicts the corresponding evaluation. The goal of this training schematic is to create an evaluation function that approximates the tree search of the engine that is training Flounder.

Chess analysis engine used for generation of training data:
Stockfish (https://stockfishchess.org/)

Parallelism Tools:
GNU-parallel (https://www.gnu.org/software/parallel/)

## Preliminary Findings:
### Filter your training data:
![Distribution of Stockfish evaluations from games between master-level players.](https://github.com/casey-martin/flounder/blob/master/figures/stockfish_eval_dist.png)  

* Increase evenness of the distribution of the training labels.
  * Over the course of many games between two master-level players, >70% of moves will have roughly an even board position (  centipawn difference ~ 0). Training on this full dataset results in the network minimizing error by randomly guessing values around 0. Reducing the relative proportion of roughly equal positions mitigates this effect. However, this means we must throw out a majority of the training data for network initialization. 
<br/><br/>
### Model performance after 250 epochs training (20M positions) on downsampled data.
![Initial network performance after fitting on downsampled training data.](https://github.com/casey-martin/flounder/blob/master/figures/cp-1360.ckpt.png)  
<br/><br/>
Evaluation function coupled with alpha-beta search 3 moves deep results in flawed and highly aggressive play.

## TODO:
### High Priority:
#### 1. Network architecture:
* 256 x 20 like Leela? Pared-down?
#### 2. Memory Usage:
* Reencode board states. Predicted ~30% memory footprint of original encoding.
    * h5py_file.create_dataset("boardStates", data=np.array(data, dtype='S')). Write labels as float to separate hdf5.
    * Use of h5df will also increase read and load speeds.
#### 3. Training Scheduling:
* Create a scheduled host process to check if training folder has been populated since last training run.
    * Train for N epochs, update saved weights, and record size of training batch and network performance.
    * Schedule minimatch with Stockfish if there's time between batches?
    * Move processed board states and labels for archival storage.
* Create a client process that submits both a parallel_eval to a worker pool, and sets up a scheduled supervisor process that scps the labelled data to the host.
    * Supervisor will merge all newly generated board states and their corresponding labels.
    * Supervisor will scp merged board states and labels to the host machine's training folder.
    * Supervisor will then remove data on the client machine that was added to the batch of merged data.
    * If scp failure. Retain data, and next batch of data to be appended to the queue. 
    * Fail state(?) if repeated scp failure condition met, halt until connection successfully reestablished?
   

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
