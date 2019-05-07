parallel -k --lb 'python stockfish_evaluator.py --pgn ./pgns/ficsgamesdb_201901_blitz_nomovetimes_65829.pgn \
    --outdir ./pgns/ficsgamesdb_scored/ \
    --job {1} --nprocs {2}' ::: {0..7} ::: 8
