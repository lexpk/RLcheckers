python -m play --player1 init.ckpt --player2 random --positions 4 --results results.txt
for i in {1..7}
do
    python -m play --player1 random --player2 stage${i}.ckpt --positions 4 --results results.txt
    python -m play --player1 init.ckpt --player2 stage${i}.ckpt --positions 4 --results results.txt
done
for i in {1..6}
do
    for j in $(seq $((i+1)) 7)
    do
        python -m play --player1 stage${i}.ckpt --player2 stage${j}.ckpt --positions 4 --results results.txt
    done
done
