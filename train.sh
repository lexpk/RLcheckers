#python -m init --size 10_000_000 --name_out init.ckpt --epochs 1
#python -m play --player1 init.ckpt --player2 random --positions 8
python -m stage --size 64 --movetime 1 --name_in init.ckpt --name_out stage1.ckpt --stage 1 --epochs 1
python -m play --player1 stage1.ckpt --player2 init.ckpt --positions 8
for i in {1..8}
do
  python -m stage --size 64 --movetime 1 --name_in stage${i}.ckpt --name_out stage$(($i+1)).ckpt --stage $(($i+1)) --epochs 1
  python -m play --player1 stage$(($i+1)).ckpt --player2 stage${i}.ckpt --positions 8
done