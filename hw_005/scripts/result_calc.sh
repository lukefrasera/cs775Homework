gamma_list=(
  '1000'
  '100'
  '10'
  '1'
  )
compute_list=(
  '1'
  '5'
  '10'
  )

for gamma in ${gamma_list[@]};do
  for compute in ${compute_list[@]};do
    ./scripts/neural_network.py $1 $compute $gamma &
  done
done
exit 0
