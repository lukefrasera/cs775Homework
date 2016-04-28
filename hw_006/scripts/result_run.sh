#!/usr/bin/env bash

c_list=(
  '10'
  # '50'
  # '100'
  # '200'
  )
train_list=(
  # '1'
  '25'
  # '50'
  # '100'
  # '200'
  )

for C in ${c_list[@]};do
  for T in ${train_list[@]};do
    ./scripts/classifier.py $C $T > "result_${C}_${T}_.txt" &
  done
done
exit 0
