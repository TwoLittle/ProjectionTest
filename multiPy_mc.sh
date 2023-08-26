#!/bin/bash
c=1
while [ $c -le $2 ]
do
    qsub << EOJ

#!/bin/bash

#PBS -A lmc8_c_g_sc_default
#PBS -N $1$c
#PBS -l nodes=1
#PBS -l walltime=80:00:00
#PBS -o $PWD/logs/$c.log

cd $PWD
python $1.py $c

EOJ

((c++))
done
