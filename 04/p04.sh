#! /bin/sh

if [ "$#" -lt "2" ]
	then
	echo "USE: p04.sh <n_threads> <res>" >&2
	exit 1
else
	n_thr="$1"
	res="$2"
fi

export OMP_NUM_THREADS=$n_thr
python Fractal.py -0.7489 -0.74925 0.1007 $res 100 $n_thr # | column -t -s ';'
