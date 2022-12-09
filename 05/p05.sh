#! /bin/sh

if [ "$#" -ne 1 ]
then
	echo "USE: p05.sh <ThPerBlk>" >&2
	exit 1
fi

./VecAdd6 10000000 $1 100000 2121
