cat $1 | cut -d ';' -f 1,6,11,12,14,15,17,18 | column -t -s ';'
