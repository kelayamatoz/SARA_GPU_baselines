#!/bin/bash

#get all execution files in ./bin
files=./bin/*
#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
EXECUTION=""
#iterate over all file names to get the largest version number
for x in $arr;
do
    output=$(grep -o "[0-9]\.[0-9]" <<< "$x")
    if [ "$output" \> "$max_ver_num" ]; then
        EXECUTION=$x
    fi
done

#put OS and Device type here
SUFFIX="GUNROCK_v1-0-0_Volta100"
DATADIR="$HOME/pldi20/v100/grock/dataset/large"

ORG_OPTIONS=""
ORG_OPTIONS="$ORG_OPTIONS --num-runs=10"
ORG_OPTIONS="$ORG_OPTIONS --validation=each"
ORG_OPTIONS="$ORG_OPTIONS --device=0"
ORG_OPTIONS="$ORG_OPTIONS --jsondir=./eval/$SUFFIX"
ORG_OPTIONS="$ORG_OPTIONS --src=random"
ORG_OPTIONS="$ORG_OPTIONS --64bit-SizeT=false,true"
ORG_OPTIONS="$ORG_OPTIONS --64bit-VertexT=false,true"
ORG_OPTIONS="$ORG_OPTIONS --idempotence=false,true"
ORG_OPTIONS="$ORG_OPTIONS --mark-pred=false,true"
ORG_OPTIONS="$ORG_OPTIONS --direction-optimized=true,false"
ORG_OPTIONS="$ORG_OPTIONS --advance-mode=LB_CULL,LB,TWC"

NAME[0]="roadNet-CA"        && OPT_UDIR[61]="--do-a=-1.0                  --queue-factor=0.01" # && I_SIZE_DIR[29]="0.01" && I_SIZE_UDIR[29]="0.01" 

mkdir -p eval/$SUFFIX

for i in {0..1}; do
    if [ "${NAME[$i]}" = "" ]; then
        continue
    fi

    for undirected in "true" "false"; do
        OPTIONS=$ORG_OPTIONS
        MARKS=""

        if [ "${GRAPH[$i]}" = "" ]; then
            GRAPH_="market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx"
        else
            GRAPH_="${GRAPH[$i]}"
        fi

        if [ "$undirected" = "true" ]; then
            OPTIONS_="$OPTIONS ${OPT_UDIR[$i]} --undirected=true"
            MARKS="UDIR"
        else
            OPTIONS_="$OPTIONS ${OPT_DIR[$i]}"
            if [ "${OPT_DIR[$i]}" = "" ]; then
                continue
            fi
            MARKS="DIR"
        fi

        echo $EXECUTION $GRAPH_ $OPTIONS_ "> ./eval/$SUFFIX/${NAME[$i]}.${MARKS}.txt"
             $EXECUTION $GRAPH_ $OPTIONS_  > ./eval/$SUFFIX/${NAME[$i]}.${MARKS}.txt
        sleep 1
    done
done
