#!/bin/bash

BASEOPTION="--quick --iteration-num=1"
BASEFLAG=""
EXECUTION="./bin/test_pr_10.0_x86_64"
DATADIR="$HOME/pldi20/v100/grock/dataset/large"


OPTION[1]="" && FLAG[1]=".undir.unnormalized.nocompensate"
OPTION[2]=" --normalized" && FLAG[2]=".default.normalized.nocompensate"
OPTION[3]=" --normalized --compensate" && FLAG[3]=".default.normalized.compensate"
OPTION[4]=" --normalized --undirected" && FLAG[4]=".undir.normalized.nocompensate"
OPTION[5]=" --normalized --undirected --compensate" && FLAG[5]=".undir.normalized.compensate"

OPTION[6]="" && FLAG[6]=".32bit_Value"
OPTION[7]=" --64bit-Value" && FLAG[7]=".64bit_Value"

OPTION[8]="" && FLAG[8]=".32bit_VertexId"
OPTION[9]=" --64bit-SizeT" && FLAG[9]=".64bit_SizeT"
OPTION[10]=" --64bit-VertexId" && FLAG[10]=".64bit_VertexId"

OPTION[11]="" && FLAG[11]=".DEF"
OPTION[12]=" --traversal-mode=LB" && FLAG[12]=".LB"
OPTION[13]=" --traversal-mode=TWC" && FLAG[13]=".TWC"
OPTION[14]=" --traversal-mode=LB_LIGHT" && FLAG[14]=".LB_LIGHT"

TEST_NAME="delaunay_n20"

for d in {1..4}
do
    SUFFIX="ubuntu14.04_K40cx${d}"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done

    for o1 in {1..5}; do for o2 in 6; do for o3 in {8..10}; do for o4 in {11..14}; do
    OPTIONS=${BASEOPTION}${OPTION[${o1}]}${OPTION[${o2}]}${OPTION[${o3}]}${OPTION[${o4}]}
    FLAGS=${BASEFLAG}${FLAG[${o1}]}${FLAG[${o2}]}${FLAG[${o3}]}${FLAG[${o4}]}

    for i in {0..14}
    do
        if [ "${o4}" -eq "13" ]; then # && [ "$i" -le "4" ]; then
            continue
        fi

        echo $EXECUTION market $DATADIR/$TEST_NAME/$TEST_NAME.mtx $OPTIONS --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/$TEST_NAME${FLAGS}.txt"
             $EXECUTION market $DATADIR/$TEST_NAME/$TEST_NAME.mtx $OPTIONS --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/$TEST_NAME${FLAGS}.txt
        sleep 1
    done

    done; done; done; done
done

