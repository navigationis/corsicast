#! /usr/bin/bash

CORSIKA_DATA=`pwd`/data/

pushd `dirname $CORSIKA_EXE`

for infile; do
    ifb=`basename $infile`
    log=$CORSIKA_DATA/${ifb%in}lst
    $CORSIKA_EXE < $infile > $log &
done

