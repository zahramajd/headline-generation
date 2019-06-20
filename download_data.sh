#!/bin/sh

cd data

if [ ! -f wiki.fa.vec ] ; then
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fa.vec
else
   echo "Data exists!"
fi