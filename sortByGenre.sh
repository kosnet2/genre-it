#!/bin/bash

mkdir -p fma_small_sorted
cd fma_small
for dir in $(ls) ; do
   cd $dir
   for file in $(ls) ; do
		genre=$(ffprobe $file 2>&1 | grep genre | cut -d ':' -f 2 | xargs echo -n) # Better read genre_top from metadata csv
		new_dir='../../fma_small_sorted/'
		new_dir+=$genre
		mkdir -p $new_dir # Watch out for spaces!
		cp $file "$new_dir/$file"
   done
   cd ..
done
