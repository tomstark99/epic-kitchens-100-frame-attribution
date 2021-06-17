mkdir epic-100/frames
touch epic-100/frames/.gitkeep
cd epic-100/frames
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts
cd epic-kitchens-download-scripts
python epic_downloader.py --rgb-frames --participants 1 --output-path $PWD
mv EPIC-KITCHENS/P01 ../
cd ..
rm -rf epic-kitchens-download-scripts
cd P01
mv rgb_frames/*.tar ./
rm -rf rgb_frames
cd ../../..
echo DONE
