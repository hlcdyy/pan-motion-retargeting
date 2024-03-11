gdown -O data_preprocess/Mixamo/test_set.tar.gz2 1_849LvuT3WBEHktBT97P2oMBzeJz7-UP
tar -jxvf data_preprocess/Mixamo/test_set.tar.gz2 -C data_preprocess/Mixamo
rm data_preprocess/Mixamo/test_set.tar.gz2

gdown -O data_preprocess/Mixamo/train_set.tar.gz2 1BYH2t5XMGWwnu5coftehU0rTXupQvFLg
tar -jxvf data_preprocess/Mixamo/train_set.tar.gz2 -C data_preprocess/Mixamo
rm data_preprocess/Mixamo/train_set.tar.gz2

mkdir -p data_preprocess/Lafan1_and_dog/Lafan1
wget -O data_preprocess/Lafan1_and_dog/lafan1.zip https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip
# Extract to data_preprocess/Lafan1_and_dog/Lafan1
unzip data_preprocess/Lafan1_and_dog/lafan1.zip -d data_preprocess/Lafan1_and_dog/Lafan1

mkdir -p data_preprocess/Lafan1_and_dog/DogSet
wget -O data_preprocess/Lafan1_and_dog/dogset.zip https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
# Extract to data_preprocess/Lafan1_and_dog/DogSet
unzip data_preprocess/Lafan1_and_dog/dogset.zip -d data_preprocess/Lafan1_and_dog/DogSet

python data_preprocess/Lafan1_and_dog/extract.py
