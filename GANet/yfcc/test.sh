cp -r ./log/main.py/train/* ./log/main.py/test/
python main.py --use_ransac=True --data_te='/data/wangyang/data_dump/yfcc-sift-2000-val.hdf5' --run_mode='test'
python main.py --use_ransac=False --data_te='/data/wangyang/data_dump/yfcc-sift-2000-val.hdf5' --run_mode='test'
mkdir ./log/main.py/test/known
mv ./log/main.py/test/*txt ./log/main.py/test/known

python main.py --use_ransac=True --data_te='/data/wangyang/data_dump/yfcc-sift-2000-test.hdf5' --run_mode='test'
python main.py --use_ransac=False --data_te='/data/wangyang/data_dump/yfcc-sift-2000-test.hdf5' --run_mode='test'
mkdir ./log/main.py/test/unknown
mv ./log/main.py/test/*txt ./log/main.py/test/unknown