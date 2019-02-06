#!/bin/bash

if [ ! -d data ]; then
  mkdir data;
fi

cd data

# Adult dataset from UCI (https://archive.ics.uci.edu/ml/datasets/adult)
echo "Adult"
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

# Communities and Crime from UCI (http://archive.ics.uci.edu/ml/datasets/communities+and+crime+unnormalized)
echo "Communities and Crime"
wget -q http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt -O communities.data

echo "Delta"
# Delta dataset (http://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html)
wget -q http://www.dcc.fc.up.pt/~ltorgo/Regression/delta_ailerons.tgz
tar xzf delta_ailerons.tgz
rm delta_ailerons.tgz
mv Ailerons/delta_ailerons.data .
mv Ailerons/delta_ailerons.domain .
rmdir Ailerons
