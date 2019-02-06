#!/bin/bash

if [ ! -d data ]; then
  mkdir data;
fi

cd data

# Adult dataset from UCI (https://archive.ics.uci.edu/ml/datasets/adult)
if [ ! -f adult.data ]; then
    echo "Adult"
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
fi

# Communities and Crime from UCI (http://archive.ics.uci.edu/ml/datasets/communities+and+crime)
if [ ! -f communities.data ]; then
    echo "Communities and Crime"
    wget -q http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data
    wget -q http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names
fi

# Delta dataset (http://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html)
if [ ! -f delta_ailerons.data ]; then
    echo "Delta"
    wget -q http://www.dcc.fc.up.pt/~ltorgo/Regression/delta_ailerons.tgz
    tar xzf delta_ailerons.tgz
    rm delta_ailerons.tgz
    mv Ailerons/delta_ailerons.data .
    mv Ailerons/delta_ailerons.domain .
    rmdir Ailerons
fi

# Comp Activ (http://www.cs.toronto.edu/~delve/data/datasets.html)
if [ ! -f comp_activ.data ]; then
    echo "Comp Activ"
    wget -q ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/comp-activ.tar.gz
    tar xzf comp-activ.tar.gz
    gunzip comp-activ/Dataset.data.gz
    mv comp-activ/Dataset.data comp_activ.data
    mv comp-activ/Dataset.spec comp_activ.spec
    rm -rf comp-activ comp-activ.tar.gz
fi

# Spambase (https://archive.ics.uci.edu/ml/datasets/spambase)
if [ ! -f spambase.data ]; then
    echo "Spambase"
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names
fi

# Magic (https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)
if [ ! -f magic.data ]; then
    echo "Magic"
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data -O magic.data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.names -O magic.names
fi

# Letter (https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/)
if [ ! -f letter.data ]; then
    echo "Letter"
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data -O letter.data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.names -O letter.names
fi

# Gisette (https://archive.ics.uci.edu/ml/datasets/Gisette)
if [ ! -f gisette_train.data ]; then
    echo "Gisette"
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_valid.data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/gisette_valid.labels
fi

# Physics (https://www.kdd.org/kdd-cup/view/kdd-cup-2004/Data)
if [ ! -f physics_train.data ]; then
    echo "Physics"
    wget -q https://www.kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz
    tar xzf data_kddcup04.tar.gz
    rm data_kddcup04.tar.gz
    mv phy_test.dat physics_test.data
    mv phy_train.dat physics_train.data
fi
