echo 'Assuming this is Ubuntu 16.04 LTS'
echo 'Installing tensorflow...'
sudo apt-get install python-pip python-dev python-virtualenv 
virtualenv --system-site-packages tensorflow
source tensorflow/bin/activate
sudo pip install --upgrade tensorflow
