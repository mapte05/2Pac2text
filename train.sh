# bash script that encapsulates control flow for training the model
echo 'starting training'

# run with printing predicted sequence versus actual
# python model.py --save_every 1 --print_every 1

# run without printing predicted
mkdir -p saved_models
python model.py --save_every 1

