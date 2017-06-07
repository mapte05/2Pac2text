# create data dir that is git ignored
mkdir -p data
mkdir -p data/wsj

# get wsj from our google drive
echo wsj_test.dat
wget -O wsj_test.dat https://doc-0g-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/tcrde8mbcfo0j28i025nckpnfdr1tciv/1496800800000/13452926836532438358/*/0B38oulL41I1NZGlYc3Z0OTNJY28?e=download
mv wsj_test.dat data/wsj

echo wsj_train.dat
wget -O wsj_train.dat https://doc-04-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/74ba472de4kdb9siphjotebqta3833th/1496800800000/13452926836532438358/*/0B38oulL41I1Nb3ppSVJVUkxQNGs?e=download
mv wsj_train.dat data/wsj

echo wsj_dev.dat
wget -O wsj_dev.dat https://doc-0s-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/ujf2ipto6rk57cecjbk72rl06up1eaq3/1496800800000/13452926836532438358/*/0B38oulL41I1NbXpSTnlzcS14LWc?e=download
mv wsj_dev.dat data/wsj

