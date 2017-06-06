# create data dir that is git ignored
mkdir -p data
mkdir -p data/wsj

# get wsj from our google drive
echo wsj_test.dat
wget https://doc-10-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/jks947oimedsgnvfj33l6u1ppbgk5sc5/1496728800000/13452926836532438358/*/0B38oulL41I1NVGlqZDdkcmZ0X1k?e=download
mv 0B38oulL41I1NVGlqZDdkcmZ0X1k?e=download data/wsj/wsj_test.dat

echo wsj_train.dat
wget https://doc-0s-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/kmp721pdsit78aplbp408cev6k52r45q/1496728800000/13452926836532438358/*/0B38oulL41I1NRTg5eER0ZHU2ME0?e=download
mv 0B38oulL41I1NRTg5eER0ZHU2ME0?e=download data/wsj/wsj_train.dat

echo wsj_dev.dat
wget https://doc-0g-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/1osdargmpvelb8uo7m5ic8bf6gesvl9q/1496728800000/13452926836532438358/*/0B38oulL41I1NV2RjUDJlRjlxWUk?e=download
mv 0B38oulL41I1NV2RjUDJlRjlxWUk?e=download data/wsj/wsj_dev.dat


