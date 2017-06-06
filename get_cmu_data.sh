
mkdir -p data
mkdir -p data/cmu


echo Getting cmu_all6_val.dat
echo ''
wget https://doc-04-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/hd6a987ibb3ig196s578okr3m568sout/1496728800000/13452926836532438358/*/0B38oulL41I1NeGktNzJraFJ4YUE?e=download
mv 0B38oulL41I1NeGktNzJraFJ4YUE?e=download data/cmu/cmu_all6_val.dat

echo Getting cmu_all6_val_noisy.dat
echo ''
wget https://doc-0g-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/qu49tvluu2lcjfuvtt37fmuhk2e4n7tn/1496728800000/13452926836532438358/*/0B38oulL41I1NenN5MVFqenJJSkk?e=download
mv 0B38oulL41I1NenN5MVFqenJJSkk?e=download data/cmu/cmu_all6_val_noisy.dat

echo Getting cmu_all6_train.dat
echo ''
wget https://doc-0o-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/a54grhlirrgo1c376qc0t0moo60qsggc/1496728800000/13452926836532438358/*/0B38oulL41I1NbHE2b25DbXZ6TU0?e=download
mv 0B38oulL41I1NbHE2b25DbXZ6TU0?e=download data/cmu/cmu_all6_train.dat

echo Getting cmu_all6_train_noisy.dat
echo ''
wget https://doc-04-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/bk34qfgmff4g2ogrfno8n4u7mfa4inti/1496728800000/13452926836532438358/*/0B38oulL41I1NdnJWYk8wSEJ2UGc?e=download
mv 0B38oulL41I1NdnJWYk8wSEJ2UGc?e=download data/cmu/cmu_all6_train_noisy.dat

echo Getting cmu_all12_val.dat
echo ''
wget https://doc-0o-48-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/uu8shvftcoicq78rsl9e40e74n9b1g6d/1496728800000/13452926836532438358/*/0B38oulL41I1NdDlmM19lR0tEQUU?e=download
mv 0B38oulL41I1NdDlmM19lR0tEQUU?e=download data/cmu/cmu_all12_val.dat

