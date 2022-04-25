cd data
../generate_parameters 15 15
../main MNT4753 preprocess MNT4753-parameters
../main MNT6753 preprocess MNT6753-parameters
../main MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output
../main MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output
cd ..