#在simpledata目录下直接执行本文件
../generate_parameters 15 15
../main MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output
../main MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output
../cuda_prover_piecewise MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output_cudap pippenger 7
../cuda_prover_piecewise MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output_cudap pippenger 7
../main MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output
../main MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output
../main MNT4753 preprocess MNT4753-parameters 
../main MNT6753 preprocess MNT6753-parameters
../cuda_prover_piecewise MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output_cudas straus MNT4753_preprocessed
../cuda_prover_piecewise MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output_cudas straus MNT6753_preprocessed
sha256sum MNT4753-output MNT6753-output MNT4753-output_cudap MNT6753-output_cudap MNT4753-output_cudas MNT6753-output_cudas