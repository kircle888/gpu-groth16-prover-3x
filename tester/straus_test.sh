cd data
rm MNT6753-output_cuda
../cuda_prover_piecewise MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output_cuda straus MNT6753_preprocessed
sha256sum MNT6753-output MNT6753-output_cuda
cd ..
cd data
rm MNT4753-output_cuda
../cuda_prover_piecewise MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output_cuda straus MNT4753_preprocessed
sha256sum MNT4753-output MNT4753-output_cuda
cd ..