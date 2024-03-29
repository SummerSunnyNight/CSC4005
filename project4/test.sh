mkdir -p build
cd src

# simple
g++ simple_ml_ext.cpp softmax_classifier.cpp -O4 -o ../build/softmax
# pgc++ -acc simple_ml_ext.cpp simple_ml_openacc.cpp softmax_classifier_openacc.cpp -o ../build/softmax_openacc
pgc++ -acc -O4 simple_ml_openacc.cpp softmax_classifier_openacc.cpp -o ../build/softmax_openacc

# # nn
g++ simple_ml_ext.cpp nn_classifier.cpp -O4 -o ../build/nn
pgc++ -acc -fast simple_ml_openacc.cpp nn_classifier_openacc.cpp  -o ../build/nn_openacc

rm *.o

cd ..

sbatch sbatch.sh
