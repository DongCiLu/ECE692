sudo apt-get install openjdk-8-jdk
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel

git clone https://github.com/tensorflow/models.git
cd models

curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzvg cifar-10-binary.tar.gz
mv cifar-10-binary cifar-10

bazel build -c opt --config=cuda resnet/...

bazel-bin/resnet/resnet_main --train_data_path=cifar10/data_batch* \
                               --log_root=/tmp/resnet_model \
                               --train_dir=/tmp/resnet_model/train \
                               --dataset='cifar10' \
                               --num_gpus=2

# bazel-bin/resnet/resnet_main --eval_data_path=cifar10/test_batch.bin \
#                               --log_root=/tmp/resnet_model \
#                               --eval_dir=/tmp/resnet_model/test \
#                               --mode=eval \
#                               --dataset='cifar10' \
#                               --num_gpus=2
