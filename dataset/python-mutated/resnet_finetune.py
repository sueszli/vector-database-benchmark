from optparse import OptionParser
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import col, udf
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image import *
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.torch import TorchModel, TorchLoss
from bigdl.dllib.nnframes import *
from bigdl.dllib.keras.metrics import Accuracy
from bigdl.dllib.utils.utils import detect_conda_env_name

class CatDogModel(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CatDogModel, self).__init__()
        self.features = torchvision.models.resnet18(pretrained=True)
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.dense1 = nn.Linear(1000, 2)

    def forward(self, x):
        if False:
            return 10
        self.features.eval()
        x = self.features(x)
        x = F.log_softmax(self.dense1(x), dim=1)
        return x
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--executor-cores', type=int, dest='cores', default=4, help='number of executor cores')
    parser.add_option('--num-executors', type=int, dest='executors', default=16, help='number of executors')
    parser.add_option('--executor-memory', type=str, dest='executorMemory', default='30g', help='executor memory')
    parser.add_option('--driver-memory', type=str, dest='driverMemory', default='30g', help='driver memory')
    parser.add_option('--deploy-mode', type=str, dest='deployMode', default='local', help='deploy mode, local, yarn-client or yarn-cluster')
    (options, args) = parser.parse_args(sys.argv)
    hadoop_conf = os.environ.get('HADOOP_CONF_DIR')
    sc = init_orca_context(cluster_mode=options.deployMode, hadoop_conf=hadoop_conf)
    model = CatDogModel()
    zoo_model = TorchModel.from_pytorch(model)

    def lossFunc(input, target):
        if False:
            for i in range(10):
                print('nop')
        return nn.NLLLoss().forward(input, target.flatten().long())
    zoo_loss = TorchLoss.from_pytorch(lossFunc)
    image_path = sys.argv[1]
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1)
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 0.0, DoubleType())
    labelDF = imageDF.withColumn('name', getName(col('image'))).withColumn('label', getLabel(col('name'))).cache()
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])
    featureTransformer = ChainedPreprocessing([RowToImageFeature(), ImageCenterCrop(224, 224), ImageChannelNormalize(123.0, 117.0, 104.0, 255.0, 255.0, 255.0), ImageMatToTensor(), ImageFeatureToTensor()])
    classifier = NNClassifier(zoo_model, zoo_loss, featureTransformer).setLearningRate(0.001).setBatchSize(16).setMaxEpoch(1).setFeaturesCol('image').setCachingSample(False).setValidation(EveryEpoch(), validationDF, [Accuracy()], 16)
    catdogModel = classifier.fit(trainingDF)
    shift = udf(lambda p: p - 1, DoubleType())
    predictionDF = catdogModel.transform(validationDF).withColumn('prediction', shift(col('prediction'))).cache()
    correct = predictionDF.filter('label=prediction').count()
    overall = predictionDF.count()
    accuracy = correct * 1.0 / overall
    predictionDF.sample(False, 0.1).show()
    print('Validation accuracy = {}, correct {},  total {}'.format(accuracy, correct, overall))