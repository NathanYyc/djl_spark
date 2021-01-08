package fdu.djl.example

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream, DataOutputStream, IOException}

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.{Block, ParameterList}
import ai.djl.training.{Trainer, TrainingConfig}
import ai.djl.training.dataset.{Batch, Dataset}
import ai.djl.translate.TranslateException
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

object DistributedTrain {
  /**
   * run an all reduced distributed train with the given SparkSession.
   * Notice that users should define a function that returns {@link Block},
   * a function returns {@link TrainingConfig} & a function that returns
   * {@link Shape} because these objects are unserializable.
   *
   * @param numEpoch            the number of epochs to train
   * @param trainingDataset     the dataset to train on
   * @param validateDataset     the dataset to validate against. Can be null for no validation
   * @param spark               the spark session to train on
   * @param getBlock            a function that returns the block to train
   * @param getConfig           a function that returns train config of the model
   * @param getShape            a function that returns the input shape of the model
   * @param partitionNum         define the number of partitions spark will create
   * @throws IOException        for various exceptions depending on the dataset
   * @throws TranslateException if there is an error while processing input
   */
    @throws[IOException]
    @throws[TranslateException]
    def fit(getBlock: () => Block,
            numEpoch: Int,
            trainingDataset: Dataset,
            validateDataset: Dataset,
            spark: SparkSession,
            getConfig: () => TrainingConfig,
            getShape: () => Shape,
            partitionNum: Int): (Model, Trainer) = {
      val originModel = Model.newInstance("")
      val block = getBlock()
      originModel.setBlock(block)
      val config = getConfig()
      val trainer = originModel.newTrainer(config)
      val shape = getShape()
      //val uninitializedBlock = block.clone().asInstanceOf[Block]
      val dataRdd = parallelize(trainer, trainingDataset, spark, partitionNum)

      trainer.initialize(shape)
      for(i <- 0 to numEpoch) {
        val (parameterByteList, batchNum)  = distributedTraining(trainer, getBlock, dataRdd, getConfig, getShape, spark)
        updateParameters(trainer.getModel, parameterByteList.toArray, batchNum)
      }

      (originModel,trainer)
    }

  /**
   * parallelize the dataset into rdd, each batch of the data is stored in a Tuple2 object
   * the first element of tuple2 is a serialized data NDList, the second element of the returned
   * value is a serialized label NDList
   *
   * @param trainer             the trainer to train for
   * @param trainingDataset     the dataset to parallelize
   * @param spark               the spark session to parallelize the data on
   * @param partitionNum        the number of partitions spark will create
   * @return                    parallelized dataset
   * @throws IOException        for various exceptions depending on the dataset
   * @throws TranslateException if there is an error while processing input
   */
  @throws[IOException]
  @throws[TranslateException]
  private def parallelize(trainer: Trainer,
                          trainingDataset: Dataset,
                          spark: SparkSession,
                          partitionNum: Int): RDD[(Array[Byte], Array[Byte])] = {
    val result = new ArrayBuffer[(Array[Byte], Array[Byte])]
    trainingDataset.getData(trainer.getManager).forEach((batch: Batch) => {
      def foo(batch: Batch) = {
        val data: Array[Byte] = batch.getData.encode()
        val label: Array[Byte] = batch.getLabels.encode
        val tuple: (Array[Byte], Array[Byte]) = new Tuple2[Array[Byte], Array[Byte]](data, label)
        result += tuple
      }

      foo(batch)
    })
    spark.sparkContext.parallelize(result, partitionNum)
  }

  /**
   * run one distributed training epoch and return the parameters of the model
   *
   * @param trainer        the trainer on which the training will processed
   * @param dataRDD        the parallelized dataset
   * @param getBlock       a function that returns a new {@link Block} object
   * @param getConfig      a function that returns a new {@link TrainingConfig} object
   * @param getShape       a function that returns a new {@link Shape} object
   * @param spark          the spark session on which the training work will run
   * @return               encoded parameters & batch numbers
   */
  private def distributedTraining(trainer: Trainer,
                                  getBlock: () => Block,
                                  dataRDD: RDD[(Array[Byte], Array[Byte])],
                                  getConfig: () => TrainingConfig,
                                  getShape: () => Shape,
                                  spark: SparkSession)
  : (ArrayBuffer[Map[String, Array[Byte]]], Int) = {
    val model = trainer.getModel

    val byteParameter = encodeParameters(model.getBlock.getParameters)
    val broadcastedParameter = spark.sparkContext.broadcast(byteParameter)

    val (parameterByte, batchCount) = dataRDD.treeAggregate(new ArrayBuffer[Map[String, Array[Byte]]], 0)(
      seqOp = (c, v) => {
        val block = getBlock()
        val broadcastedModel = Model.newInstance("broadcastedModel")
        broadcastedModel.setBlock(block)
        val trainingConfig = getConfig()
        val broadTrainer = broadcastedModel.newTrainer(trainingConfig)
        val shapes = getShape()

        broadTrainer.initialize(shapes)

        val paramList = broadcastedModel.getBlock.getParameters
        val itr = broadcastedParameter.value.iterator

        while(itr.hasNext){
          val tempMap = itr.next()

          val bis = new ByteArrayInputStream(tempMap._2)
          val dis = new DataInputStream(bis)

          paramList.get(tempMap._1).load(broadcastedModel.getNDManager, dis)

          dis.close()
          bis.close()
        }

        val gradientCollector = broadTrainer.newGradientCollector()

        val data = NDList.decode(broadTrainer.getManager, v._1)
        val label = NDList.decode(broadTrainer.getManager, v._2)
        val pred = broadTrainer.forward(data, label)
        val loss = broadTrainer.getLoss.evaluate(label, pred)

        gradientCollector.backward(loss)
        broadTrainer.step()

        c._1 += encodeParameters(broadcastedModel.getBlock.getParameters)

        gradientCollector.close()
        broadTrainer.close()
        broadcastedModel.close()

        (c._1, c._2 + 1)
      },
      combOp = (c, v) => {
        (c._1 ++ v._1, c._2 + v._2)
      }
    )
    (parameterByte, batchCount)
  }

  /**
   * encode the {@link ParameterList} to byte Arrays
   *
   * @param parameterList the parameter list to be encoded
   * @return              the encoded parameter list
   */
  private def encodeParameters(parameterList: ParameterList): Map[String, Array[Byte]] = {

    var result = Map[String, Array[Byte]]()

    parameterList.forEach(param => {
      val bos = new ByteArrayOutputStream()
      val dos = new DataOutputStream(bos)
      result += (param.getKey -> {
        param.getValue.save(dos)
        bos.toByteArray
      })
      dos.close()
      bos.close()
    })
    result
  }

  /**
   * update the parameters in the model from a set of encoded parameter lists
   *
   * @param model     the model in which the parameters will be update
   * @param byteParam Array of parameter lists, each element stands for one parameter list
   * @param batchNum  the num of batches in training
   */
  private def updateParameters(model: Model,
                               byteParam: Array[Map[String, Array[Byte]]],
                               batchNum: Int): Unit = {
    val paramList = model.getBlock.getParameters


    for(params <- byteParam){
      val itr = params.iterator
      while(itr.hasNext){
        val tempMap = itr.next()

        val bis = new ByteArrayInputStream(tempMap._2)
        val dis = new DataInputStream(bis)

        val oldArray = paramList.get(tempMap._1).getArray
        paramList.get(tempMap._1).load(model.getNDManager, dis)
        paramList.get(tempMap._1).setArray(oldArray.add(paramList.get(tempMap._1).getArray))
      }
    }
    paramList.forEach(param => {
      param.getValue.setArray(param.getValue.getArray.div(batchNum))
    })

  }
}
