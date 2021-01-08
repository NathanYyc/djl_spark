package fdu.djl.example

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream, DataOutputStream}

import ai.djl.Model
import ai.djl.basicdataset.Mnist
import ai.djl.ndarray.{NDArray, NDList}
import ai.djl.ndarray.types.Shape
import ai.djl.nn.{Blocks, ParameterList, SequentialBlock}
import ai.djl.nn.core.Linear
import ai.djl.training.{DefaultTrainingConfig, Trainer}
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.loss.Loss
import ai.djl.util.PairList
import fdu.djl.MyTranslator
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

/**
 * original code of {@link DistributedTrain}, build blocks & shapes in
 */
object App extends App {

  val input_size: Long = 28 * 28
  val output_size = 10

  val batchSize = 16
  val mnist: Mnist = Mnist.builder.setSampling(batchSize, true).build


  val block = new SequentialBlock
  block.add(Blocks.batchFlattenBlock(input_size))
  block.add(Linear.builder.setUnits(64).build)
  block.add(Linear.builder.setUnits(output_size).build)

  val model: Model = Model.newInstance("mlp")
  //Criteria[Row, Classifications]
  model.setBlock(block)

  val config: DefaultTrainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss).addEvaluator(new Accuracy) //.addTrainingListeners(TrainingListener.Defaults.logging)
  //softmaxCrossEntropyLoss is a standard loss for classification problems(new Accuracy).addTrainingListeners // Use accuracy so we humans can understand how accurate the model is

  // Now that we have our training configuration, we should create a new trainer for our model
  val trainer: Trainer = model.newTrainer(config)

  trainer.initialize(new Shape(1, 28 * 28))

  val epoch = 15
  val start_time: Long = System.currentTimeMillis


  //EasyTrain.fit(trainer, 15, mnist, null)
  val a = ArrayBuffer[(Array[Byte], Array[Byte])]()
  mnist.getData(model.getNDManager).forEach(batch => {
    val data = batch.getData.encode()
    val label = batch.getLabels.encode()
    val tempTuple = (data, label)
    a += tempTuple
    // a+ = (data, label)
  })


  val spark: SparkSession = SparkSession.builder()
    .appName("djl Test")
    .master("local[3]")
    .getOrCreate()

  val rdd = spark.sparkContext.parallelize(a)

  /**
   * train the model separately in different nodes.
   * data were carry to different nodes in `scala.collection.mutable.ArrayBuffer`.
   * return an rdd that carry the separated & serialized data {@link a}
   */
  val parameterRdd = rdd.mapPartitions(batchList => {
    val broadcastModel = Model.newInstance("broadcastModel")
    val block = new SequentialBlock
    block.add(Blocks.batchFlattenBlock(input_size))
    block.add(Linear.builder.setUnits(64).build)
    block.add(Linear.builder.setUnits(output_size).build)

    val model: Model = Model.newInstance("mlp")
    //Criteria[Row, Classifications]
    model.setBlock(block)


    val Config: DefaultTrainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss).addEvaluator(new Accuracy) //.addTrainingListeners(TrainingListener.Defaults.logging)
    val broadcastTrainer = model.newTrainer(Config)
    broadcastTrainer.initialize(new Shape(1, 28 * 28))

    val collector = broadcastTrainer.newGradientCollector()

    for (i <- 1 to epoch) {

      for (batch <- batchList) {

        val data = NDList.decode(broadcastTrainer.getManager, batch._1)
        val label = NDList.decode(broadcastTrainer.getManager, batch._2)

        val preds = broadcastTrainer.forward(data, label)
        val lossVal = broadcastTrainer.getLoss.evaluate(label, preds)

        collector.backward(lossVal)
        broadcastTrainer.step()
      }
    }
    collector.close()
    broadcastTrainer.close()
    broadcastModel.close()

    val a: ArrayBuffer[ParameterList] = new ArrayBuffer[ParameterList]()
    a += model.getBlock.getParameters
    a.iterator
  })

  print(parameterRdd.foreach(print(_)))

  /**
   * aggregate serialized parameters into {@link scala.collection.mutable.ArrayBuffer}
   * in form of {@link scala.collection.mutable.ArrayBuffer[ Array[ Byte ]}
   * The order of parameters are definitive while the order of different parameter sets are random
   *
   * return serialized parameter set {@link parameterByte}
   */
  val (parameterByte, miniBatchSize) = parameterRdd.treeAggregate(new ArrayBuffer[Map[String, Array[Byte]]], 0)(
    seqOp = (c, v) => {
      var byteMap = Map[String, Array[Byte]]()

      v.forEach(param => {
        val bos = new ByteArrayOutputStream()
        val dos = new DataOutputStream(bos)
        param.getValue.save(dos)
        byteMap += (param.getKey -> bos.toByteArray)
      })

      c._1 += byteMap
      (c._1, c._2 + 1)
    },
    combOp = (c, v) => {
      (c._1 ++ v._1, c._2 + v._2)
    })

  val parameterList = new ArrayBuffer[PairList[String, NDArray]]
  val itr = parameterByte.iterator
  var pairList = new PairList[String, NDArray]()
  while (itr.hasNext) {
    val map = itr.next()
    val mapItr = map.iterator

    val paramList = block.getParameters
    while (mapItr.hasNext) {
      val next = mapItr.next()

      val bis = new ByteArrayInputStream(next._2)
      val dis = new DataInputStream(bis)

      val oldParam = paramList.get(next._1).getArray
      paramList.get(next._1).load(model.getNDManager, dis)
      paramList.get(next._1).setArray(oldParam.add(paramList.get(next._1).getArray))
    }
  }
  block.getParameters.forEach(param => {
    param.getValue.setArray(param.getValue.getArray.div(miniBatchSize))
  })


  mnist.getData(model.getNDManager).forEach(batch => {
    val myTranslator = new MyTranslator
    val predictor = model.newPredictor(myTranslator)
    println(predictor.predict(batch.getData))
  })
}








