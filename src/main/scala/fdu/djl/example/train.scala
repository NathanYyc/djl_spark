package fdu.djl.example

import ai.djl.Model
import ai.djl.basicdataset.Mnist
import ai.djl.ndarray.types.Shape
import ai.djl.nn.{Blocks, SequentialBlock}
import ai.djl.nn.core.Linear
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.loss.Loss
import ai.djl.training.{DefaultTrainingConfig, Trainer}
import fdu.djl.example.App.{input_size, output_size}
import org.apache.spark.sql.SparkSession

object train extends App {
  val spark = SparkSession.builder()
    .master("local[1]")
    .appName("Distributed Training DJL")
    .getOrCreate()

  val batchSize = 16
  val mnist = Mnist.builder().setSampling(batchSize, true).build()
  val input_size: Long = 28 * 28
  val output_size = 10
  def getBlock = () => {
    val block = new SequentialBlock
    block.add(Blocks.batchFlattenBlock(input_size))
    block.add(Linear.builder.setUnits(64).build)
    block.add(Linear.builder.setUnits(output_size).build)
  }

  def getConfig = () => {
    new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss).addEvaluator(new Accuracy)
  }

  def getShape = () => {
    new Shape(1, 28*28)
  }

  lazy val config: DefaultTrainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss).addEvaluator(new Accuracy)

  val (model, trainer) = DistributedTrain.fit(getBlock, 5, mnist, mnist, spark, getConfig, getShape, 5)
  trainer.getTrainingResult
}
