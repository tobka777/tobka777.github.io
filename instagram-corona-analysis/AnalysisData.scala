import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

object Main {
  val spark: SparkSession = SparkSession.builder().master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "localhost")
    .appName("First spark app").getOrCreate()
  val sc: SparkContext = spark.sparkContext

  import org.apache.spark.sql.functions._
  import spark.implicits._

  val emoji2: DataFrame = spark.read
    .parquet("data/EmojiSentiment/Emoji_Sentiment_Data_v1.0.parquet")
    .withColumn("sentimentScore", (col("Positive").cast("Int")-col("Negative").cast("Int")) / col("Occurrences").cast("Int"))

  val tmpem: Array[Row] = emoji2.rdd.collect()
  val emojiScore: Array[Double] = tmpem.map(em => em.getAs[Double]("sentimentScore"))
  val emojiEmo: Array[String] = tmpem.map(em => em.getAs[String]("Emoji"))
  val emoji: List[(String, Double)] = (emojiEmo zip emojiScore).toList;
  val broadcastEmoji = sc.broadcast(emoji)

  def main(args: Array[String]): Unit = {
    run(read())
    run_week(read())
  }

  /**
   * read json from api and convert to parquet
   */
  def read():DataFrame = {
    val files = Array("data/instagram/data_70018_01.json","data/instagram/data_250027_02.json", "data/instagram/data_430055_03.json", "data/instagram/data_710045_04.json", "data/instagram/data_840065_05.json", "data/instagram/data_950054_06.json")
    val d = files.map(r => spark.read.option("multiline",true).json(r)).reduce((df1, df2) => df1.union(df2))
    d
  }

  def progress(instagram:DataFrame): DataFrame = {
    val data = instagram
      .groupBy("date")
      .agg(
        count("date").as("posts"),
        round(sum("sentiment"), 4).as("sentimentSum"),
        round(avg("sentiment"), 4).as("sentimentScore"),
        sum("relationFakenews").as("fakenewsSum"),
        round(avg("relationFakenews"), 4).as("fakenewsScore"),
        round(avg("likes")).as("likeScore")
      )
      .sort($"date".desc)
    data
  }

  /**
   * analyse data and save result csv
   */
  def run(instagram:DataFrame): Unit = {
    progress(
      instagram
        .select($"shortcode", $"taken_at_timestamp", from_unixtime($"taken_at_timestamp", "YYYY-MM-dd").as("date"), $"text", findEmoticonsUDF(col("text")).as("sentiment"), lower(col("text")).contains("#fakenews").as("relationFakenews").cast("int"), col("edge_liked_by.count").as("likes").cast("int"))
    )
      .repartition(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("result/20200605_instagram_950054_sentiment.csv")
  }

  /**
   * analyse data for one week and save result csv
   */
  def run_week(instagram:DataFrame): Unit = {
    progress(
      instagram
        .select($"shortcode", $"taken_at_timestamp", from_unixtime($"taken_at_timestamp", "YYYY-MM-dd_HH").as("date"), $"text", findEmoticonsUDF(col("text")).as("sentiment"), lower(col("text")).contains("#fakenews").as("relationFakenews").cast("int"), col("edge_liked_by.count").as("likes").cast("int"))
        .where("date > '2020-05-25_00'")
    )
      .repartition(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("result/20200605_instagram_sentiment_20200602-20200525.csv")
  }

  def findEmoticons(s: String): Option[Double] = {
    var count = 0
    val filterEmo: (String, Double) = broadcastEmoji.value
      .reduce[(String, Double)](
        (em: (String, Double), em2: (String, Double)) => {
          if (s != null && em != null && em2._1 != null && em2._2 != null && (s contains em2._1)) {
            count += 1
            ("", em2._2 + em._2)
          }
          else {
            em
          }
        }
      )
    var sentiment = 0.0
    if (filterEmo != null)
      sentiment = (math rint (filterEmo._2 / (count+1)) * 10000) / 10000
    Some(sentiment)
  }

  val findEmoticonsUDF = udf[Option[Double], String](findEmoticons)
}
